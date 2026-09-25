import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .backends import Encoder, Index
from .common import model_root, normalize_entity, synchronize, unit
from .models import ContinuousPlan, Operators, Proposal, ValueModel
from .query import QueryRealizer, query_prompt
from .training import load_network


def fuse(rankings, constant=60):
    scores = {}
    for ranking in rankings:
        for rank, identity in enumerate(dict.fromkeys(ranking), 1):
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (constant + rank)
    return sorted(scores, key=lambda identity: (-scores[identity], identity))


@dataclass
class State:
    question: str
    q: np.ndarray
    rankings: list
    memory: list
    budget: int
    history: list = field(default_factory=list)


class Retriever:
    def __init__(self, cfg, controller=True):
        self.cfg = cfg
        self.device = cfg["device"]
        self.index = Index(cfg["artifacts"])
        self.encoder = Encoder(cfg)
        self.dimension = self.index.embeddings.shape[1]
        root = model_root(cfg)
        self.variant = cfg["variant"]
        self.operators = self.proposal = self.continuous = None
        if self.variant == "continuous":
            self.continuous = load_network(root / "continuous.pt", ContinuousPlan, self.device)
        elif self.variant != "direct":
            self.operators = load_network(root / "operators.pt", Operators, self.device)
            self.proposal = load_network(root / "proposal.pt", Proposal, self.device)
        self.realizer = QueryRealizer(cfg, self.dimension, cfg["operator"]["hidden"], checkpoint=root / "query")
        self.controller = None
        if controller and cfg["controller"]["mode"] != "fixed":
            self.controller = load_network(root / f"controller_{cfg['controller_seed']}.pt", ValueModel, self.device)
        self.stats = {}
        self.initial_actions = None

    def charge(self, usage):
        self.stats["generation_calls"] += int(usage["input_tokens"] > 0)
        for key in ("input_tokens", "output_tokens"):
            self.stats[key] += usage[key]

    def initialize(self, question):
        self.encoder.cache.clear()
        self.realizer.entity_cache.clear()
        self.initial_actions = None
        self.stats = {"searches": 0, "generation_calls": 0, "input_tokens": 0,
                      "output_tokens": 0, "returned_candidates": 0, "proposed_candidates": 0}
        vector = self.encoder.one(question, query=True)
        initial = self.index.search(vector, self.cfg["retrieval"]["top_k"])
        self.stats["searches"] += 1
        self.stats["returned_candidates"] += len(initial)
        return State(question, vector, [initial], initial, self.cfg["retrieval"]["budget"] - 1)

    def memory_vector(self, state):
        memory = state.memory[:self.cfg["retrieval"]["score_k"]]
        return unit(np.mean([self.index.vector(identity) for identity in memory], axis=0)) if memory else np.zeros(self.dimension, np.float32)

    def pivots(self, state):
        mentions, usage = self.realizer.entities(state.question, query=True)
        self.charge(usage)
        mentions.sort(key=lambda mention: (state.question.lower().find(mention.lower()), mention))
        candidates = [(mention, "question", state.question.lower().find(mention.lower())) for mention in mentions]
        for identity in state.memory[:self.cfg["retrieval"]["score_k"]]:
            text = self.index.text(identity)
            mentions, usage = self.realizer.entities(text)
            self.charge(usage)
            mentions.sort(key=lambda mention: (text.lower().find(mention.lower()), mention))
            candidates.extend((mention, identity, text.lower().find(mention.lower())) for mention in mentions)
        seen, result = set(), []
        for mention, occurrence, offset in candidates:
            key = normalize_entity(mention)
            signature = (key, occurrence, offset)
            if not key or signature in seen or offset < 0:
                continue
            seen.add(signature)
            context = next((self.index.text(identity) for identity in state.memory
                            if key in normalize_entity(self.index.text(identity))), state.question + "\n" + mention)
            anchor = self.encoder.one(context, max_length=self.cfg["retrieval"]["context_tokens"])
            result.append((mention, anchor))
            if len(result) == self.cfg["retrieval"]["pivots"]:
                break
        return result

    @torch.inference_mode()
    def latent_plans(self, state):
        settings = self.cfg["retrieval"]
        observed = torch.tensor(np.concatenate((state.q, self.memory_vector(state)))[None], device=self.device)
        plans = []
        for pivot_order, (pivot, vector) in enumerate(self.pivots(state)):
            anchor = torch.tensor(vector[None], device=self.device)
            beam = [(0.0, (), self.operators.encoder(anchor), anchor)]
            pool = []
            for depth in range(self.cfg["operator"]["depth"]):
                expanded = []
                for score, sequence, hidden, endpoint in beam:
                    probabilities = self.proposal(observed, anchor, endpoint, torch.tensor([depth], device=self.device)).log_softmax(-1)[0]
                    for code in range(len(self.operators.codes)):
                        next_hidden, next_endpoint = self.operators.step(hidden, anchor, torch.tensor([code], device=self.device))
                        expanded.append((score + float(probabilities[code]), sequence + (code,), next_hidden, next_endpoint))
                beam = sorted(expanded, key=lambda item: (-item[0], item[1]))[:settings["beam"]]
                pool.extend(beam)
            pool.sort(key=lambda item: (-item[0] / len(item[1]) ** settings["length_penalty"], item[1]))
            retained = []
            for score, sequence, _, endpoint in pool:
                vector_endpoint = endpoint[0].cpu().numpy()
                if settings["merge_mode"] == "endpoint" and any(float(vector_endpoint @ old["endpoint"]) >= settings["merge"] for old in retained):
                    continue
                retained.append({"kind": "plan", "pivot": pivot, "pivot_order": pivot_order,
                                 "anchor": vector, "endpoint": vector_endpoint, "codes": list(sequence),
                                 "soft_codes": self.operators.codes[list(sequence)].cpu().numpy(),
                                 "score": score / len(sequence) ** settings["length_penalty"], "focus": ""})
            plans.extend(retained)
        plans.sort(key=lambda plan: (-plan["score"], plan["pivot_order"], plan["codes"]))
        return plans[:settings["plans"]]

    @torch.inference_mode()
    def actions(self, state):
        if state.budget <= 0:
            return []
        if not self.cfg["retrieval"]["replan"] and self.initial_actions is not None:
            actions = copy.deepcopy(self.initial_actions)
            for action in actions:
                action["features"] = self.features(state, action)
            return actions
        passages = [self.index.text(identity) for identity in state.memory[:self.cfg["retrieval"]["score_k"]]]
        if self.variant in {"direct", "continuous"}:
            plans = []
            conditions = [""] + [self.index.text(identity) for identity in state.memory[:self.cfg["retrieval"]["plans"]]]
            observed = torch.tensor(np.concatenate((state.q, self.memory_vector(state)))[None], device=self.device)
            for rank, focus in enumerate(conditions):
                anchor = self.encoder.one(focus, max_length=self.cfg["retrieval"]["context_tokens"]) if focus else state.q
                endpoint = soft_codes = None
                if self.continuous is not None:
                    predicted, hidden = self.continuous(observed, torch.tensor(anchor[None], device=self.device))
                    endpoint, soft_codes = predicted[0].cpu().numpy(), hidden.cpu().numpy()
                plans.append({"kind": "direct" if self.continuous is None else "continuous", "anchor": anchor,
                              "endpoint": endpoint, "soft_codes": soft_codes, "codes": [], "pivot": "",
                              "focus": focus, "score": -float(rank), "pivot_order": rank})
        else:
            plans = self.latent_plans(state)
            plans.append({"kind": "dense", "anchor": state.q, "endpoint": None, "soft_codes": None,
                          "codes": [], "pivot": "", "focus": "", "pivot_order": len(plans),
                          "score": min([plan["score"] for plan in plans], default=1.0) - 1})
        if self.variant == "endpoint_shuffle" and len(plans) > 2:
            endpoints = [plan["endpoint"] for plan in plans if plan["endpoint"] is not None]
            for i, plan in enumerate(plans[:-1]):
                plan["endpoint"] = endpoints[(i + 1) % len(endpoints)]
        actions, seen = [], set()
        for plan in plans:
            if plan["kind"] == "dense":
                query = state.question
            elif self.variant == "endpoint_only":
                query = ""
            else:
                prompt = query_prompt(state.question, passages, plan["pivot"], plan["focus"])
                query, usage = self.realizer.generate_query(prompt,
                    None if self.variant == "text_only" else plan["endpoint"], plan["soft_codes"])
                self.charge(usage)
            self.stats["proposed_candidates"] += 1
            key = query.strip().casefold() if self.variant != "endpoint_only" else tuple(plan["codes"])
            if key in seen and self.cfg["retrieval"]["merge_mode"] != "none":
                continue
            seen.add(key)
            plan["query"] = query
            plan["query_vector"] = plan["endpoint"] if self.variant == "endpoint_only" and plan["endpoint"] is not None else self.encoder.one(query or state.question, query=True)
            plan["features"] = self.features(state, plan)
            actions.append(plan)
        if self.initial_actions is None:
            self.initial_actions = copy.deepcopy(actions)
        return actions

    def features(self, state, action):
        history = unit(np.mean([item["query_vector"] for item in state.history], axis=0)) if state.history else np.zeros(self.dimension, np.float32)
        vectors = [state.q, self.memory_vector(state), history, action["anchor"],
                   action["endpoint"] if action["endpoint"] is not None else action["query_vector"], action["query_vector"]]
        scalars = [action["score"], len(action["codes"]), len(state.memory) / 60,
                   len(state.history) / 3, len(state.rankings) / 3, state.budget]
        return np.concatenate(vectors + [np.asarray(scalars, np.float32)]).astype(np.float32)

    def execute(self, state, action):
        ranking = self.index.search(action["query_vector"], self.cfg["retrieval"]["top_k"])
        self.stats["searches"] += 1
        self.stats["returned_candidates"] += len(ranking)
        rankings = state.rankings + [ranking]
        history = state.history + [{"query": action["query"], "query_vector": action["query_vector"],
                                    "kind": action["kind"], "codes": action["codes"]}]
        return State(state.question, state.q, rankings, fuse(rankings, self.cfg["retrieval"]["rrf"]), state.budget - 1, history)

    @torch.inference_mode()
    def values(self, actions):
        inputs = torch.tensor(np.stack([action["features"] for action in actions]), device=self.device)
        return self.controller(inputs).cpu().tolist()

    def retrieve(self, question):
        synchronize()
        start = time.perf_counter()
        state = self.initialize(question)
        trace = []
        while state.budget > 0:
            actions = self.actions(state)
            if not actions:
                break
            mode = self.cfg["controller"]["mode"]
            scores = [action["score"] for action in actions] if mode == "fixed" else self.values(actions)
            chosen = max(range(len(actions)), key=lambda i: (scores[i], -i))
            stop = mode not in {"fixed", "fixed_stopping"} and scores[chosen] <= 0
            trace.append({"budget": state.budget, "candidates": [
                {"kind": action["kind"], "query": action["query"], "pivot": action["pivot"],
                 "codes": action["codes"], "proposal_score": action["score"], "value": score}
                for action, score in zip(actions, scores)], "selected": "STOP" if stop else chosen})
            if stop:
                break
            state = self.execute(state, actions[chosen])
        synchronize()
        self.stats["retrieval_seconds"] = time.perf_counter() - start
        return {"ranking": state.memory, "rankings": state.rankings, "trace": trace, **self.stats}
