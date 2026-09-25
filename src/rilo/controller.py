import copy
import math
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .common import read_json, read_rows, save_json, save_model, seed_all
from .models import ValueModel
from .runtime import Retriever
from .training import load_network


def support_metrics(ranking, supports, k=10):
    if not supports:
        return None
    gold = set(supports)
    found = set(list(dict.fromkeys(ranking))[:k])
    return {"recall": len(gold & found) / len(gold), "complete": float(gold <= found)}


def utility(ranking, supports, beta=1.0):
    metrics = support_metrics(ranking, supports)
    if metrics is None:
        raise ValueError("Support utility is undefined without nonempty support annotations")
    return (metrics["recall"] + beta * metrics["complete"]) / (1 + beta)


def sampled_actions(actions, count, rng):
    ranked = sorted(actions, key=lambda action: (-action["score"], action["pivot_order"], action["codes"]))
    count = min(count, len(ranked))
    top = count // 2
    selected = ranked[:top] + rng.sample(ranked[top:], count - top)
    return [{**action, "selection_probability": 1.0 if i < top else (count - top) / (len(ranked) - top)}
            for i, action in enumerate(selected)]


class ReplayWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.files = []

    def add(self, features, reward, next_features, question_id, budget, probability, transition):
        self.rows.append({"features": torch.as_tensor(features, dtype=torch.float16),
                          "reward": float(reward), "next": torch.as_tensor(next_features, dtype=torch.float16),
                          "question_id": question_id, "budget": budget, "selection_probability": probability,
                          "transition": transition})
        if len(self.rows) == 128:
            self.flush()

    def flush(self):
        if self.rows:
            name = f"replay_{len(self.files):05d}.pt"
            torch.save(self.rows, self.path / name)
            self.files.append({"file": name, "count": len(self.rows)})
            self.rows = []


def collect(cfg):
    seed_all(cfg["seed"])
    if cfg["retrieval"]["budget"] != 3:
        raise ValueError("The two-level collection protocol requires retrieval.budget=3")
    root = Path(cfg["artifacts"])
    retriever = Retriever(cfg, controller=False)
    questions = [row for row in read_rows(Path(cfg["data"]) / "fit.jsonl")
                 if row.get("support_ids") and set(row["support_ids"]) <= retriever.index.lookup.keys()]
    if not questions:
        raise ValueError("No fitting question has a nonempty fully mapped support set")
    settings = cfg["controller"]
    writer = ReplayWriter(root / "replay")
    rng = random.Random(cfg["seed"])
    frontier = []
    counts = {"initialization": 0, "remaining_2": 0, "remaining_1": 0, "eligible_questions": len(questions)}
    first_budget = min(settings["first_level_searches"], settings["searches"])
    for i, row in enumerate(tqdm(questions, desc="Collecting remaining-budget 2")):
        state = retriever.initialize(row["question"])
        counts["initialization"] += 1
        actions = retriever.actions(state)
        count = math.ceil((first_budget - counts["remaining_2"]) / (len(questions) - i))
        for action in sampled_actions(actions, count, rng):
            next_state = retriever.execute(state, action)
            next_actions = retriever.actions(next_state)
            future = np.stack([candidate["features"] for candidate in next_actions]) if next_actions else np.empty((0, len(action["features"])), np.float32)
            reward = utility(next_state.memory, row["support_ids"], settings["beta"]) - utility(state.memory, row["support_ids"], settings["beta"]) - settings["cost"]
            writer.add(action["features"], reward, future, row["id"], 2, action["selection_probability"],
                       {"before": state.memory, "after": next_state.memory, "returned": next_state.rankings[-1],
                        "query": action["query"], "kind": action["kind"], "codes": action["codes"]})
            counts["remaining_2"] += 1
            entry = (row, next_state)
            if len(frontier) < settings["frontier"]:
                frontier.append(entry)
            else:
                position = rng.randrange(counts["remaining_2"])
                if position < len(frontier):
                    frontier[position] = entry
    second_budget = settings["searches"] - counts["remaining_2"]
    rng.shuffle(frontier)
    for i, (row, state) in enumerate(tqdm(frontier, desc="Collecting remaining-budget 1")):
        retriever.initial_actions = None
        actions = retriever.actions(state)
        count = math.ceil((second_budget - counts["remaining_1"]) / (len(frontier) - i))
        for action in sampled_actions(actions, count, rng):
            next_state = retriever.execute(state, action)
            reward = utility(next_state.memory, row["support_ids"], settings["beta"]) - utility(state.memory, row["support_ids"], settings["beta"]) - settings["cost"]
            writer.add(action["features"], reward, np.empty((0, len(action["features"])), np.float32), row["id"], 1, action["selection_probability"],
                       {"before": state.memory, "after": next_state.memory, "returned": next_state.rankings[-1],
                        "query": action["query"], "kind": action["kind"], "codes": action["codes"]})
            counts["remaining_1"] += 1
    writer.flush()
    counts["additional_searches"] = counts["remaining_1"] + counts["remaining_2"]
    counts["requested_additional_searches"] = settings["searches"]
    counts["frontier_states"] = len(frontier)
    counts["files"] = writer.files
    counts["dimension"] = retriever.dimension
    counts["seed"] = cfg["seed"]
    counts["frontier_inclusion_probability"] = min(1.0, settings["frontier"] / max(1, counts["remaining_2"]))
    save_json(root / "replay" / "manifest.json", counts)
    save_json(root / "collection_config.json", cfg)


class Replay:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = read_json(self.path / "manifest.json")
        self.positions = [(item["file"], i) for item in self.manifest["files"] for i in range(item["count"])]
        self.cache = OrderedDict()

    def get(self, index):
        name, offset = self.positions[index]
        if name not in self.cache:
            self.cache[name] = torch.load(self.path / name, map_location="cpu", weights_only=True)
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        return self.cache[name][offset]


def train_controller(cfg):
    seed_all(cfg["controller_seed"])
    root = Path(cfg["artifacts"])
    replay = Replay(root / "replay")
    if not replay.positions:
        raise ValueError("No executed transitions in replay")
    settings = cfg["controller"]
    device = cfg["device"]
    model = ValueModel(replay.manifest["dimension"]).to(device)
    target = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    rng = random.Random(cfg["controller_seed"])
    interval = max(1, math.ceil(settings["steps"] / settings["rounds"]))
    history = []
    for step in tqdm(range(settings["steps"]), desc="Fitted value iteration"):
        batch = [replay.get(rng.randrange(len(replay.positions))) for _ in range(settings["batch_size"])]
        inputs = torch.stack([row["features"] for row in batch]).to(device=device, dtype=torch.float32)
        expected = torch.tensor([row["reward"] for row in batch], device=device)
        with torch.no_grad():
            if settings["mode"] != "immediate":
                for i, row in enumerate(batch):
                    if len(row["next"]):
                        future = target(row["next"].to(device=device, dtype=torch.float32))
                        expected[i] += future.max().clamp_min(0)
        prediction = model(inputs)
        loss = F.huber_loss(prediction, expected, delta=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if (step + 1) % interval == 0 or step + 1 == settings["steps"]:
            target.load_state_dict(model.state_dict())
            checkpoint = root / "controller_candidates" / f"{cfg['controller_seed']}_{step + 1}.pt"
            save_model(checkpoint, model, model.metadata)
            history.append({"step": step + 1, "loss": float(loss.detach()), "checkpoint": str(checkpoint)})
    save_model(root / f"controller_{cfg['controller_seed']}.pt", model, model.metadata)
    save_json(root / f"controller_{cfg['controller_seed']}_history.json", history)


def select_controller(cfg):
    root = Path(cfg["artifacts"])
    retriever = Retriever(cfg)
    questions = [row for row in read_rows(Path(cfg["data"]) / "tune.jsonl")
                 if row.get("support_ids") and set(row["support_ids"]) <= retriever.index.lookup.keys()]
    if not questions:
        raise ValueError("Controller selection requires disjoint tuning questions with mapped supports")
    candidates = read_json(root / f"controller_{cfg['controller_seed']}_history.json")
    best, chosen, results = -float("inf"), None, []
    for candidate in candidates:
        retriever.controller = load_network(candidate["checkpoint"], ValueModel, cfg["device"])
        scores = []
        for row in tqdm(questions, desc="Selecting controller " + str(candidate["step"])):
            result = retriever.retrieve(row["question"])
            scores.append(utility(result["ranking"], row["support_ids"], cfg["controller"]["beta"]) - cfg["controller"]["cost"] * (result["searches"] - 1))
        score = float(np.mean(scores))
        results.append({**candidate, "tuning_utility_minus_cost": score, "questions": len(questions)})
        if score > best:
            best, chosen = score, candidate
    selected = load_network(chosen["checkpoint"], ValueModel, cfg["device"])
    save_model(root / f"controller_{cfg['controller_seed']}.pt", selected, selected.metadata)
    save_json(root / f"controller_{cfg['controller_seed']}_selection.json", {"selected": chosen, "candidates": results})
