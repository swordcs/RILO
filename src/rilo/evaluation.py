import importlib
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .backends import Encoder, Index, TextModel
from .common import read_rows, save_json, synchronize
from .controller import support_metrics
from .runtime import Retriever, fuse


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(prediction, answers, dataset):
    predicted = normalize_answer(prediction)
    scores = []
    for answer in answers:
        gold = normalize_answer(str(answer))
        exact = float(predicted == gold)
        if dataset == "hotpotqa" and predicted != gold and ({predicted, gold} & {"yes", "no", "noanswer"}):
            scores.append((exact, 0.0))
            continue
        overlap = sum((Counter(predicted.split()) & Counter(gold.split())).values())
        denominator = len(predicted.split()) + len(gold.split())
        f1 = 2 * overlap / denominator if denominator else exact
        scores.append((exact, f1))
    return {"exact_match": max((value[0] for value in scores), default=0.0),
            "f1": max((value[1] for value in scores), default=0.0)}


class Baseline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.index = Index(cfg["artifacts"])
        self.encoder = Encoder(cfg)
        self.bm25 = None
        self.generator = None
        if cfg["variant"] == "hybrid":
            self.bm25 = BM25Okapi([self.index.text(identity).lower().split() for identity in self.index.ids])
        if cfg["variant"] == "hyde":
            self.generator = TextModel(cfg)

    def retrieve(self, question):
        self.encoder.cache.clear()
        synchronize()
        start = time.perf_counter()
        cfg = self.cfg
        ranking = self.index.search(self.encoder.one(question, query=True), cfg["retrieval"]["top_k"])
        rankings = [ranking]
        stats = {"generation_calls": 0, "input_tokens": 0, "output_tokens": 0, "trace": []}
        if self.bm25 is not None:
            scores = self.bm25.get_scores(question.lower().split())
            ordered = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.index.ids[i]))
            rankings.append([self.index.ids[i] for i in ordered[:cfg["retrieval"]["top_k"]]])
        if self.generator is not None:
            for step in range(1, cfg["retrieval"]["budget"]):
                observed = fuse(rankings, cfg["retrieval"]["rrf"])[:cfg["retrieval"]["score_k"]]
                prompt = "Write a hypothetical Wikipedia passage that answers the question.\nQuestion: " + question
                if step > 1:
                    prompt += "\nRetrieved context:\n" + "\n\n".join(self.index.text(identity) for identity in observed)
                hypothesis, usage = self.generator.generate(prompt, limit=cfg["query"]["output_tokens"], input_limit=cfg["query"]["input_tokens"])
                for key in ("input_tokens", "output_tokens"):
                    stats[key] += usage[key]
                stats["generation_calls"] += 1
                rankings.append(self.index.search(self.encoder.one(hypothesis), cfg["retrieval"]["top_k"]))
                stats["trace"].append({"hypothesis": hypothesis})
        synchronize()
        return {"ranking": fuse(rankings, cfg["retrieval"]["rrf"]), "rankings": rankings,
                "searches": len(rankings), "returned_candidates": sum(map(len, rankings)),
                "retrieval_seconds": time.perf_counter() - start, **stats}


def load_retriever(cfg):
    if cfg.get("native_adapter"):
        module, name = cfg["native_adapter"].split(":", 1)
        return getattr(importlib.import_module(module), name)(cfg)
    if cfg["variant"] in {"dense", "hybrid", "hyde"}:
        if cfg["variant"] == "hybrid" and cfg["retrieval"]["budget"] < 2:
            raise ValueError("Hybrid requires two separately charged index searches")
        return Baseline(cfg)
    return Retriever(cfg)


def retrieve(cfg):
    retriever = load_retriever(cfg)
    destination = Path(cfg["output"])
    destination.mkdir(parents=True, exist_ok=True)
    questions = list(read_rows(Path(cfg["data"]) / "eval.jsonl"))
    save_json(destination / "config.json", cfg)
    with (destination / "retrieval.jsonl").open("w", encoding="utf-8") as stream:
        for row in tqdm(questions, desc="Retrieving"):
            result = retriever.retrieve(row["question"])
            if result["searches"] > cfg["retrieval"]["budget"]:
                raise ValueError("Retriever exceeded the declared search budget")
            result["ranking"] = list(dict.fromkeys(result["ranking"]))
            if any(identity not in retriever.index.lookup for identity in result["ranking"]):
                raise ValueError("Retriever returned a passage outside the canonical corpus")
            result.update({"id": row["id"], "question": row["question"]})
            metrics = support_metrics(result["ranking"], row.get("support_ids"), cfg["retrieval"]["score_k"])
            result["support"] = metrics
            result["hop"] = row.get("hop")
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
    summarize(destination / "retrieval.jsonl", destination / "retrieval_summary.json")


def reader_context(model, question, ranking, index, settings):
    header = "Answer the question using the evidence. Return only the short answer without explanation.\nQuestion: " + question + "\nEvidence:\n"
    if len(model.prompt_ids(header)) > settings["input_tokens"]:
        raise ValueError("The question alone exceeds the reader input limit")
    prompt, spans = header, []
    for rank, identity in enumerate(ranking[:10], 1):
        prefix = f"\n[{rank}] "
        text = index.text(identity)
        complete = prompt + prefix + text
        if len(model.prompt_ids(complete)) <= settings["input_tokens"]:
            spans.append({"id": identity, "start": len(prompt + prefix), "end": len(complete), "complete": True})
            prompt = complete
            continue
        tokens = model.tokenizer.encode(text, add_special_tokens=False)
        low, high = 0, len(tokens)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = prompt + prefix + model.tokenizer.decode(tokens[:middle], skip_special_tokens=True)
            if len(model.prompt_ids(candidate)) <= settings["input_tokens"]:
                low = middle
            else:
                high = middle - 1
        if low:
            clipped = model.tokenizer.decode(tokens[:low], skip_special_tokens=True)
            spans.append({"id": identity, "start": len(prompt + prefix), "end": len(prompt + prefix + clipped), "complete": False})
            prompt += prefix + clipped
        break
    return prompt, spans


def answer(cfg):
    destination = Path(cfg["output"])
    index = Index(cfg["artifacts"])
    model = TextModel(cfg, reader=True)
    questions = {row["id"]: row for row in read_rows(Path(cfg["data"]) / "eval.jsonl")}
    rows = list(read_rows(destination / "retrieval.jsonl"))
    if {row["id"] for row in rows} != questions.keys() or len(rows) != len(questions):
        raise ValueError("Retrieval results must cover exactly the locked evaluation IDs")
    with (destination / "results.jsonl").open("w", encoding="utf-8") as stream:
        for row in tqdm(rows, desc="Reading"):
            question = questions[row["id"]]
            if row["question"] != question["question"]:
                raise ValueError("Question text changed after retrieval")
            prompt, spans = reader_context(model, question["question"], row["ranking"], index, cfg["reader"])
            synchronize()
            start = time.perf_counter()
            prediction, usage = model.generate(prompt, limit=cfg["reader"]["output_tokens"])
            synchronize()
            support = question.get("support_ids")
            result = {**row, "prediction": prediction, "answer": answer_scores(prediction, question["answers"], cfg["dataset"]),
                      "reader_seconds": time.perf_counter() - start, "reader_usage": usage,
                      "visible_spans": spans, "reader_prompt": prompt,
                      "all_support_passages_fully_visible": bool(set(support) <= {span["id"] for span in spans if span["complete"]}) if support else None}
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
    summarize(destination / "results.jsonl", destination / "summary.json")


def summarize(path, output):
    rows = list(read_rows(path))
    result = {"questions": len(rows), "support_questions": sum(row.get("support") is not None for row in rows)}
    for key in ("searches", "generation_calls", "input_tokens", "output_tokens", "returned_candidates", "proposed_candidates"):
        values = [row[key] for row in rows if key in row]
        if values:
            result[key] = float(np.mean(values))
    for key in ("retrieval_seconds", "reader_seconds"):
        values = [row[key] for row in rows if key in row]
        if values:
            result[key] = {"mean": float(np.mean(values)), "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
    for parent, key, name in (("support", "recall", "R@10"), ("support", "complete", "SC@10"),
                              ("answer", "f1", "F1"), ("answer", "exact_match", "EM")):
        values = [row[parent][key] for row in rows if row.get(parent) is not None]
        result[name] = float(np.mean(values) * 100) if values else None
    save_json(output, result)
    return result
