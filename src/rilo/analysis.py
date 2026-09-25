import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

from .backends import Index
from .common import chunks, read_json, read_rows, save_json, save_rows
from .models import Operators
from .training import load_network, routed_vectors, sequence_batch


def evaluate_paths(cfg):
    root = Path(cfg["artifacts"])
    index = Index(root)
    rows = list(read_rows(root / "paths.jsonl"))
    training = [row for row in rows if row["split"] == "train"]
    seen_sequences = {tuple(row["relations"]) for row in training}
    seen_relations = {relation for row in training for relation in row["relations"]}
    groups = {}
    for row in rows:
        if row["split"] != "val":
            continue
        sequence = tuple(row["relations"])
        key = (row["group"], sequence)
        if key not in groups:
            category = "seen" if sequence in seen_sequences else "compositional" if set(sequence) <= seen_relations else "unseen_atomic"
            groups[key] = {"source": row["source"], "relations": row["relations"], "targets": [], "category": category}
        groups[key]["targets"].append(row["target"])
    categories = defaultdict(list)
    for key in sorted(groups):
        categories[groups[key]["category"]].append(groups[key])
    selected = []
    rng = random.Random(cfg["seed"])
    for category in sorted(categories):
        pool = categories[category]
        rng.shuffle(pool)
        selected.extend(pool[:500])
    relation_ids = {relation: i for i, relation in enumerate(read_json(root / "relations.json"))}
    student_vectors = routed_vectors(root, cfg["device"])
    teacher_vectors = torch.tensor(np.load(root / "relations.npy"), device=cfg["device"])
    results = []
    for name in ("teacher", "operators"):
        model = load_network(root / (name + ".pt"), Operators, cfg["device"])
        for batch in chunks(selected, cfg["operator"]["batch_size"]):
            ids, lengths = sequence_batch(batch, relation_ids, cfg["device"])
            anchor = torch.tensor(np.stack([index.vector(row["source"]) for row in batch]), device=cfg["device"])
            with torch.inference_mode():
                prediction, _ = model(anchor, ids, lengths, teacher_vectors if name == "teacher" else student_vectors)
            for row, vector in zip(batch, prediction.cpu().numpy()):
                ranking = index.search(vector, 10, [row["source"]])
                targets = set(row["targets"])
                results.append({**row, "model": name, "ranking": ranking,
                                **{f"recall@{k}": len(set(ranking[:k]) & targets) / len(targets) for k in (1, 5, 10)}})
    destination = Path(cfg["output"])
    save_rows(destination / "path_results.jsonl", results)
    summary = {}
    for name in ("teacher", "operators"):
        for category in sorted(categories):
            subset = [row for row in results if row["model"] == name and row["category"] == category]
            summary[name + "/" + category] = {"contexts": len(subset), **{
                f"R@{k}": 100 * float(np.mean([row[f"recall@{k}"] for row in subset])) for k in (1, 5, 10)}}
    save_json(destination / "path_summary.json", summary)


def metric(row, name):
    if name == "calls":
        return row["searches"]
    parent, field = {"recall": ("support", "recall"), "complete": ("support", "complete"), "f1": ("answer", "f1")}[name]
    return row[parent][field] if row.get(parent) is not None else None


def paired(args):
    def aggregate(paths):
        runs = [{row["id"]: row for row in read_rows(path)} for path in paths]
        identities = set(runs[0])
        if any(set(run) != identities for run in runs):
            raise ValueError("All seeds must use exactly the same question IDs")
        return {identity: np.mean([metric(run[identity], args.metric) for run in runs]) for identity in identities
                if all(metric(run[identity], args.metric) is not None for run in runs)}
    first, second = aggregate(args.first), aggregate(args.second)
    if first.keys() != second.keys():
        raise ValueError("Paired results must contain identical eligible question IDs")
    identities = sorted(first)
    if not identities:
        raise ValueError("No eligible paired observations")
    values = np.array([first[identity] - second[identity] for identity in identities])
    rng = np.random.default_rng(args.seed)
    bootstraps = np.empty(args.samples)
    for start in range(0, args.samples, 100):
        size = min(100, args.samples - start)
        bootstraps[start:start + size] = values[rng.integers(0, len(values), (size, len(values)))].mean(1)
    scale = 1 if args.metric == "calls" else 100
    save_json(args.output, {"metric": args.metric, "paired_questions": len(values), "samples": args.samples,
                           "difference": float(values.mean() * scale),
                           "ci95": (np.quantile(bootstraps, [0.025, 0.975]) * scale).tolist(),
                           "unit": "calls" if scale == 1 else "percentage_points", "seed": args.seed,
                           "conditioning": "fixed evaluated runs; paired question resampling"})


def report(args):
    grouped = defaultdict(list)
    for path in args.inputs:
        path = Path(path)
        cfg = read_json(path.parent / "config.json")
        summary = read_json(path)
        grouped[(cfg["dataset"], cfg["variant"])].append(summary)
    result = []
    for (dataset, method), runs in sorted(grouped.items()):
        row = {"dataset": dataset, "method": method, "runs": len(runs)}
        for key in ("R@10", "SC@10", "F1", "searches"):
            values = [run[key] for run in runs if run.get(key) is not None]
            row[key] = float(np.mean(values)) if values else None
            row[key + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        result.append(row)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / "table.json", result)
    if not result:
        raise ValueError("No summaries provided")
    with (destination / "table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    lines = ["# Experiment Results", "", "| Dataset | Method | Runs | R@10 | SC@10 | F1 | Calls |",
             "|:--|:--|--:|--:|--:|--:|--:|"]
    for row in result:
        fields = []
        for key in ("R@10", "SC@10", "F1", "searches"):
            value, sd = row[key], row[key + "_sd"]
            fields.append("n/a" if value is None else f"{value:.2f}" + (f" +/- {sd:.2f}" if sd is not None else ""))
        lines.append(f"| {row['dataset']} | {row['method']} | {row['runs']} | " + " | ".join(fields) + " |")
    (destination / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    datasets = sorted({row["dataset"] for row in result})
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        for row in result:
            if row["dataset"] == dataset and row["F1"] is not None:
                axis.scatter(row["searches"], row["F1"], label=row["method"], s=45)
        axis.set(title=dataset, xlabel="Mean index searches", ylabel="Answer F1 (%)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "quality_cost.pdf", bbox_inches="tight")
    plt.close(fig)


def reader_strata(args):
    first = {row["id"]: row for row in read_rows(args.first)}
    second = {row["id"]: row for row in read_rows(args.second)}
    if first.keys() != second.keys():
        raise ValueError("Reader diagnostic requires identical question IDs")
    groups = defaultdict(list)
    for identity in sorted(first):
        left, right = first[identity], second[identity]
        if left.get("support") is None or right.get("support") is None:
            continue
        key = (bool(left["support"]["complete"]), bool(right["support"]["complete"]))
        groups[key].append((left, right))
    result = []
    for key, rows in sorted(groups.items()):
        result.append({"first_complete": key[0], "second_complete": key[1], "questions": len(rows),
                       "first_f1": float(np.mean([row[0]["answer"]["f1"] for row in rows]) * 100),
                       "second_f1": float(np.mean([row[1]["answer"]["f1"] for row in rows]) * 100)})
    save_json(args.output, result)
