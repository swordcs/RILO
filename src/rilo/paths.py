import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import faiss

from .backends import Encoder
from .common import normalize_entity, read_rows, save_json, save_rows


def build_paths(cfg):
    root = Path(cfg["artifacts"])
    graph = list(read_rows(root / "triples.jsonl"))
    outgoing = defaultdict(list)
    incidence = defaultdict(set)
    source_edges = defaultdict(list)
    for row in graph:
        for subject, relation, target in row["triples"]:
            s, r, t = map(normalize_entity, (subject, relation, target))
            if not s or not r or not t:
                continue
            edge = {"subject": s, "relation": relation.strip(), "target": t, "passage": row["id"], "surface": subject}
            outgoing[s].append(edge)
            source_edges[row["id"]].append(edge)
            incidence[s].add(row["id"])
            incidence[t].add(row["id"])
    records = []
    for source, edges in source_edges.items():
        levels = [[edge] for edge in edges[:32]]
        for depth in range(1, 4):
            unique = {}
            for path in levels:
                key = (path[0]["subject"], tuple(normalize_entity(edge["relation"]) for edge in path), path[-1]["target"])
                unique.setdefault(key, path)
            retained = list(unique.values())[:32 if depth == 1 else 64]
            for path in retained:
                for target in [identity for identity in sorted(incidence[path[-1]["target"]]) if identity != source][:16]:
                    if target != source:
                        records.append({
                            "source": source, "target": target, "pivot": path[0]["surface"],
                            "relations": [edge["relation"] for edge in path],
                            "witnesses": [edge["passage"] for edge in path],
                            "group": source + ":" + str(depth),
                        })
            branch_cap = 16 if depth == 1 else 8
            levels = [path + [edge] for path in retained for edge in outgoing[path[-1]["target"]][:branch_cap]]
    if not records:
        raise ValueError("No directed non-self source-target paths were extracted")
    counts = Counter(row["target"] for row in records)
    buckets = Counter(int(math.log2(counts[row["target"]])) for row in records)
    for row in records:
        bucket = int(math.log2(counts[row["target"]]))
        probability = buckets[bucket] / len(records)
        row["weight"] = float(np.clip((0.5 * probability + 0.5 / len(buckets)) / probability, 0.1, 10))
    groups = sorted({row["group"] for row in records})
    random.Random(42).shuffle(groups)
    validation = set(groups[:max(1, int(len(groups) * 0.1))])
    for row in records:
        row["split"] = "val" if row["group"] in validation else "train"
    relations = sorted({relation for row in records for relation in row["relations"]})
    encoder = Encoder(cfg)
    vectors = encoder.encode(relations)
    np.save(root / "relations.npy", vectors)
    save_json(root / "relations.json", relations)
    save_rows(root / "paths.jsonl", records)


def build_aliases(cfg):
    root = Path(cfg["artifacts"])
    graph = list(read_rows(root / "triples.jsonl"))
    entities = sorted({normalize_entity(value) for row in graph for triple in row["triples"]
                       for value in (triple[0], triple[2]) if normalize_entity(value)})
    if not entities:
        raise ValueError("No entities available for alias candidates")
    encoder = Encoder(cfg)
    embeddings = encoder.encode(entities)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    def rows():
        for start in range(0, len(entities), 128):
            scores, neighbors = index.search(embeddings[start:start + 128], min(len(entities), 2048))
            for offset, (similarities, candidates) in enumerate(zip(scores, neighbors)):
                source = start + offset
                if sum(char.isalnum() for char in entities[source]) <= 2:
                    continue
                found = [(int(target), float(score)) for target, score in zip(candidates, similarities)
                         if target >= 0 and target != source][:2047]
                for target, score in found:
                    if score >= 0.8:
                        yield {"source": entities[source], "target": entities[target], "cosine": score}
    save_rows(root / "aliases.jsonl", rows())
    save_json(root / "entities.json", entities)
    np.save(root / "entities.npy", embeddings)
