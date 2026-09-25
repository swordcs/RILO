import copy
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.nn import functional as F
from tqdm import tqdm

from .backends import Encoder, Index
from .common import chunks, load_checkpoint, model_root, read_json, read_rows, save_json, save_model, save_rows, seed_all, unit
from .models import ContinuousPlan, Operators, Proposal
from .query import QueryRealizer, query_prompt


def load_network(path, cls, device):
    checkpoint = load_checkpoint(path)
    model = cls(**checkpoint["metadata"]).to(device)
    model.load_state_dict(checkpoint["state"])
    return model.eval()


def routed_vectors(root, device):
    vectors = torch.tensor(np.load(root / "relations.npy"), device=device)
    if (root / "relation_permutation.json").exists():
        vectors = vectors[torch.tensor(read_json(root / "relation_permutation.json"), device=device)]
    return vectors


def sequence_batch(rows, relation_ids, device):
    lengths = torch.tensor([len(row["relations"]) for row in rows], device=device)
    ids = torch.zeros((len(rows), int(lengths.max())), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        ids[index, :lengths[index]] = torch.tensor([relation_ids[item] for item in row["relations"]], device=device)
    return ids, lengths


def train_operators(cfg):
    seed_all(cfg["seed"])
    root = Path(cfg["artifacts"])
    index = Index(root)
    op = cfg["operator"]
    relations = read_json(root / "relations.json")
    relation_ids = {relation: i for i, relation in enumerate(relations)}
    vectors = torch.from_numpy(np.load(root / "relations.npy")).to(cfg["device"])
    paths = [row for row in read_rows(root / "paths.jsonl") if len(row["relations"]) <= op["train_depth"]]
    training = [row for row in paths if row["split"] == "train"]
    validation = [row for row in paths if row["split"] == "val"]
    if not training or not validation:
        raise ValueError("Operator fitting requires nonempty train and validation path groups")
    rng = np.random.default_rng(op["data_seed"])
    if len(training) > op["max_examples"]:
        weights = np.array([row["weight"] for row in training])
        selected = rng.choice(len(training), op["max_examples"], replace=False, p=weights / weights.sum())
        training = [training[i] for i in sorted(selected)]
    save_rows(root / "operator_training.jsonl", training)
    dimension = index.embeddings.shape[1]
    teacher = Operators(dimension, op["hidden"], op["codes"], continuous=True).to(cfg["device"])
    mapping = None
    code_count = op["codes"]
    if cfg["variant"] == "random":
        mapping = rng.integers(0, code_count, len(relations)).tolist()
    elif cfg["variant"] == "clustered":
        mapping = KMeans(n_clusters=code_count, random_state=cfg["seed"], n_init=10).fit_predict(vectors.cpu().numpy()).tolist()
    elif cfg["variant"] == "surface":
        code_count = len(relations)
        mapping = list(range(code_count))
    student = Operators(dimension, op["hidden"], code_count, mapping=mapping).to(cfg["device"])
    if op["route_noise"]:
        permutation = np.arange(len(relations))
        affected = rng.choice(len(relations), int(len(relations) * op["route_noise"]), replace=False)
        permutation[affected] = rng.permutation(affected)
        student_vectors = vectors[torch.tensor(permutation, device=vectors.device)]
    else:
        student_vectors = vectors
    for name, model in (("teacher", teacher), ("operators", student)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=op["learning_rate"], weight_decay=op["weight_decay"])
        best = -1
        history = []
        for epoch in range(op["epochs"]):
            model.train()
            random.Random(cfg["seed"] + epoch).shuffle(training)
            temperature = 1.0 - 0.8 * epoch / max(1, op["epochs"] - 1)
            losses = []
            for batch in tqdm(list(chunks(training, op["batch_size"])), desc=f"{name} {epoch + 1}"):
                ids, lengths = sequence_batch(batch, relation_ids, cfg["device"])
                anchor = torch.from_numpy(np.stack([index.vector(row["source"]) for row in batch])).to(cfg["device"])
                target_ids = np.array([index.lookup[row["target"]] for row in batch])
                target = torch.tensor(np.array(index.embeddings[target_ids]), device=cfg["device"])
                prediction, extra = model(anchor, ids, lengths, vectors if name == "teacher" else student_vectors, temperature)
                negatives = rng.integers(0, len(index.ids), op["negatives"])
                negative_vectors = torch.tensor(np.array(index.embeddings[negatives]), device=cfg["device"])
                scores = prediction @ negative_vectors.T
                mask = torch.tensor(target_ids[:, None] == negatives[None, :], device=cfg["device"])
                scores = scores.masked_fill(mask, -1e4)
                scores = torch.cat(((prediction * target).sum(-1, keepdim=True), (prediction * anchor).sum(-1, keepdim=True), scores), 1)
                per_example = F.cross_entropy(scores / op["temperature"], torch.zeros(len(batch), dtype=torch.long, device=cfg["device"]), reduction="none")
                weights = torch.tensor([row["weight"] for row in batch], device=cfg["device"])
                loss = (per_example * weights / weights.mean()).mean()
                if name == "operators":
                    with torch.no_grad():
                        expected, _ = teacher(anchor, ids, lengths, vectors)
                    loss = loss + op["distill"] * (1 - (prediction * expected).sum(-1)).mean()
                    loss = loss + op["balance"] * extra[0] + op["entropy"] * extra[1]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            hits = []
            with torch.inference_mode():
                for batch in chunks(validation, op["batch_size"]):
                    ids, lengths = sequence_batch(batch, relation_ids, cfg["device"])
                    anchor = torch.tensor(np.stack([index.vector(row["source"]) for row in batch]), device=cfg["device"])
                    predictions, _ = model(anchor, ids, lengths, vectors if name == "teacher" else student_vectors)
                    for row, prediction in zip(batch, predictions.cpu().numpy()):
                        hits.append(row["target"] in index.search(prediction, 10, [row["source"]]))
            score = float(np.mean(hits))
            history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_r10": score})
            if score > best:
                best = score
                save_model(root / (name + ".pt"), model, model.metadata)
        model.load_state_dict(load_checkpoint(root / (name + ".pt"))["state"])
        model.eval()
        if name == "teacher":
            model.requires_grad_(False)
        save_json(root / (name + "_history.json"), history)
    if op["route_noise"]:
        save_json(root / "relation_permutation.json", permutation.tolist())
    save_json(root / "operator_config.json", cfg)


def make_supervision(cfg):
    root = Path(cfg["artifacts"])
    index = Index(root)
    encoder = Encoder(cfg)
    paths_by_source = defaultdict(list)
    for row in read_rows(root / "paths.jsonl"):
        if row["split"] == "train" and len(row["relations"]) <= cfg["operator"]["depth"]:
            paths_by_source[row["source"]].append(row)
    for split in ("fit", "tune"):
        records = []
        state_vectors, anchors = [], []
        for question in tqdm(list(read_rows(Path(cfg["data"]) / (split + ".jsonl"))), desc="Supervision " + split):
            support = set(question["support_ids"] or [])
            q_vector = encoder.one(question["question"], query=True)
            initial = index.search(q_vector, cfg["retrieval"]["top_k"])
            memory = initial[:cfg["retrieval"]["score_k"]]
            observed_support = [identity for identity in memory if identity in support]
            for source in observed_support:
                compatible = [row for row in paths_by_source[source] if row["target"] in support and row["target"] not in memory]
                grouped = defaultdict(list)
                for path in compatible:
                    grouped[(path["pivot"], path["target"])].append(path["relations"])
                for (pivot, target), sequences in grouped.items():
                    record = {"question_id": question["id"], "question": question["question"], "memory": memory,
                              "pivot": pivot, "source": source, "target": target,
                              "sequences": [list(sequence) for sequence in sorted({tuple(sequence) for sequence in sequences})]}
                    records.append(record)
                    state_vectors.append(np.concatenate((q_vector, unit(np.mean([index.vector(identity) for identity in memory], axis=0)))))
                    anchors.append(encoder.one(index.text(source), max_length=cfg["retrieval"]["context_tokens"]))
        if not records:
            raise ValueError(f"No support-compatible {split} continuations. Check corpus access and question splits")
        rng = np.random.default_rng(cfg["operator"]["data_seed"])
        count = cfg["query"]["examples"] if split == "fit" else min(len(records), 512)
        chosen = rng.choice(len(records), count, replace=len(records) < count)
        save_rows(root / (split + "_queries.jsonl"), [records[i] for i in chosen])
        np.savez(root / (split + "_states.npz"), state=np.stack(state_vectors)[chosen], anchor=np.stack(anchors)[chosen])
        save_json(root / (split + "_supervision.json"), {"eligible_continuations": len(records), "examples": len(chosen),
                  "unique_targets": len({records[i]["target"] for i in chosen}),
                  "selection_seed": cfg["operator"]["data_seed"], "memory_source": "executed initial retrieval"})


def train_proposal(cfg):
    seed_all(cfg["seed"])
    root = Path(cfg["artifacts"])
    records = list(read_rows(root / "fit_queries.jsonl"))
    states = np.load(root / "fit_states.npz")
    operators = load_network(root / "operators.pt", Operators, cfg["device"]).requires_grad_(False)
    relations = read_json(root / "relations.json")
    vectors = routed_vectors(root, cfg["device"])
    relation_ids = {relation: i for i, relation in enumerate(relations)}
    dimension = states["anchor"].shape[1]
    model = Proposal(dimension, cfg["operator"]["hidden"], len(operators.codes), cfg["operator"]["depth"]).to(cfg["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["proposal"]["learning_rate"])
    generator = random.Random(cfg["seed"])
    for _ in tqdm(range(cfg["proposal"]["steps"]), desc="Training proposal"):
        losses = []
        for _ in range(cfg["proposal"]["batch_size"]):
            i = generator.randrange(len(records))
            state = torch.tensor(states["state"][i:i + 1], device=cfg["device"])
            anchor = torch.tensor(states["anchor"][i:i + 1], device=cfg["device"])
            sequence_losses = []
            for sequence in records[i]["sequences"]:
                ids = torch.tensor([relation_ids[item] for item in sequence], device=cfg["device"])
                codes = operators.route(ids, vectors[ids])
                hidden = operators.encoder(anchor)
                endpoint = anchor
                token_losses = []
                for depth, code in enumerate(codes):
                    logits = model(state, anchor, endpoint, torch.tensor([depth], device=cfg["device"]))
                    token_losses.append(F.cross_entropy(logits, code[None]))
                    with torch.no_grad():
                        hidden, endpoint = operators.step(hidden, anchor, code[None])
                sequence_losses.append(torch.stack(token_losses).mean())
            losses.append(torch.stack(sequence_losses).mean())
        optimizer.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    save_model(root / "proposal.pt", model, model.metadata)


def train_continuous(cfg):
    seed_all(cfg["seed"])
    root = Path(cfg["artifacts"])
    index = Index(root)
    rows = list(read_rows(root / "fit_queries.jsonl"))
    states = np.load(root / "fit_states.npz")
    teacher = load_network(root / "teacher.pt", Operators, cfg["device"]).requires_grad_(False)
    relation_ids = {relation: i for i, relation in enumerate(read_json(root / "relations.json"))}
    vectors = torch.tensor(np.load(root / "relations.npy"), device=cfg["device"])
    model = ContinuousPlan(states["anchor"].shape[1], cfg["operator"]["hidden"], cfg["operator"]["depth"], cfg["operator"]["codes"]).to(cfg["device"])
    model.world.load_state_dict(teacher.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["proposal"]["learning_rate"])
    rng = random.Random(cfg["seed"])
    for _ in tqdm(range(cfg["proposal"]["steps"]), desc="Training continuous state"):
        chosen = [rng.randrange(len(rows)) for _ in range(cfg["proposal"]["batch_size"])]
        state = torch.tensor(states["state"][chosen], device=cfg["device"])
        anchor = torch.tensor(states["anchor"][chosen], device=cfg["device"])
        full_state = torch.tensor([rng.random() < 0.5 for _ in chosen], device=cfg["device"])
        anchor = torch.where(full_state[:, None], state[:, :anchor.shape[1]], anchor)
        targets = torch.tensor(np.stack([index.vector(rows[i]["target"]) for i in chosen]), device=cfg["device"])
        paths = [{"relations": rng.choice(rows[i]["sequences"])} for i in chosen]
        ids, lengths = sequence_batch(paths, relation_ids, cfg["device"])
        with torch.no_grad():
            expected, _ = teacher(anchor, ids, lengths, vectors)
        predicted, _ = model(state, anchor)
        loss = (1 - (predicted * targets).sum(-1)).mean() + cfg["operator"]["distill"] * (1 - (predicted * expected).sum(-1)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    save_model(root / "continuous.pt", model, model.metadata)


def train_query(cfg):
    seed_all(cfg["seed"])
    root = Path(cfg["artifacts"])
    index = Index(root)
    rows = list(read_rows(root / "fit_queries.jsonl"))
    states = np.load(root / "fit_states.npz")
    device = cfg["device"]
    model = load_network(root / ("continuous.pt" if cfg["variant"] == "continuous" else "operators.pt"),
                         ContinuousPlan if cfg["variant"] == "continuous" else Operators, device).requires_grad_(False)
    relations = read_json(root / "relations.json")
    relation_ids = {relation: i for i, relation in enumerate(relations)}
    vectors = routed_vectors(root, device)
    realizer = QueryRealizer(cfg, states["anchor"].shape[1], cfg["operator"]["hidden"], training=True)
    optimizer = torch.optim.AdamW([parameter for parameter in realizer.parameters() if parameter.requires_grad],
                                  lr=cfg["query"]["learning_rate"], weight_decay=cfg["query"]["weight_decay"])
    rng = random.Random(cfg["seed"])
    history = []
    condition_rng = random.Random(cfg["seed"] + 100003)
    for step in tqdm(range(cfg["query"]["steps"]), desc="Training query realizer"):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(cfg["query"]["batch_size"]):
            i = rng.randrange(len(rows))
            row = rows[i]
            endpoint = codes = None
            full_state = cfg["variant"] in {"direct", "continuous"} and condition_rng.random() < 0.5
            with torch.no_grad():
                anchor = torch.tensor(states["anchor"][i:i + 1], device=device)
                if full_state:
                    anchor = torch.tensor(states["state"][i:i + 1, :index.embeddings.shape[1]], device=device)
                if cfg["variant"] == "continuous":
                    endpoint, hidden = model(torch.tensor(states["state"][i:i + 1], device=device), anchor)
                    endpoint, codes = endpoint[0].cpu().numpy(), hidden.cpu().numpy()
                elif cfg["variant"] != "direct":
                    sequence = condition_rng.choice(row["sequences"])
                    ids = torch.tensor([relation_ids[item] for item in sequence], device=device)
                    assigned = model.route(ids, vectors[ids])
                    hidden = model.encoder(anchor)
                    endpoint = anchor
                    for code in assigned:
                        hidden, endpoint = model.step(hidden, anchor, code[None])
                    endpoint, codes = endpoint[0].cpu().numpy(), model.codes[assigned].cpu().numpy()
            if cfg["variant"] == "text_only":
                endpoint = None
            prompt = query_prompt(row["question"], [index.text(identity) for identity in row["memory"]],
                                  row["pivot"] if cfg["variant"] not in {"direct", "continuous"} else "",
                                  index.text(row["source"]) if cfg["variant"] in {"direct", "continuous"} and not full_state else "")
            loss = realizer(prompt, endpoint, codes, index.text(row["target"]), index.vector(row["target"]))
            total += float(loss.detach())
            (loss / cfg["query"]["batch_size"]).backward()
        torch.nn.utils.clip_grad_norm_([parameter for parameter in realizer.parameters() if parameter.requires_grad], 1)
        optimizer.step()
        if (step + 1) % 100 == 0:
            history.append({"step": step + 1, "loss": total / cfg["query"]["batch_size"]})
            realizer.save(root / "query")
            save_json(root / "query_history.json", history)
    realizer.save(root / "query")
