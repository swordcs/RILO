import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .common import normalize_question, passage_id, read_json, read_rows, save_json, save_rows


def raw_rows(path):
    path = Path(path)
    if path.suffix in {".jsonl", ".ndjson"}:
        return list(read_rows(path))
    if path.suffix in {".csv", ".tsv"}:
        with path.open(encoding="utf-8") as stream:
            return list(csv.DictReader(stream, delimiter="\t" if path.suffix == ".tsv" else ","))
    value = read_json(path)
    return value["data"] if isinstance(value, dict) and "data" in value else value


def convert(row, dataset):
    if "support_ids" in row and "id" in row:
        return row, row.get("passages", [])
    question = row.get("question", row.get("Question", ""))
    if not question:
        raise ValueError("Every question must contain question text")
    identity = str(row.get("_id", row.get("id", normalize_question(question))))
    answers = row.get("answers", row.get("answer", row.get("Answer", "")))
    if isinstance(answers, str):
        answers = [answers]
    passages = []
    supports = []
    if dataset in {"2wiki", "hotpotqa"}:
        supporting_titles = {pair[0] for pair in row.get("supporting_facts", [])}
        for title, sentences in row.get("context", []):
            text = " ".join(sentences) if isinstance(sentences, list) else sentences
            identity_p = passage_id(title, text)
            passages.append({"id": identity_p, "title": title, "text": text})
            if title in supporting_titles:
                supports.append(identity_p)
    elif dataset == "musique":
        for paragraph in row.get("paragraphs", []):
            title, text = paragraph["title"], paragraph["paragraph_text"]
            identity_p = passage_id(title, text)
            passages.append({"id": identity_p, "title": title, "text": text})
            if paragraph.get("is_supporting", False):
                supports.append(identity_p)
    elif dataset != "bamboogle":
        raise ValueError(f"Unsupported dataset: {dataset}")
    aliases = row.get("answer_aliases", [])
    answers = list(dict.fromkeys([*answers, *aliases]))
    return {
        "id": dataset + ":" + identity,
        "question": question,
        "answers": answers,
        "support_ids": list(dict.fromkeys(supports)) if dataset != "bamboogle" else None,
        "hop": len(row.get("question_decomposition", [])) or row.get("hop", None),
    }, passages


def prepare(args):
    destination = Path(args.output)
    dataset = args.dataset
    converted = [convert(row, dataset) for row in raw_rows(args.evaluation)]
    by_id = {row["id"]: (row, passages) for row, passages in converted}
    if len(by_id) != len(converted):
        raise ValueError("Duplicate evaluation question IDs")
    if args.eval_ids:
        ids = read_json(args.eval_ids)
        ids = [identity if identity in by_id else dataset + ":" + str(identity) for identity in ids]
        if len(set(ids)) != len(ids):
            raise ValueError("Evaluation ID manifest contains duplicates")
        chosen = [by_id[identity] for identity in ids]
    else:
        chosen = sorted(converted, key=lambda item: item[0]["id"])
        random.Random(args.seed).shuffle(chosen)
        if len(chosen) < args.count:
            raise ValueError(f"Requested {args.count} evaluation questions, found {len(chosen)}")
        chosen = chosen[:args.count]
        if args.hop_counts:
            chosen = []
            for specification in args.hop_counts.split(","):
                hop, count = map(int, specification.split(":"))
                pool = sorted([item for item in converted if item[0]["hop"] == hop], key=lambda item: item[0]["id"])
                random.Random(args.seed + hop).shuffle(pool)
                if len(pool) < count:
                    raise ValueError(f"Insufficient {hop}-hop evaluation questions")
                chosen.extend(pool[:count])
            if len(chosen) != args.count:
                raise ValueError("Hop counts must sum to --count")
    corpus = {}
    if args.corpus:
        for row in raw_rows(args.corpus):
            title, text = row.get("title", ""), row["text"]
            identity = passage_id(title, text)
            corpus[identity] = {"id": identity, "title": title, "text": text}
    else:
        for _, passages in chosen:
            corpus.update({passage["id"]: passage for passage in passages})
    if not corpus:
        raise ValueError("No corpus passages. Supply --corpus for answer-only datasets")
    if args.train:
        excluded = {normalize_question(row["question"]) for row, _ in chosen}
        excluded_ids = {row["id"] for row, _ in chosen}
        groups = defaultdict(list)
        for raw in raw_rows(args.train):
            row, _ = convert(raw, dataset)
            key = normalize_question(row["question"])
            if key not in excluded and row["id"] not in excluded_ids:
                groups[key].append(row)
        keys = sorted(groups)
        random.Random(args.seed).shuffle(keys)
        boundary = int(len(keys) * 0.9)
        fitting = [row for key in keys[:boundary] for row in groups[key]][:args.fit_count]
        tuning = [row for key in keys[boundary:] for row in groups[key]][:args.tune_count]
        if len(fitting) < args.fit_count or len(tuning) < args.tune_count:
            raise ValueError("Training source cannot supply requested disjoint fit/tune counts")
        save_rows(destination / "fit.jsonl", fitting)
        save_rows(destination / "tune.jsonl", tuning)
    save_rows(destination / "corpus.jsonl", sorted(corpus.values(), key=lambda row: row["id"]))
    save_rows(destination / "eval.jsonl", [row for row, _ in chosen])
    save_json(destination / "eval_ids.json", [row["id"] for row, _ in chosen])
    save_rows(destination / "passage_mapping.jsonl", (
        {"question_id": row["id"], "position": position, "title": passage["title"], "canonical_id": passage["id"]}
        for row, passages in chosen for position, passage in enumerate(passages)
    ))
    save_json(destination / "preparation.json", {
        "dataset": dataset, "seed": args.seed, "questions": len(chosen),
        "passages": len(corpus), "provided_eval_ids": bool(args.eval_ids),
        "corpus_source": str(args.corpus) if args.corpus else "evaluation supplied contexts",
    })


def scale_corpus(args):
    base = list(read_rows(Path(args.base) / "corpus.jsonl"))
    seen = {row["id"] for row in base}
    if args.size < len(base):
        raise ValueError("Scale size cannot remove base passages")
    result = list(base)
    for row in read_rows(args.snapshot):
        if len(result) >= args.size:
            break
        title, text = row.get("title", ""), row["text"]
        identity = passage_id(title, text)
        if identity not in seen:
            seen.add(identity)
            result.append({"id": identity, "title": title, "text": text})
    if len(result) != args.size:
        raise ValueError("Snapshot has insufficient distinct passages")
    destination = Path(args.output)
    save_rows(destination / "corpus.jsonl", sorted(result, key=lambda row: row["id"]))
    save_rows(destination / "eval.jsonl", read_rows(Path(args.base) / "eval.jsonl"))
