import copy
import hashlib
import json
import os
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def read_rows(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def config(path, overrides=()):
    path = Path(path).resolve()
    value = read_json(path)
    parent = value.pop("extends", None)
    value = merge(config(path.parent / parent), value) if parent else value
    for setting in overrides:
        key, raw = setting.split("=", 1)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        target = value
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = parsed
    return value


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synchronize():
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)


def normalize_entity(text):
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def normalize_question(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def passage_id(title, text=None):
    content = title if text is None else title + "\n" + text
    return "chunk-" + hashlib.md5(content.encode("utf-8")).hexdigest()


def unit(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def model_root(cfg):
    return Path(cfg.get("source_artifacts") or cfg["artifacts"])


def save_model(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]
