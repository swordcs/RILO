import json
import os
from pathlib import Path

import faiss
import numpy as np
import requests
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .common import chunks, passage_id, read_json, read_rows, save_json, save_rows, unit
from .openie import entities_from, ner_messages, triple_messages, triples_from


class Encoder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg["embedding_device"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["embedding_model"], revision=cfg["embedding_revision"], padding_side="left"
        )
        self.model = AutoModel.from_pretrained(
            cfg["embedding_model"], revision=cfg["embedding_revision"],
            torch_dtype=getattr(torch, cfg["dtype"])
        ).to(self.device).eval()
        self.cache = {}

    @torch.inference_mode()
    def encode(self, texts, query=False, max_length=None):
        instruction = "Instruct: Given a question, retrieve relevant documents that best answer the question.\nQuery: "
        texts = [instruction + text if query else text for text in texts]
        length = max_length or self.cfg["embedding_tokens"]
        outputs = []
        for batch in chunks(texts, self.cfg["embedding_batch"]):
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=length, return_tensors="pt").to(self.device)
            hidden = self.model(**encoded).last_hidden_state
            indices = torch.arange(hidden.shape[1], device=self.device).expand(hidden.shape[0], -1)
            last = indices.masked_fill(~encoded.attention_mask.bool(), -1).max(dim=1).values
            vectors = hidden[torch.arange(len(batch), device=self.device), last].float()
            if self.cfg.get("embedding_dimension"):
                vectors = vectors[:, :self.cfg["embedding_dimension"]]
            outputs.append(torch.nn.functional.normalize(vectors, dim=-1).cpu().numpy())
        if not outputs:
            dimension = self.cfg.get("embedding_dimension") or self.model.config.hidden_size
            return np.empty((0, dimension), dtype=np.float32)
        return np.concatenate(outputs)

    def one(self, text, query=False, max_length=None):
        key = (text, query, max_length)
        if key not in self.cache:
            if len(self.cache) >= 4096:
                self.cache.clear()
            self.cache[key] = self.encode([text], query=query, max_length=max_length)[0]
        return self.cache[key]


class TextModel:
    def __init__(self, cfg, reader=False):
        self.cfg = cfg
        self.reader = reader
        model_key = "reader_model" if reader else "language_model"
        revision = cfg["reader_revision" if reader else "language_revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg[model_key], revision=revision, padding_side="left")
        self.model = None
        if not reader or not cfg.get("reader_url"):
            kwargs = {"device_map": cfg["reader_device_map"]} if reader else {}
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg[model_key], revision=revision, torch_dtype=getattr(torch, cfg["dtype"]), **kwargs
            )
            if not reader:
                self.model.to(cfg["query_device"])
            self.model.eval()

    def prompt_ids(self, prompt):
        return self.tokenizer.apply_chat_template(
            prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False
        )

    @torch.inference_mode()
    def generate(self, prompt, limit=2048, input_limit=None):
        ids = self.prompt_ids(prompt)
        if input_limit and len(ids) > input_limit:
            ids = ids[:input_limit - 16] + ids[-16:]
        if self.model is None:
            key = os.environ.get(self.cfg["reader_api_key_env"], "EMPTY")
            response = requests.post(
                self.cfg["reader_url"].rstrip("/") + "/completions",
                headers={"Authorization": "Bearer " + key},
                json={"model": self.cfg["reader_model"], "prompt": ids,
                      "max_tokens": limit, "temperature": 0}, timeout=600
            )
            response.raise_for_status()
            payload = response.json()
            text = payload["choices"][0]["text"]
            output_count = payload.get("usage", {}).get("completion_tokens", len(self.tokenizer.encode(text)))
        else:
            inputs = torch.tensor([ids], device=self.model.get_input_embeddings().weight.device)
            output = self.model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                do_sample=False, max_new_tokens=limit, pad_token_id=self.tokenizer.eos_token_id
            )[0, len(ids):]
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            output_count = len(output)
        return text.strip(), {"input_tokens": len(ids), "output_tokens": output_count}


class Index:
    def __init__(self, root):
        root = Path(root)
        self.passages = list(read_rows(root / "passages.jsonl"))
        self.ids = [row["id"] for row in self.passages]
        self.lookup = {identity: index for index, identity in enumerate(self.ids)}
        self.embeddings = np.load(root / "passages.npy", mmap_mode="r")
        self.index = faiss.read_index(str(root / "index.faiss"))

    def search(self, vector, k=20, exclude=()):
        blocked = set(exclude)
        count = min(len(self.ids), k + len(blocked))
        scores, indices = self.index.search(unit(np.asarray(vector).reshape(1, -1)), count)
        candidates = [(self.ids[i], float(score)) for i, score in zip(indices[0], scores[0]) if i >= 0 and self.ids[i] not in blocked]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return [identity for identity, _ in candidates[:k]]

    def text(self, identity):
        row = self.passages[self.lookup[identity]]
        return row["title"] + "\n" + row["text"]

    def vector(self, identity):
        return np.array(self.embeddings[self.lookup[identity]], dtype=np.float32, copy=True)


def embed_corpus(cfg):
    root = Path(cfg["artifacts"])
    root.mkdir(parents=True, exist_ok=True)
    passages = sorted(read_rows(Path(cfg["data"]) / "corpus.jsonl"), key=lambda row: row["id"])
    if not passages:
        raise ValueError("Cannot build an index from an empty corpus")
    if len({row["id"] for row in passages}) != len(passages):
        raise ValueError("Corpus passage IDs must be unique")
    encoder = Encoder(cfg)
    first = encoder.encode([passages[0]["title"] + "\n" + passages[0]["text"]])
    embeddings = np.lib.format.open_memmap(root / "passages.npy", mode="w+", dtype=np.float32, shape=(len(passages), first.shape[1]))
    index = faiss.IndexFlatIP(first.shape[1])
    offset = 0
    for batch in tqdm(list(chunks(passages, cfg["embedding_batch"])), desc="Embedding corpus"):
        vectors = encoder.encode([row["title"] + "\n" + row["text"] for row in batch])
        embeddings[offset:offset + len(batch)] = vectors
        index.add(vectors)
        offset += len(batch)
    embeddings.flush()
    faiss.write_index(index, str(root / "index.faiss"))
    save_rows(root / "passages.jsonl", passages)


def extract(cfg):
    root = Path(cfg["artifacts"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "triples.jsonl"
    known = {row["id"] for row in read_rows(path)} if path.exists() else set()
    passages = list(read_rows(Path(cfg["data"]) / "corpus.jsonl"))
    corpus_ids = {row["id"] for row in passages}
    imported = []
    if cfg.get("openie_cache"):
        for row in read_json(cfg["openie_cache"]).get("docs", []):
            identity = passage_id(row["passage"])
            if identity in corpus_ids and identity not in known:
                entities = entities_from(json.dumps({"named_entities": row.get("extracted_entities", [])}))
                triples = triples_from(json.dumps({"triples": row.get("extracted_triples", [])}))
                if entities or triples:
                    imported.append({"id": identity, "entities": entities, "triples": triples})
                    known.add(identity)
    model = TextModel(cfg) if corpus_ids - known else None
    with path.open("a", encoding="utf-8") as stream:
        for row in imported:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        for passage in tqdm(passages, desc="Extracting graph"):
            if passage["id"] in known:
                continue
            text = passage["title"] + "\n" + passage["text"]
            ner_raw, ner_usage = model.generate(ner_messages(text), limit=2048)
            entities = entities_from(ner_raw)
            triple_raw, triple_usage = model.generate(triple_messages(text, entities), limit=2048)
            triples = triples_from(triple_raw)
            stream.write(json.dumps({"id": passage["id"], "entities": entities, "triples": triples,
                                    "ner_response": ner_raw, "triple_response": triple_raw,
                                    "ner_usage": ner_usage, "triple_usage": triple_usage}, ensure_ascii=False) + "\n")
            stream.flush()
    extracted = {row["id"]: row for row in read_rows(path)}
    docs = [{"idx": row["id"], "passage": row["title"] + "\n" + row["text"],
             "extracted_entities": extracted[row["id"]]["entities"],
             "extracted_triples": extracted[row["id"]]["triples"]} for row in passages]
    save_json(root / ("openie_results_ner_" + cfg["language_model"].replace("/", "_") + ".json"), {"docs": docs})
