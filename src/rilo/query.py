from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .common import save_json
from .openie import entities_from, ner_messages


def query_prompt(question, passages, pivot="", focus="", action_ids=None):
    plan = "" if not action_ids else "\nOperators: " + " ".join(f"o_{index}" for index in action_ids)
    return (
        "Write a concise passage-search query for missing evidence. Do not answer the question.\n"
        + "Question: " + question + "\nObserved evidence:\n" + "\n\n".join(passages)
        + "\nPivot: " + pivot + "\nFocus: " + focus + plan
    )


class QueryRealizer(nn.Module):
    def __init__(self, cfg, dimension, hidden=256, checkpoint=None, training=False):
        super().__init__()
        self.cfg = cfg
        self.device_name = cfg["query_device"]
        self.dimension = dimension
        self.prefix_tokens = cfg["query"]["prefix_tokens"]
        source = str(Path(checkpoint) / "model") if checkpoint else cfg["query_model"]
        revision = None if checkpoint else cfg["query_revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
        self.lm = AutoModelForSeq2SeqLM.from_pretrained(
            source, revision=revision, torch_dtype=getattr(torch, cfg["dtype"])
        ).to(self.device_name)
        width = self.lm.config.d_model
        self.endpoint_norm = nn.LayerNorm(dimension).to(self.device_name)
        self.endpoint_projection = nn.Linear(dimension, self.prefix_tokens * width, bias=False).to(self.device_name)
        self.endpoint_gate = nn.Linear(width, 1).to(self.device_name)
        if checkpoint:
            heads = torch.load(Path(checkpoint) / "heads.pt", map_location=self.device_name, weights_only=True)
            self.endpoint_norm.load_state_dict(heads["endpoint_norm"])
            self.endpoint_projection.load_state_dict(heads["endpoint_projection"])
            self.endpoint_gate.load_state_dict(heads["endpoint_gate"])
        if training and cfg["query"]["gradient_checkpointing"]:
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.lm.config.use_cache = False
        self.requires_grad_(training)
        self.train(training)
        self.entity_cache = {}
        self.entity_extractor = None

    def inputs(self, prompt, endpoint, target=None):
        prefix = None
        text_limit = self.cfg["query"]["input_tokens"]
        if endpoint is not None:
            vector = torch.as_tensor(np.asarray(endpoint), device=self.device_name, dtype=torch.float32)
            prefix = self.endpoint_projection(self.endpoint_norm(vector)).reshape(self.prefix_tokens, -1)
            text_limit -= self.prefix_tokens
        ids = self.tokenizer.encode(prompt, truncation=True, max_length=text_limit, add_special_tokens=True)
        embeddings = self.lm.get_input_embeddings()(torch.tensor(ids, device=self.device_name))
        if prefix is not None:
            pooled = prefix.mean(0)
            embeddings = embeddings + (torch.sigmoid(self.endpoint_gate(pooled)) * pooled).to(embeddings.dtype)
            embeddings = torch.cat((prefix.to(embeddings.dtype), embeddings))
        labels = None
        if target is not None:
            target_ids = self.tokenizer.encode(
                target, add_special_tokens=False
            )[:self.cfg["query"]["target_tokens"] - 1]
            target_ids.append(self.tokenizer.eos_token_id)
            labels = torch.tensor([target_ids], device=self.device_name)
        return embeddings[None], labels

    def forward(self, prompt, endpoint, target, target_vector):
        inputs, labels = self.inputs(prompt, endpoint, target)
        output = self.lm(
            inputs_embeds=inputs,
            attention_mask=torch.ones(inputs.shape[:2], device=inputs.device, dtype=torch.long),
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
        )
        projection = self.endpoint_projection.weight.reshape(self.prefix_tokens, -1, self.dimension).mean(0).T
        predicted = F.linear(output.decoder_hidden_states[-1][0, -1].float(), projection)
        gold = torch.as_tensor(target_vector, device=self.device_name, dtype=torch.float32)
        alignment = 1 - F.cosine_similarity(predicted[None], gold[None]).mean()
        return output.loss + self.cfg["query"]["alignment"] * alignment

    @torch.inference_mode()
    def generate_query(self, prompt, endpoint=None):
        inputs, _ = self.inputs(prompt, endpoint)
        output = self.lm.generate(
            inputs_embeds=inputs,
            attention_mask=torch.ones(inputs.shape[:2], device=inputs.device, dtype=torch.long),
            do_sample=False,
            num_beams=self.cfg["query"]["num_beams"],
            max_new_tokens=self.cfg["query"]["output_tokens"],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )[0]
        generated = output[1:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip(), {
            "input_tokens": inputs.shape[1],
            "output_tokens": len(generated),
        }

    def entities(self, text, query=False):
        key = (text, query)
        if key in self.entity_cache:
            return self.entity_cache[key], {"input_tokens": 0, "output_tokens": 0}
        if self.entity_extractor is None:
            from .backends import TextModel

            self.entity_extractor = TextModel(self.cfg)
        raw, usage = self.entity_extractor.generate(ner_messages(text, query=query), limit=2048)
        entities = entities_from(raw)
        self.entity_cache[key] = entities
        return entities, usage

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(path / "model")
        self.tokenizer.save_pretrained(path / "model")
        torch.save(
            {
                "endpoint_norm": self.endpoint_norm.state_dict(),
                "endpoint_projection": self.endpoint_projection.state_dict(),
                "endpoint_gate": self.endpoint_gate.state_dict(),
            },
            path / "heads.pt",
        )
        save_json(path / "config.json", self.cfg)
