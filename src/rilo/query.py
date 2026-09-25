import contextlib
import json
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .openie import entities_from, ner_messages
from .common import load_checkpoint, save_json


def query_prompt(question, passages, pivot="", focus=""):
    return (
        "Write a concise passage-search query for missing evidence. Do not answer the question.\n"
        + "Question: " + question + "\nObserved evidence:\n" + "\n\n".join(passages)
        + "\nPivot: " + pivot + "\nFocus: " + focus
    )


class QueryRealizer(nn.Module):
    def __init__(self, cfg, dimension, hidden=256, checkpoint=None, training=False):
        super().__init__()
        self.cfg = cfg
        self.device_name = cfg["query_device"]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["language_model"], revision=cfg["language_revision"], padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            cfg["language_model"], revision=cfg["language_revision"],
            torch_dtype=getattr(torch, cfg["dtype"])
        ).to(self.device_name)
        if checkpoint:
            self.lm = PeftModel.from_pretrained(base, str(Path(checkpoint) / "adapter"), is_trainable=training)
        else:
            self.lm = get_peft_model(base, LoraConfig(
                r=cfg["query"]["lora_rank"], lora_alpha=cfg["query"]["lora_alpha"],
                lora_dropout=0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
            ))
        width = base.config.hidden_size
        self.endpoint_projection = nn.Linear(dimension, width).to(self.device_name)
        self.code_projection = nn.Linear(hidden, width).to(self.device_name)
        self.alignment = nn.Linear(width, dimension).to(self.device_name)
        if checkpoint:
            heads = torch.load(Path(checkpoint) / "heads.pt", map_location=self.device_name, weights_only=True)
            self.endpoint_projection.load_state_dict(heads["endpoint"])
            self.code_projection.load_state_dict(heads["code"])
            self.alignment.load_state_dict(heads["alignment"])
        if training and cfg["query"]["gradient_checkpointing"]:
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.lm.enable_input_require_grads()
            self.lm.config.use_cache = False
        self.train(training)
        self.entity_cache = {}
        self.dimension = dimension

    def prompt_ids(self, prompt):
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False
        )
        limit = self.cfg["query"]["input_tokens"]
        if len(ids) > limit:
            ids = ids[:limit - 32] + ids[-32:]
        return ids

    def inputs(self, prompt, endpoint, codes, target=None):
        ids = self.prompt_ids(prompt)
        soft = []
        if endpoint is not None:
            vector = torch.as_tensor(np.asarray(endpoint), device=self.device_name, dtype=torch.float32)
            soft.append(self.endpoint_projection(vector)[None])
        if codes is not None and len(codes):
            vectors = torch.as_tensor(np.asarray(codes), device=self.device_name, dtype=torch.float32)
            soft.append(self.code_projection(vectors))
        prefix_length = sum(item.shape[0] for item in soft)
        limit = self.cfg["query"]["input_tokens"] - prefix_length
        if len(ids) > limit:
            ids = ids[:limit - 32] + ids[-32:]
        target_ids = [] if target is None else self.tokenizer.encode(target, add_special_tokens=False)[:self.cfg["query"]["target_tokens"] - 1] + [self.tokenizer.eos_token_id]
        ids_tensor = torch.tensor(ids + target_ids, device=self.device_name)
        embeddings = self.lm.get_input_embeddings()(ids_tensor)
        if soft:
            embeddings = torch.cat([item.to(embeddings.dtype) for item in soft] + [embeddings])
        prompt_end = prefix_length + len(ids) - 1
        labels = torch.full((len(embeddings),), -100, device=self.device_name, dtype=torch.long)
        if target_ids:
            labels[prompt_end + 1:] = torch.tensor(target_ids, device=self.device_name)
        return embeddings[None], labels[None], prompt_end, len(ids), len(target_ids)

    def forward(self, prompt, endpoint, codes, target, target_vector):
        inputs, labels, prompt_end, _, _ = self.inputs(prompt, endpoint, codes, target)
        output = self.lm(inputs_embeds=inputs, attention_mask=torch.ones(inputs.shape[:2], device=inputs.device, dtype=torch.long),
                         labels=labels, output_hidden_states=True, use_cache=False)
        predicted = self.alignment(output.hidden_states[-1][0, prompt_end].float())
        gold = torch.as_tensor(target_vector, device=self.device_name, dtype=torch.float32)
        return output.loss + self.cfg["query"]["alignment"] * (1 - F.cosine_similarity(predicted[None], gold[None]).mean())

    @torch.inference_mode()
    def generate_query(self, prompt, endpoint=None, codes=None):
        inputs, _, _, input_count, _ = self.inputs(prompt, endpoint, codes)
        output = self.lm.generate(
            inputs_embeds=inputs, attention_mask=torch.ones(inputs.shape[:2], device=inputs.device, dtype=torch.long),
            do_sample=False, max_new_tokens=self.cfg["query"]["output_tokens"],
            pad_token_id=self.tokenizer.eos_token_id, use_cache=True
        )[0]
        return self.tokenizer.decode(output, skip_special_tokens=True).strip(), {
            "input_tokens": input_count + (inputs.shape[1] - input_count), "output_tokens": len(output)
        }

    @torch.inference_mode()
    def entities(self, text, query=False):
        key = (text, query)
        if key in self.entity_cache:
            return self.entity_cache[key], {"input_tokens": 0, "output_tokens": 0}
        ids = self.tokenizer.apply_chat_template(ner_messages(text, query=query), tokenize=True,
                                                  add_generation_prompt=True, enable_thinking=False)
        ids = ids[:self.cfg["query"]["input_tokens"]]
        inputs = torch.tensor([ids], device=self.device_name)
        with self.lm.disable_adapter():
            output = self.lm.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                      max_new_tokens=2048, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id, use_cache=True)[0, len(ids):]
        entities = entities_from(self.tokenizer.decode(output, skip_special_tokens=True))
        self.entity_cache[key] = entities
        return entities, {"input_tokens": len(ids), "output_tokens": len(output)}

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(path / "adapter")
        self.tokenizer.save_pretrained(path / "adapter")
        torch.save({"endpoint": self.endpoint_projection.state_dict(), "code": self.code_projection.state_dict(),
                    "alignment": self.alignment.state_dict()}, path / "heads.pt")
        save_json(path / "config.json", self.cfg)
