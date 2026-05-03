from __future__ import annotations

import penman
import torch
from transformers import BartForConditionalGeneration
from penman.models.amr import model as amr_model

from semgraph.models.amrbart.model_interface.tokenization_bart import AMRBartTokenizer

__all__ = [
    "AMRBart",
]


class AMRBart:
    tokenizer: AMRBartTokenizer

    def __init__(
        self,
        model: BartForConditionalGeneration,
        tokenizer: AMRBartTokenizer,
        batch_size: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | None = None,
        batch_size: int = 1,
        device: str | None = None,
    ):
        model = BartForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            device_map="auto" if device is None else device,
        )
        tokenizer = AMRBartTokenizer.from_pretrained(
            pretrained_model_name_or_path,
        )
        return cls(model=model, tokenizer=tokenizer, batch_size=batch_size)

    def parse_sents(self, texts: list[str] | str) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]
        outputs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            outputs.extend(self._parse_sents(batch))
        return outputs

    def _parse_sents(self, batch: list[str]) -> list[str]:
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if self.model.device.type != "cpu":
            inputs = {
                k: v.to(self.model.device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }
        results = self.model.generate(**inputs)
        outputs = []
        for token_ids in results:
            graph, status, _ = self.tokenizer.decode_amr(tokens=token_ids)
            gs = penman.encode(graph, model=amr_model)
            outputs.append(gs)
        return outputs
