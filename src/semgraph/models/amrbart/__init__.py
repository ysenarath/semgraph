from __future__ import annotations

import penman
import torch
from penman.models.amr import model as amr_model
from tqdm import auto as tqdm
from transformers import BartForConditionalGeneration

from semgraph.models.amrbart.model_interface.tokenization_bart import AMRBartTokenizer
from semgraph.utils import logging

__all__ = [
    "AMRBart",
]

logger = logging.get_logger(__name__)

# TODO: consider using a more informative fallback AMR,
# e.g., one that includes the original sentence as a comment
_FALLBACK_AMR = "(z0 / amr-empty)"


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
        for i in tqdm.trange(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            outputs.extend(self._parse_sents(batch))
        return outputs

    def _parse_sents(self, batch: list[str]) -> list[str]:
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,  # explicit, BART's hard limit
        )
        if self.model.device.type != "cpu":
            inputs = {
                k: v.to(self.model.device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }
        results = self.model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=5,
            early_stopping=True,
        )
        outputs = []
        for text, token_ids in zip(batch, results):
            try:
                graph, status, _ = self.tokenizer.decode_amr(tokens=token_ids)
            except Exception as e:
                logger.warning(
                    "Failed to decode AMR for input %r: %s. Using fallback.", text, e
                )
                outputs.append(_FALLBACK_AMR)
                continue
            try:
                gs = penman.encode(graph, model=amr_model)
            except Exception as e:
                logger.warning(
                    "Failed to encode AMR for input %r: %s. Using fallback.", text, e
                )
                outputs.append(_FALLBACK_AMR)
                continue
            outputs.append(gs)
        return outputs
