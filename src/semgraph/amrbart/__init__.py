from __future__ import annotations

from penman.graph import Graph
from transformers import BartForConditionalGeneration

from semgraph.amrbart.model_interface.tokenization_bart import AMRBartTokenizer

__all__ = [
    "AMRBart",
]


class AMRBart:
    tokenizer: AMRBartTokenizer

    def __init__(self):
        self.model = BartForConditionalGeneration.from_pretrained(
            "xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2"
        )
        self.tokenizer = AMRBartTokenizer.from_pretrained(
            "xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2"
        )

    def parse(self, texts: list[str] | str, batch_size: int = 1) -> list[Graph]:
        if isinstance(texts, str):
            texts = [texts]
        outputs = []
        for i in range(0, len(texts), batch_size):
            sentences = texts[i : i + batch_size]
            inputs = self.tokenizer(
                sentences, return_tensors="pt", padding=True, truncation=True
            )
            results = self.model.generate(**inputs)
            for token_ids in results:
                graph, status, _ = self.tokenizer.decode_amr(tokens=token_ids)
                outputs.append(graph)
        return outputs


if __name__ == "__main__":
    parser = AMRBart()
    sentence = [
        "The cat sat on the mat .",
        "The dog barked loudly .",
    ]
    amr = parser.parse(sentence)
    print(amr)
