from typing import Protocol

import amrlib
from penman.graph import Graph

from semgraph import config
from semgraph.models.amrbart import AMRBart
from semgraph.utils import logging
from semgraph.utils.device import get_device

logger = logging.get_logger(__name__)

AMRLIB_MODEL_NAMES = [
    "model_parse_spring-v0_1_0",
    "model_parse_xfm_bart_base-v0_1_0",
    "model_parse_gsii-v0_1_0",
    "model_parse_t5-v0_2_0",
    "model_parse_xfm_bart_large-v0_1_0",
]


class Pipeline(Protocol):
    def parse_sents(self, texts: list[str] | str, **kwargs) -> list[Graph]: ...


def load_pipeline(model_name_or_path: str, **kwargs) -> Pipeline:
    logger.info(f"Loading AMR parsing model from {model_name_or_path}...")
    if model_name_or_path in AMRLIB_MODEL_NAMES:
        amrlib_data_dir = config.cache_dir / "amrlib" / "data" / model_name_or_path
        model = amrlib.load_stog_model(
            model_dir=amrlib_data_dir,
            device=get_device(config.device),
            batch_size=config.batch_size,
        )
    elif "AMRBART".lower() in model_name_or_path.lower():
        model = AMRBart.from_pretrained(
            model_name_or_path, batch_size=config.batch_size
        )
    else:
        raise ValueError(f"Model name '{model_name_or_path}' is not recognized.")
    logger.info("Model loaded successfully.")
    return model


if __name__ == "__main__":
    models = [
        "model_parse_spring-v0_1_0",
        "xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2",
    ]
    for model_name in models:
        print(f"Testing model: {model_name}")
        pipe = load_pipeline(model_name)
        # sentences = [
        #     "The cat sat on the mat.",
        #     "The dog barked loudly.",
        # ]
        sentences = [
            "U.S. and European officials may impose a 4th round of sanctions on Tehran when the U.N. Security Council considers the issue of Iran's nuclear energy program most likely in September 2007."
        ]
        graphs = pipe.parse_sents(sentences)
        for graph in graphs:
            print(f"Model: {model_name}")
            print("AMR Graph:")
            print(graph)
