import json

import penman
from amrlib.evaluate.smatch_enhanced import compute_smatch
from penman.models.amr import model as amr_model

from semgraph import config
from semgraph.data import load_dataset
from semgraph.evaluate import get_entries_from_string
from semgraph.models import load_pipeline
from semgraph.utils import logging

logger = logging.get_logger(__name__)

allowed_parse_model_names = [
    "model_parse_spring-v0_1_0",
    "model_parse_xfm_bart_base-v0_1_0",
    "model_parse_gsii-v0_1_0",
    "model_parse_t5-v0_2_0",
    "model_parse_xfm_bart_large-v0_1_0",
    "xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2",
]

allowed_generate_model_names = [
    "model_generate_t5wtense-v0_1_0",
]


def evaluate_model(model_name: str, dataset_name: str) -> dict:
    dataset = load_dataset(dataset_name)
    pipe = load_pipeline(model_name)

    input_graphs = []
    graphs_gold = []

    test_dataset = dataset["test"]

    for i in range(len(test_dataset)):
        item = test_dataset[i]
        graphs_gold.append(item)
        snt = item.metadata["snt"]
        input_graphs.append(snt)

    num_beams = 5 if "spring" in model_name else 4

    try:
        gen = pipe.parse_sents(
            input_graphs, disable_progress=False, num_beams=num_beams
        )
    except Exception as e:
        try:
            logger.warning(
                f"Error during parsing with disable_progress=False and num_beams={num_beams}: {e}. Retrying without disable_progress."
            )
            gen = pipe.parse_sents(input_graphs, num_beams=num_beams)
        except Exception as e:
            logger.warning(
                f"Error during parsing with num_beams={num_beams}: {e}. Retrying without num_beams."
            )
            gen = pipe.parse_sents(input_graphs)

    graphs_gen = [penman.decode(g, model=amr_model) for g in gen]

    # Instead of penman.dump to files, serialize to strings in memory
    gold_string = penman.dumps(graphs_gold, indent=4, model=amr_model)
    pred_string = penman.dumps(graphs_gen, indent=4, model=amr_model)

    # get_entries accepts either a filepath or an iterable of strings
    # Check what it expects — if it reads lines, wrap in StringIO:
    gold_entries = get_entries_from_string(gold_string)
    test_entries = get_entries_from_string(pred_string)

    precision, recall, f_score = compute_smatch(test_entries, gold_entries)

    logger.info(f"Evaluation results for model {model_name} on dataset {dataset_name}:")
    logger.info(f"\tPrecision: {precision:.4f}")
    logger.info(f"\tRecall: {recall:.4f}")
    logger.info(f"\tF-score: {f_score:.4f}")

    return {
        "model": model_name,
        "dataset": dataset_name,
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
    }


def main():
    output_file = config.cache_dir / "results" / "parse_model_results.jsonl"
    executed_model_results = {}
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    executed_model_results[result["model"]] = result
                except json.JSONDecodeError:
                    logger.warning(
                        f"Skipping malformed line in results file: {line.strip()}"
                    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a") as f:
        for parse_model_name in allowed_parse_model_names:
            if parse_model_name in executed_model_results:
                logger.info(
                    f"Skipping already evaluated model: {parse_model_name} (found in results file)"
                )
                continue
            logger.info(f"Evaluating model: {parse_model_name}")
            results = evaluate_model(
                model_name=parse_model_name,
                dataset_name="LDC2020T02",
            )
            f.write(json.dumps(results) + "\n")


if __name__ == "__main__":
    main()
