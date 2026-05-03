from pathlib import Path

from amrlib.graph_processing.amr_loading_raw import load_raw_amr

from semgraph import config
from semgraph.utils import logging

logger = logging.get_logger(__name__)

amrlib_data_dir = config.cache_dir / "amrlib" / "data"

paths = [
    amrlib_data_dir / "amr_annotation_3.0" / "data" / "amrs" / "split",
]

name2index = {
    "ldc2020t02": 0,
    "amr_annotation_3": 0,
    "amr_annotation_3.0": 0,
}


def preprocess(name: str) -> dict[str, Path]:
    base_dir = paths[name2index[name.lower()]]
    out_dir = amrlib_data_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    # Loop through the dirctories
    for dirname in ("dev", "test", "training"):
        if dirname == "training":
            split = "train"
        elif dirname == "dev":
            split = "valid"
        else:
            split = dirname
        out_path = out_dir / f"{split}.txt"
        splits[split] = out_path
        if out_path.exists():
            logger.info("File %s already exists, skipping", out_path)
            continue
        entries = []
        dn = base_dir / dirname
        logger.info("Loading data from %s", dn)
        fpaths = [fn for fn in dn.iterdir() if fn.is_file()]
        for fpath in fpaths:
            entries += load_raw_amr(fpath)
        logger.info("Loaded {:,} entries".format(len(entries)))
        # Save the collated data
        logger.info("Saving data to %s", out_path)
        with open(out_path, "w") as f:
            for entry in entries:
                f.write("%s\n\n" % entry)
    return splits


if __name__ == "__main__":
    for name in name2index.keys():
        preprocess(name)
