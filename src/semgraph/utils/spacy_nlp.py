import spacy.cli
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "load",
]


def load(model_name: str = "en_core_web_sm"):
    try:
        nlp = spacy.load(model_name)
    except Exception:
        logging.warning(f"Spacy model '{model_name}' not found. Downloading...")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
    return nlp
