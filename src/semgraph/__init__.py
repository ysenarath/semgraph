from semgraph.utils import spacy_nlp

from .conf import config
from .utils.logging import get_logger

logger = get_logger(__name__)

__version__ = "0.0.1-dev"

__all__ = [
    "config",
    "nlp",
]


nlp = spacy_nlp.load("en_core_web_sm")
