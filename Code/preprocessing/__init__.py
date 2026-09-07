"""Turn the MS MARCO release into a query set and a knowledge base."""

from .filters import categorise, gold_passages, well_formed_answers
from .loader import load_rows
from .corpus import build_corpus

__all__ = ["load_rows", "well_formed_answers", "gold_passages",
           "categorise", "build_corpus"]
