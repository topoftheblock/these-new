"""Building the vector database from the MS MARCO passages."""

from .corpus_source import count, gold_ids, iter_passages, passage_id
from .vector_store import FaissStore

__all__ = ["FaissStore", "iter_passages", "gold_ids", "passage_id", "count"]
