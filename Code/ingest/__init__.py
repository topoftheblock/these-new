"""Building the vector database from the MS MARCO passages.

``FaissStore`` is exported lazily. Importing it pulls in faiss, which links its
own copy of libomp and on macOS aborts the process when another copy is already
loaded (see :mod:`ingest.vector_store` for the details and the mitigation).
Modules that only read the parquet -- :mod:`ingest.export_corpus`, anything
touching :mod:`ingest.corpus_source` -- have no use for faiss, and eagerly
importing it here made them fail for a reason that had nothing to do with what
they were doing. So the name resolves on first access instead.
"""

from .corpus_source import count, gold_ids, iter_passages, passage_id

__all__ = ["FaissStore", "iter_passages", "gold_ids", "passage_id", "count"]


def __getattr__(name):
    """Resolve ``FaissStore`` on first use, so faiss loads only when needed."""
    if name == "FaissStore":
        from .vector_store import FaissStore
        return FaissStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
