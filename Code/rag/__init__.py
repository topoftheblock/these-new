"""
A simple retrieval-augmented generation pipeline.

Single-shot: retrieve once, generate once. No query rewriting, no reranking, no
self-correction, no agent loop. That is a deliberate restriction, not a
simplification for convenience: the system prompt reaches the answer along
exactly one path, which is what makes its effect separable from everything else.

The package is usable on its own::

    from rag import SimpleRAG
    rag = SimpleRAG.from_corpus({"p1": "some passage text", ...})
    print(rag.answer("a question"))

The experiment in ``experiment/`` drives the two stages separately instead.
"""

from .embedder import Embedder
from .generator import Generator
from .index import VectorIndex
from .pipeline import SimpleRAG
from .retriever import Retriever, RetrievalResult

__all__ = ["SimpleRAG", "Embedder", "VectorIndex", "Retriever",
           "RetrievalResult", "Generator"]
