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

Two generator backends exist behind one interface. :func:`make_generator`
picks the one named in :data:`config.GENERATOR_BACKEND`, so the rest of the
code never has to know which is in use.
"""

import config

from .embedder import Embedder
from .generator import Generator
from .index import VectorIndex
from .pipeline import SimpleRAG
from .retriever import Retriever, RetrievalResult

__all__ = ["SimpleRAG", "Embedder", "VectorIndex", "Retriever",
           "RetrievalResult", "Generator", "make_generator"]


def make_generator(model_id=None, backend=None, **kwargs):
    """Return a generator for ``backend``.

    Parameters
    ----------
    model_id : str, optional
        Overrides the configured model. Only for a smoke test or the judge.
    backend : str, optional
        ``"openai"`` or ``"local"``. Defaults to
        :data:`config.GENERATOR_BACKEND`.
    **kwargs
        Passed to the backend's constructor (``temperature``,
        ``max_new_tokens``, ...).

    Both backends expose the same call: ``generator(prompt) -> (text, n_tokens)``
    and the attributes ``model_id``, ``temperature``, ``dtype``, ``backend``
    and ``last_sanitised``, which the runners write into every row.
    """
    backend = backend or config.GENERATOR_BACKEND
    if backend == "openai":
        from .openai_generator import OpenAIGenerator
        return OpenAIGenerator(model_id=model_id, **kwargs)
    if backend == "local":
        generator = Generator(model_id=model_id or config.LOCAL_GENERATOR_MODEL,
                              **kwargs)
        generator.backend = "local"
        return generator
    raise ValueError(f"unknown generator backend {backend!r}; "
                     "config.GENERATOR_BACKEND must be 'openai' or 'local'")
