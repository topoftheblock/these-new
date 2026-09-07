"""
The two stages wired together.

:class:`SimpleRAG` is the convenience face of the package: give it a corpus and
a question, get an answer. It is what you would use to try the system by hand.

The experiment does **not** use :meth:`SimpleRAG.answer`. It calls
:meth:`retrieve` for every query first, writes the result to disk, and only then
calls :meth:`generate`. Running the stages in separate passes is what makes it
impossible for a prompt condition to influence retrieval, rather than merely
unlikely. ``answer()`` collapses them for interactive use, and is therefore the
one method the experiment must avoid.
"""

from . import prompts
from .embedder import Embedder
from .generator import Generator
from .index import VectorIndex
from .retriever import Retriever


class SimpleRAG:
    """Retrieve once, generate once."""

    def __init__(self, retriever, generator=None):
        self.retriever = retriever
        self._generator = generator

    @classmethod
    def from_corpus(cls, corpus, embedder=None, generator=None, k=None):
        """Build from a ``{passage_id: text}`` mapping.

        Encoding the corpus is the expensive step and happens once here.
        """
        embedder = embedder or Embedder()
        ids = list(corpus)
        vectors = embedder.encode([corpus[i] for i in ids])
        index = VectorIndex(vectors, ids)
        return cls(Retriever(embedder, index, k=k), generator)

    @property
    def generator(self):
        """Loaded lazily, so retrieval-only runs never pay for the LLM."""
        if self._generator is None:
            self._generator = Generator()
        return self._generator

    def retrieve(self, question, gold_ids=None):
        """Stage 1. Returns a :class:`~rag.retriever.RetrievalResult`."""
        return self.retriever.retrieve(question, [gold_ids or []])[0]

    def generate(self, question, passages, system=None):
        """Stage 2. Returns ``(answer, n_tokens)``.

        Takes passages as an argument rather than fetching them, so the caller
        controls exactly what evidence reaches the model.
        """
        return self.generator(prompts.build(question, passages, system))

    def answer(self, question, system=None, corpus=None):
        """Both stages in one call. For interactive use only.

        The experiment must not use this: it hides the boundary between
        retrieval and generation that the design depends on.
        """
        result = self.retrieve(question)
        texts = ([corpus[p] for p in result.retrieved_ids] if corpus
                 else result.retrieved_ids)
        text, _ = self.generate(question, texts, system)
        return text
