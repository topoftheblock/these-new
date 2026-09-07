"""
The retrieval stage.

Retrieval depends on the query and the corpus, and on nothing else. In
particular it does not take the system prompt as an argument. That absence is
the whole point: it is what lets a change in the answer be attributed to the
prompt rather than to a change in the evidence.

The retriever also reports whether retrieval *succeeded*, meaning that every
passage a query needs was actually returned. Success is a property of the
query, not of any prompt condition, because the same passages are returned for
every condition.
"""

from dataclasses import dataclass, field

import config


@dataclass
class RetrievalResult:
    """What retrieval returns for one query.

    Attributes
    ----------
    query_id, query : str
        Identifiers carried through to generation.
    retrieved_ids : list of str
        Passage ids, best first.
    scores : list of float
        Similarity of each retrieved passage.
    gold_ids : list of str
        Passages the annotation marks as necessary. Empty when unknown, in
        which case ``success`` is ``None``.
    """

    query_id: object
    query: str
    retrieved_ids: list
    scores: list = field(default_factory=list)
    gold_ids: list = field(default_factory=list)

    @property
    def n_gold_retrieved(self):
        return len(set(self.gold_ids) & set(self.retrieved_ids))

    @property
    def success(self):
        """True when every required passage is present, None when unknown.

        Note the requirement is *all* gold passages, not merely one. A
        multi-hop query whose answer needs two passages is not answerable from
        one of them, so partial retrieval is a failure.
        """
        if not self.gold_ids:
            return None
        return set(self.gold_ids).issubset(set(self.retrieved_ids))

    def to_dict(self):
        return {
            "query_id": self.query_id,
            "query": self.query,
            "retrieved_ids": self.retrieved_ids,
            "scores": [round(float(s), 5) for s in self.scores],
            "gold_ids": self.gold_ids,
            "n_gold_retrieved": self.n_gold_retrieved,
            "retrieval_success": self.success,
        }


class Retriever:
    """Encode a query, search the index, return the top K passages."""

    def __init__(self, embedder, index, k=None):
        self.embedder = embedder
        self.index = index
        self.k = k or config.K

    def retrieve(self, queries, gold_ids=None):
        """Retrieve for one query string or a list of them.

        Parameters
        ----------
        queries : str or list of str
        gold_ids : list of list of str, optional
            Required passages per query, used only to compute ``success``.
            Retrieval itself never sees them.

        Returns
        -------
        list of RetrievalResult
        """
        if isinstance(queries, str):
            queries = [queries]
        vectors = self.embedder.encode(queries)
        ids, scores = self.index.search(vectors, self.k)
        gold_ids = gold_ids or [[] for _ in queries]
        return [
            RetrievalResult(query_id=i, query=q, retrieved_ids=r,
                            scores=list(s), gold_ids=g)
            for i, (q, r, s, g) in enumerate(zip(queries, ids, scores, gold_ids))
        ]
