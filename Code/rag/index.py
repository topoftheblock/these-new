"""
The vector store.

A flat inner-product index: every query is compared against every passage, with
no approximation. At the corpus size used here that is both exact and fast, and
exactness matters more than speed because an approximate index would make
retrieval depend on the index build rather than on the query alone.

FAISS is used when it is installed. The numpy path returns the same neighbours,
because a flat FAISS index performs the same exhaustive comparison. The two are
interchangeable; only the constant factor differs.
"""

import numpy as np


class VectorIndex:
    """Exact nearest-neighbour search over normalised vectors.

    Built once and never rebuilt during a run. Rebuilding mid-experiment would
    change what is retrieved and break the comparison between conditions.
    """

    def __init__(self, vectors, ids):
        if len(vectors) != len(ids):
            raise ValueError(f"{len(vectors)} vectors but {len(ids)} ids")
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        self.ids = list(ids)
        self._faiss = None
        try:
            import faiss
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)
            self.backend = "faiss"
        except ImportError:
            self.backend = "numpy"

    def __len__(self):
        return len(self.ids)

    def search(self, queries, k):
        """Return ``(ids, scores)`` for each query row.

        Parameters
        ----------
        queries : ndarray
            ``(n, dim)`` normalised query vectors.
        k : int
            Neighbours per query. Clipped to the corpus size.

        Returns
        -------
        (list of list of str, ndarray)
            Passage ids, best first, and the matching similarity scores.
        """
        k = min(k, len(self.ids))
        if self._faiss is not None:
            scores, idx = self._faiss.search(
                np.ascontiguousarray(queries, dtype="float32"), k)
        else:
            sims = queries @ self.vectors.T
            idx = np.argsort(-sims, axis=1)[:, :k]
            scores = np.take_along_axis(sims, idx, axis=1)
        return [[self.ids[i] for i in row] for row in idx], scores
