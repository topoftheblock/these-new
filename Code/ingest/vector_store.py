"""
The vector database.

At 8.8 million passages the flat index the small corpus used is no longer an
option. Raw vectors would be 27 GB at float32, or 13.6 GB at float16, and the
machine this was written for has under 6 GB free. The store therefore never
materialises raw vectors: passages are embedded in batches and added straight to
a compressed index, and the vectors are discarded.

Two index types
---------------

``flat``: exhaustive inner-product search. Every query is compared against every
passage, with no approximation. This is the default, because the corpus built
from the well-formed rows is about 122,700 passages, whose vectors occupy 0.38 GB
at float32. That fits in memory, and exactness is worth more than speed at this
size: an approximate index would make retrieval depend on how the index was
built rather than on the query alone.

``ivfpq``: an inverted file with product quantisation. For a corpus large enough
that flat no longer fits.

The inverted file splits the space into ``nlist`` cells and searches only
``nprobe`` of them, so a query touches a fraction of the corpus rather than all
of it. Product quantisation then compresses each vector from 768 float32 values
(3072 bytes) into ``m`` bytes, a factor of about 32 at ``m = 96``.

That makes 8.8 million passages fit in roughly 850 MB. The cost is that search
is **approximate**: raising ``nprobe`` trades speed for recall. Set it high
enough that the retrieval gate is not measuring the index.

For a corpus small enough to fit in memory, use :class:`rag.index.VectorIndex`
instead. It is exact, and exactness is worth more than speed at that size.

Training
--------

An IVF index must see a sample of the data before it can assign cells. FAISS
wants at least 39 vectors per cell and prefers 256. With ``nlist = 4096`` that
is 160k to 1M vectors, so :meth:`FaissStore.train` takes a sample and fits on
that. Training happens once; adding is incremental after it.
"""

import json
import os
from pathlib import Path

# faiss and torch each ship their own libomp. On macOS the second one to
# initialise aborts the process with "OMP: Error #15" before any of this
# module's code runs, so the variable is set here, before either import, rather
# than left to the caller's shell: a run that dies at import time produces no
# output at all and the cause is easy to misread. The documented risk of this
# flag is unsafe *parallel* execution, which the omp_set_num_threads(1) below
# then rules out for faiss.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# torch must be imported before faiss. Both link their own copy of libomp, and
# on macOS loading faiss first makes the process die inside the first
# IndexFlat.add with no traceback and no exit status: stdout is lost with it, so
# the failure looks like the script producing nothing at all. Importing torch
# first pins the runtime that gets loaded and the crash goes away.
import torch  # noqa: F401  (import order matters, see above)

import faiss

# The same libomp conflict that dictates the import order above also crashes
# IndexFlat.search, which is where faiss parallelises across queries. Loading
# the index and encoding both succeed; the process then dies inside search with
# SIGSEGV and no traceback. Restricting faiss to one thread avoids it. Search is
# not the bottleneck here (100 queries against 122,678 passages), so the cost is
# not measurable, and correctness of the retrieval gate depends on it.
faiss.omp_set_num_threads(1)
import numpy as np

DEFAULT_NLIST = 4096      # ~sqrt(8.8M), rounded to a power of two
DEFAULT_M = 96            # subquantisers; 768 / 96 = 8 dims each
DEFAULT_NBITS = 8         # 256 centroids per subquantiser
DEFAULT_NPROBE = 64       # cells searched per query


class FaissStore:
    """A compressed, on-disk vector index with an external passage-id map.

    Passage ids are kept in a numpy array parallel to the index, rather than in
    FAISS itself. At this scale a Python dict of 8.8 million ids would cost more
    memory than the index.
    """

    def __init__(self, index, pids=None, nprobe=DEFAULT_NPROBE, dim=None):
        self.index = index
        self.pids = list(pids) if pids is not None else []
        self.dim = dim or index.d
        self.nprobe = nprobe
        if hasattr(index, "nprobe"):
            index.nprobe = nprobe

    # ------------------------------------------------------------- building
    @classmethod
    def flat(cls, dim):
        """Create an exact inner-product index. No training needed.

        Vectors are stored uncompressed, so memory is ``4 * dim`` bytes each.
        Search is exhaustive and therefore exact.
        """
        return cls(faiss.IndexFlatIP(dim), [], dim=dim)

    @classmethod
    def ivfpq(cls, dim, nlist=DEFAULT_NLIST, m=DEFAULT_M, nbits=DEFAULT_NBITS,
              nprobe=DEFAULT_NPROBE):
        """Create an untrained IVF-PQ index over inner-product similarity."""
        if dim % m:
            raise ValueError(f"dim {dim} is not divisible by m {m}")
        quantiser = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantiser, dim, nlist, m, nbits,
                                 faiss.METRIC_INNER_PRODUCT)
        return cls(index, [], nprobe=nprobe, dim=dim)

    @property
    def is_trained(self):
        return self.index.is_trained

    @property
    def is_exact(self):
        """True when search is exhaustive rather than approximate."""
        return isinstance(self.index, faiss.IndexFlat)

    def min_training_vectors(self):
        """Lower bound on the training sample.

        Two things are fitted, and each has its own floor. The inverted file
        needs about 39 points per cell. The product quantiser needs about 39 per
        centroid, and it has ``2 ** nbits`` centroids per subquantiser. FAISS
        warns rather than fails when either is short, and a warning is easy to
        miss in a job that runs for hours, so the larger floor is enforced here.
        """
        if not hasattr(self.index, "nlist"):
            return 0                      # flat index needs no training
        ivf = 39 * self.index.nlist
        pq = 39 * (2 ** self.index.pq.nbits) if hasattr(self.index, "pq") else 0
        return max(ivf, pq)

    def train(self, vectors):
        """Fit the cell assignment and the quantiser on a sample.

        A no-op for a flat index, which has nothing to fit.
        """
        if self.index.is_trained:
            return
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        needed = self.min_training_vectors()
        if len(vectors) < needed:
            raise ValueError(
                f"{len(vectors)} training vectors, but nlist="
                f"{self.index.nlist} with {2 ** self.index.pq.nbits} PQ "
                f"centroids needs at least {needed}. Sample more, or lower "
                f"nlist / nbits.")
        self.index.train(vectors)

    def add(self, vectors, pids):
        """Append a batch. Vectors are consumed and not retained."""
        if not self.is_trained:
            raise RuntimeError("train the index before adding to it")
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.pids.extend(pids)

    def __len__(self):
        return self.index.ntotal

    # -------------------------------------------------------------- search
    def search(self, queries, k):
        """Return ``(pids, scores)``, matching the small index's interface."""
        queries = np.ascontiguousarray(queries, dtype="float32")
        scores, idx = self.index.search(queries, k)
        out = [[self.pids[i] for i in row if 0 <= i < len(self.pids)]
               for row in idx]
        return out, scores

    # ------------------------------------------------------- serialisation
    def save(self, directory):
        """Write the index, the id map and the settings."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        np.save(directory / "pids.npy", np.asarray(self.pids))
        (directory / "meta.json").write_text(json.dumps({
            "dim": self.dim, "n_vectors": int(self.index.ntotal),
            "nlist": int(getattr(self.index, "nlist", 0)),
            "nprobe": int(self.nprobe),
            "index_type": type(self.index).__name__,
            "exact": self.is_exact,
        }, indent=2))
        return directory

    @classmethod
    def load(cls, directory, nprobe=None):
        directory = Path(directory)
        index = faiss.read_index(str(directory / "index.faiss"))
        pids = np.load(directory / "pids.npy", allow_pickle=True).tolist()
        meta = json.loads((directory / "meta.json").read_text())
        return cls(index, pids, nprobe=nprobe or meta.get("nprobe", DEFAULT_NPROBE),
                   dim=meta["dim"])
