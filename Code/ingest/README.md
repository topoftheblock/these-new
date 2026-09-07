# ingest

Builds the vector database: embeds the corpus and writes a searchable index.

    corpus_source.py   which passages the knowledge base contains
    vector_store.py    the FAISS index and its passage-id map
    build_index.py     command-line entry point, resumable

## Run

    python3 -m preprocessing.fetch                  # first, if not already done
    python3 -m ingest.build_index --limit 2000      # trial
    python3 -m ingest.build_index                   # the whole corpus
    python3 -m ingest.build_index --resume          # continue after a stop

Output goes to `data/vector_store/`: `index.faiss`, `pids.npy`, `meta.json` and
`checkpoint.json`.

## What is in the knowledge base

Every passage attached to a row that has a well-formed answer.

MS MARCO v2.1 ships about ten candidate passages per query, retrieved by Bing and
then judged. Taking all of them from every usable row gives **122,678 unique
passages** from the validation split: the gold evidence for each query, plus the
near-misses that were retrieved alongside it. 124,316 before deduplication, so
about 1,600 texts appear under more than one query and are indexed once.

Two reasons this beats sampling a fixed-size corpus.

The knowledge base is defined by a **rule**, not by a sample. Every passage MS
MARCO retrieved for a usable query is in it and nothing else is, so there is no
distractor budget to justify and no sampling seed to defend.

Retrieval is hard in the right way. The non-gold passages are Bing's own
candidates for these queries, so they are topical near-misses rather than
off-domain filler that any encoder separates trivially.

## Index type

**Flat, and therefore exact**, by default. Every query is compared against every
passage with no approximation. 122,678 vectors at 768 dimensions occupy 0.38 GB,
which fits in memory, and exactness is worth more than speed here: an approximate
index would make retrieval depend on how the index was built rather than on the
query alone.

`--index ivfpq` builds an inverted file with product quantisation instead, for a
corpus too large for flat. It compresses roughly 32-fold, so 8.8 million passages
would fit in about 850 MB rather than 27 GB, at the cost of approximate search.
It needs training: at least `39 x nlist` vectors for the inverted file and
`39 x 2^nbits` for the quantiser, whichever is larger. `min_training_vectors()`
enforces the larger floor, because FAISS only warns and a warning is easy to miss
in a job that runs for hours.

If you use `ivfpq`, record it. Results then come from an approximate index, and
`meta.json` carries `exact: false` to make that checkable afterwards.

## Cost

Measured on this machine: **60 passages per second** on Apple MPS with
`all-mpnet-base-v2`, so the full corpus takes about **34 minutes**. Embedding
dominates; indexing is negligible.

The job checkpoints every `--checkpoint-every` batches and `--resume` continues
from the last checkpoint rather than re-embedding. Use it: a 34-minute job gets
interrupted.

## A crash worth knowing about

`vector_store.py` imports `torch` before `faiss`, and the import order is
load-bearing.

Both libraries link their own copy of `libomp`. On macOS, loading faiss first
makes the process die inside the first `IndexFlat.add` with **no traceback, no
exception and no exit status**. Standard output is lost with it, so the failure
looks like the script printing nothing at all, which sends you hunting for a bug
in your own code.

Verified by removing the guard and re-running: the script prints its first line
and then stops. With the guard it completes. If you add a module that touches
faiss, import torch first there too.
