"""
The knowledge base: every passage attached to a well-formed row.

MS MARCO v2.1 ships ten candidate passages per query on average, retrieved by
Bing and then judged. Taking every passage from every row that carries a
well-formed answer gives about 122,700 unique passages: the gold evidence for
each query, plus the near-misses that were retrieved alongside it.

Why this corpus rather than a sampled one
-----------------------------------------

The knowledge base is then defined by a **rule** rather than by a sample. Every
passage that MS MARCO retrieved for a usable query is in it, and nothing else is.
There is no distractor budget to choose and no sampling seed to defend, which
removes a free parameter from the design.

It also makes retrieval genuinely hard in the right way. The non-gold passages
are Bing's own candidates for these queries, so they are topical near-misses
rather than off-domain filler that any encoder separates trivially.

Deduplication
-------------

The same passage text can appear under several queries. Texts are deduplicated
so a passage is embedded and indexed once; the first identifier that carried it
wins. About 1,600 of 124,300 are duplicates.
"""

import pyarrow.parquet as pq


def passage_id(query_id, position):
    """Stable id: which row a passage came from, and where in that row."""
    return f"{query_id}-p{position}"


def iter_passages(path, batch_size=10_000, deduplicate=True, min_words=0):
    """Yield ``(pids, texts)`` batches from a well-formed-filtered parquet.

    Parameters
    ----------
    path : str or Path
        Output of :mod:`preprocessing.fetch`, already filtered to rows with a
        well-formed answer.
    batch_size : int
        Passages per yielded batch.
    deduplicate : bool
        Drop texts already seen. Keeps the index free of duplicates, at the cost
        of holding a set of seen texts in memory.
    min_words : int
        Skip passages shorter than this. Zero keeps everything, which is the
        default: a passage that MS MARCO retrieved is part of the corpus whether
        or not it is long.

    Yields
    ------
    (list of str, list of str)
        Passage ids and their texts.
    """
    seen = set()
    pids, texts = [], []

    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=1000, columns=["query_id", "passages"]):
        for query_id, passages in zip(batch.column("query_id").to_pylist(),
                                      batch.column("passages").to_pylist()):
            for position, text in enumerate(passages.get("passage_text") or []):
                if not text or len(text.split()) < min_words:
                    continue
                if deduplicate:
                    if text in seen:
                        continue
                    seen.add(text)
                pids.append(passage_id(query_id, position))
                texts.append(text)
                if len(pids) >= batch_size:
                    yield pids, texts
                    pids, texts = [], []

    if pids:
        yield pids, texts


def gold_ids(path):
    """Map each query id to the ids of its annotated supporting passages.

    Read from the same file, so the gold ids always line up with the corpus ids
    rather than being derived separately and drifting.
    """
    out = {}
    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=1000, columns=["query_id", "passages"]):
        for query_id, passages in zip(batch.column("query_id").to_pylist(),
                                      batch.column("passages").to_pylist()):
            selected = passages.get("is_selected") or []
            out[query_id] = [passage_id(query_id, i)
                             for i, flag in enumerate(selected) if flag == 1]
    return out


def canonical_ids(path, min_words=0):
    """Map passage text to the id it is indexed under.

    Deduplication keeps one copy of a repeated text and the first id wins, so a
    passage can be *in* the corpus under an id that is not the one its own row
    would mint. That matters for gold passages specifically. The retrieval gate
    asks whether the passage carrying the answer arrived; if the same text is
    indexed under another id, retrieving that id *is* retrieving the passage,
    but an uncanonicalised gold id would never match it and the query would be
    recorded as a retrieval failure for a reason that has nothing to do with
    the retriever.

    Returns
    -------
    dict
        ``{passage_text: passage_id}`` for the corpus as indexed.
    """
    out = {}
    for pids, texts in iter_passages(path, deduplicate=True, min_words=min_words):
        for pid, text in zip(pids, texts):
            out.setdefault(text, pid)
    return out


def count(path, deduplicate=True):
    """Total passages the corpus will contain. Cheap enough to call first."""
    return sum(len(p) for p, _ in iter_passages(path, deduplicate=deduplicate))
