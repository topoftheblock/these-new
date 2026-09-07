"""
Assembling the knowledge base.

The corpus is *collected*, not written. Every passage comes from the MS MARCO
release; nothing is authored for the experiment. It has two parts:

**Gold passages.** For each sampled query, the passages its annotation marks as
supporting the answer. These are what retrieval must find, and what the
retrieval gate checks for.

**Distractor pool.** Passages from rows that were *not* sampled, and which were
never marked ``is_selected`` for their own query. Drawing distractors from the
same collection keeps the topical distribution of the corpus close to that of
the queries, so retrieval difficulty comes from genuine near-misses rather than
from off-domain filler.

The pool is sized so that the top K contains, on average, one gold passage and
four distractors on direct queries, and two gold and three distractors on
multi-hop queries (thesis Section 5.2).
"""

import config
from ingest.corpus_source import passage_id


def _usable(text, min_words):
    return bool(text) and len(text.split()) >= min_words


def build_corpus(sampled, spare_texts, rng,
                 n_passages=None, min_words=None):
    """Build the passage store and attach gold identifiers to each query.

    Parameters
    ----------
    sampled : list of dict
        The rows chosen for the query set, each already carrying ``query_id``
        and its gold passages.
    spare_texts : list of str
        Never-selected passage texts from rows that were not sampled.
    rng : random.Random
        Seeded, so the distractor pool is reproducible.
    n_passages, min_words : int, optional
        Default to :data:`config.N_PASSAGES` and
        :data:`config.PASSAGE_MIN_WORDS`.

    Notes on the length filter
    --------------------------
    ``min_words`` applies to distractors only. Gold passages are kept at any
    length, because dropping one would change the query's category without
    changing its label.

    Returns
    -------
    (corpus, queries) : (dict, list)
        ``corpus`` maps passage id to text. ``queries`` is the query set with a
        ``gold_ids`` list added to each entry.

    Notes
    -----
    Gold passage ids use the shared scheme from
    :func:`ingest.corpus_source.passage_id`, ``"<query_id>-p<position>"``, so
    that a gold id here names the same text as the corresponding entry in the
    vector store. Distractors, which exist only in the small sampled corpus,
    are ``"d<n>"``. A gold passage keeps its id across runs with the
    same seed, which is what lets a retrieval result be compared between runs.
    """
    n_passages = n_passages or config.N_PASSAGES
    min_words = min_words if min_words is not None else config.PASSAGE_MIN_WORDS

    corpus, queries = {}, []

    # Gold passages are kept whatever their length. The minimum-length rule
    # exists to keep fragments out of the distractor pool, and applying it to
    # gold would silently change a query's category: a multi-hop query that
    # loses one of its two gold passages is no longer multi-hop, but would
    # still be labelled as such and gated as though it needed only one.
    for row in sampled:
        gold_ids = []
        for passage in row["gold"]:
            text = passage.get("passage_text", "")
            if not text:
                continue
            pid = passage_id(row["query_id"], passage["position"])
            corpus[pid] = text
            gold_ids.append(pid)
        queries.append({
            "query_id": row["query_id"],
            "query": row["query"],
            "category": row["category"],
            "answer": row["answer"],
            "gold_ids": gold_ids,
        })

    rng.shuffle(spare_texts)
    for i, text in enumerate(spare_texts):
        if len(corpus) >= n_passages:
            break
        if _usable(text, min_words):
            corpus[f"d{i}"] = text

    return corpus, queries
