"""
Write the passage store the generation stage reads.

Retrieval returns passage *ids*; generation needs the *text* behind them. This
module writes that lookup, and it must cover the same corpus the index was
built over or the two drift apart.

Why this exists
---------------

An earlier route built the store from a sampled distractor pool of a few
hundred passages. That contradicts Section 5.2 -- "No distractor pool is
sampled: the gold passages for a query compete against the whole corpus" -- and
it also breaks, quietly. Retrieval runs against the full index, so it returns
ids from all 122,678 passages, and a store holding only the sample cannot
resolve most of them. The generation stage then either raises or, worse, drops
the passages it cannot find and generates against a truncated context that
nothing in the output records.

So the store is built by the same rule as the index, from the same file:
every passage attached to a validation row with a non-empty wellFormedAnswers
field, deduplicated by text, first id wins.

Usage::

    python3 -m ingest.export_corpus                    # full corpus
    python3 -m ingest.export_corpus --used-only        # only ids some cell needs
"""

import argparse
import json

import config

from .corpus_source import iter_passages


def used_ids(runs=None):
    """Every passage id that some cell of the design will actually ask for.

    That is the gold passages of every query, plus everything retrieved at every
    depth. Reading the retrieval files rather than guessing means the set grows
    automatically when a new depth is run.
    """
    runs = runs or config.RUNS
    wanted = set()

    queries = json.loads((runs / "queries.json").read_text())["queries"]
    for q in queries:
        wanted.update(q["gold_ids"])

    for path in sorted(runs.glob("retrieval*.json")):
        payload = json.loads(path.read_text())
        for record in payload["records"]:
            wanted.update(record["retrieved_ids"])
            wanted.update(record["gold_ids"])
    return wanted


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=str(config.CORPUS_PARQUET))
    ap.add_argument("--out", default="corpus.json")
    ap.add_argument("--used-only", action="store_true",
                    help="keep only ids the retrieval files reference; the "
                         "corpus is unchanged, the lookup is just smaller")
    args = ap.parse_args(argv)

    wanted = used_ids() if args.used_only else None
    if wanted is not None:
        print(f"restricting to {len(wanted):,} ids referenced by the run")

    corpus, seen = {}, 0
    for pids, texts in iter_passages(args.parquet,
                                     min_words=config.PASSAGE_MIN_WORDS):
        for pid, text in zip(pids, texts):
            seen += 1
            if wanted is None or pid in wanted:
                corpus[pid] = text

    path = config.RUNS / args.out
    path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")

    print(f"read {seen:,} passages from {args.parquet}")
    print(f"wrote {len(corpus):,} to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    if wanted is not None:
        missing = wanted - set(corpus)
        if missing:
            print(f"WARNING: {len(missing)} referenced ids are not in the "
                  f"parquet, e.g. {sorted(missing)[:3]}")
        else:
            print("every referenced id resolved")
    elif seen != config.N_PASSAGES:
        print(f"note: {seen:,} passages, config.N_PASSAGES says "
              f"{config.N_PASSAGES:,}")


if __name__ == "__main__":
    main()
