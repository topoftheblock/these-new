"""
Command-line entry point for preprocessing.

Reads an MS MARCO split, applies the filters in :mod:`preprocessing.filters`,
samples the query set under a fixed seed, and writes two files to ``runs/``:

``queries.json``
    The query set. Each entry has ``query_id``, ``query``, ``category``,
    the well-formed ``answer`` used as the correctness reference, and
    ``gold_ids`` pointing into the corpus.

``corpus.json``
    The knowledge base, mapping passage id to text.

Usage::

    python3 -m preprocessing.build --input ../data/dev_v2.1.json

The report printed at the end states how many rows were read, how many fell
into each category, and how many were dropped by each rule. Those counts belong
in the thesis: they document what fraction of the release the query set was
drawn from.
"""

import argparse
import json
import random
import sys
from collections import Counter

import config

from .corpus import build_corpus
from .filters import categorise, gold_passages, well_formed_answers
from .loader import load_rows


def collect(path):
    """Partition a split into category pools plus a distractor reservoir.

    Each pool entry keeps its non-selected passage texts as well as its gold
    ones. Those become distractors if the row is not sampled, which is what
    lets the corpus reach its target size: the dropped rows alone are far too
    few once the input has already been filtered to well-formed answers.
    """
    pools = {"direct": [], "multi-hop": []}
    spare, stats = [], Counter()

    for row in load_rows(path):
        stats["rows"] += 1
        others = [p.get("passage_text", "") for p in row.get("passages") or []
                  if p.get("is_selected") != 1]

        label = categorise(row)
        if label is None:
            stats["dropped_no_wfa" if not well_formed_answers(row)
                  else "dropped_no_gold"] += 1
            spare.extend(others)          # dropped rows are pure distractors
            continue

        pools[label].append({
            "query_id": row["query_id"],
            "query": row["query"],
            "category": label,
            "answer": well_formed_answers(row)[0],
            "gold": gold_passages(row),
            "others": others,             # distractors, if this row is not sampled
        })
        stats[label] += 1

    return pools, spare, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="dev_v2.1.json or a .jsonl split")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--n-direct", type=int, default=config.N_DIRECT)
    ap.add_argument("--n-multihop", type=int, default=config.N_MULTIHOP)
    args = ap.parse_args(argv)

    pools, spare, stats = collect(args.input)
    want = {"direct": args.n_direct, "multi-hop": args.n_multihop}

    for label, n in want.items():
        if len(pools[label]) < n:
            sys.exit(f"error: {len(pools[label])} '{label}' rows available, "
                     f"need {n}. Use a larger split.")

    rng = random.Random(args.seed)
    sampled, sampled_ids = [], set()
    for label, n in want.items():
        ordered = sorted(pools[label], key=lambda r: str(r["query_id"]))
        chosen = rng.sample(ordered, n)
        sampled.extend(chosen)
        sampled_ids.update(r["query_id"] for r in chosen)

    # distractors: non-selected passages from every row that was not sampled
    for label in pools:
        for row in pools[label]:
            if row["query_id"] not in sampled_ids:
                spare.extend(row["others"])

    corpus, queries = build_corpus(sampled, spare, rng)

    config.RUNS.mkdir(parents=True, exist_ok=True)
    (config.RUNS / "queries.json").write_text(json.dumps(
        {"seed": args.seed, "source": args.input,
         "n_queries": len(queries), "queries": queries},
        indent=2, ensure_ascii=False), encoding="utf-8")
    (config.RUNS / "corpus.json").write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"read {stats['rows']} rows from {args.input}")
    print(f"  direct                : {stats['direct']}")
    print(f"  multi-hop             : {stats['multi-hop']}")
    print(f"  dropped, no well-formed answer : {stats['dropped_no_wfa']}")
    print(f"  dropped, no gold passage       : {stats['dropped_no_gold']}")
    print(f"sampled {len(queries)} queries, corpus of {len(corpus)} passages "
          f"(seed {args.seed})")
    if len(corpus) < config.N_PASSAGES:
        print(f"note: corpus is {len(corpus)}, below the {config.N_PASSAGES} "
              "target. Use a larger split for the full run.")


if __name__ == "__main__":
    main()
