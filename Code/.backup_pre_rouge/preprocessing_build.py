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

#: what a row with no query_type is filed under, so the stratifier never has to
#: handle a missing key. No well-formed validation row lacks one.
UNKNOWN_TYPE = "UNKNOWN"
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
            "query_type": row.get("query_type") or UNKNOWN_TYPE,
            "answer": well_formed_answers(row)[0],
            "gold": gold_passages(row),
            "others": others,             # distractors, if this row is not sampled
        })
        stats[label] += 1
        stats[f"{label}/{row.get('query_type') or UNKNOWN_TYPE}"] += 1

    return pools, spare, stats


def canonicalise_gold(queries, corpus, parquet):
    """Rewrite gold ids to the ids the corpus actually indexes them under.

    The knowledge base deduplicates by text and keeps the first id, so a gold
    passage whose text also appears under an earlier query is indexed under
    that earlier id. Left alone, the gate would look for an id that is not in
    the index and record a retrieval failure that no retriever could avoid.

    Returns the number of ids rewritten, which belongs in the run log: it is
    small but it is not zero, and a reader should be able to see that the gold
    ids in ``queries.json`` are not always the ones the row itself would mint.
    """
    if not str(parquet).endswith(".parquet"):
        return 0                       # only the indexed corpus needs this

    from ingest.corpus_source import canonical_ids
    canonical = canonical_ids(parquet, min_words=config.PASSAGE_MIN_WORDS)

    rewritten = 0
    for query in queries:
        fixed = []
        for pid in query["gold_ids"]:
            text = corpus.get(pid)
            target = canonical.get(text, pid) if text else pid
            if target != pid:
                rewritten += 1
            fixed.append(target)
        query["gold_ids"] = fixed
    return rewritten


def stratified_sample(pool, n, rng, strata=None):
    """Draw ``n`` rows from one category, balanced across ``query_type``.

    MS MARCO labels every query with a coarse intent type, and those types are
    distributed very unevenly: on the well-formed validation rows DESCRIPTION is
    44% and PERSON 7%. An unstratified draw reproduces that imbalance, which
    leaves the rarer types with too few queries to say anything about and lets
    the aggregate be dominated by one type's behaviour.

    Balancing makes the type a usable covariate rather than a confound. It also
    costs something worth stating: the query set is then no longer a sample of
    what people ask Bing, so a figure averaged over it is an average over the
    *design*, not an estimate of performance on the natural distribution. Since
    every prompt contrast is taken within a query, that trade is a good one
    here -- the pairing is unaffected -- but any marginal figure reported over
    the query set has to be read as a balanced average.

    Parameters
    ----------
    pool : list of dict
        Candidate rows for one category, each carrying ``query_type``.
    n : int
        How many to draw in total.
    rng : random.Random
    strata : list of str, optional
        The types to balance over. Defaults to every type present in the pool,
        sorted, so the split is deterministic.

    Raises
    ------
    SystemExit
        If a stratum has fewer rows than the quota. Falling back to an
        unbalanced draw would produce a query set that silently fails the
        property it was built for.
    """
    by_type = {}
    for row in pool:
        by_type.setdefault(row["query_type"], []).append(row)

    types = strata or sorted(by_type)
    if not types:
        raise SystemExit("no query types found in the pool")

    quota, remainder = divmod(n, len(types))
    if remainder:
        raise SystemExit(
            f"{n} queries do not divide evenly over {len(types)} query types "
            f"({', '.join(types)}). Choose a multiple of {len(types)}.")

    short = [(t, len(by_type.get(t, []))) for t in types
             if len(by_type.get(t, [])) < quota]
    if short:
        detail = ", ".join(f"{t}: {have} of {quota}" for t, have in short)
        raise SystemExit(f"not enough rows to balance the query set: {detail}")

    chosen = []
    for query_type in types:
        # sort before sampling so the draw depends on the seed and not on the
        # order rows happened to arrive in
        ordered = sorted(by_type[query_type], key=lambda r: str(r["query_id"]))
        chosen.extend(rng.sample(ordered, quota))
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="dev_v2.1.json or a .jsonl split")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--n-direct", type=int, default=config.N_DIRECT)
    ap.add_argument("--n-multihop", type=int, default=config.N_MULTIHOP)
    ap.add_argument("--strata", nargs="*", default=list(config.QUERY_TYPES),
                    help="query types to balance over; the default is the five "
                         "MS MARCO intent labels")
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
        chosen = stratified_sample(pools[label], n, rng, args.strata)
        sampled.extend(chosen)
        sampled_ids.update(r["query_id"] for r in chosen)

    # distractors: non-selected passages from every row that was not sampled
    for label in pools:
        for row in pools[label]:
            if row["query_id"] not in sampled_ids:
                spare.extend(row["others"])

    corpus, queries = build_corpus(sampled, spare, rng)
    n_rewritten = canonicalise_gold(queries, corpus, args.input)

    config.RUNS.mkdir(parents=True, exist_ok=True)
    (config.RUNS / "queries.json").write_text(json.dumps(
        {"seed": args.seed, "source": args.input,
         "n_queries": len(queries), "queries": queries},
        indent=2, ensure_ascii=False), encoding="utf-8")
    # The passage store is NOT written here. Section 5.2 defines the knowledge
    # base by a rule over the whole collection, not by a sample, and
    # ingest.export_corpus builds it from the same parquet the index was built
    # from. Writing a sampled corpus here as well produced two files that
    # disagreed: retrieval returned ids from the full index that the sampled
    # store could not resolve.
    gold_only = {pid: text for pid, text in corpus.items()
                 if not pid.startswith("d")}
    (config.RUNS / "gold_passages.json").write_text(
        json.dumps(gold_only, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"read {stats['rows']} rows from {args.input}")
    print(f"  direct                : {stats['direct']}")
    print(f"  multi-hop             : {stats['multi-hop']}")
    print(f"  dropped, no well-formed answer : {stats['dropped_no_wfa']}")
    print(f"  dropped, no gold passage       : {stats['dropped_no_gold']}")
    print(f"\nsampled {len(queries)} queries (seed {args.seed})")
    counts = Counter((q["category"], q["query_type"]) for q in queries)
    types = sorted({t for _, t in counts})
    print(f"  {'':<12s}" + "".join(f"{t:>13s}" for t in types))
    for category in sorted({c for c, _ in counts}):
        row = [counts.get((category, t), 0) for t in types]
        print(f"  {category:<12s}" + "".join(f"{v:>13d}" for v in row)
              + f"   = {sum(row)}")
    if n_rewritten:
        print(f"\n  {n_rewritten} gold id(s) rewritten to the id the corpus "
              "indexes that text under (deduplication)")
    print(f"\n  gold passages written to runs/gold_passages.json")
    print("  the knowledge base itself: python3 -m ingest.export_corpus")


if __name__ == "__main__":
    main()
