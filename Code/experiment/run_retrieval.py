"""
Stage 1 of the experiment: retrieval.

Runs once for every query, before any prompt condition exists, and writes the
result to ``runs/retrieval.json``. Generation reads that file. Because the two
stages are separate passes over the data, there is no execution path along which
a prompt could reach the retriever.

Also applies the retrieval gate. A query succeeds when every passage its
annotation marks as necessary appears in the top K. Failures are recorded here
and excluded later, at the retrieved evidence level only.

Depth
-----

Section 5.1 runs the retrieved level at more than one depth, so this stage runs
once per depth and writes one file per depth, ``retrieval_k5.json`` and
``retrieval_k10.json``. The gate is recomputed at each: a query whose gold
evidence is missed at K = 5 may be recovered at K = 10, and pooling the two
would report a denominator that belongs to neither.

K = 0 is not run here. A retrieval of depth zero returns nothing, which is the
empty evidence level, and that level is already in the design; running it would
duplicate those cells under a second name. See :mod:`experiment.design`.

Usage::

    python3 -m experiment.run_retrieval              # every depth in config
    python3 -m experiment.run_retrieval --k 10       # one depth
"""

import argparse
import json

import config
from rag import Embedder, Retriever, VectorIndex


def load_index(embedder, store_dir=None):
    """Return ``(index, description)``.

    With ``--store``, the prebuilt vector database is loaded and nothing is
    re-embedded. Without it, a small in-memory index is built from
    ``runs/corpus.json``, which is only practical for the sampled corpus.
    """
    if store_dir:
        from ingest.vector_store import FaissStore
        store = FaissStore.load(store_dir)
        return store, (f"{store_dir}, {len(store):,} passages, "
                       f"exact={store.is_exact}")

    corpus = json.loads((config.RUNS / "corpus.json").read_text())
    ids = list(corpus)
    print(f"encoding {len(ids)} passages with {config.EMBEDDING_MODEL}")
    index = VectorIndex(embedder.encode([corpus[i] for i in ids]), ids)
    return index, f"in-memory {index.backend}, {len(index)} passages"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, action="append", dest="depths",
                    help="retrieval depth; repeat for several. "
                         "Default: every depth in config.RETRIEVED_DEPTHS")
    ap.add_argument("--store", default=str(config.VECTOR_STORE),
                    help="prebuilt vector database. The full corpus lives here; "
                         "without it a small in-memory index is built instead")
    args = ap.parse_args(argv)

    depths = args.depths or list(config.RETRIEVED_DEPTHS)
    if 0 in depths:
        raise SystemExit(
            "K = 0 is the empty evidence level, not a retrieval depth. "
            "It is already in the design; see experiment/design.py")

    queries = json.loads((config.RUNS / "queries.json").read_text())["queries"]
    embedder = Embedder()
    index, described = load_index(embedder, args.store)
    print(f"index: {described}")

    for k in depths:
        run_one_depth(queries, embedder, index, described, k)

    print("\nfailed queries are skipped at the retrieved level of their own "
          "depth only; every other level runs over the full query set")


def run_one_depth(queries, embedder, index, described, k):
    """Retrieve at one depth, apply the gate, and write that depth's file."""
    retriever = Retriever(embedder, index, k=k)
    results = retriever.retrieve([q["query"] for q in queries],
                                 [q["gold_ids"] for q in queries])

    records = []
    for query, result in zip(queries, results):
        record = result.to_dict()
        record["query_id"] = query["query_id"]     # real id, not the batch index
        record["category"] = query["category"]
        record["depth"] = k
        records.append(record)

    n_ok = sum(bool(r["retrieval_success"]) for r in records)
    by_cat = {}
    for r in records:
        hit, total = by_cat.get(r["category"], (0, 0))
        by_cat[r["category"]] = (hit + bool(r["retrieval_success"]), total + 1)

    # gold recall over all annotated passages, the second figure Section 5.2
    # reports: how many required passages arrived, not how many queries had all
    # of theirs arrive
    found = sum(len(set(r["gold_ids"]) & set(r["retrieved_ids"])) for r in records)
    needed = sum(len(r["gold_ids"]) for r in records)

    path = config.RUNS / f"retrieval_k{k}.json"
    path.write_text(json.dumps({
        "embedding_model": config.EMBEDDING_MODEL,
        "index": described,
        "k": k,
        "n_queries": len(records),
        "n_success": n_ok,
        "gold_recall": {"found": found, "needed": needed,
                        "rate": round(found / needed, 4) if needed else None},
        "records": records,
    }, indent=2), encoding="utf-8")

    print(f"\nK = {k}")
    print(f"  gate passed  : {n_ok}/{len(records)} queries")
    for cat, (hit, total) in sorted(by_cat.items()):
        print(f"    {cat:10s} {hit}/{total}")
    if needed:
        print(f"  gold recall  : {found}/{needed} = {found / needed:.1%}")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
