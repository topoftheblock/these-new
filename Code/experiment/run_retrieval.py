"""
Stage 1 of the experiment: retrieval.

Runs once for every query, before any prompt condition exists, and writes the
result to ``runs/retrieval.json``. Generation reads that file. Because the two
stages are separate passes over the data, there is no execution path along which
a prompt could reach the retriever.

Also applies the retrieval gate. A query succeeds when every passage its
annotation marks as necessary appears in the top K. Failures are recorded here
and excluded later, at the retrieved evidence level only.

Usage::

    python3 -m experiment.run_retrieval
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
    ap.add_argument("--k", type=int, default=config.K)
    ap.add_argument("--store", default=None,
                    help="prebuilt vector database, e.g. ../data/vector_store")
    args = ap.parse_args(argv)

    queries = json.loads((config.RUNS / "queries.json").read_text())["queries"]
    embedder = Embedder()
    index, described = load_index(embedder, args.store)
    print(f"index: {described}")

    retriever = Retriever(embedder, index, k=args.k)
    results = retriever.retrieve([q["query"] for q in queries],
                                 [q["gold_ids"] for q in queries])

    records = []
    for query, result in zip(queries, results):
        record = result.to_dict()
        record["query_id"] = query["query_id"]     # real id, not the batch index
        record["category"] = query["category"]
        records.append(record)

    n_ok = sum(r["retrieval_success"] for r in records)
    by_cat = {}
    for r in records:
        hit, total = by_cat.get(r["category"], (0, 0))
        by_cat[r["category"]] = (hit + bool(r["retrieval_success"]), total + 1)

    path = config.RUNS / "retrieval.json"
    path.write_text(json.dumps({
        "embedding_model": config.EMBEDDING_MODEL,
        "index": described,
        "k": args.k,
        "n_queries": len(records),
        "n_success": n_ok,
        "records": records,
    }, indent=2), encoding="utf-8")

    print(f"\nretrieval success: {n_ok}/{len(records)}")
    for cat, (hit, total) in sorted(by_cat.items()):
        print(f"  {cat:10s} {hit}/{total}")
    print(f"wrote {path}")
    print("failed queries will be skipped at the retrieved level only")


if __name__ == "__main__":
    main()
