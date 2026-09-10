"""
Check retrieval against the raw annotation.

The gold passages of a query are the entries of its ``passages`` struct whose
``is_selected`` flag is 1. Everything downstream refers to them by an id minted
as ``"<query_id>-p<position>"``, and that id is built in two different modules:
:func:`preprocessing.filters.gold_passages` enumerates the transposed passage
list, while :func:`ingest.corpus_source.iter_passages` enumerates the raw
``passage_text`` array when building the index. If those two ever disagree
about what ``position`` means, every id would still look well-formed, retrieval
would still return five passages, and the gate would still report a number --
it would just be measuring the wrong thing.

So this compares *texts*, not ids. Ids are an implementation detail and one of
them is deliberately rewritten (see ``canonicalise_gold``); the text the
annotation marked is the ground truth.

Four checks:

1. the gold ids in queries.json resolve to exactly the texts the raw parquet
   marks is_selected == 1
2. the reference answer stored for a query is the row's own well-formed answer
3. the gate's verdict matches an independent recount from the texts
4. retrieved passages resolve, are distinct, and rank the gold ones sensibly

Usage::

    python3 test_retrieval.py                 # summary over all 100 queries
    python3 test_retrieval.py --show 3        # plus 3 worked examples
    python3 test_retrieval.py --query 445094  # one query in full
"""

import argparse
import json

import pyarrow.parquet as pq

import config


def raw_rows(parquet, wanted):
    """``{query_id: {"query":…, "gold_texts":[…], "all_texts":[…], "answer":…}}``.

    Read straight from the parquet struct, transposing it here rather than
    reusing the pipeline's loader: a test that shares the code it is testing
    cannot catch a fault in that code.
    """
    out = {}
    for batch in pq.ParquetFile(parquet).iter_batches(
            batch_size=1000,
            columns=["query_id", "query", "query_type", "passages",
                     "wellFormedAnswers"]):
        for qid, query, qtype, passages, wfa in zip(
                batch.column("query_id").to_pylist(),
                batch.column("query").to_pylist(),
                batch.column("query_type").to_pylist(),
                batch.column("passages").to_pylist(),
                batch.column("wellFormedAnswers").to_pylist()):
            if qid not in wanted:
                continue
            texts = passages.get("passage_text") or []
            flags = passages.get("is_selected") or []
            out[qid] = {
                "query": query,
                "query_type": qtype,
                "all_texts": list(texts),
                "gold_texts": [t for t, f in zip(texts, flags) if f == 1],
                "gold_positions": [i for i, f in enumerate(flags) if f == 1],
                "answer": (wfa or [None])[0],
            }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=str(config.CORPUS_PARQUET))
    ap.add_argument("--show", type=int, default=0, help="worked examples")
    ap.add_argument("--query", type=int, help="inspect one query id in full")
    args = ap.parse_args(argv)

    queries = json.loads((config.RUNS / "queries.json").read_text())["queries"]
    corpus = json.loads((config.RUNS / "corpus.json").read_text())
    retrieval = {}
    for depth in config.RETRIEVED_DEPTHS:
        payload = json.loads((config.RUNS / f"retrieval_k{depth}.json").read_text())
        retrieval[depth] = {r["query_id"]: r for r in payload["records"]}

    raw = raw_rows(args.parquet, {q["query_id"] for q in queries})
    failures = []

    for q in queries:
        qid = q["query_id"]
        row = raw.get(qid)
        if row is None:
            failures.append(f"{qid}: not found in the parquet")
            continue

        # 1. gold ids resolve to exactly the annotated texts
        resolved = [corpus.get(pid) for pid in q["gold_ids"]]
        if any(t is None for t in resolved):
            failures.append(f"{qid}: gold id does not resolve in the corpus")
            continue
        if set(resolved) != set(row["gold_texts"]):
            failures.append(
                f"{qid}: gold ids name {len(set(resolved))} distinct texts, the "
                f"annotation marks {len(set(row['gold_texts']))}; the position "
                "scheme disagrees between preprocessing and ingest")
            continue

        # 2. the stored reference answer is the row's own
        if q["answer"] != row["answer"]:
            failures.append(f"{qid}: reference answer differs from the parquet")

        # 3. category matches the gold count
        expected = "direct" if len(row["gold_texts"]) == 1 else "multi-hop"
        if q["category"] != expected:
            failures.append(f"{qid}: labelled {q['category']}, has "
                            f"{len(row['gold_texts'])} gold passages")

        # 4. retrieval: recount the gate from texts, independently
        gold_set = set(row["gold_texts"])
        for depth, records in retrieval.items():
            rec = records[qid]
            got = [corpus.get(pid) for pid in rec["retrieved_ids"]]
            if any(t is None for t in got):
                failures.append(f"{qid} k={depth}: a retrieved id does not resolve")
                continue
            if len(rec["retrieved_ids"]) != len(set(rec["retrieved_ids"])):
                failures.append(f"{qid} k={depth}: retrieved a duplicate id")
            if len(rec["retrieved_ids"]) > depth:
                failures.append(f"{qid} k={depth}: returned "
                                f"{len(rec['retrieved_ids'])} passages")
            hit_by_text = gold_set.issubset(set(got))
            if bool(rec["retrieval_success"]) != hit_by_text:
                failures.append(
                    f"{qid} k={depth}: gate says "
                    f"{bool(rec['retrieval_success'])}, recount from texts says "
                    f"{hit_by_text}")
            scores = rec.get("scores") or []
            if scores != sorted(scores, reverse=True):
                failures.append(f"{qid} k={depth}: scores are not descending")

    print(f"checked {len(queries)} queries against {args.parquet}")
    print(f"  gold ids resolve to the annotated texts")
    print(f"  reference answers match the annotation")
    print(f"  categories match the gold count")
    print(f"  gate recounted from texts at K = {sorted(retrieval)}")
    print(f"\n{len(failures)} failures")
    for f in failures[:20]:
        print("  FAIL", f)

    # worked examples
    shown = [q for q in queries if args.query in (None, q["query_id"])]
    for q in shown[:(1 if args.query else args.show)]:
        qid = q["query_id"]
        row = raw[qid]
        print("\n" + "=" * 74)
        print(f"query_id {qid}  [{row['query_type']}, {q['category']}]")
        print(f"  query   : {row['query']}")
        print(f"  answer  : {(row['answer'] or '')[:100]}")
        print(f"  the row ships {len(row['all_texts'])} candidate passages; "
              f"is_selected == 1 at position(s) {row['gold_positions']}")
        for pid, text in zip(q["gold_ids"], (corpus[p] for p in q["gold_ids"])):
            mark = "OK " if text in row["gold_texts"] else "BAD"
            print(f"    {mark} gold id {pid}")
            print(f"        {text[:110]}")
        for depth in sorted(retrieval):
            rec = retrieval[depth][qid]
            print(f"  retrieved at K = {depth}: "
                  f"gate {'PASS' if rec['retrieval_success'] else 'FAIL'}")
            for rank, (pid, score) in enumerate(
                    zip(rec["retrieved_ids"], rec["scores"]), 1):
                text = corpus[pid]
                tag = "<-- GOLD" if text in row["gold_texts"] else ""
                print(f"    {rank:2d}. {score:.3f}  {pid:<16s} "
                      f"{text[:62]}... {tag}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
