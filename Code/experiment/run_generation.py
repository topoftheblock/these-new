"""
Stage 2 of the experiment: generation.

Reads the stored retrieval output and never calls the retriever. That is not an
implementation convenience: Section 4.7 assumes the system prompt has no effect
on which documents are retrieved, and running retrieval to completion before any
prompt condition exists is what enforces the assumption rather than asserting
it. No execution path here can reach the retriever.

Runs every cell of :mod:`experiment.design` over every query, ``N_REPEATS``
times. The generator is whatever :func:`rag.make_generator` returns for the
backend in :mod:`config`; with the API backend the key is read from the
environment or ``Code/.env`` and never from source.

Cell order is randomised per query so drift over the collection period cannot
align with a condition, and every generation is an independent call with its
timestamp recorded.

The *context*, by contrast, is not drawn from that stream. Passage order and
the corrupted level's substitutions are keyed on the query, the evidence level,
the depth and the repeat, so that every prompt condition sees a byte-identical
context and the contrast is between prompts rather than between evidence.

The retrieval gate is applied at the retrieved level only, and per depth: a
query that failed the gate at K = 5 is skipped in the K = 5 ``ret`` cells but
still runs in the K = 10 ones if it passed there. The three levels that set the
context directly cannot fail retrieval and run over the full query set.

Concurrency
-----------

Requests are I/O-bound, so ``--workers`` runs several in flight at once through
a thread pool. This changes how long the run takes and nothing else that the
experiment measures: each generation is still an independent request built from
the same context, the work is still submitted in the per-query randomised cell
order, and the timestamp recorded on a row is still the moment that row was
produced. The generator returns its provenance per call rather than storing it,
so two threads cannot attribute one request's fingerprint to another's answer,
and the output file is written under a lock so rows cannot interleave.

Resuming
--------

A full run is 30,000 generations. ``--resume`` reads the rows already written
and skips them, so an interrupted run continues instead of starting again. Rows
are keyed by query, condition, evidence level, depth and repeat, which is
exactly the cell coordinate, so a resumed run cannot silently duplicate or drop
a cell.

Usage::

    python3 -m experiment.run_generation --limit 2   # smoke test
    python3 -m experiment.run_generation             # full run
    python3 -m experiment.run_generation --resume    # continue an interrupted run
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import config
from rag import make_generator, prompts

from .conditions import CONDITIONS
from .contexts import build_context
from .design import cells, n_generations, summary


def load_retrieval(runs=None):
    """Return ``{depth: {query_id: record}}`` from the per-depth files.

    Reads ``retrieval_k*.json``. Falls back to a depth-less ``retrieval.json``
    if that is all there is, treating it as the reference depth, so a run made
    before the depth axis existed can still be read.
    """
    runs = runs or config.RUNS
    by_depth = {}

    for path in sorted(runs.glob("retrieval_k*.json")):
        payload = json.loads(path.read_text())
        by_depth[int(payload["k"])] = {r["query_id"]: r
                                       for r in payload["records"]}

    if not by_depth:
        legacy = runs / "retrieval.json"
        if not legacy.exists():
            raise SystemExit(
                "no retrieval output found. Run experiment.run_retrieval first")
        payload = json.loads(legacy.read_text())
        by_depth[int(payload.get("k", config.K))] = {
            r["query_id"]: r for r in payload["records"]}
        print(f"note: no per-depth files; read {legacy.name} "
              f"as depth K = {list(by_depth)[0]}")
    return by_depth


def gold_record(query):
    """A retrieval-shaped record carrying only what the annotation fixes.

    The gold, corrupted and empty levels build their context from ``gold_ids``
    alone. Passing them a real retrieval record would work but would also hand
    them ``retrieved_ids`` they must not read, so they are given a record that
    does not have any: if one of those levels ever reached for the retriever's
    output, it would raise here rather than quietly use it.
    """
    return {"query_id": query["query_id"], "gold_ids": query["gold_ids"]}


def row_key(row):
    """The cell coordinate of one written row, used to resume without gaps."""
    return (row["query_id"], row["condition"], row["evidence_level"],
            row["depth"], row["repeat"])


def already_done(path):
    """Cell coordinates already present in an output file."""
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                done.add(row_key(json.loads(line)))
    return done


def build_plan(queries, retrieval, plan, done, rng):
    """Every generation this run still has to make, in submission order.

    The per-query cell shuffle happens here, exactly as it did when the loop was
    sequential, so the order work is submitted in is still randomised within a
    query and drift over the collection period cannot align with a condition.

    Returns ``(items, skipped)``: the work, and how many cells the retrieval
    gate removed.
    """
    items, skipped = [], 0
    for query in queries:
        order = list(plan)
        rng.shuffle(order)
        for cell in order:
            for repeat in range(config.N_REPEATS):
                key = (query["query_id"], cell.condition, cell.level,
                       cell.depth, repeat)
                if key in done:
                    continue
                if cell.level == "ret":
                    record = retrieval[cell.depth][query["query_id"]]
                    if not record["retrieval_success"]:
                        skipped += 1
                        continue
                else:
                    record = gold_record(query)
                items.append((query, cell, repeat, record))
    return items, skipped


def generate_one(item, corpus, generator, seed):
    """Produce one row. Pure with respect to the run: touches no shared state.

    The context RNG is keyed on the query, evidence level, depth and repeat and
    never on the condition, so every condition sees a byte-identical context and
    a concurrent run builds exactly the contexts a sequential one would.
    """
    query, cell, repeat, record = item
    context_rng = random.Random(
        f"{seed}/{query['query_id']}/{cell.level}/{cell.depth}/{repeat}")
    passages, meta = build_context(record, corpus, cell.level, context_rng)
    prompt = prompts.build(query["query"], passages, CONDITIONS[cell.condition])

    started = time.time()
    out = generator.generate(prompt)
    return {
        "query_id": query["query_id"],
        "category": query["category"],
        "query_type": query.get("query_type"),
        "condition": cell.condition,
        "evidence_level": cell.level,
        "depth": cell.depth,
        "cell": cell.key,
        "repeat": repeat,
        "question": query["query"],
        "reference_answer": query["answer"],
        "gold_ids": query["gold_ids"],
        "context_passage_ids": meta["passage_ids"],
        "gold_positions": meta["gold_positions"],
        "n_substitutions": meta["n_substitutions"],
        "system_prompt": CONDITIONS[cell.condition],
        "generator_backend": generator.backend,
        "generator_model": generator.model_id,
        "temperature": generator.temperature,
        "max_new_tokens": generator.max_new_tokens,
        "dtype": str(generator.dtype),
        "system_fingerprint": out.get("system_fingerprint"),
        "finish_reason": out.get("finish_reason"),
        "output": out["text"],
        "n_output_tokens": out["n_tokens"],
        "n_sanitised_steps": out.get("n_sanitised_steps", 0),
        "latency_s": round(time.time() - started, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="run only the first N queries")
    ap.add_argument("--queries", default="queries.json",
                    help="query set to read. Point a smoke test at a copy "
                         "rather than editing the real one in place: the "
                         "design checks read queries.json too, and a run that "
                         "swaps it temporarily makes them fail for a reason "
                         "that has nothing to do with the design")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", default="generations.jsonl")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells already present in the output file")
    ap.add_argument("--model", default=None,
                    help="SMOKE TEST ONLY. Overrides config.GENERATOR_MODEL. "
                         "The generator is held constant for the whole of data "
                         "collection (Section 5.4), so a run made with this "
                         "flag is not comparable with one made without it and "
                         "must not be reported. The model actually used is "
                         "written into every row.")
    ap.add_argument("--backend", default=None, choices=["openai", "local"],
                    help="SMOKE TEST ONLY. Overrides config.GENERATOR_BACKEND. "
                         "Same caveat as --model.")
    ap.add_argument("--workers", type=int, default=config.N_WORKERS,
                    help="requests in flight at once. Affects how long the run "
                         "takes and nothing the experiment measures. 1 is the "
                         "old sequential behaviour.")
    args = ap.parse_args(argv)

    queries = json.loads((config.RUNS / args.queries).read_text())["queries"]
    corpus = json.loads((config.RUNS / "corpus.json").read_text())
    retrieval = load_retrieval()

    if args.limit:
        queries = queries[:args.limit]

    print(summary())
    print(f"\ncorpus: {len(corpus):,} passages")
    print(f"retrieval depths available: {sorted(retrieval)}")
    print(f"upper bound this run: "
          f"{n_generations(n_queries=len(queries)):,} generations\n")

    path = config.RUNS / args.out
    done = already_done(path) if args.resume else set()
    if done:
        print(f"resuming: {len(done):,} generations already written")

    if args.model or args.backend:
        print(f"WARNING: generator overridden to backend={args.backend!r} "
              f"model={args.model!r}. This is a smoke test, not a run of the "
              f"experiment.\n")
    generator = make_generator(model_id=args.model, backend=args.backend)
    print(f"generator: {generator.backend} / {generator.model_id}, "
          f"temperature {generator.temperature}, "
          f"max {generator.max_new_tokens} new tokens\n")
    rng = random.Random(args.seed)     # cell visiting order only
    items, skipped = build_plan(queries, retrieval, cells(), done, rng)
    print(f"{len(items):,} generations to make, {args.workers} at a time")
    if skipped:
        print(f"{skipped:,} cells skipped: the query failed the gate at that depth")

    write_lock = threading.Lock()
    written = 0
    failures = []
    t0 = time.time()

    with open(path, "a" if args.resume else "w", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(generate_one, item, corpus, generator, args.seed): item
                for item in items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    row = future.result()
                except Exception as exc:              # noqa: BLE001
                    query, cell, repeat, _ = item
                    failures.append((query["query_id"], cell.key, repeat, str(exc)))
                    continue
                # One lock, held only for the write, so rows from different
                # threads cannot interleave inside a line.
                with write_lock:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    written += 1
                    n = written
                if n % 200 == 0:
                    rate = n / max(time.time() - t0, 1e-9)
                    left = (len(items) - n) / rate if rate else 0
                    print(f"  {n:,}/{len(items):,}  {rate:.1f}/s  "
                          f"eta {left / 60:.0f} min")

    elapsed = time.time() - t0
    print(f"\nwrote {written:,} generations to {path} in {elapsed / 60:.1f} min")
    if done:
        print(f"skipped {len(done):,} already present (resumed)")
    if skipped:
        print(f"skipped {skipped:,} cells at the retrieved level: "
              "the query failed the gate at that depth")
    print(f"api calls {generator.n_calls:,}, retries {generator.n_retries:,}")
    if failures:
        print(f"\n{len(failures)} generations FAILED and were not written. "
              "Re-run with --resume to fill the gaps.")
        for qid, cell, repeat, err in failures[:5]:
            print(f"  q{qid} {cell} r{repeat}: {err[:110]}")


if __name__ == "__main__":
    main()
