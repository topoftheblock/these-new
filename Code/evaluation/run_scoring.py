"""
Score every generation.

Reads ``runs/generations.jsonl``, applies the four measures of Section 5.7 and
writes ``runs/scores.jsonl``, one row per generation.

There is no cost function. Each measure is kept separate, because the analysis
estimates the treatment effect once per measure (Equation 4.12) rather than on
a pooled score.

Records are shuffled before judging so that position in the judging sequence
cannot correlate with condition, and the judge never sees a condition label.

Judging is I/O-bound, so ``--workers`` runs several records at once. Each record
is scored independently of every other, so concurrency changes only the wall
clock. The judge's prompts are module constants and the encoder is guarded by a
lock, since a single torch model is not safe to call from several threads.

Usage::

    python3 -m evaluation.run_scoring                    # judged, needs a key
    python3 -m evaluation.run_scoring --resume           # continue an interrupted run
    python3 -m evaluation.run_scoring --program-only     # no judge, no key

Resuming
--------

Rows are written as they are scored, not collected and dumped at the end, and
each carries its cell coordinate. ``--resume`` reads what is already there and
skips it. Judging the full set takes hours and is rate-limited, so a run that
loses everything on an interruption is not usable; this is the same guarantee
generation already had.
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from evaluation.judge import make_judge
from evaluation.metrics import (Adherence, AnswerRelevance, Correctness,
                                Faithfulness)


def attach_passages(record, corpus):
    """Resolve the recorded passage ids back to their text."""
    record["context_passages"] = [corpus[p] for p in
                                  record.get("context_passage_ids", [])
                                  if p in corpus]
    return record


def build_metrics(program_only, embedder=None):
    metrics = [Correctness(), Adherence()]
    if not program_only:
        metrics = [Faithfulness(), AnswerRelevance(embedder)] + metrics
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", default="generations.jsonl")
    ap.add_argument("--out", default="scores.jsonl")
    ap.add_argument("--program-only", action="store_true",
                    help="skip the judged measures; no model is loaded")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--workers", type=int, default=config.N_WORKERS,
                    help="records scored at once; affects only the wall clock")
    ap.add_argument("--resume", action="store_true",
                    help="skip records already present in the output file")
    args = ap.parse_args(argv)

    corpus = json.loads((config.RUNS / "corpus.json").read_text())
    with open(config.RUNS / args.generations, encoding="utf-8") as fh:
        records = [attach_passages(json.loads(line), corpus) for line in fh if line.strip()]

    path = config.RUNS / args.out

    def coord(row):
        """The cell coordinate of a record or a scored row."""
        return (row["query_id"], row["condition"], row["evidence_level"],
                row.get("depth"), row["repeat"])

    done_keys = set()
    if args.resume and path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done_keys.add(coord(json.loads(line)))
        print(f"resuming: {len(done_keys):,} records already scored")

    # blinding: judged in random order, never with a condition label
    order = [i for i in range(len(records)) if coord(records[i]) not in done_keys]
    random.Random(args.seed).shuffle(order)
    print(f"{len(order):,} records to score, {args.workers} at a time")

    judge = None
    embedder = None
    if not args.program_only:
        from rag import Embedder
        embedder = Embedder()          # relevance compares questions in this space
        judge = make_judge()
        print(f"judge: {judge.backend_name} / {judge.model_id}, "
              f"temperature {config.JUDGE_TEMPERATURE}")

    metrics = build_metrics(args.program_only, embedder)

    # One torch model behind the relevance measure, and torch forward passes
    # are not safe to run concurrently on one module. The lock serialises only
    # the encode, which is local and fast; the judge calls, which are the slow
    # part, stay parallel.
    encode_lock = threading.Lock()
    if embedder is not None:
        raw_encode = embedder.encode
        def locked_encode(*a, **kw):
            with encode_lock:
                return raw_encode(*a, **kw)
        embedder.encode = locked_encode

    def score_one(i):
        record = records[i]
        results = {m.name: m.score(record, judge if m.needs_judge else None)
                   for m in metrics}
        return i, {
            "query_id": record["query_id"],
            "category": record["category"],
            "query_type": record.get("query_type"),
            "condition": record["condition"],
            "evidence_level": record["evidence_level"],
            "depth": record.get("depth"),
            "cell": record.get("cell"),
            "repeat": record["repeat"],
            "generator_model": record.get("generator_model"),
            "judge_model": judge.model_id if judge else None,
            # one entry per measure: breaches, score, applicable, detail. The
            # analysis reads "score" as M(Y) and skips a measure that does not
            # apply, rather than scoring it as a pass.
            "metrics": {k: v.to_dict() for k, v in results.items()},
        }

    done = 0
    failures = []
    t0 = time.time()
    write_lock = threading.Lock()

    with open(path, "a" if args.resume else "w", encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(score_one, i) for i in order]
            for future in as_completed(futures):
                try:
                    _, row = future.result()
                except Exception as exc:               # noqa: BLE001
                    failures.append(str(exc))
                    continue
                # Written as it is produced, so an interrupted run keeps its
                # work. Order in the file is completion order; nothing reads
                # this file positionally.
                with write_lock:
                    out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    done += 1
                    n = done
                if n % 100 == 0:
                    rate = n / max(time.time() - t0, 1e-9)
                    print(f"  scored {n:,}/{len(order):,}  {rate:.2f}/s  "
                          f"eta {(len(order) - n) / rate / 60:.0f} min", flush=True)

    print(f"\nscored {done:,} records in {(time.time() - t0) / 60:.1f} min")
    if judge:
        print(f"judge calls: {judge.n_calls:,}")
    if failures:
        print(f"{len(failures)} records FAILED to score, e.g. {failures[0][:140]}")
        print("re-run with --resume to fill the gaps")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
