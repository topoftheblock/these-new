"""
Stage 2 of the experiment: generation.

Reads ``runs/retrieval.json`` and never calls the retriever.

Runs the full crossing of prompt conditions with evidence levels, over every
query, repeated ``N_REPEATS`` times. Cell order is randomised per query so that
drift over the collection period cannot align with a condition, and every
generation is an independent call with its timestamp recorded.

Queries whose retrieval failed are skipped at the retrieved evidence level only.
The other three levels set the context directly, so retrieval cannot fail in
them and they run over the full query set.

Usage::

    python3 -m experiment.run_generation --limit 2   # smoke test
    python3 -m experiment.run_generation             # full run
"""

import argparse
import itertools
import json
import random
import time
from datetime import datetime, timezone

import config
from rag import Generator, prompts

from .conditions import CONDITIONS
from .contexts import build_context


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="run only the first N queries")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", default="generations.jsonl")
    args = ap.parse_args(argv)

    queries = json.loads((config.RUNS / "queries.json").read_text())["queries"]
    corpus = json.loads((config.RUNS / "corpus.json").read_text())
    retrieval = json.loads((config.RUNS / "retrieval.json").read_text())["records"]
    by_id = {r["query_id"]: r for r in retrieval}

    if args.limit:
        queries = queries[:args.limit]

    expected = (len(queries) * len(config.PROMPT_CONDITIONS)
                * len(config.EVIDENCE_LEVELS) * config.N_REPEATS)
    print(f"{len(queries)} queries, {expected} cells before the retrieval gate")

    generator = Generator()
    rng = random.Random(args.seed)
    path = config.RUNS / args.out
    done = skipped = 0

    with open(path, "w", encoding="utf-8") as fh:
        for query in queries:
            record = by_id[query["query_id"]]
            cells = list(itertools.product(
                config.PROMPT_CONDITIONS, config.EVIDENCE_LEVELS,
                range(config.N_REPEATS)))
            rng.shuffle(cells)

            for condition, level, repeat in cells:
                if level == "ret" and not record["retrieval_success"]:
                    skipped += 1
                    continue

                passages, meta = build_context(record, corpus, level, rng)
                prompt = prompts.build(query["query"], passages,
                                       CONDITIONS[condition])

                started = time.time()
                answer, n_tokens = generator(prompt)
                fh.write(json.dumps({
                    "query_id": query["query_id"],
                    "category": query["category"],
                    "condition": condition,
                    "evidence_level": level,
                    "repeat": repeat,
                    "question": query["query"],
                    "reference_answer": query["answer"],
                    "gold_ids": query["gold_ids"],
                    "context_passage_ids": meta["passage_ids"],
                    "gold_positions": meta["gold_positions"],
                    "n_substitutions": meta["n_substitutions"],
                    "output": answer,
                    "n_output_tokens": n_tokens,
                    "latency_s": round(time.time() - started, 3),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")
                fh.flush()
                done += 1
                if done % 50 == 0:
                    print(f"  {done} generations")

    print(f"\nwrote {done} generations to {path}")
    print(f"skipped {skipped} cells at the retrieved level (retrieval failed)")


if __name__ == "__main__":
    main()
