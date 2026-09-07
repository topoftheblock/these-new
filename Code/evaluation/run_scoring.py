"""
Score every generation.

Reads ``runs/generations.jsonl``, applies the four measures, computes the cost,
and writes ``runs/scores.jsonl``.

Records are shuffled before judging so that position in the judging sequence
cannot correlate with condition, and the judge never sees a condition label.

Usage::

    python3 -m evaluation.run_scoring                    # judged, needs a model
    python3 -m evaluation.run_scoring --program-only     # no judge, no download
    python3 -m evaluation.run_scoring --weights grounding
"""

import argparse
import json
import random

import config
from evaluation.cost import SCHEMES, cost
from evaluation.judge import Judge
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
    ap.add_argument("--weights", default="equal", choices=sorted(SCHEMES))
    ap.add_argument("--program-only", action="store_true",
                    help="skip the judged measures; no model is loaded")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args(argv)

    corpus = json.loads((config.RUNS / "corpus.json").read_text())
    with open(config.RUNS / args.generations, encoding="utf-8") as fh:
        records = [attach_passages(json.loads(line), corpus) for line in fh if line.strip()]

    # blinding: judged in random order, never with a condition label
    order = list(range(len(records)))
    random.Random(args.seed).shuffle(order)

    judge = None
    embedder = None
    if not args.program_only:
        from rag import Embedder, Generator
        embedder = Embedder()
        generator = Generator()
        judge = Judge(lambda p: generator(p)[0])

    metrics = build_metrics(args.program_only, embedder)
    weights = SCHEMES[args.weights]
    scored = [None] * len(records)

    for n, i in enumerate(order, 1):
        record = records[i]
        results = {m.name: m.score(record, judge if m.needs_judge else None)
                   for m in metrics}
        c = cost(results, weights)
        scored[i] = {
            "query_id": record["query_id"],
            "category": record["category"],
            "condition": record["condition"],
            "evidence_level": record["evidence_level"],
            "repeat": record["repeat"],
            "cost": c["value"],
            "n_applicable": c["n_applicable"],
            "excluded": c["excluded"],
            "metrics": {k: v.to_dict() for k, v in results.items()},
        }
        if n % 50 == 0:
            print(f"  scored {n}/{len(records)}")

    path = config.RUNS / args.out
    with open(path, "w", encoding="utf-8") as fh:
        for row in scored:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nscored {len(scored)} generations with weights '{args.weights}'")
    if judge:
        print(f"judge calls: {judge.n_calls}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
