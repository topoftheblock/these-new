"""
Compute the reported effect from the scores.

Reads ``runs/scores.jsonl`` and writes ``runs/results.json``, printing the same
numbers as a table.

Usage::

    python3 -m evaluation.run_analysis
"""

import argparse
import json

import config
from evaluation.analysis import (BASELINE, empirical_risk, evidence_check,
                                 interaction, prompt_effects)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="scores.jsonl")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)

    with open(config.RUNS / args.scores, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    conditions = config.PROMPT_CONDITIONS
    levels = config.EVIDENCE_LEVELS

    risks = {}
    print("empirical risk, mean cost per cell\n")
    header = f"{'':6s}" + "".join(f"{lv:>12s}" for lv in levels)
    print(header)
    for condition in conditions:
        line = f"{condition:6s}"
        for level in levels:
            mean, n = empirical_risk(rows, condition, level)
            risks[f"{condition}/{level}"] = {"mean_cost": mean, "n": n}
            line += f"{mean:12.3f}" if n else f"{'-':>12s}"
        print(line)

    print(f"\ntau_hat: cost relative to {BASELINE}, within each evidence level\n")
    effects = {}
    for level in levels:
        effects[level] = prompt_effects(rows, level, conditions, args.alpha)
        print(f"  {level}")
        for condition, res in effects[level].items():
            if res["test"] is None:
                print(f"    {condition}: too few paired observations")
                continue
            mark = "*" if res.get("significant") else " "
            print(f"    {condition}: tau={res['mean_difference']:+.3f} "
                  f"p={res['p']:.4f} p_holm={res['p_holm']:.4f} {mark}")

    print("\nmanipulation check: evidence levels at the baseline prompt\n")
    check = evidence_check(rows, levels, args.alpha)
    for level, res in check.items():
        if res["test"] is None:
            print(f"  {level}: too few paired observations")
            continue
        mark = "*" if res.get("significant") else " "
        print(f"  {level} vs {res['compared_with']}: "
              f"diff={res['mean_difference']:+.3f} p_holm={res['p_holm']:.4f} {mark}")

    print("\ninteraction: does the prompt matter more under corrupted evidence?\n")
    inter = interaction(rows, conditions)
    for condition, res in inter.items():
        print(f"  {condition}: delta={res['delta']:+.3f} "
              f"(corrupted {res['corrupted_effect']:+.3f}, "
              f"clean {res['clean_effect']:+.3f})")

    path = config.RUNS / args.out
    path.write_text(json.dumps(
        {"alpha": args.alpha, "baseline": BASELINE, "empirical_risk": risks,
         "prompt_effects": effects, "manipulation_check": check,
         "interaction": inter}, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    print("* marks significance after Holm correction within a family")


if __name__ == "__main__":
    main()
