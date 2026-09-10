"""
Estimate the average treatment effect and decide the hypothesis of Section 5.1.

Reads ``runs/scores.jsonl`` and writes ``runs/results.json``, printing the same
numbers as tables.

For each cell of the design and each of the four measures, tau_hat is the mean
within-query difference between a rewording and the canonical form of its own
baseline. Positive means the rewording scored higher, since every measure is
oriented so that higher is better.

Usage::

    python3 -m evaluation.run_analysis
    python3 -m evaluation.run_analysis --alpha 0.01
"""

import argparse
import json

import config
from evaluation.analysis import (ANY_DEPTH, MEASURES, cell_means, effects,
                                 verdict)
from experiment.design import evidence_depths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="scores.jsonl")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)

    with open(config.RUNS / args.scores, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    conditions = config.PROMPT_CONDITIONS
    # one column per cell of the design: the retrieved level once per depth,
    # gold once, which is the 36-cell shape of Section 5.4
    columns = [(lv, d) for lv in config.EVIDENCE_LEVELS
               for d in evidence_depths(lv)]

    def label(level, depth):
        return level if depth is None else f"{level}/k{depth}"

    # ------------------------------------------------ descriptive cell means
    print("mean of each measure per condition\n")
    means = {}
    for level, depth in columns:
        print(f"  {label(level, depth)}")
        header = "".join(f"{m:>15s}" for m in MEASURES)
        print(f"    {'':6s}{header}")
        for condition in conditions:
            line = f"    {condition:6s}"
            for measure in MEASURES:
                cells = cell_means(rows, [condition], level, measure,
                                   ANY_DEPTH if depth is None else depth)
                if condition in cells:
                    means[f"{condition}/{label(level, depth)}/{measure}"] = \
                        cells[condition]
                    line += f"{cells[condition]['mean']:15.3f}"
                else:
                    line += f"{'n/a':>15s}"
            print(line)
        print()

    # -------------------------------------------------------------- tau_hat
    print("tau_hat: each rewording against the canonical form of its own "
          "baseline\n"
          "         Equation 4.12, one estimate per measure, 95% CI, "
          "Holm within each measure\n")
    all_effects = {}
    for level, depth in columns:
        for measure in MEASURES:
            family = effects(rows, level, measure, conditions,
                             ANY_DEPTH if depth is None else depth, args.alpha)
            if not family:
                continue
            all_effects[(level, depth, measure)] = family
            print(f"  {label(level, depth)}  /  {measure}")
            for condition, res in family.items():
                mark = "*" if res.get("significant") else " "
                lo, hi = res["ci"]
                print(f"    {condition} vs {res['compared_with']}: "
                      f"tau={res['tau']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                      f"n={res['n']:3d}  p_holm={res.get('p_holm', float('nan')):.4f} {mark}")
            print()

    # ------------------------------------------------------------- verdict
    decision = verdict(all_effects)
    print("=" * 70)
    print(f"H0 (Section 5.1): every rewording leaves every measure unchanged")
    print(f"  estimates made      : {decision['n_tested']}")
    print(f"  significant (Holm)  : {decision['n_significant']}")
    if decision["h0_rejected"]:
        print(f"  -> H0 REJECTED. The rewordings below changed the measure named.")
        for hit in decision["significant"]:
            print(f"     {hit['condition']:4s} {hit['cell']:>9s} "
                  f"{hit['measure']:<14s} tau={hit['tau']:+.3f} "
                  f"p_holm={hit['p_holm']:.4f}")
    else:
        print("  -> H0 NOT rejected. No rewording moved any measure by more "
              "than this design can\n     distinguish from decoding noise. "
              "That is not the same as showing the effect is\n     zero; see "
              "the power discussion in Section 5.8.")
    print("=" * 70)

    path = config.RUNS / args.out
    path.write_text(json.dumps({
        "alpha": args.alpha,
        "measures": list(MEASURES),
        "cell_means": means,
        "tau_hat": {f"{label(lv, d)}/{m}": fam
                    for (lv, d, m), fam in all_effects.items()},
        "hypothesis": decision,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {path}")
    print("* marks an interval that excludes zero after Holm correction")


if __name__ == "__main__":
    main()
