"""
Judge agreement with human labelling.

Section 5.4.1: faithfulness and answer relevance are scored by a language
model, and "since this thesis assumes that language models are sensitive to
their instructions, the judge cannot be assumed reliable". Three precautions
follow Zheng et al. (2023). Two are enforced elsewhere -- the judge's prompt is
a module constant in :mod:`evaluation.judge`, and :mod:`evaluation.run_scoring`
shuffles records and never shows a condition label. The third is measured here.

What is measured
----------------

Cohen's kappa between the judge's verdict and a human label, on about 100
held-out outputs, **reported per condition rather than pooled**. The reason for
the split is specific and worth keeping in view: the format condition is
recognisable from its surface alone, so a judge that treated list-formatted
answers differently would produce an error correlated with condition, and a
pooled kappa would hide exactly that. A pooled figure can look reassuring while
the per-condition figures disagree.

Kappa rather than raw agreement, because two labellers who both say "supported"
most of the time will agree often by chance, and the correction is what makes
the number comparable across conditions with different base rates.

Workflow
--------

1. ``sample`` draws the held-out outputs and writes a labelling sheet with the
   judge's verdict withheld, so a human label cannot be anchored by it.
2. A person fills in the ``human`` column.
3. ``report`` reads the completed sheet and computes kappa per condition.

The sample is drawn *before* the main analysis and its rows are excluded from
nothing: they are ordinary generations, scored like any other. What is held out
is the labelling effort, not the data.

Usage::

    python3 -m evaluation.agreement sample --n 100
    # ... a human fills in runs/agreement_sheet.jsonl ...
    python3 -m evaluation.agreement report
"""

import argparse
import json
import random
from collections import defaultdict

import config

#: measures that are judged, and therefore need an agreement figure
JUDGED = ("faithfulness", "relevance")

#: what a human writes in the ``human`` field
LABELS = {0: "the measure's requirement was broken", 1: "it was met"}


def cohens_kappa(a, b):
    """Cohen's kappa for two lists of binary labels.

    Returns ``None`` when it is undefined, which happens when one labeller used
    a single category throughout: chance agreement is then 1 and the correction
    divides by zero. That is reported rather than papered over, because a
    constant judge is a finding about the judge.
    """
    if not a or len(a) != len(b):
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n

    expected = 0.0
    for label in (0, 1):
        expected += (sum(x == label for x in a) / n) * (sum(y == label for y in b) / n)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def judged_verdict(score_row, measure):
    """The judge's binary verdict for one measure on one scored row.

    ``breaches == 0`` is the judge saying the requirement was met. Rows where
    the measure did not apply carry no verdict and return ``None``.
    """
    result = (score_row.get("metrics") or {}).get(measure)
    if not result or not result.get("applicable", True):
        return None
    return 0 if result.get("breaches", 0) > 0 else 1


def sample(scores, n, seed, measures=JUDGED):
    """Draw a stratified sample of scored rows for human labelling.

    Stratified by condition, so every one of the fifteen contributes and the
    per-condition kappas rest on comparable numbers of items. Within a
    condition the draw is uniform.
    """
    by_condition = defaultdict(list)
    for row in scores:
        for measure in measures:
            if judged_verdict(row, measure) is not None:
                by_condition[row["condition"]].append((row, measure))

    if not by_condition:
        return []

    rng = random.Random(seed)
    per = max(1, n // len(by_condition))
    out = []
    for condition in sorted(by_condition):
        items = by_condition[condition]
        out.extend(rng.sample(items, min(per, len(items))))
    rng.shuffle(out)                      # labelling order carries no condition
    return out[:n]


def write_sheet(items, path):
    """Write the labelling sheet, with the judge's verdict withheld.

    The sheet carries the condition id because the report needs it to group,
    but a labeller has no reason to read it and the judge's own verdict is not
    in the file at all: an anchored human label would make the kappa
    meaningless.
    """
    with open(path, "w", encoding="utf-8") as fh:
        for row, measure in items:
            fh.write(json.dumps({
                "query_id": row["query_id"],
                "condition": row["condition"],
                "evidence_level": row["evidence_level"],
                "depth": row.get("depth"),
                "repeat": row["repeat"],
                "measure": measure,
                "human": None,          # <- fill in: 1 requirement met, 0 broken
                "guidance": LABELS,
            }, ensure_ascii=False) + "\n")
    return path


def report(sheet, scores, measures=JUDGED):
    """Kappa per condition and per measure, plus the pooled figure.

    The pooled number is reported last and only for completeness. Section 5.4.1
    asks for the per-condition figures precisely because pooling can conceal an
    error that varies with condition.
    """
    index = {(r["query_id"], r["condition"], r["evidence_level"],
              r.get("depth"), r["repeat"]): r for r in scores}

    pairs = defaultdict(lambda: ([], []))
    unlabelled = missing = 0

    for line in sheet:
        if line.get("human") is None:
            unlabelled += 1
            continue
        key = (line["query_id"], line["condition"], line["evidence_level"],
               line.get("depth"), line["repeat"])
        row = index.get(key)
        if row is None:
            missing += 1
            continue
        verdict = judged_verdict(row, line["measure"])
        if verdict is None:
            continue
        human, judge = pairs[(line["condition"], line["measure"])]
        human.append(int(line["human"]))
        judge.append(verdict)

    out = {"per_condition": {}, "unlabelled": unlabelled, "unmatched": missing}
    all_human, all_judge = [], []

    for (condition, measure), (human, judge) in sorted(pairs.items()):
        k = cohens_kappa(human, judge)
        out["per_condition"][f"{condition}/{measure}"] = {
            "n": len(human), "kappa": k,
            "raw_agreement": (sum(x == y for x, y in zip(human, judge)) / len(human)
                              if human else None),
        }
        all_human += human
        all_judge += judge

    out["pooled"] = {"n": len(all_human), "kappa": cohens_kappa(all_human, all_judge)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="draw held-out outputs for labelling")
    s.add_argument("--scores", default="scores.jsonl")
    s.add_argument("--out", default="agreement_sheet.jsonl")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=config.SEED)

    r = sub.add_parser("report", help="kappa from a completed sheet")
    r.add_argument("--scores", default="scores.jsonl")
    r.add_argument("--sheet", default="agreement_sheet.jsonl")
    r.add_argument("--out", default="agreement.json")

    args = ap.parse_args(argv)

    def read(name):
        with open(config.RUNS / name, encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]

    if args.command == "sample":
        items = sample(read(args.scores), args.n, args.seed)
        path = write_sheet(items, config.RUNS / args.out)
        print(f"wrote {len(items)} items to {path}")
        print("fill in the 'human' field: 1 requirement met, 0 broken")
        return

    result = report(read(args.sheet), read(args.scores))
    print("Cohen's kappa, judge against human, per condition\n")
    for name, res in result["per_condition"].items():
        k = "undefined" if res["kappa"] is None else f"{res['kappa']:+.3f}"
        print(f"  {name:20s} n={res['n']:3d}  kappa={k}")
    pooled = result["pooled"]
    k = "undefined" if pooled["kappa"] is None else f"{pooled['kappa']:+.3f}"
    print(f"\n  {'pooled':20s} n={pooled['n']:3d}  kappa={k}")
    if result["unlabelled"]:
        print(f"\n{result['unlabelled']} rows still have no human label")

    path = config.RUNS / args.out
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
