"""
Robustness checks on the confirmatory analysis.

Nothing here changes a reported estimate or the verdict on H0. Each check asks
whether a conclusion drawn in Chapter 6 depends on a choice that could
reasonably have been made differently, and every number the thesis quotes from
this audit is reproduced by running this module.

    python3 -m evaluation.sensitivity

1. The Holm family. The confirmatory analysis corrects within each measure at
   each evidence level (8 families). Correcting all 66 estimates as one family
   is the most conservative alternative.

2. The justification rule for baseline C. The rule as run counts attribution
   phrases (``according to``, ``based on``) and a few other phrases as well as
   causal connectives. Re-scoring with the three connectives the thesis names
   shows which C effects depend on the attribution phrases.

3. The sentence splitter for baseline B. A citation written after the full stop
   is credited to the next sentence, and list items are not split. Re-scoring
   with a splitter that fixes both shows whether that matters.

4. The API backend. The provider reports a ``system_fingerprint`` per request.
   Repeats of the same query, condition and level sometimes land on different
   backends, which allows a within-cell test of whether the backend itself
   moves a score.

5. Truncated answers. Generations that hit the output limit are dropped and the
   adherence estimates recomputed.
"""

import collections
import copy
import json
import re

import numpy as np
from scipy import stats

import config
from evaluation.analysis import MEASURES, baseline_of, effects, holm

LEVELS = (("gold", None), ("ret", 10))


def load():
    gens = {}
    with open(config.RUNS / "generations.jsonl", encoding="utf-8") as fh:
        for line in fh:
            g = json.loads(line)
            gens[(g["query_id"], g["condition"], g["evidence_level"],
                  g["depth"], g["repeat"])] = g
    with open(config.RUNS / "scores.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return gens, rows


def gen_of(gens, row):
    return gens[(row["query_id"], row["condition"], row["evidence_level"],
                 row["depth"], row["repeat"])]


def rescored(rows, gens, baselines, rule):
    """Copies of the rows for these baselines, with adherence re-scored by ``rule``."""
    out = []
    for r in rows:
        if r["condition"][0] not in baselines:
            continue
        r = copy.deepcopy(r)
        r["metrics"]["adherence"]["score"] = rule(gen_of(gens, r)["output"] or "")
        r["metrics"]["adherence"]["applicable"] = True
        out.append(r)
    return out


def show(family, label):
    for c, e in family.items():
        print(f"    {label:4s} {c} vs {baseline_of(c)}: tau={e['tau']:+.3f}  "
              f"[{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]  p_holm={e['p_holm']:.4f}"
              f" {'*' if e['significant'] else ' '}")


# ----------------------------------------------------------------- 1. family
def global_holm(rows):
    print("1. Holm over all confirmatory estimates as one family")
    names, ps, taus = [], [], []
    per_family = 0
    for level, depth in LEVELS:
        for m in MEASURES:
            fam = effects(rows, level, m, config.PROMPT_CONDITIONS, depth=depth)
            per_family += sum(e["significant"] for e in fam.values())
            for c, e in fam.items():
                names.append(f"{c} {level} {m}"); ps.append(e["p"]); taus.append(e["tau"])
    rejected, adjusted = holm(ps)
    print(f"   {len(ps)} estimates; significant per family: {per_family}; "
          f"significant as one family: {sum(rejected)}")
    for i in np.argsort(adjusted):
        if rejected[i]:
            print(f"     {names[i]:28s} tau={taus[i]:+.3f}  p_holm={adjusted[i]:.4f}")
    print(f"   H0 rejected under both definitions: {per_family > 0 and any(rejected)}\n")


# --------------------------------------------------------------- 2. C rule
CAUSAL = re.compile(r"\b(because|since|therefore)\b", re.I)


def c_rule(rows, gens):
    print("2. Baseline C re-scored with only 'because', 'since', 'therefore'")
    fired = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["condition"].startswith("C"):
            m = (r["metrics"]["adherence"]["detail"].get("justify") or {}).get("marker")
            fired[r["condition"]][(m or "<none>").lower()] += 1
    for c in sorted(fired):
        print(f"   rule as run fired on, {c}: " +
              ", ".join(f"{w}={n}" for w, n in fired[c].most_common(6)))
    rr = rescored(rows, gens, "C", lambda t: 1.0 if CAUSAL.search(t) else 0.0)
    rate = collections.defaultdict(list)
    for r in rr:
        rate[r["condition"]].append(r["metrics"]["adherence"]["score"])
    print("   pass rate: " + "  ".join(f"{c} {np.mean(v):.3f}" for c, v in sorted(rate.items())))
    for level, depth in LEVELS:
        show(effects(rr, level, "adherence", config.PROMPT_CONDITIONS, depth=depth), level)
    print()


# --------------------------------------------------------------- 3. B rule
_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITE = re.compile(r"\[(\d+)\]")


def b_robust(text):
    text = re.sub(r"([.!?])\s+((?:\[\d+\])+)", r" \2\1", (text or "").strip())
    parts = [p for line in text.splitlines() for p in _SPLIT.split(line) if p.strip()]
    parts = [p for p in parts if re.search(r"[A-Za-z]{2,}", p)]
    return 1.0 if parts and all(1 <= len(_CITE.findall(p)) <= 3 for p in parts) else 0.0


def b_rule(rows, gens):
    print("3. Baseline B re-scored with trailing citations re-attached and list items split")
    rr = rescored(rows, gens, "B", b_robust)
    for level, depth in LEVELS:
        show(effects(rr, level, "adherence", config.PROMPT_CONDITIONS, depth=depth), level)
    print()


# ---------------------------------------------------------------- 4. backend
def backend(rows, gens):
    print("4. API backend (system_fingerprint)")
    by = collections.defaultdict(collections.Counter)
    for g in gens.values():
        by[g["condition"]][g["system_fingerprint"]] += 1
    for c in sorted(by):
        f, n = by[c].most_common(1)[0]
        print(f"   {c}: {len(by[c]):3d} distinct fingerprints, modal share {n / sum(by[c].values()):.0%}")
    modal = by["B3"].most_common(1)[0][0]
    print(f"   within-cell test on {modal}: residual from the query-condition-level mean")
    for m in ("faithfulness", "relevance", "correctness_f1", "adherence"):
        groups = collections.defaultdict(list)
        for r in rows:
            src = r["metrics"]["correctness" if m == "correctness_f1" else m]
            if not src["applicable"]:
                continue
            v = src["detail"]["token_f1"] if m == "correctness_f1" else src["score"]
            groups[(r["query_id"], r["condition"], r["evidence_level"])].append(
                (v, gen_of(gens, r)["system_fingerprint"] == modal))
        on, off = [], []
        for vals in groups.values():
            if len({b for _, b in vals}) < 2:
                continue
            mu = np.mean([v for v, _ in vals])
            for v, b in vals:
                (on if b else off).append(v - mu)
        p = stats.ttest_ind(on, off, equal_var=False).pvalue
        print(f"     {m:15s} {len(on):3d} rows on, {len(off):3d} off: "
              f"{np.mean(on):+.4f} vs {np.mean(off):+.4f}  p={p:.3f}")
    print()


# -------------------------------------------------------------- 5. truncation
def truncation(rows, gens):
    print("5. Adherence with truncated generations dropped")
    kept = [r for r in rows if gen_of(gens, r).get("finish_reason") != "length"]
    print(f"   dropped {len(rows) - len(kept)} of {len(rows)}")
    for level, depth in LEVELS:
        show(effects(kept, level, "adherence", config.PROMPT_CONDITIONS, depth=depth), level)


def main():
    gens, rows = load()
    global_holm(rows)
    c_rule(rows, gens)
    b_rule(rows, gens)
    backend(rows, gens)
    truncation(rows, gens)


if __name__ == "__main__":
    main()
