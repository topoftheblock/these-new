"""
From scored generations to the reported effect.

The main result is the difference in average cost between a prompt condition and
the baseline, computed **within** each evidence level:

    tau_hat(j,k) = R_hat(S_j, c_k) - R_hat(S_0, c_k)

where R_hat is the empirical risk, the mean cost over the queries and repeats in
that cell.

All comparisons are paired at query level. Each query is run in every condition,
so the paired difference removes between-query variation, which is the whole
reason the design runs every query everywhere.

Three families of tests are pre-specified and each is corrected separately by
Holm's method:

1. the effect of the prompt at the retrieved level, which is the result of
   interest
2. a manipulation check, that the evidence levels differ at the baseline prompt.
   Reported as a validity check on the design, not as a claim about which
   evidence is better
3. one interaction contrast, asking whether the prompt matters more when the
   evidence is corrupted

Correcting the families separately rather than pooling them keeps the
manipulation check from inflating the correction on the result of interest.
"""

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

#: Each baseline is its own reference. A0, B0 and C0 state different
#: requirement sets, so comparing B1 against A0 would confound a change of
#: wording with a change of what was asked for. Every contrast is therefore
#: taken within a baseline, against that baseline's canonical condition.
from experiment.conditions import BASELINE_OF, CANONICAL


def baseline_of(condition):
    """The canonical condition a given condition should be compared against."""
    return CANONICAL[BASELINE_OF[condition]]


def empirical_risk(rows, condition, level):
    """Mean cost in one cell. Returns ``(mean, n)``."""
    costs = [r["cost"] for r in rows
             if r["condition"] == condition and r["evidence_level"] == level]
    return (float(np.mean(costs)) if costs else float("nan")), len(costs)


def paired_costs(rows, condition, level):
    """Mean cost per query in one cell, keyed by query id.

    Averaging the repeats before pairing is deliberate: the repeats measure
    decoding noise, not between-query variation, and treating them as
    independent observations would overstate the sample size.
    """
    by_query = {}
    for r in rows:
        if r["condition"] == condition and r["evidence_level"] == level:
            by_query.setdefault(r["query_id"], []).append(r["cost"])
    return {q: float(np.mean(c)) for q, c in by_query.items()}


def paired_test(treatment, baseline, alpha=0.05):
    """Paired test on two ``{query_id: cost}`` mappings.

    Uses a paired t-test where the differences look normal by Shapiro-Wilk, and
    Wilcoxon signed-rank otherwise. Which test ran is reported, because the
    choice is data-dependent and a reader should be able to see it.
    """
    shared = sorted(set(treatment) & set(baseline))
    if len(shared) < 3:
        return {"n": len(shared), "test": None, "p": float("nan"),
                "statistic": float("nan"), "mean_difference": float("nan")}

    a = np.array([treatment[q] for q in shared])
    b = np.array([baseline[q] for q in shared])
    diff = a - b

    if np.allclose(diff, 0):
        return {"n": len(shared), "test": "none", "p": 1.0,
                "statistic": 0.0, "mean_difference": 0.0,
                "note": "all paired differences are zero"}

    # Constant non-zero differences have no variance, so a normality test is
    # undefined and a t statistic is infinite. Every pair moved the same way, so
    # the sign test underlying Wilcoxon is the honest report.
    if np.std(diff) == 0:
        result = stats.wilcoxon(a, b)
        return {"n": len(shared), "test": "wilcoxon",
                "statistic": float(result.statistic), "p": float(result.pvalue),
                "mean_difference": float(np.mean(diff)),
                "note": "paired differences are constant; no variance to test"}

    normal = len(diff) >= 3 and stats.shapiro(diff).pvalue > alpha
    if normal:
        result = stats.ttest_rel(a, b)
        name = "paired t"
    else:
        result = stats.wilcoxon(a, b)
        name = "wilcoxon"

    return {"n": len(shared), "test": name,
            "statistic": float(result.statistic), "p": float(result.pvalue),
            "mean_difference": float(np.mean(diff))}


def holm(pvalues, alpha=0.05):
    """Holm correction over one family. Returns ``(rejected, adjusted)``."""
    clean = [p if np.isfinite(p) else 1.0 for p in pvalues]
    if not clean:
        return [], []
    rejected, adjusted, _, _ = multipletests(clean, alpha=alpha, method="holm")
    return list(rejected), [float(p) for p in adjusted]


def prompt_effects(rows, level, conditions, alpha=0.05):
    """Family 1: every variant against its own baseline, at one evidence level.

    Holm correction is applied across the whole family rather than per baseline,
    because the family is the set of tests the result of interest rests on.
    """
    out = {}
    for condition in conditions:
        reference = baseline_of(condition)
        if condition == reference:
            continue
        out[condition] = paired_test(paired_costs(rows, condition, level),
                                     paired_costs(rows, reference, level), alpha)
        out[condition]["compared_with"] = reference
    names = list(out)
    rejected, adjusted = holm([out[n]["p"] for n in names], alpha)
    for name, rej, adj in zip(names, rejected, adjusted):
        out[name].update(p_holm=adj, significant=bool(rej))
    return out


def evidence_check(rows, levels, alpha=0.05):
    """Family 2: do the evidence levels differ at the baseline prompt?

    A manipulation check. It establishes that the evidence manipulation moved
    the outcome at all. No claim about the relative merits of the levels is
    drawn from it.
    """
    reference = levels[0]
    base = paired_costs(rows, "A0", reference)
    out = {}
    for level in levels[1:]:
        out[level] = paired_test(paired_costs(rows, BASELINE, level), base, alpha)
    names = list(out)
    rejected, adjusted = holm([out[n]["p"] for n in names], alpha)
    for name, rej, adj in zip(names, rejected, adjusted):
        out[name].update(p_holm=adj, significant=bool(rej),
                         compared_with=reference)
    return out


def interaction(rows, conditions, corrupted="corr", clean="gold"):
    """Family 3: does the prompt matter more when the evidence is corrupted?

        delta_j = [R(S_j,corr) - R(S_0,corr)] - [R(S_j,gold) - R(S_0,gold)]
    """
    out = {}
    for condition in conditions:
        reference = baseline_of(condition)
        if condition == reference:
            continue
        a, _ = empirical_risk(rows, condition, corrupted)
        b, _ = empirical_risk(rows, reference, corrupted)
        c, _ = empirical_risk(rows, condition, clean)
        d, _ = empirical_risk(rows, reference, clean)
        out[condition] = {"delta": (a - b) - (c - d), "compared_with": reference,
                          "corrupted_effect": a - b, "clean_effect": c - d}
    return out
