"""
From scored generations to the average treatment effect.

The reported quantity is Equation 4.12 of the thesis, the average causal effect
of a prompt condition relative to its baseline, estimated once for each of the
four evaluation measures of Section 5.7:

    tau_hat(j, M) = mean over queries of
                        [ mean M(Y) under S_j  -  mean M(Y) under S_0 ]

There is no cost function and no pooling. M is instantiated four times, as
faithfulness, answer relevance, correctness and instruction adherence, and a
separate tau_hat is reported for each. Pooling them into one number would hide
direction: a rewording that improves grounding and damages relevance would
average to nothing.

Pairing
-------

Every comparison is paired at query level, because every query is run in every
condition and pairing removes the variation between queries. The n repetitions
of a cell are averaged first, so one query contributes one number per cell. The
repetitions measure decoding noise, not variation between queries, and treating
them as independent observations would overstate the sample size.

Uncertainty
-----------

tau_hat is a sample mean of paired differences, so it carries a standard error
and a confidence interval. They are reported with it because the hypothesis of
Section 5.1 is a statement about whether tau is zero, and a point estimate
alone cannot answer that: tau_hat is almost never exactly zero even when tau is.

Multiplicity
------------

H1 is an "at least one" claim over nine rewordings. Reading nine intervals and
declaring H1 confirmed because one of them excluded zero would report noise as
a finding. Holm's method is therefore applied across the nine rewordings within
each measure, and both the raw and the adjusted verdict are reported.
"""

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

#: Each baseline is its own reference. A0, B0 and C0 state different
#: requirement sets, so comparing B1 against A0 would confound a change of
#: wording with a change of what was asked for. Every contrast is therefore
#: taken within a baseline, against that baseline's canonical condition.
from experiment.conditions import BASELINE_OF, CANONICAL

#: the four measures, in the order Section 5.7 introduces them
MEASURES = ("faithfulness", "relevance", "correctness", "adherence")

#: "do not filter on depth". Distinct from None, which is the real depth value
#: of the gold cells, so the two must not collide.
ANY_DEPTH = object()


def baseline_of(condition):
    """The canonical condition a given condition is compared against."""
    return CANONICAL[BASELINE_OF[condition]]


def in_cell(row, condition, level, depth=ANY_DEPTH):
    """True when a scored row belongs to the cell asked for.

    ``depth`` is compared only at the retrieved level. The gold level sets the
    context directly and carries no depth, so asking for one there would match
    nothing; ``ANY_DEPTH`` means "do not filter on it".
    """
    if row["condition"] != condition or row["evidence_level"] != level:
        return False
    if depth is ANY_DEPTH or level != "ret":
        return True
    return row.get("depth") == depth


def measure_score(row, measure):
    """The value of M for one generation, or None when M does not apply.

    Adherence is undefined for baseline A, whose only requirement is grounding;
    an inapplicable measure is skipped rather than counted as a pass, which
    would put a free point into the mean.
    """
    result = (row.get("metrics") or {}).get(measure)
    if not result or not result.get("applicable", True):
        return None
    return float(result["score"])


def paired_scores(rows, condition, level, measure, depth=ANY_DEPTH):
    """Mean of M per query in one cell, as ``{query_id: score}``.

    The repeats are averaged here, before any pairing.
    """
    by_query = {}
    for row in rows:
        if not in_cell(row, condition, level, depth):
            continue
        value = measure_score(row, measure)
        if value is not None:
            by_query.setdefault(row["query_id"], []).append(value)
    return {q: float(np.mean(v)) for q, v in by_query.items() if v}


def ate(rows, condition, reference, level, measure, depth=ANY_DEPTH,
        confidence=0.95):
    """tau_hat for one condition and one measure: Equation 4.12, estimated.

    Returns the estimate, its standard error, a confidence interval and the
    number of queries it rests on. ``p`` is the two-sided p-value of the paired
    difference against zero, from a one-sample t-test on the differences, which
    is the same thing the interval says and is reported so the Holm correction
    has something to adjust.

    A positive tau_hat means the rewording scored *higher* than the canonical
    prompt on that measure, since every measure is oriented so that higher is
    better.
    """
    treatment = paired_scores(rows, condition, level, measure, depth)
    control = paired_scores(rows, reference, level, measure, depth)
    shared = sorted(set(treatment) & set(control))
    n = len(shared)

    empty = {"tau": float("nan"), "se": float("nan"), "ci": (float("nan"),) * 2,
             "p": float("nan"), "n": n, "measure": measure,
             "compared_with": reference,
             "depth": None if depth is ANY_DEPTH else depth}
    if n < 2:
        return empty

    diff = np.array([treatment[q] - control[q] for q in shared], dtype=float)
    tau = float(diff.mean())

    if np.allclose(diff, 0):
        # Every query moved by exactly nothing. The estimate is zero with no
        # uncertainty; a t statistic would be 0/0.
        return {**empty, "tau": 0.0, "se": 0.0, "ci": (0.0, 0.0), "p": 1.0,
                "note": "every paired difference is zero"}
    if np.std(diff, ddof=1) == 0:
        # Constant non-zero difference: no variance, so no interval. Report the
        # estimate and say why the interval is missing rather than inventing one.
        return {**empty, "tau": tau, "se": 0.0,
                "ci": (tau, tau), "p": 0.0,
                "note": "every paired difference is identical; no variance"}

    se = float(stats.sem(diff))
    half = se * stats.t.ppf(0.5 + confidence / 2, n - 1)
    result = stats.ttest_1samp(diff, 0.0)
    return {"tau": tau, "se": se, "ci": (tau - half, tau + half),
            "p": float(result.pvalue), "n": n, "measure": measure,
            "compared_with": reference,
            "depth": None if depth is ANY_DEPTH else depth}


def holm(pvalues, alpha=0.05):
    """Holm correction over one family. Returns ``(rejected, adjusted)``."""
    clean = [p if np.isfinite(p) else 1.0 for p in pvalues]
    if not clean:
        return [], []
    rejected, adjusted, _, _ = multipletests(clean, alpha=alpha, method="holm")
    return list(rejected), [float(p) for p in adjusted]


def effects(rows, level, measure, conditions, depth=ANY_DEPTH, alpha=0.05):
    """tau_hat for every rewording of every baseline, on one measure.

    Holm is applied across the rewordings in this call, which is the family the
    "at least one" hypothesis of Section 5.1 ranges over for this measure.
    """
    out = {}
    for condition in conditions:
        reference = baseline_of(condition)
        if condition == reference:
            continue                      # a canonical prompt is the reference
        result = ate(rows, condition, reference, level, measure, depth)
        if not np.isnan(result["tau"]):
            out[condition] = result

    names = list(out)
    rejected, adjusted = holm([out[n]["p"] for n in names], alpha)
    for name, rej, adj in zip(names, rejected, adjusted):
        out[name].update(p_holm=adj, significant=bool(rej))
    return out


def cell_means(rows, conditions, level, measure, depth=ANY_DEPTH):
    """Mean of M in each cell, for the descriptive table that precedes tau_hat.

    Returns ``{condition: {"mean": float, "n_queries": int}}``.
    """
    out = {}
    for condition in conditions:
        scores = paired_scores(rows, condition, level, measure, depth)
        if scores:
            out[condition] = {"mean": float(np.mean(list(scores.values()))),
                              "n_queries": len(scores)}
    return out


def verdict(all_effects):
    """Does the evidence reject H0?

    H0 (Section 5.1): every rewording leaves every measure unchanged. H1: at
    least one does not. The answer is a single sentence, and it is Holm-adjusted
    because H1 ranges over the whole family rather than over one chosen member.

    Parameters
    ----------
    all_effects : dict
        ``{(level, depth, measure): {condition: result}}``.

    Returns
    -------
    dict
        ``rejected`` and the conditions responsible.
    """
    hits = [(key, name, res) for key, family in all_effects.items()
            for name, res in family.items() if res.get("significant")]
    return {
        "h0_rejected": bool(hits),
        "n_significant": len(hits),
        "n_tested": sum(len(f) for f in all_effects.values()),
        "significant": [
            {"cell": f"{k[0]}" + (f"/k{k[1]}" if k[1] is not None else ""),
             "measure": k[2], "condition": name,
             "tau": res["tau"], "ci": list(res["ci"]),
             "p_holm": res.get("p_holm")}
            for k, name, res in sorted(hits, key=lambda h: h[2]["p_holm"])],
    }
