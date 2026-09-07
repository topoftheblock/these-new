"""
The cost function.

    l(y) = sum_k lambda_k * n_k(y)

n_k is the breach count from measure k, and lambda_k says how serious a breach of
requirement k is in the setting the system is built for. Zero cost means the
answer met every requirement; larger is worse.

The weights are not fixed by the thesis
---------------------------------------

Section 4.9 argues at length that the four requirements are not equally serious
and that their relative weight is a property of the deployment setting, but no
values are assigned anywhere. Until they are, this module cannot know them, so
it does two things instead of guessing.

``EQUAL`` is the default: every weight is 1, and the cost is a straight count of
requirement breaches. It has no free parameters, which makes it the honest
primary analysis.

``GROUNDING_WEIGHTED`` doubles faithfulness. It exists so the primary result can
be re-run under a different weighting and reported as a sensitivity analysis. If
the ordering of the prompt conditions survives reweighting, that is worth
stating; if it flips, that is worth stating more.

Whichever is used, record the choice with the results. A cost is not comparable
across weightings.

Inapplicable measures
---------------------

A measure that cannot be computed contributes nothing and is excluded from the
sum rather than counted as a pass. Faithfulness under the empty evidence level is
the case that matters: with no passages, every claim is trivially unsupported, so
scoring it would report the absence of context as a failure of grounding.

That makes raw costs comparable **within** an evidence level but not across
them, since the empty level sums over fewer terms. This is not a problem for the
main result, which compares prompt conditions inside a level, but it does mean a
cost averaged over all four levels is meaningless. ``n_applicable`` is recorded
on every cost so the boundary is visible rather than implicit.
"""

from .metrics import ORDER

#: no free parameters; the cost is a count of breaches
EQUAL = {"faithfulness": 1.0, "relevance": 1.0,
         "correctness": 1.0, "adherence": 1.0}

#: sensitivity scheme: grounding weighted double
GROUNDING_WEIGHTED = {"faithfulness": 2.0, "relevance": 1.0,
                      "correctness": 1.0, "adherence": 1.0}

SCHEMES = {"equal": EQUAL, "grounding": GROUNDING_WEIGHTED}


def cost(results, weights=None):
    """Compute l(y) from a mapping of measure name to :class:`MetricResult`.

    Parameters
    ----------
    results : dict
        ``{measure_name: MetricResult}``.
    weights : dict, optional
        Measure name to lambda. Defaults to :data:`EQUAL`.

    Returns
    -------
    dict
        ``value`` is l(y). ``contributions`` gives the weighted term per
        measure. ``n_applicable`` and ``excluded`` record what was summed, so a
        cost can be checked against the terms that produced it.
    """
    weights = weights or EQUAL
    contributions, excluded = {}, []
    total = 0.0

    for name in ORDER:
        result = results.get(name)
        if result is None:
            excluded.append(name)
            continue
        if not result.applicable:
            excluded.append(name)
            continue
        term = weights.get(name, 1.0) * result.breaches
        contributions[name] = term
        total += term

    return {"value": total,
            "contributions": contributions,
            "n_applicable": len(contributions),
            "excluded": excluded}
