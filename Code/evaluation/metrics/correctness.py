"""
Correctness: does the answer agree with the reference?

Computed by program, not judged. The reference is the well-formed answer from
the MS MARCO annotation, which the preprocessing step required every query to
have.

Accuracy is read in the sense of Chen et al. (2024): an answer is correct when
it contains the reference answer, rather than when it matches it exactly. MS
MARCO's well-formed answers are complete sentences, so exact match would fail on
a correct answer phrased differently, and the measure would report a difference
in verbosity as a difference in correctness.

Token F1 and exact match are computed as well and kept in ``detail``. They are
not what enters the cost, but they let a reader see how far the containment
judgement is carrying the result.
"""

import re
import string

from .base import Metric, MetricResult

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalise(text):
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    text = text.lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def token_f1(prediction, reference):
    """Harmonic mean of token precision and recall after normalisation."""
    pred, ref = normalise(prediction).split(), normalise(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = {}
    for t in pred:
        if t in ref:
            common[t] = min(pred.count(t), ref.count(t))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


class Correctness(Metric):
    """Agreement with the annotated reference answer."""

    name = "correctness"
    needs_judge = False

    def score(self, record, judge=None):
        reference = (record.get("reference_answer") or "").strip()
        answer = (record.get("output") or "").strip()

        if not reference:
            return MetricResult(self.name, applicable=False,
                                detail={"reason": "no reference answer"})

        ref_n, ans_n = normalise(reference), normalise(answer)
        contains = bool(ref_n) and ref_n in ans_n
        exact = ref_n == ans_n
        f1 = token_f1(answer, reference)

        return MetricResult(
            name=self.name,
            breaches=0 if contains else 1,
            score=1.0 if contains else 0.0,
            detail={"contains_reference": contains,
                    "exact_match": exact,
                    "token_f1": round(f1, 4)},
        )
