"""
Correctness: does the answer agree with the reference answers?

Computed by program, not judged, with the two metrics MS MARCO's free-form
answer task is scored with, and that Lewis et al. (2020) report on it:
ROUGE-L and BLEU-1. Both are computed against *every* well-formed answer the
annotation provides for the query, not only the first, because a correct answer
may be phrased like any one of them.

Tokens
------
Lowercased, citation markers such as ``[3]`` removed, punctuation replaced by
spaces, split on whitespace. Citation markers are removed because baseline B
asks for them: left in, they would lower the precision of an answer for obeying
its prompt, and correctness would partly measure adherence.

ROUGE-L
-------
The F-measure of the longest common subsequence, as the MS MARCO / coco-caption
scorer computes it: LCS precision and LCS recall are each maximised over the
references and then combined with beta = 1.2, which weights recall slightly
above precision.

BLEU-1
------
Sentence-level BLEU with unigrams only (Papineni et al., 2002): unigram
precision with each token's count clipped at its largest count in any single
reference, multiplied by the brevity penalty against the reference closest in
length. The official scorer computes BLEU over the whole corpus; a per-answer
value is needed here because the estimand averages a per-answer measure, so the
sentence-level form is used.

Both scores lie on [0, 1], higher is better. Neither checks truth: both reward
overlapping words, so they are proxies for agreement with the reference.
"""

import math
import re
import string
from collections import Counter

from .base import Metric, MetricResult

_CITATION = re.compile(r"\[\d+\]")
_PUNCT = str.maketrans(string.punctuation, " " * len(string.punctuation))

#: ROUGE-L recall weight, as in the MS MARCO / coco-caption scorer
BETA = 1.2


def tokens(text):
    """Lowercase, drop citation markers, punctuation to spaces, split."""
    text = _CITATION.sub(" ", text or "")
    return text.lower().translate(_PUNCT).split()


def _lcs(a, b):
    """Length of the longest common subsequence of two token lists."""
    row = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = row[j]
            row[j] = prev + 1 if x == y else max(row[j], row[j - 1])
            prev = cur
    return row[-1]


def rouge_l(candidate, references, beta=BETA):
    """ROUGE-L F against several references, coco-caption style."""
    cand = tokens(candidate)
    refs = [r for r in (tokens(x) for x in references) if r]
    if not cand or not refs:
        return 0.0
    lcs = [_lcs(cand, r) for r in refs]
    precision = max(l / len(cand) for l in lcs)
    recall = max(l / len(r) for l, r in zip(lcs, refs))
    if precision == 0 or recall == 0:
        return 0.0
    return ((1 + beta ** 2) * precision * recall) / (recall + beta ** 2 * precision)


def bleu1(candidate, references):
    """Sentence-level BLEU-1 against several references."""
    cand = tokens(candidate)
    refs = [r for r in (tokens(x) for x in references) if r]
    if not cand or not refs:
        return 0.0
    ceiling = Counter()
    for r in refs:
        for tok, n in Counter(r).items():
            ceiling[tok] = max(ceiling[tok], n)
    clipped = sum(min(n, ceiling[tok]) for tok, n in Counter(cand).items())
    precision = clipped / len(cand)
    # the reference closest in length; a tie goes to the shorter one
    ref_len = min((abs(len(r) - len(cand)), len(r)) for r in refs)[1]
    penalty = 1.0 if len(cand) >= ref_len else math.exp(1 - ref_len / len(cand))
    return penalty * precision


def references_of(record):
    """Every reference answer for the record. Raises if none was attached.

    A silent fall-back to the single reference stored on a generation would
    score against one answer when the design says all of them, so the scoring
    runner must attach the full list and a record without one is an error.
    """
    if "references" not in record:
        raise ValueError(
            "record has no 'references'; the scoring runner attaches every "
            "well-formed answer from queries.json before scoring correctness")
    return [r for r in record["references"] if r and r.strip()]


class _Overlap(Metric):
    """Shared shape of the two correctness metrics."""

    needs_judge = False
    _fn = None

    def score(self, record, judge=None):
        refs = references_of(record)
        if not refs:
            return MetricResult(self.name, applicable=False,
                                detail={"reason": "no reference answer"})
        value = type(self)._fn(record.get("output") or "", refs)
        # a continuous score with no threshold, so there is no breach to count
        return MetricResult(self.name, breaches=0, score=value,
                            detail={"n_references": len(refs)})


class RougeL(_Overlap):
    """ROUGE-L F (beta 1.2) against every well-formed reference."""
    name = "rouge_l"
    _fn = staticmethod(rouge_l)


class Bleu1(_Overlap):
    """Sentence-level BLEU-1 against every well-formed reference."""
    name = "bleu1"
    _fn = staticmethod(bleu1)
