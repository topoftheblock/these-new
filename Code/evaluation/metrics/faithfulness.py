"""
Faithfulness: is every claim traceable to a supplied passage?

The measure the experiment turns on. Every prompt condition states a grounding
constraint, four of them in identical terms and the fifth with its modal force
weakened, and faithfulness is what shows whether that constraint is obeyed or
merely present in the wording.

Scored reference-free, in two judge steps after Es et al. (2023):

1. decompose the answer into atomic claims
2. for each claim, decide whether the passages support it

n_1 is the number of unsupported claims. The reported score is the supported
fraction, because a raw count grows with answer length and prompts change how
much a model writes.

The empty evidence level is a special case. With no passages, every claim is
unsupported by construction, so faithfulness there measures nothing about
grounding: it measures whether the model spoke at all. The measure reports
itself inapplicable rather than returning a misleading zero.
"""

from .base import Metric, MetricResult


class Faithfulness(Metric):
    """Fraction of the answer's claims that the context supports."""

    name = "faithfulness"
    needs_judge = True

    def score(self, record, judge=None):
        if judge is None:
            raise ValueError("faithfulness requires a judge")

        answer = (record.get("output") or "").strip()
        passages = record.get("context_passages") or []

        if not passages:
            return MetricResult(
                self.name, applicable=False,
                detail={"reason": "no context supplied; grounding is undefined "
                                  "when there is nothing to be grounded in"})
        if not answer:
            return MetricResult(self.name, breaches=0, score=1.0,
                                detail={"reason": "empty answer, no claims"})

        claims = judge.extract_claims(answer)
        if not claims:
            return MetricResult(self.name, breaches=0, score=1.0,
                                detail={"n_claims": 0,
                                        "reason": "answer makes no factual claim"})

        context = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
        verdicts = [judge.verdict(c, context) for c in claims]
        unsupported = [c for c, ok in zip(claims, verdicts) if not ok]

        return MetricResult(
            name=self.name,
            breaches=len(unsupported),
            score=sum(verdicts) / len(verdicts),
            detail={"n_claims": len(claims),
                    "n_supported": int(sum(verdicts)),
                    "unsupported_claims": unsupported},
        )
