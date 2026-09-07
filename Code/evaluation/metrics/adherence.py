"""
Instruction adherence: did the answer do what the prompt told it to?

Prompt-level strict accuracy in the sense of Zhou et al. (2023): the answer
counts as adherent only if it satisfies every checked requirement at once, so
the score is binary.

Only requirements that can be **verified by rule** are checked. That is the
design principle of verifiable instructions: a rule either fires or it does not,
so the measure needs no judge and carries no judge bias.

Which requirements are checkable depends on the baseline
--------------------------------------------------------

The three baselines state different things, so adherence checks different
things for each, and for one of them it checks nothing at all.

======== ================================ =========================
baseline states                           rule-checkable
======== ================================ =========================
A        grounding                        nothing
B        grounding, citation, concision,  citation syntax and
         register                         density
C        grounding, comprehension,        justification present
         justification
======== ================================ =========================

Two exclusions are deliberate.

**Grounding is never checked here.** Every baseline states it, and it is exactly
what faithfulness measures. Checking it again would count one failure in both
n_1 and n_4 and silently weight grounding double in the cost.

**Baseline A has nothing left.** Its only requirement is grounding, so adherence
reports itself inapplicable rather than returning a free pass. That is a real
property of the design, not a gap: a prompt stating one requirement gives a
wording change almost nothing to act on, which is why three baselines of
different density are used.

Register and comprehension are not rule-checkable at all and are recorded in
``not_checked`` rather than silently ignored.
"""

import re

from experiment.conditions import BASELINE_OF

from .base import Metric, MetricResult

#: bracketed citation markers, as ALCE writes them: [1], [2][3]
_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
#: markers that an answer explains its choice, as baseline C requires
_JUSTIFY = re.compile(
    r"\b(because|since|therefore|as the (?:passage|document|text)|"
    r"this is why|the reason|according to|based on)\b", re.I)

NOT_CHECKED = {
    "ground": "measured by faithfulness; checking it here would count the same "
              "failure twice in the cost",
    "register": "tone is not decidable by rule",
    "comprehend": "understanding is not observable in the output",
    "concision": "stated without a threshold, so no rule can fire",
}


def sentences(text):
    return [s for s in _SENTENCE_SPLIT.split((text or "").strip()) if s.strip()]


class Adherence(Metric):
    """Prompt-level strict accuracy over the rule-verifiable requirements.

    Parameters
    ----------
    min_citations, max_citations : int
        Baseline B requires at least one and at most three documents cited per
        sentence. Both come from the published prompt, not from a choice here.
    """

    name = "adherence"
    needs_judge = False

    def __init__(self, min_citations=1, max_citations=3):
        self.min_citations = min_citations
        self.max_citations = max_citations

    # ------------------------------------------------------------ per rule
    def _check_citations(self, answer):
        """Every sentence carries between min and max bracketed markers."""
        lines = sentences(answer)
        if not lines:
            return False, {"reason": "empty answer"}
        counts = [len(_CITATION.findall(s)) for s in lines]
        ok = all(self.min_citations <= c <= self.max_citations for c in counts)
        return ok, {"per_sentence": counts,
                    "bounds": [self.min_citations, self.max_citations]}

    def _check_justification(self, answer):
        found = _JUSTIFY.search(answer or "")
        return bool(found), {"marker": found.group(0) if found else None}

    # --------------------------------------------------------------- score
    def score(self, record, judge=None):
        condition = record.get("condition", "")
        baseline = BASELINE_OF.get(condition)
        answer = (record.get("output") or "").strip()

        if baseline == "A":
            return MetricResult(
                self.name, applicable=False,
                detail={"baseline": baseline,
                        "reason": "baseline A states only grounding, which "
                                  "faithfulness measures; no rule-checkable "
                                  "requirement remains",
                        "not_checked": NOT_CHECKED})

        checked, detail = {}, {"baseline": baseline}
        if baseline == "B":
            ok, info = self._check_citations(answer)
            checked["cite"] = ok
            detail["cite"] = info
        elif baseline == "C":
            ok, info = self._check_justification(answer)
            checked["justify"] = ok
            detail["justify"] = info
        else:
            return MetricResult(self.name, applicable=False,
                                detail={"reason": f"unknown condition {condition!r}"})

        failed = [k for k, ok in checked.items() if not ok]
        detail.update(checked=checked, failed=failed, not_checked=NOT_CHECKED)
        return MetricResult(name=self.name, breaches=len(failed),
                            score=1.0 if not failed else 0.0, detail=detail)
