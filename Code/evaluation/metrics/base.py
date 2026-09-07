"""
The interface every measure implements.

A measure reports two things about one generation. ``breaches`` is the count
that enters the cost function, written n_k in the thesis. ``score`` is the rate
reported alongside it, on [0, 1] with higher meaning better, because a raw count
grows with answer length and is not comparable between answers.

Keeping both is deliberate. The cost needs the count; the results table needs
the rate.
"""

from dataclasses import dataclass, field


@dataclass
class MetricResult:
    """What one measure reports about one generation.

    Attributes
    ----------
    name : str
        The measure's identifier, matching the key used in the cost function.
    breaches : int
        n_k. How many times this generation broke the requirement. Zero means
        the requirement was met.
    score : float
        The reported rate on [0, 1], higher is better. For a measure with no
        natural denominator this is ``1.0 - min(breaches, 1)``.
    detail : dict
        Whatever the measure wants preserved for auditing: extracted claims,
        judge verdicts, which rule failed. Written to disk with the score so a
        result can be traced back to its evidence.
    applicable : bool
        False when the measure cannot be computed for this generation, for
        example correctness with no reference answer. An inapplicable measure
        contributes nothing to the cost and is excluded from its denominator
        rather than counted as a pass.
    """

    name: str
    breaches: int = 0
    score: float = 1.0
    detail: dict = field(default_factory=dict)
    applicable: bool = True

    def to_dict(self):
        return {"name": self.name, "breaches": self.breaches,
                "score": round(float(self.score), 4),
                "applicable": self.applicable, "detail": self.detail}


class Metric:
    """Base class. Subclasses implement :meth:`score`.

    ``needs_judge`` tells the runner whether to pass a judge in. Correctness and
    adherence are computed by program and set it False; faithfulness and answer
    relevance are judged and set it True.
    """

    name = "unnamed"
    needs_judge = False

    def score(self, record, judge=None):
        """Return a :class:`MetricResult` for one generation record."""
        raise NotImplementedError
