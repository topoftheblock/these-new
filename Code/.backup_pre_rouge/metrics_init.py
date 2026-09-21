"""
The four measures.

Two are computed by program and two are judged:

============== ========= ==============================================
faithfulness   judged    claims traceable to the supplied passages
relevance      judged    answer addresses the question asked
correctness    program   agreement with the annotated reference
adherence      program   every rule-verifiable requirement satisfied
============== ========= ==============================================

Each returns a :class:`~evaluation.metrics.base.MetricResult` carrying the
breach count that enters the cost and the rate that goes in the results table.
"""

from .adherence import Adherence
from .base import Metric, MetricResult
from .correctness import Correctness
from .faithfulness import Faithfulness
from .relevance import AnswerRelevance

#: the order the cost function expects, matching n_1 to n_4 in the thesis
ORDER = ("faithfulness", "relevance", "correctness", "adherence")

__all__ = ["Metric", "MetricResult", "Faithfulness", "AnswerRelevance",
           "Correctness", "Adherence", "ORDER"]
