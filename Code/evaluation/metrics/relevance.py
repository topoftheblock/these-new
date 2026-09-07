"""
Answer relevance: does the answer address the question that was asked?

Reference-free, after Es et al. (2023). The judge writes several questions that
the answer would answer well, and those are compared with the question actually
asked by cosine similarity in the encoder's space. An answer that drifts to a
neighbouring question, hedges, or pads produces questions unlike the original and
scores low.

This is where abstention is read. Under the empty evidence level nothing in the
context supports an answer, so declining is the responsive behaviour and is
scored as relevant. None of the published baselines mandates a fixed abstention
string, so a refusal is detected by pattern rather than by exact match. Producing unrelated material instead is not. Without this
case a correct abstention would be punished for failing to answer a question it
was right not to answer.

The measure needs both a judge and an encoder. The encoder is the same one
retrieval uses, so no second embedding space is introduced.
"""

import numpy as np

import re

from experiment.conditions import DECLINE_PATTERNS

from .base import Metric, MetricResult

#: at or above this cosine, the answer is treated as addressing the question
THRESHOLD = 0.5

#: None of the published baselines mandates a fixed abstention string, so a
#: refusal is detected by pattern rather than by equality. The list is narrow,
#: so a paraphrase it misses is scored as an attempt to answer rather than as a
#: refusal: its hits are more trustworthy than its misses.
_DECLINE = re.compile("|".join(DECLINE_PATTERNS), re.I)


def declined(answer):
    """True when the answer reads as a refusal to answer."""
    return bool(_DECLINE.search(answer or ""))


class AnswerRelevance(Metric):
    """Similarity between the question asked and the questions the answer fits.

    Parameters
    ----------
    embedder : rag.Embedder
        The retrieval encoder, reused so both live in one space.
    n_questions : int
        How many questions the judge generates per answer.
    threshold : float
        Cosine at or above which the answer counts as relevant.
    """

    name = "relevance"
    needs_judge = True

    def __init__(self, embedder, n_questions=3, threshold=THRESHOLD):
        self.embedder = embedder
        self.n_questions = n_questions
        self.threshold = threshold

    def score(self, record, judge=None):
        if judge is None:
            raise ValueError("answer relevance requires a judge")

        answer = (record.get("output") or "").strip()
        question = (record.get("question") or "").strip()
        level = record.get("evidence_level")

        if not answer:
            return MetricResult(self.name, breaches=1, score=0.0,
                                detail={"reason": "empty answer"})

        # abstention is the responsive behaviour when nothing supports an answer
        abstained = declined(answer)
        if level == "empty":
            return MetricResult(
                name=self.name,
                breaches=0 if abstained else 1,
                score=1.0 if abstained else 0.0,
                detail={"mode": "abstention", "abstained": abstained,
                        "reason": "empty context; declining is responsive"})

        if abstained:
            # declined although evidence was supplied: unresponsive, and the
            # decision is recorded so it can be told apart from a low-similarity
            # answer that at least tried
            return MetricResult(self.name, breaches=1, score=0.0,
                                detail={"mode": "abstained_with_evidence"})

        generated = judge.generate_questions(answer, self.n_questions)
        if not generated:
            return MetricResult(self.name, applicable=False,
                                detail={"reason": "judge produced no questions"})

        vectors = self.embedder.encode([question] + generated)
        sims = vectors[1:] @ vectors[0]
        mean_sim = float(np.mean(sims))

        return MetricResult(
            name=self.name,
            breaches=0 if mean_sim >= self.threshold else 1,
            score=max(0.0, min(1.0, mean_sim)),
            detail={"mode": "similarity",
                    "mean_cosine": round(mean_sim, 4),
                    "threshold": self.threshold,
                    "generated_questions": generated},
        )
