"""
The language-model judge.

Faithfulness and answer relevance have no reference string to compare against,
so they are scored by a second model. It is a different model from the
generator (:data:`config.JUDGE_MODEL`), run at temperature 0, so the generator
never grades its own output and a verdict does not depend on a draw. This
thesis assumes language models are sensitive to their instructions, which means
the judge cannot be assumed reliable either. Three precautions follow Zheng et
al. (2023).

**The judge's own prompt is fixed.** It is a module constant, not a parameter.
Varying it between conditions would confound the judge with the thing being
measured.

**Outputs are shown blind.** :meth:`Judge.ask` receives the answer and the
context, never the condition label, the query id, or the evidence level. The
format condition is recognisable from its surface alone, so any judge error that
varied with condition would bias the estimated effect.

**Order is randomised.** The scoring runner shuffles records before judging, so
position in the judging sequence cannot correlate with condition.

Agreement with human labelling is measured separately, on held-out outputs, and
reported per condition rather than pooled. See :mod:`evaluation.agreement`.
"""

import re

import config

#: Fixed. Do not vary between conditions or runs.
CLAIM_PROMPT = """Break the following answer into atomic factual claims.
One claim per line, no numbering, no commentary.
If the answer makes no factual claim, output exactly: NONE

Answer: {answer}"""

VERDICT_PROMPT = """Decide whether the claim is supported by the passages.
Reply with exactly one word: SUPPORTED or UNSUPPORTED.
A claim is SUPPORTED only if the passages state it or directly entail it.
General knowledge that the passages do not state is UNSUPPORTED.

Passages:
{context}

Claim: {claim}"""

QUESTION_PROMPT = """Write {n} different questions that the answer below would
be a good answer to. One question per line, no numbering, no commentary.

Answer: {answer}"""


class Judge:
    """Wraps any callable that maps a prompt string to a response string.

    Parameters
    ----------
    backend : callable
        ``backend(prompt) -> str``. A :class:`rag.Generator` works directly if
        wrapped to return only the text.

    Notes
    -----
    The judge is deliberately dumb about the experiment. It never sees a
    condition label, so it cannot be influenced by one.
    """

    def __init__(self, backend):
        self.backend = backend
        self.n_calls = 0

    def ask(self, prompt):
        self.n_calls += 1
        out = self.backend(prompt)
        return out[0] if isinstance(out, tuple) else out

    # ---------------------------------------------------------------- claims
    def extract_claims(self, answer):
        """Decompose an answer into atomic claims. Returns a list of strings."""
        if not (answer or "").strip():
            return []
        raw = self.ask(CLAIM_PROMPT.format(answer=answer))
        if raw.strip().upper().startswith("NONE"):
            return []
        claims = [re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
                  for line in raw.splitlines()]
        return [c for c in claims if len(c.split()) >= 3]

    def verdict(self, claim, context):
        """True when the passages support the claim."""
        raw = self.ask(VERDICT_PROMPT.format(context=context, claim=claim))
        return "UNSUPPORTED" not in raw.strip().upper()

    # ------------------------------------------------------------- questions
    def generate_questions(self, answer, n=3):
        """Questions the answer would answer well. Used for answer relevance."""
        raw = self.ask(QUESTION_PROMPT.format(n=n, answer=answer))
        qs = [re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
              for line in raw.splitlines()]
        return [q for q in qs if q.endswith("?") or len(q.split()) >= 3][:n]


def make_judge(model_id=None, backend=None):
    """The judge the experiment uses: a fixed second model at temperature 0.

    Built through :func:`rag.make_generator` so the same key handling and the
    same backends apply. The judge's model id is recorded on the returned
    object so the scoring runner can write it into every row.
    """
    from rag import make_generator
    generator = make_generator(
        model_id=model_id or config.JUDGE_MODEL,
        backend=backend or config.JUDGE_BACKEND,
        temperature=config.JUDGE_TEMPERATURE,
        max_new_tokens=config.JUDGE_MAX_TOKENS)
    judge = Judge(lambda p: generator(p)[0])
    judge.model_id = generator.model_id
    judge.backend_name = generator.backend    # ``backend`` is the callable
    return judge
