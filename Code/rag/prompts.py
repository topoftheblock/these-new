"""
Assembling the model input.

The system prompt and the user turn are kept apart, and that separation is the
whole point of the module.

Chapter 1 defines S as "the standing instruction supplied in the system role,
fixed across requests, and ... distinct from the query q, which the user
supplies anew with each request". Chapter 3 rests the contribution on the same
distinction: the prompt-sensitivity literature varies the *user* instruction,
and what has not been tested is the *standing system* prompt. Concatenating the
two into one user turn would erase exactly the difference the thesis claims to
be measuring, and would do it silently, since the model would still answer and
the numbers would still look like results.

So :func:`build` returns the two parts separately and the generator puts each in
its own chat role. Passages travel with the user turn, because they are supplied
anew with each query in the way the query is, and because a citation instruction
in S has to refer to something the user turn provides.

Passages are numbered from 1 so a citation instruction has something to refer
to and so a passage's position is recoverable from the output. Their order is
preserved exactly: shuffling happens in :mod:`experiment.contexts`, before this
call, and is recorded there, because accuracy depends on where the relevant
passage sits (Liu et al., 2024).
"""

from dataclasses import dataclass

DEFAULT_SYSTEM = (
    "Answer the question using only the numbered passages below.\n"
    "If the passages do not contain the answer, say so.\n"
    "Answer in at most three sentences."
)


@dataclass(frozen=True)
class Prompt:
    """A system turn and a user turn, kept separate all the way to the model.

    Attributes
    ----------
    system : str
        S. The standing instruction, one of the fifteen conditions.
    user : str
        The passages and the question, supplied anew with each request.
    """

    system: str
    user: str

    def as_messages(self):
        """Chat messages, system role first. What the generator is given."""
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": self.user}]

    def as_text(self):
        """The two turns flattened, for logging and for a model with no roles.

        Recorded with each generation so the exact input can be reconstructed,
        but never used to build the model input when a chat template exists.
        """
        return f"[system]\n{self.system}\n\n[user]\n{self.user}"


def format_passages(passages):
    """Number the passages, or return a placeholder when there are none."""
    if not passages:
        return "(none)"
    return "\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))


def build(question, passages, system=None):
    """Return the :class:`Prompt` for one request.

    Parameters
    ----------
    question : str
    passages : list of str
        In the order they should appear. May be empty, which is the empty
        evidence level.
    system : str, optional
        The system prompt S. Defaults to :data:`DEFAULT_SYSTEM`; the experiment
        passes one of its fifteen conditions instead.
    """
    user = (f"Passages:\n{format_passages(passages)}\n\n"
            f"Question: {question}\nAnswer:")
    return Prompt(system=system or DEFAULT_SYSTEM, user=user)
