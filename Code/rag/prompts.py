"""
Assembling the model input.

One function, so that every condition in the experiment builds its input the
same way and only the system prompt differs. Passages are numbered from 1 so
that a citation instruction has something to refer to, and so that the position
of a passage in the context is recoverable from the output.

The ordering of ``passages`` is preserved exactly. Shuffling, when the
experiment wants it, happens before this call and is recorded there, because
answer accuracy depends on where the relevant passage sits.
"""

DEFAULT_SYSTEM = (
    "Answer the question using only the numbered passages below.\n"
    "If the passages do not contain the answer, say so.\n"
    "Answer in at most three sentences."
)


def format_passages(passages):
    """Number the passages, or return a placeholder when there are none."""
    if not passages:
        return "(none)"
    return "\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))


def build(question, passages, system=None):
    """Return the full prompt string.

    Parameters
    ----------
    question : str
    passages : list of str
        In the order they should appear. May be empty.
    system : str, optional
        The system prompt. Defaults to :data:`DEFAULT_SYSTEM`; the experiment
        passes one of its own conditions instead.
    """
    return (f"{system or DEFAULT_SYSTEM}\n\n"
            f"Passages:\n{format_passages(passages)}\n\n"
            f"Question: {question}\nAnswer:")
