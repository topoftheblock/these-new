"""
The four evidence levels.

Each is an intervention on the context. All four set the context *directly*
rather than by changing the corpus and retrieving again, so the index is
identical for every level and retrieval never has to run twice.

======== ==========================================================
gold     only the passages the annotation marks as supporting
ret      the real top-K output of the fixed retriever
corr     the gold passages with their supporting facts negated
empty    no context at all
======== ==========================================================

Positions of gold passages inside the context are randomised and recorded,
because answer accuracy depends on where the relevant passage sits.
"""

import re

#: one pass over years and other numbers, so a year is never rescaled twice
_TOKEN = re.compile(r"\b(1[89]\d{2}|20\d{2})\b|\b(\d+(?:\.\d+)?)\b")
_PROPER = re.compile(r"\b[A-Z][a-z]{3,}\b")


def corrupt(text, rng):
    """Negate the facts a passage states. Returns ``(text, n_substitutions)``.

    Substitutions stay plausible. A year becomes another year, a count becomes
    another count. An implausible value would be detectable without consulting
    the passage at all, which is the opposite of what the corrupted level tests.

    ``n_substitutions == 0`` means no counterfactual could be formed, usually
    because the passage states no numeric fact and has fewer than two proper
    nouns. Those cells carry no memory-against-evidence contrast and should be
    excluded from any claim that rests on one. The count is returned rather
    than raised so the caller can record it.
    """
    n = 0

    def substitute(match):
        nonlocal n
        n += 1
        if match.group(1):                    # a year stays a year
            return str(int(match.group(1)) + rng.choice([-13, -9, 9, 13]))
        value = float(match.group(2))
        return str(int(value) + 7) if value == int(value) else f"{value + 7:.1f}"

    out = _TOKEN.sub(substitute, text)

    if n == 0:                                 # fall back to swapping two names
        names = list(dict.fromkeys(_PROPER.findall(out)))
        if len(names) >= 2:
            first, second = names[0], names[1]
            out = re.sub(rf"\b{re.escape(first)}\b", "\x00", out)
            out = re.sub(rf"\b{re.escape(second)}\b", first, out)
            out = out.replace("\x00", second)
            n = 2
    return out, n


def build_context(record, corpus, level, rng):
    """Build the passages for one query at one evidence level.

    Parameters
    ----------
    record : dict
        A retrieval record, carrying ``gold_ids`` and ``retrieved_ids``.
    corpus : dict
        Passage id to text.
    level : str
        One of ``gold``, ``ret``, ``corr``, ``empty``.
    rng : random.Random
        Seeded, so passage order is reproducible.

    Returns
    -------
    (list of str, dict)
        The passages in presentation order, and metadata recording which
        positions hold gold evidence and how many substitutions were made.
    """
    gold_ids = record["gold_ids"]

    if level == "empty":
        return [], {"passage_ids": [], "gold_positions": [], "n_substitutions": 0}
    if level in ("gold", "corr"):
        ids = list(gold_ids)
    elif level == "ret":
        ids = list(record["retrieved_ids"])
    else:
        raise ValueError(f"unknown evidence level: {level!r}")

    rng.shuffle(ids)

    passages, substitutions = [], 0
    for pid in ids:
        text = corpus[pid]
        if level == "corr" and pid in gold_ids:
            text, made = corrupt(text, rng)
            substitutions += made
        passages.append(text)

    positions = [i for i, pid in enumerate(ids, 1) if pid in gold_ids]
    return passages, {"passage_ids": ids, "gold_positions": positions,
                      "n_substitutions": substitutions}
