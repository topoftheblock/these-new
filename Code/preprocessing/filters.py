"""
The selection rules.

Two filters are applied, in this order.

1. **Well-formed answer.** Only rows with a non-empty ``wellFormedAnswers``
   are kept. MS MARCO's well-formed answers are rewritten by annotators to be
   readable without the question, which is what makes them usable as a
   reference string for a free-text correctness measure. Rows whose only answer
   is ``"No Answer Present."`` carry no reference and are dropped here.

2. **Evidence distribution.** Rows are split by how many passages carry
   ``is_selected == 1``:

   ===================  =============================================
   exactly one          ``direct``     answer in one passage
   two or more          ``multi-hop``  answer needs several passages
   zero                 dropped, no gold evidence to retrieve
   ===================  =============================================

   The count comes from the annotation, not from any judgement made here, so
   the category of a query is a property of the release.
"""

#: MS MARCO's sentinel for an unanswerable row
NO_ANSWER = "no answer present."


def well_formed_answers(row):
    """Return the row's well-formed answers, normalised to a list of strings.

    MS MARCO stores this field inconsistently: usually a list, sometimes the
    literal string ``"[]"``. Both are handled. The sentinel ``"No Answer
    Present."`` is treated as absent.
    """
    wfa = row.get("wellFormedAnswers") or []
    if isinstance(wfa, str):
        wfa = [] if wfa.strip() in ("[]", "") else [wfa]
    return [a for a in wfa
            if a and a.strip() and a.strip().lower() != NO_ANSWER]


def gold_passages(row):
    """Return the passages the annotators marked as supporting the answer.

    Each carries the ``position`` it occupied in the row's passage list. The
    position, not the rank among gold passages, is what identifies a passage:
    the vector store built by :mod:`ingest` mints ids the same way, so gold ids
    from here and corpus ids from there refer to the same text.
    """
    out = []
    for position, passage in enumerate(row.get("passages") or []):
        if passage.get("is_selected") == 1:
            out.append({**passage, "position": position})
    return out


def categorise(row):
    """Return ``'direct'``, ``'multi-hop'``, or ``None`` if out of scope.

    ``None`` means one of: no well-formed answer, or no gold passage. Both are
    counted separately by the caller so the drop reasons stay visible.
    """
    if not well_formed_answers(row):
        return None
    n = len(gold_passages(row))
    if n == 1:
        return "direct"
    if n >= 2:
        return "multi-hop"
    return None
