"""
The design, enumerated in one place.

Section 5.4 of the thesis describes a crossing that is easy to state in prose
and easy to get subtly wrong in code, because the depth axis does not apply
uniformly. This module is the single definition of which cells exist, so that
the generation runner, the analysis and the tests all read the same answer
rather than each rebuilding it.

The crossing
------------

Three factors, one of which is the treatment:

    prompt condition   12 levels, the independent variable
    evidence level      2 levels, a controlled robustness factor

Retrieval depth is held constant at K = 10, so it is no longer an axis. The
depth field survives on a cell because the two evidence levels differ in
whether a retrieval happened at all: ``ret`` records the depth it was retrieved
at, and ``gold`` records ``None``, since its context is set directly and
writing a depth there would imply a retrieval that did not occur.

Counting the cells
------------------

    24   12 conditions x 2 evidence levels, at K = 10

At 100 queries and n = 3 repetitions that is 7,200 generations, which is the
figure Section 5.4 states.

The gate
--------

The retrieval gate is applied per depth, because a query whose gold evidence is
missed at K = 5 may be recovered at K = 10. A cell at ``ret`` is skipped for a
query that failed the gate *at that depth*; the other levels are never gated.
The exclusion is applied to all twelve conditions at once, so the pairing that
the whole design rests on survives.
"""

from dataclasses import dataclass

import config


#: The depth recorded for a cell whose context does not come from the retriever.
#: Written to every record so that "depth" is never null and grouping never has
#: to special-case a missing value.
NO_RETRIEVAL = None


@dataclass(frozen=True)
class Cell:
    """One cell of the design: a prompt condition, an evidence level, a depth.

    Attributes
    ----------
    condition : str
        Prompt condition id, e.g. ``"B3"``.
    level : str
        Evidence level: ``gold``, ``ret``, ``corr`` or ``empty``.
    depth : int or None
        The retrieval depth this cell was run at. An int for ``ret``; ``None``
        for the three levels that set the context directly, since no retrieval
        happened and recording a depth there would imply one did.
    """

    condition: str
    level: str
    depth: object = NO_RETRIEVAL

    @property
    def is_gated(self):
        """True when the retrieval gate can exclude a query from this cell."""
        return self.level == "ret"

    @property
    def key(self):
        """A stable string id, used for grouping and for reading a log."""
        return (f"{self.condition}/{self.level}/k{self.depth}"
                if self.depth is not None else f"{self.condition}/{self.level}")

    def __str__(self):
        return self.key


def evidence_depths(level, depths=None):
    """The depths one evidence level is run at.

    ``ret`` is run at every depth that requires a retrieval pass. The other
    three levels set the context directly and are run once, with no depth.
    """
    if level != "ret":
        return [NO_RETRIEVAL]
    return list(config.RETRIEVED_DEPTHS if depths is None else depths)


def cells(conditions=None, levels=None, depths=None):
    """Every cell of the design, in a deterministic order.

    Order is condition, then evidence level, then depth. The generation runner
    shuffles a copy of this per query; the order here is only so that two runs
    enumerate the same set.
    """
    conditions = list(conditions or config.PROMPT_CONDITIONS)
    levels = list(levels or config.EVIDENCE_LEVELS)
    return [Cell(c, lv, d)
            for c in conditions
            for lv in levels
            for d in evidence_depths(lv, depths)]


def n_cells(**kw):
    """How many cells the design has. 24 with the thesis' settings."""
    return len(cells(**kw))


def n_generations(n_queries=None, repeats=None, **kw):
    """Generations before the retrieval gate removes any.

    The gate only ever removes cells at ``ret``, so this is an upper bound and
    the number actually written will be smaller.
    """
    n_queries = config.N_Q if n_queries is None else n_queries
    repeats = config.N_REPEATS if repeats is None else repeats
    return n_cells(**kw) * n_queries * repeats


def summary():
    """A short report of the design, printed by the runners before they start."""
    by_level = {}
    for cell in cells():
        by_level.setdefault(cell.level, set()).add(cell.depth)
    lines = [
        f"prompt conditions : {len(config.PROMPT_CONDITIONS)}",
        f"evidence levels   : {len(config.EVIDENCE_LEVELS)} "
        f"({', '.join(config.EVIDENCE_LEVELS)})",
        f"retrieval depths  : K in {config.DEPTHS} "
        f"(retrieval runs at {config.RETRIEVED_DEPTHS})",
    ]
    for level in config.EVIDENCE_LEVELS:
        depths = sorted(d for d in by_level[level] if d is not None)
        where = f"K = {depths}" if depths else "context set directly"
        lines.append(f"  {level:<6s} {where}")
    lines += [
        f"cells             : {n_cells()}",
        f"queries           : {config.N_Q}",
        f"repeats           : {config.N_REPEATS}",
        f"generations       : {n_generations():,} before the gate",
    ]
    return "\n".join(lines)


def check():
    """Assert the design matches the thesis. Called at import.

    A miscount here would not raise anywhere else: the runner would simply do
    the wrong number of things and report the number it did. So the arithmetic
    of Section 5.4 is asserted rather than trusted.
    """
    expected_cells = (len(config.PROMPT_CONDITIONS) * len(config.EVIDENCE_LEVELS)
                      + len(config.PROMPT_CONDITIONS)
                      * (len(config.RETRIEVED_DEPTHS) - 1))
    got = n_cells()
    if got != expected_cells:
        raise ValueError(f"design has {got} cells, expected {expected_cells}")
    if 0 in config.RETRIEVED_DEPTHS:
        raise ValueError("K = 0 returns no passage and is not a level here")
    if set(config.EVIDENCE_LEVELS) != {"gold", "ret"}:
        raise ValueError(
            f"evidence levels are {config.EVIDENCE_LEVELS}; this design has "
            "gold and ret only")


check()
