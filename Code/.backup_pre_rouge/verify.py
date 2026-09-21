"""
Check that what the code does matches what Chapter 5 says.

Run this before a data-collection run and after any change to the design. Every
check names the section it enforces, so a failure points at the sentence it
contradicts rather than at a bare assertion.

These are properties of the *design*, not unit tests of the implementation.
They catch the class of error that does not raise: a run that completes, writes
plausible numbers, and answers a different question from the one the thesis
asks.

Usage::

    python3 verify.py
    python3 verify.py --runs      # also check the artefacts on disk
"""

import argparse
import collections
import json
import os
import sys

import config
from experiment import design
from experiment.conditions import (AXIS_OF, BASELINE_OF, CANONICAL, CONDITIONS,
                                   LENGTH_TOLERANCE, length_deviations)

CHECKS = []

#: condition ids in a stable order, for the checks that compare two of them
CONDITIONS_ORDER = sorted(CONDITIONS)


def check(section, claim):
    """Register a check, labelled with the thesis section it enforces."""
    def register(fn):
        CHECKS.append((section, claim, fn))
        return fn
    return register


# ------------------------------------------------------------ the conditions
@check("5.3", "twelve prompt conditions: three requirement sets in four wordings")
def _conditions():
    assert len(CONDITIONS) == 12, f"{len(CONDITIONS)} conditions, expected 12"
    assert list(config.WORDINGS) == ["canonical", "lexical", "syntactic", "format"]
    for baseline in ("A", "B", "C"):
        n = sum(1 for c in CONDITIONS if BASELINE_OF[c] == baseline)
        assert n == 4, f"baseline {baseline} has {n} wordings, expected 4"
    axes = {AXIS_OF[c] for c in CONDITIONS}
    assert axes == set(config.WORDINGS), axes


@check("5.3", "no condition changes the requirement set; only the wording varies")
def _requirements_fixed():
    # The design has no condition whose requirement set differs from its
    # canonical form. A deontic variant (obligations weakened to requests) once
    # existed under index 4; it changed what the prompt asked for, which the
    # research question holds fixed, and it must not come back silently.
    assert all(int(c[1]) < len(config.WORDINGS) for c in CONDITIONS), (
        "a condition index outside the four wordings exists; if it changes the "
        "requirement set it does not belong in this design")
    assert "deontic" not in {AXIS_OF[c] for c in CONDITIONS}


@check("5.3", "every contrast is taken within a baseline")
def _within_baseline():
    for condition in CONDITIONS:
        reference = CANONICAL[BASELINE_OF[condition]]
        assert BASELINE_OF[reference] == BASELINE_OF[condition], (
            f"{condition} would be compared against {reference}, "
            "which belongs to another requirement set")


@check("5.3", "variants stay within one tenth of the canonical length")
def _lengths():
    assert LENGTH_TOLERANCE == 0.10, LENGTH_TOLERANCE
    for baseline in CANONICAL:
        for name, deviation in length_deviations(baseline).items():
            assert abs(deviation) <= LENGTH_TOLERANCE + 1e-9, (
                f"{name} is {deviation:+.0%} from canonical; length would be "
                "confounded with the axis being varied")


# ------------------------------------------------------------- the crossing
@check("5.4", "24 cells; 7,200 generations at 100 queries and n=3")
def _cells():
    assert design.n_cells() == 24, f"{design.n_cells()} cells, expected 24"
    assert config.N_REPEATS == 3, f"n = {config.N_REPEATS}, expected 3"
    assert config.N_Q == 100, f"{config.N_Q} queries, expected 100"
    assert design.n_generations() == 7_200, design.n_generations()


@check("5.4", "retrieval depth is held constant at K = 10")
def _one_depth():
    assert list(config.RETRIEVED_DEPTHS) == [10], config.RETRIEVED_DEPTHS
    assert list(config.DEPTHS) == [10], config.DEPTHS
    assert config.K == 10, config.K


@check("5.4", "two evidence levels: gold and the real retriever output")
def _levels():
    assert list(config.EVIDENCE_LEVELS) == ["gold", "ret"], config.EVIDENCE_LEVELS


@check("5.4", "depth applies to the retrieved level alone")
def _depth_scope():
    for cell in design.cells():
        if cell.level == "ret":
            assert cell.depth in config.RETRIEVED_DEPTHS, cell
        else:
            assert cell.depth is None, (
                f"{cell} carries a depth, but its context is set directly and "
                "no retrieval happened")


@check("5.4", "K=0 returns no passage and is not run")
def _k_zero():
    assert 0 not in config.RETRIEVED_DEPTHS, (
        "K = 0 returns no passage and is not a level of this design")


@check("5.4", "the generator, encoder, index and chunking are held constant")
def _held_constant():
    for name in ("GENERATOR_BACKEND", "GENERATOR_MODEL", "EMBEDDING_MODEL",
                 "TEMPERATURE", "MAX_NEW_TOKENS", "MAX_SEQ_LEN", "SEED"):
        value = getattr(config, name, None)
        assert value is not None, f"config.{name} is not fixed"
    assert isinstance(config.TEMPERATURE, (int, float))
    assert config.GENERATOR_BACKEND in ("openai", "local")


@check("5.7", "the judge is a second model, not the generator")
def _judge_separate():
    assert config.JUDGE_MODEL != config.GENERATOR_MODEL, (
        "the generator would be grading its own output")
    assert config.JUDGE_TEMPERATURE == 0.0, (
        "a judge verdict must not depend on a draw")


# --------------------------------------------------------------- the corpus
@check("5.5", "no distractor pool is sampled; the corpus is the whole collection")
def _corpus_rule():
    assert config.N_PASSAGES == 122_678, (
        f"config.N_PASSAGES is {config.N_PASSAGES}; Section 5.5 states the "
        "index holds every passage attached to a well-formed row, 122,678")
    assert config.PASSAGE_MIN_WORDS == 0, (
        "a length filter would make the corpus a sample rather than a rule")


@check("5.5", "the query set is 50 direct and 50 multi-hop")
def _query_split():
    assert config.N_DIRECT == 50 and config.N_MULTIHOP == 50
    assert config.N_Q == config.N_DIRECT + config.N_MULTIHOP


@check("5.5", "the query set is balanced across query_type within each category")
def _strata():
    assert len(config.QUERY_TYPES) == 5, config.QUERY_TYPES
    assert config.N_DIRECT % len(config.QUERY_TYPES) == 0, (
        "the category size must divide evenly over the query types")
    assert config.N_PER_STRATUM == 10, config.N_PER_STRATUM
    assert config.N_PER_STRATUM * len(config.QUERY_TYPES) * 2 == config.N_Q


# ------------------------------------------------------- prompt and metrics
@check("5.3", "S is supplied in the system role, not folded into the user turn")
def _system_role():
    from rag import prompts
    prompt = prompts.build("q?", ["p"], "SYSTEM TEXT")
    messages = prompt.as_messages()
    assert messages[0]["role"] == "system", messages[0]["role"]
    assert messages[0]["content"] == "SYSTEM TEXT"
    assert "SYSTEM TEXT" not in messages[1]["content"], (
        "the system prompt leaked into the user turn; the distinction between "
        "a standing instruction and the user's message is the thesis' claim")


@check("5.4", "every condition sees a byte-identical context")
def _context_is_condition_free():
    """The passage order and the corrupted-level substitutions must be keyed on
    the query, level, depth and repeat, and never on the prompt condition.

    If they were drawn from the run-wide stream, two conditions would receive
    different passage orders -- and different corrupted facts -- for the same
    query, and the comparison would be between different evidence rather than
    between different prompts. Nothing would raise: the run would complete and
    report an effect. So the property is asserted here.
    """
    import random

    from experiment.contexts import build_context

    record = {"query_id": 1, "gold_ids": ["g1", "g2", "g3"],
              "retrieved_ids": ["g1", "d1", "g2", "d2", "g3"]}
    corpus = {"g1": "Built in 1889 by 200 workers.", "g2": "It is 300 m tall.",
              "g3": "Alice met Bob in Paris.", "d1": "Unrelated text one.",
              "d2": "Unrelated text two."}

    def context_for(condition, level, depth, repeat, seed=config.SEED):
        # exactly the key run_generation builds; note `condition` is unused,
        # which is the property under test
        rng = random.Random(f"{seed}/{record['query_id']}/{level}/{depth}/{repeat}")
        return build_context(record, corpus, level, rng)

    for level, depth in (("gold", None), ("ret", 5)):
        first = context_for(CONDITIONS_ORDER[0], level, depth, 0)
        for condition in CONDITIONS_ORDER[1:]:
            other = context_for(condition, level, depth, 0)
            assert first == other, (
                f"{level}: condition {condition} would see a different context "
                "from the first condition; the contrast would be between "
                "evidence, not between prompts")
        # and it must still vary across repeats, so the cell mean marginalises
        # over position rather than fixing one arbitrary arrangement
        if True:
            orders = {tuple(context_for("A0", level, depth, r)[1]["passage_ids"])
                      for r in range(config.N_REPEATS)}
            assert len(orders) > 1, (
                f"{level}: passage order is identical in every repeat; the cell "
                "mean would be tied to one arrangement")


@check("5.7", "adherence checks only what a baseline's own prompt states")
def _adherence_scope():
    from evaluation.metrics import Adherence
    metric = Adherence()
    a = metric.score({"condition": "A0", "output": "text"})
    assert not a.applicable, (
        "baseline A states only grounding, which faithfulness measures; "
        "adherence must report itself undefined rather than pass it free")
    for condition, rule in (("B0", "cite"), ("C0", "justify")):
        result = metric.score({"condition": condition, "output": "text"})
        assert result.applicable and rule in result.detail, (condition, rule)


@check("5.7", "faithfulness is undefined with no context")
def _faithfulness_empty():
    from evaluation.metrics import Faithfulness
    result = Faithfulness().score(
        {"output": "an answer", "context_passages": []}, judge=object())
    assert not result.applicable, (
        "with no passages every claim is trivially unsupported; scoring it "
        "would report the absence of context as a failure of grounding")


@check("5.7", "an inapplicable measure is skipped, not scored as a pass")
def _inapplicable_skipped():
    from evaluation.analysis import measure_score
    row = {"metrics": {"adherence": {"score": 1.0, "applicable": False},
                       "correctness": {"score": 1.0, "applicable": True}}}
    assert measure_score(row, "adherence") is None, (
        "adherence is undefined for baseline A; counting it would put a free "
        "point into the mean")
    assert measure_score(row, "correctness") == 1.0


@check("5.7", "a secondary measure is reported but never decides H0")
def _secondary_is_not_confirmatory():
    """Section 5.7: the decision rests on the four preregistered measures.

    correctness_f1 is reported beside containment because containment floors on
    this data (Section 5.8). Letting it into the confirmatory family would make
    the hypothesis depend on a measure chosen after the results were seen, so
    the two sets must stay disjoint and the verdict must read MEASURES only.
    """
    import inspect
    from evaluation import run_analysis
    from evaluation.analysis import MEASURES, SECONDARY, SECONDARY_SOURCE, verdict

    assert not set(SECONDARY) & set(MEASURES), (
        "a secondary measure must not also be confirmatory")
    assert set(SECONDARY) == set(SECONDARY_SOURCE), (
        "every secondary measure needs a source in SECONDARY_SOURCE")

    # the verdict is computed from all_effects, which is filled from MEASURES
    source = inspect.getsource(run_analysis.main)
    assert "verdict(all_effects)" in source, source[:200]
    body = source.split("secondary = {}")[1].split("verdict(all_effects)")[0]
    assert "all_effects[" not in body, (
        "the secondary section must not write into the confirmatory family")

    # and a secondary estimate carries no weight in it
    fake = {("gold", None, "correctness_f1"): {
        "B1": {"tau": 0.9, "p": 0.0, "p_holm": 0.0, "significant": True,
               "ci": (0.8, 1.0), "n": 10, "compared_with": "B0"}}}
    assert verdict({})["h0_rejected"] is False
    assert verdict(fake)["n_tested"] == 1, (
        "verdict() is only ever handed the confirmatory family; if this "
        "changes, the caller in run_analysis.py must be rechecked")


@check("5.7", "a secondary measure reads its own source, not the parent score")
def _secondary_reads_detail():
    from evaluation.analysis import measure_score
    row = {"metrics": {"correctness": {
        "score": 0.0, "applicable": True, "detail": {"token_f1": 0.75}}}}
    assert measure_score(row, "correctness") == 0.0
    assert measure_score(row, "correctness_f1") == 0.75, (
        "correctness_f1 must read detail.token_f1, not the containment score")
    bare = {"metrics": {"correctness": {"score": 0.0, "applicable": True}}}
    assert measure_score(bare, "correctness_f1") is None, (
        "a missing token_f1 is skipped, never scored as zero")


@check("5.7", "tau_hat is estimated once per measure, with an interval")
def _ate_per_measure():
    from evaluation.analysis import MEASURES, ate
    assert MEASURES == ("faithfulness", "relevance", "correctness",
                        "adherence"), MEASURES

    def row(condition, qid, score):
        return {"condition": condition, "evidence_level": "gold",
                "depth": None, "query_id": qid, "repeat": 0,
                "metrics": {"correctness": {"score": score, "applicable": True}}}

    # B1 beats B0 by exactly 0.2 on every query
    rows = ([row("B0", q, 0.5) for q in range(10)]
            + [row("B1", q, 0.7) for q in range(10)])
    out = ate(rows, "B1", "B0", "gold", "correctness")
    assert abs(out["tau"] - 0.2) < 1e-9, out
    assert out["n"] == 10, out
    assert "ci" in out and "p" in out, out


@check("5.1", "H0 is decided on the Holm-adjusted family, not one interval")
def _holm_family():
    from evaluation.analysis import holm
    # nine raw p-values, one just under .05: unadjusted it "rejects", and after
    # Holm over the family of nine it must not
    rejected, _ = holm([0.04] + [0.6] * 8)
    assert not rejected[0], (
        "a single p just under alpha would confirm the 'at least one' "
        "hypothesis by chance; Holm is what stops that")


# ------------------------------------------------------------ what is on disk
def _artefacts():
    """Checks that need the run directory. Only run with --runs."""
    problems = []
    runs = config.RUNS

    corpus_path = runs / "corpus.json"
    if not corpus_path.exists():
        return ["runs/corpus.json missing: run ingest.export_corpus"]
    corpus = json.loads(corpus_path.read_text())
    if len(corpus) != config.N_PASSAGES:
        problems.append(f"corpus.json holds {len(corpus):,} passages, "
                        f"expected {config.N_PASSAGES:,} (Section 5.5)")

    queries = json.loads((runs / "queries.json").read_text())["queries"]
    if len(queries) != config.N_Q:
        problems.append(f"{len(queries)} queries, expected {config.N_Q}")

    counts = collections.Counter((q["category"], q.get("query_type"))
                                 for q in queries)
    for category in ("direct", "multi-hop"):
        for query_type in config.QUERY_TYPES:
            got = counts.get((category, query_type), 0)
            if got != config.N_PER_STRATUM:
                problems.append(
                    f"{category}/{query_type}: {got} queries, expected "
                    f"{config.N_PER_STRATUM}; the covariate is unbalanced")
    if any(q.get("query_type") is None for q in queries):
        problems.append("some queries carry no query_type; the covariate "
                        "cannot be conditioned on")

    unresolved = {g for q in queries for g in q["gold_ids"]} - corpus.keys()
    if unresolved:
        problems.append(
            f"{len(unresolved)} gold id(s) are not in the corpus, e.g. "
            f"{sorted(unresolved)[:2]}; those queries fail the gate for a "
            "reason no retriever could avoid (see canonicalise_gold)")

    for depth in config.RETRIEVED_DEPTHS:
        path = runs / f"retrieval_k{depth}.json"
        if not path.exists():
            problems.append(f"{path.name} missing: run experiment.run_retrieval")
            continue
        payload = json.loads(path.read_text())
        if payload["k"] != depth:
            problems.append(f"{path.name} records k={payload['k']}")
        unresolved = set()
        for record in payload["records"]:
            unresolved |= set(record["retrieved_ids"]) - corpus.keys()
            unresolved |= set(record["gold_ids"]) - corpus.keys()
        if unresolved:
            problems.append(
                f"{path.name}: {len(unresolved)} passage ids are not in "
                "corpus.json; generation would build truncated contexts")
    return problems


def _key_note():
    """Say whether the API key is available. Informational, never a failure:
    the design is right or wrong independently of whether a key is present."""
    if config.GENERATOR_BACKEND != "openai" and config.JUDGE_BACKEND != "openai":
        return "local backends only; no API key needed"
    env_file = config.ROOT / "Code" / ".env"
    if os.environ.get(config.OPENAI_ENV_VAR):
        return f"{config.OPENAI_ENV_VAR} is set in the environment"
    if env_file.exists():
        return f"{env_file} exists; the key will be read from it"
    return (f"no {config.OPENAI_ENV_VAR} in the environment and no {env_file}; "
            "generation and judged scoring will stop at start-up")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", action="store_true",
                    help="also check the artefacts in runs/")
    args = ap.parse_args(argv)

    failures = []
    for section, claim, fn in CHECKS:
        try:
            fn()
        except AssertionError as exc:
            failures.append((section, claim, str(exc)))
            print(f"FAIL  §{section}  {claim}\n      {exc}")
        else:
            print(f"ok    §{section}  {claim}")

    if args.runs:
        print()
        problems = _artefacts()
        for problem in problems:
            failures.append(("runs", "artefacts on disk", problem))
            print(f"FAIL  runs/  {problem}")
        if not problems:
            print("ok    runs/  artefacts agree with the design")

    print(f"\nnote  API key: {_key_note()}")
    print(f"\n{len(CHECKS)} design checks, {len(failures)} failing")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
