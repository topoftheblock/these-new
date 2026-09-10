# experiment

The prompt-structure study, built on top of `rag/`.

    design.py           the cells of the design, enumerated once
    conditions.py       the twelve system prompts
    contexts.py         the evidence levels
    run_retrieval.py    stage 1
    run_generation.py   stage 2

## Run

    python3 -m experiment.run_retrieval
    python3 -m experiment.run_generation --limit 2    # smoke test
    python3 -m experiment.run_generation              # full run

Retrieval must finish first. Generation reads its output from disk and will not
run without it. Generation needs the API key (see the top-level README).

## The design

Twelve prompt conditions (three published baselines, four wordings each)
crossed with two evidence levels, plus the retrieved level again at K = 10:
36 cells. Over 100 queries, repeated five times, that is 18,000 generations.
`experiment/design.py` is the definition; this paragraph is a description of it.

### Prompt conditions

| id | varies | what it isolates |
|----|--------|------------------|
| b0 | canonical, the published form | the reference form |
| b1 | lexical, synonym substitution | word choice |
| b2 | syntactic, clause order and voice | sentence structure |
| b3 | format, prose rewritten as a list | layout |

`b` is the baseline, `A`, `B` or `C`. All four conditions of a baseline state
the **same requirements**; only the wording differs, so any difference between
them is attributable to wording alone. That is the independent variable. No
condition changes what the prompt asks for.

Each baseline is a prompt published with a retrieval-augmented system:

| baseline | source | requirements stated |
|----------|--------|---------------------|
| A | Friel et al. 2024, RAGBench | 1 (grounding) |
| B | Gao et al. 2023, ALCE, VANILLA | about 10 (grounding, citations, tone, ...) |
| C | Mao et al. 2024, FIT-RAG | 3 (grounding, read carefully, justify) |

`SIGMA` in `conditions.py` names those requirements per baseline. Naming them
is what lets adherence be checked by rule for the ones a rule can decide.
Within a baseline the four wordings stay within one tenth of the canonical
word count (`LENGTH_TOLERANCE`), checked at import, so length is not confounded
with the axis being varied.

### Evidence levels

| id | context supplied | what it isolates |
|----|------------------|------------------|
| `gold` | only the annotated supporting passages | the answer is certainly present, so a failure is the generator's |
| `ret` | the real top-K from the retriever | the deployment condition |

Both set the context directly rather than by changing the corpus and retrieving
again. The index is identical for both, so retrieval runs once for the whole
experiment. `corr` and `empty` remain implemented in `contexts.py` but are not
in `config.EVIDENCE_LEVELS` and no cell requests them.

Passage order within the context is shuffled with a seeded generator and the
gold positions are recorded, because answer accuracy depends on where the
relevant passage sits.

## The retrieval gate

`run_generation.py` skips a cell when the evidence level is `ret` and retrieval
failed for that query at that depth. It skips nothing at the gold
level, which sets the context directly and so cannot fail.

The exclusion is uniform across prompt conditions. Retrieval is identical in all
twelve, so success is a property of the query rather than of the condition, and
a failed query drops from all twelve at once. That keeps the pairing intact:
every comparison is still made within a query.

Consequence for reporting: the denominator at `ret` is smaller than at the gold
level. `run_retrieval.py` prints the success count, overall and per category,
and it belongs in the results.

## Corrupting evidence (not in this design)

`contexts.corrupt()` is still implemented and still exercised by the tests, but
no cell requests it: the `corr` level asked about evidence against parametric
memory rather than about prompt structure, and it is not in
`config.EVIDENCE_LEVELS`. Adding `"corr"` back there is all that is needed to
run it. `n_substitutions` is written on every generation and is 0 at both
levels this design uses.

## Output contract

`runs/generations.jsonl`, one JSON object per line:

    query_id, category, query_type   which query, its category, its intent type
    condition, evidence_level, depth which cell; `cell` is the joined key
    repeat                           0 .. N_REPEATS-1
    question, reference_answer       input and the correctness reference
    gold_ids                         passages the query needs
    context_passage_ids              what was actually in the context, in order
    gold_positions                   1-indexed positions of gold passages
    n_substitutions                  0 means no counterfactual was formed
    system_prompt                    the exact S sent in the system role
    generator_backend, generator_model, temperature, max_new_tokens, dtype
    system_fingerprint, finish_reason   API provenance (null for local)
    output, n_output_tokens          what the model produced
    latency_s, timestamp             for drift checking

Cell order is randomised per query, so drift over the collection period cannot
align with a condition. Timestamps let you check for it afterwards.
