# experiment

The prompt-sensitivity study, built on top of `rag/`.

    conditions.py       the five system prompts
    contexts.py         the four evidence levels
    run_retrieval.py    stage 1
    run_generation.py   stage 2

## Run

    python3 -m experiment.run_retrieval
    python3 -m experiment.run_generation --limit 2    # smoke test
    python3 -m experiment.run_generation              # full run

Retrieval must finish first. Generation reads its output from disk and will not
run without it.

## The design

Fifteen prompt conditions (three published baselines, five wordings each)
crossed with four evidence levels, over 100 queries, repeated five times:
30,000 generations. The repetition count is `config.N_REPEATS`.

### Prompt conditions

| id | varies | what it isolates |
|----|--------|------------------|
| S0 | canonical, plain imperative prose | the reference form |
| S1 | lexical, synonym substitution | word choice |
| S2 | syntactic, clause order and voice | sentence structure |
| S3 | format, prose rewritten as a list | layout |
| S4 | deontic, obligations weakened to requests | modal force |

S0 through S3 state the **same requirements** in four wordings, so any
difference between them is attributable to wording alone. S4 changes the
requirement set itself while keeping S0's wording, which is why it isolates
modal force rather than phrasing and why it is reported separately.

`SIGMA` in `conditions.py` names the four requirements the prompts state:
grounding, abstention, reporting disagreement, and length. Naming them is what
lets adherence be checked by rule instead of by judgement. `ABSTAIN_STRING` is
fixed for the same reason: an exact string is checkable, a paraphrase is not.

### Evidence levels

| id | context supplied | what it isolates |
|----|------------------|------------------|
| `gold` | only the annotated supporting passages | behaviour on ideal evidence |
| `ret` | the real top-K from the retriever | the deployment condition |
| `corr` | gold passages with their facts negated | evidence against memory |
| `empty` | nothing | parametric knowledge alone |

All four set the context directly rather than by changing the corpus and
retrieving again. The index is identical for every level, so retrieval runs once
for the whole experiment.

Gold positions within the context are shuffled and recorded, because answer
accuracy depends on where the relevant passage sits.

## The retrieval gate

`run_generation.py` skips a cell when the evidence level is `ret` and retrieval
failed for that query. It skips nothing at the other three levels, which set the
context directly and so cannot fail.

The exclusion is uniform across prompt conditions. Retrieval is identical in all
five, so success is a property of the query rather than of the condition, and a
failed query drops from all five at once. That keeps the pairing intact: every
comparison is still made within a query.

Consequence for reporting: the denominator at `ret` is smaller than at the other
levels. `run_retrieval.py` prints the success count, overall and per category,
and it belongs in the results.

## Corrupting evidence

`contexts.corrupt()` negates the facts a passage states, substituting years for
other years and counts for other counts in a single pass. Substitutions stay
plausible on purpose: an implausible value would be detectable without reading
the passage at all, which is the opposite of what the corrupted level tests.

It returns `n_substitutions`, and **zero is meaningful**. It means no
counterfactual could be formed, usually because the passage states no numeric
fact and has fewer than two proper nouns. Those cells carry no
memory-against-evidence contrast and should be excluded from any claim resting
on one. The count is recorded on every generation so this is checkable after the
fact rather than invisible.

## Output contract

`runs/generations.jsonl`, one JSON object per line:

    query_id, category            which query, and its category
    condition, evidence_level     which cell
    repeat                        0 .. N_REPEATS-1
    question, reference_answer    input and the correctness reference
    gold_ids                      passages the query needs
    context_passage_ids           what was actually in the context, in order
    gold_positions                1-indexed positions of gold passages
    n_substitutions               0 means no counterfactual was formed
    output, n_output_tokens       what the model produced
    latency_s, timestamp          for drift checking

Cell order is randomised per query, so drift over the collection period cannot
align with a condition. Timestamps let you check for it afterwards.
