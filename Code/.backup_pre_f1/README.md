# Experiment code

Implements the experiment of Chapter 5: does changing the *structure* of the
system prompt of a retrieval-augmented generation system change its behaviour,
when the prompt's meaning and everything else in the system are held fixed?

Five packages, in the order data flows through them.

    preprocessing/   MS MARCO in, query set out
    ingest/          the knowledge base: embed the corpus, build the index
    rag/             the retrieval-augmented generation system itself
    experiment/      the study that drives that system across conditions
    evaluation/      scoring the generations and computing the effect

The split is not cosmetic. `rag/` knows nothing about the experiment: it is a
working RAG system you could point at any corpus. `experiment/` knows about
prompt conditions and evidence levels but implements no retrieval or generation
of its own. `preprocessing/` knows about MS MARCO and nothing downstream.

## The design in one place

`experiment/design.py` enumerates the cells and is the only definition of what
the crossing is. Everything else asks it.

    12 prompt conditions   3 published requirement sets x 4 wordings
                           (canonical, lexical, syntactic, format);
                           every wording states the same requirements
     2 evidence levels     gold (annotated passages), ret (the real top-K)
     2 retrieval depths    K = 5, 10, applying to `ret` alone
    ---
    36 cells               24 at K=5, +12 for ret at K=10

    36 cells x 100 queries x n=5 repeats = 18,000 generations

Run `python3 verify.py --runs` to check the implementation against Chapter 5.
Every check is labelled with the section it enforces. Run it before collecting
data and after any change to the design.

## Install

Python 3.10 or newer, and the packages in `requirements.txt`:

    torch, transformers, numpy, faiss-cpu, pyarrow, scipy, statsmodels

`python-dotenv` is optional (reads `.env`); `accelerate` is optional and only
matters for the local generator backend.

The encoder (`all-mpnet-base-v2`, ~420 MB) is downloaded on first use. The
generator and the judge are API models and download nothing.

## The API key

Generation and judged scoring call the OpenAI chat completions API. The key is
read from the environment variable `OPENAI_API_KEY`, or from `Code/.env`:

    cp .env.example .env
    # edit .env: OPENAI_API_KEY=sk-...

`.env` is git-ignored. The key is never written into source and never logged.
The HTTP call is made with the standard library (`rag/openai_generator.py`), so
the `openai` package is not needed; on this machine it cannot be imported
anyway because of a pydantic version conflict.

Which models are used is fixed in `config.py`:

    GENERATOR_MODEL = "gpt-4o-mini"     temperature 1.0, max 256 new tokens
    JUDGE_MODEL     = "gpt-4.1-mini"    temperature 0

The judge is a different model from the generator on purpose: the generator
must not grade its own output. Both ids are written into every row so a result
can always be traced to the model that produced it.

`GENERATOR_BACKEND = "local"` swaps in a Hugging Face checkpoint
(`LOCAL_GENERATOR_MODEL`) for smoke tests without network access. A run made
with one backend is not comparable with a run made with the other.

## Run

    # once: corpus, index, query set
    python3 -m preprocessing.fetch                     # MS MARCO -> parquet
    python3 -m ingest.build_index                      # the vector database
    python3 -m ingest.export_corpus                    # passage id -> text
    python3 -m preprocessing.build \
        --input ../data/ms_marco_v2.1_validation_wellformed.parquet

    # stage 1: retrieval, at every depth, before any prompt exists
    python3 -m experiment.run_retrieval

    # stage 2: generation
    python3 -m experiment.run_generation --limit 2     # smoke test first
    python3 -m experiment.run_generation               # the full run
    python3 -m experiment.run_generation --resume      # continue if interrupted

    # stage 3: scoring and analysis
    python3 -m evaluation.run_scoring
    python3 -m evaluation.run_analysis

    # the judge audit (Section 5.7)
    python3 -m evaluation.agreement sample --n 100     # writes a labelling sheet
    # ... a human fills in the 'human' column ...
    python3 -m evaluation.agreement report

Run these from inside `Code/`. Everything is written to `runs/`.

## What comes out

    runs/queries.json          100 queries, reference answers, gold ids
    runs/corpus.json           passage id -> text, all 122,678
    data/vector_store/         index.faiss, pids.npy, meta.json
    runs/retrieval_k5.json     top-5 per query, plus that depth's gate result
    runs/retrieval_k10.json    top-10 per query, plus that depth's gate result
    runs/generations.jsonl     one line per generation, 18,000 on a full run
    runs/scores.jsonl          per-generation scores for the four measures
    runs/results.json          cell means, tau_hat with 95% CIs, the verdict
    runs/agreement.json        Cohen's kappa per condition

## Configuration

`config.py` holds every parameter the study fixes, and it is the only place any
of them appears. The rule: if a value is in that file it must not differ
between conditions, and if it is not in that file it is not a controlled
parameter. Changing a value there invalidates comparison with earlier runs, so
record `SEED` and the model identifiers alongside any results.

## Design constraints the code enforces

**Only the wording changes.** The four conditions of a baseline state the same
requirement set. `experiment/conditions.py` documents the rule each wording
was written under, and `verify.py` refuses a condition outside the four
wordings. A variant that changed what the prompt asks for would change the
requirement set, and a difference it produced could not be read as an effect of
structure; no such variant is in the design.

**Retrieval and generation are separate passes.** `run_retrieval.py` finishes
and writes to disk before `run_generation.py` starts. There is no execution
path along which a prompt condition could influence what was retrieved. This is
the assumption the causal claim rests on, so it is enforced structurally rather
than asserted.

**The system prompt goes in the system role.** `rag/prompts.py` returns the
system turn and the user turn separately and the generator puts each in its own
chat role. Concatenating them would erase the distinction between a standing
instruction and the user's message, which is the distinction the thesis is
built on, and it would do it silently.

**The retrieval gate, per depth.** A query succeeds when *every* passage its
annotation marks as necessary is in the top K. Partial retrieval on a multi-hop
query is a failure, because an answer needing two passages cannot be produced
from one. The gate is recomputed at each depth, since a query missed at K=5 may
be recovered at K=10, and failed queries are excluded from the retrieved level
of that depth only, uniformly across all twelve conditions so the pairing
survives.

**The context never depends on the condition.** Passage order is drawn from a
generator seeded on the query, evidence level, depth and repeat, and not on the
prompt condition, so all twelve conditions receive a byte-identical user turn.
Without that, the comparison would be between different evidence rather than
between different prompts.

**One encoder, one index, built once.** Rebuilding the index mid-run would
change what is retrieved and break the comparison between conditions.

**Adherence is checked per baseline.** A condition is held only to what its own
prompt states. Baseline A states only grounding, which faithfulness already
measures, so its adherence measure is undefined by construction rather than a
free pass.

## The query set

100 queries, balanced on two axes at once:

                DESCRIPTION  ENTITY  LOCATION  NUMERIC  PERSON
    direct               10      10        10       10      10   = 50
    multi-hop            10      10        10       10      10   = 50

`query_type` is MS MARCO's coarse intent label. It is balanced so it can be
entered as a covariate rather than confounded with the category, and it is
carried through generation and scoring into `runs/scores.jsonl`. The
`--covariate` flag on `run_analysis` breaks the prompt effect down by it.

That balance is not free: the natural distribution is uneven (DESCRIPTION 44%,
PERSON 7% on the well-formed rows), so a figure averaged over this query set is
an average over the design, not an estimate on the natural distribution. The
prompt effect is unaffected, because every contrast is taken within a query.

## Measured retrieval performance

A property of the fixed pipeline, reported ahead of the results (Section 5.6):

    K = 5     gate 74/100  (direct 41/50, multi-hop 33/50)  gold recall 123/152 = 80.9%
    K = 10    gate 88/100  (direct 45/50, multi-hop 43/50)  gold recall 139/152 = 91.4%

By query type, out of 20 each:

    K = 5     DESCRIPTION 16  ENTITY 15  LOCATION 15  NUMERIC 15  PERSON 13
    K = 10    DESCRIPTION 17  ENTITY 16  LOCATION 19  NUMERIC 18  PERSON 18

## Not settled yet

**Judge agreement needs a human.** `evaluation/agreement.py` draws the sample
and computes kappa; the labelling itself is manual and has not been done.
