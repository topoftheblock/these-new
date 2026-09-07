# Experiment code

Code for the thesis experiment: does changing the system prompt of a
retrieval-augmented generation system change its behaviour, when everything else
is held fixed?

Three packages, in the order data flows through them.

    preprocessing/   MS MARCO in, query set out
    ingest/          the knowledge base: embed the corpus, build the index
    rag/             the retrieval-augmented generation model itself
    experiment/      the study that drives the model across conditions
    evaluation/      scoring the generations and computing the effect

The split is not cosmetic. `rag/` knows nothing about the experiment: it is a
working RAG system you could point at any corpus. `experiment/` knows about
prompt conditions and evidence levels but implements no retrieval or generation
of its own. `preprocessing/` knows about MS MARCO and nothing downstream.

## Install

Python 3.10 or newer. The project environment already has everything needed:

    torch, transformers, numpy

`faiss-cpu` is optional. At this corpus size a flat index is exact search either
way, and `rag/index.py` falls back to numpy with identical results. See
`requirements.txt`.

The first run downloads two models from Hugging Face, about 420 MB for the
encoder and several gigabytes for the generator. Set `HF_HOME` if you want them
somewhere other than `~/.cache/huggingface`.

## Input

Download the MS MARCO v2.1 QnA release and place a split here:

    data/dev_v2.1.json

The ranking release will not work: it has no `wellFormedAnswers` field and no
`is_selected` flags, both of which the filtering depends on.

## Run

    python3 -m preprocessing.build --input ../data/dev_v2.1.json
    python3 -m experiment.run_retrieval
    python3 -m experiment.run_generation --limit 2     # smoke test first
    python3 -m experiment.run_generation               # full run
    python3 -m evaluation.run_scoring                  # score them
    python3 -m evaluation.run_analysis                 # the reported effect

Run these from inside `code/`. Everything is written to `runs/`.

## What comes out

    runs/queries.json       40 queries with their reference answers and gold ids
    runs/corpus.json        passage id to text
    data/vector_store/      the vector database: index.faiss, pids.npy, meta.json
    runs/retrieval.json     top-K per query, plus the retrieval gate result
    runs/generations.jsonl  one line per generation, 30000 on a full run
    runs/scores.jsonl       per-generation measure scores and cost
    runs/results.json       empirical risks, effects, tests

## Configuration

`config.py` holds every parameter the study fixes, and it is the only place any
of them appears. The rule is simple: if a value is in that file it must not
differ between conditions, and if it is not in that file it is not a controlled
parameter. Changing a value there invalidates comparison with earlier runs, so
record `SEED` and both model identifiers alongside any results.

## Design constraints the code enforces

**Retrieval and generation are separate passes.** `run_retrieval.py` finishes
and writes to disk before `run_generation.py` starts. There is no execution
path along which a prompt condition could influence what was retrieved. This is
the assumption the causal claim rests on, so it is enforced structurally rather
than asserted.

**The retrieval gate.** A query counts as retrieved successfully when *every*
passage its annotation marks as necessary is in the top K. Partial retrieval on
a multi-hop query is a failure, because an answer needing two passages cannot be
produced from one. Failed queries are excluded from the retrieved evidence level
only; the other three levels set the context directly and cannot fail.

**One encoder, one index, built once.** Rebuilding the index mid-run would
change what is retrieved and break the comparison between conditions.

## Not settled yet

**The cost weights.** Section 4.9 of the thesis argues the four requirements are
not equally serious but assigns no values to lambda_k. `evaluation/cost.py`
defaults to equal weights, so the headline number has no free parameters, and
offers a grounding-weighted scheme for a sensitivity analysis. Fix the weights
in the thesis and record which scheme produced any reported result.

**Judge agreement.** Section 5.4.1 requires Cohen's kappa against human labels on
about 100 held-out outputs, reported per condition. That needs hand labelling
and is not implemented.
