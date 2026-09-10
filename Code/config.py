"""
Fixed experimental parameters.

Every value the study holds constant lives here and nowhere else. If a value is
not in this file it is not a controlled parameter; if it is in this file it must
not differ between conditions. This mirrors Section 5.4 of the thesis, which
lists the components held constant and the reason each one is fixed.

Changing anything under "held constant" invalidates comparison with runs made
before the change. Record SEED and the model identifiers in the thesis.
"""

from pathlib import Path

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"      # inputs: the MS MARCO release, downloaded separately
RUNS = ROOT / "runs"      # outputs: everything the pipeline writes

# ----------------------------------------------- held constant: generator
# The generator is one instruction-tuned model behind one API, fixed for the
# whole of data collection (Section 5.2). "openai" calls the chat completions
# API; "local" loads a Hugging Face checkpoint and is kept for smoke tests and
# for running without network access. The two are not comparable with one
# another: a run made with one backend must not be pooled with a run made with
# the other. The backend and model actually used are written into every row.
GENERATOR_BACKEND = "openai"           # "openai" or "local"
GENERATOR_MODEL = "gpt-4o-mini"        # the model id sent to the API
LOCAL_GENERATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # only for the local backend

# Decoding. Temperature is the API default and is recorded, never varied. The
# repeats (N_REPEATS) exist because sampling at this temperature is stochastic.
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 256

# Local backend only: the dtype the checkpoint is loaded in. "auto" honours the
# checkpoint. Irrelevant to the API backend, where it is recorded as "api".
GENERATOR_DTYPE = "auto"

# --------------------------------------------------- held constant: judge
# Faithfulness and answer relevance are scored by a second language model with
# a fixed prompt (Section 5.7). It is a different model from the generator, so
# the generator is not grading its own output, and it is run at temperature 0
# so that a verdict depends on the answer and the passages and not on a draw.
JUDGE_BACKEND = "openai"
JUDGE_MODEL = "gpt-4.1-mini"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 256

# ------------------------------------------------------------- the API
# The key is never written into source. It is read from the environment
# variable below, or from a ``.env`` file next to this module (see
# ``.env.example``), which is git-ignored.
OPENAI_ENV_VAR = "OPENAI_API_KEY"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_TIMEOUT_S = 90
OPENAI_MAX_RETRIES = 10   # raised after 12 workers exhausted 6 on 429s

# How many requests are in flight at once. This is NOT an experimental
# parameter: it changes how long a run takes and nothing it measures, because
# every generation is an independent request built from a context that does not
# depend on what else is running. It is kept here so there is one place to
# change it. Raise it only as far as the account's rate limit allows.
N_WORKERS = 8   # 12 hit the per-model rate limit; 8 did not

# --------------------------------------------- held constant: retrieval
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
K = 10                    # retrieval depth, held constant (Section 5.4)
MAX_SEQ_LEN = 384         # encoder truncation, in tokens

# ------------------------------------------------------- the query set
N_DIRECT = 50             # exactly one passage marked is_selected
N_MULTIHOP = 50           # two or more passages marked is_selected
N_Q = N_DIRECT + N_MULTIHOP

# MS MARCO's coarse intent labels. The query set is balanced across them within
# each category -- ten per (category, type) cell, 5 x 2 x 10 = 100 -- so that
# the type can be entered as a covariate instead of being confounded with the
# category. The natural distribution is very uneven (DESCRIPTION 44%, PERSON 7%
# on the well-formed rows), so a marginal figure over this query set is a
# balanced average, not an estimate on the natural distribution.
QUERY_TYPES = ("DESCRIPTION", "ENTITY", "LOCATION", "NUMERIC", "PERSON")
N_PER_STRATUM = N_DIRECT // len(QUERY_TYPES)   # 10

# ------------------------------------------------------- the corpus
# Section 5.5: the knowledge base is defined by a rule -- every passage attached
# to a validation row with a non-empty wellFormedAnswers field -- not by a
# sample, so there is no distractor budget and no sampling seed to defend.
# N_PASSAGES is what that rule yields on the v2.1 validation split after
# deduplication; it is recorded, not chosen.
N_PASSAGES = 122_678
PASSAGE_MIN_WORDS = 0     # a passage MS MARCO retrieved is in the corpus
CORPUS_PARQUET = DATA / "ms_marco_v2.1_validation_wellformed.parquet"
VECTOR_STORE = DATA / "vector_store"

# ------------------------------------------------------- the design
# Three published baselines, each in four wordings. The four wordings state the
# same requirement set; only how it is written changes. See
# experiment/conditions.py for the texts and the rules they were written under.
BASELINES = ["A", "B", "C"]
WORDINGS = ["canonical", "lexical", "syntactic", "format"]
PROMPT_CONDITIONS = [f"{b}{i}" for b in BASELINES for i in range(len(WORDINGS))]
# Two evidence levels. "gold" supplies the annotated supporting passages, so
# the answer is certainly present and a failure belongs to the generator;
# "ret" supplies the real top-K, which is the deployed system. A corrupted
# and an empty level were part of an earlier design; they asked about
# evidence against parametric memory rather than about prompt structure, and
# they are not in this one.
EVIDENCE_LEVELS = ["gold", "ret"]

# Retrieval depth is held constant at K = 10. It admits more of the annotated
# evidence than a shallower depth -- the gate passes 88 of 100 queries at K =
# 10 against 74 at K = 5 -- so the retrieved-level comparison rests on a larger
# set of queries, and it gives the prompt a longer and noisier context to
# govern. A second depth was part of an earlier design as a robustness factor;
# it doubled the run for a question secondary to the one being asked, and is
# not in this one. run_retrieval takes --k, so another depth can still be
# retrieved without changing the design.
DEPTHS = [10]
RETRIEVED_DEPTHS = [10]   # the depths that require a retrieval pass

N_REPEATS = 3             # n in the thesis
# 24 cells (12 conditions x 2 evidence levels, at K = 5)
# 24 x 100 x 3 = 7,200 generations

SEED = 20260904           # fixes the query sample and the cell order
