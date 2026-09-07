"""
Fixed experimental parameters.

Every value the study holds constant lives here and nowhere else. If a value is
not in this file it is not a controlled parameter; if it is in this file it must
not differ between conditions. This mirrors Section 5.1 of the thesis, which
lists the components held constant and the reason each one is fixed.

Changing anything under "held constant" invalidates comparison with runs made
before the change. Record SEED and the model identifiers in the thesis.
"""

from pathlib import Path

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"      # inputs: the MS MARCO release, downloaded separately
RUNS = ROOT / "runs"      # outputs: everything the pipeline writes

# ------------------------------------------------------- held constant
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
GENERATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"
K = 5                     # reference retrieval depth; the default for run_retrieval
MAX_SEQ_LEN = 384         # encoder truncation
TEMPERATURE = 0.7         # fixed and low; recorded, never varied
MAX_NEW_TOKENS = 256

# ------------------------------------------------------- the query set
N_DIRECT = 50             # exactly one passage marked is_selected
N_MULTIHOP = 50           # two or more passages marked is_selected
N_Q = N_DIRECT + N_MULTIHOP

# ------------------------------------------------------- the corpus
N_PASSAGES = 800          # gold passages plus a distractor pool
PASSAGE_MIN_WORDS = 20    # drop fragments that carry no evidence

# ------------------------------------------------------- the design
# Three published baselines, each in five wordings. See experiment/conditions.py.
BASELINES = ["A", "B", "C"]
PROMPT_CONDITIONS = [f"{b}{i}" for b in BASELINES for i in range(5)]
EVIDENCE_LEVELS = ["gold", "ret", "corr", "empty"]

# Retrieval depth is a controlled robustness factor, not a held-constant: the
# prompt contrast is estimated separately within each depth and the depths are
# never compared with one another. It applies to the "ret" level only, since the
# other three set the context directly and never consult the retriever.
# K = 0 is the same cell as the "empty" level and so adds no run of its own;
# only K = 10 adds cells beyond the reference depth.
DEPTHS = [0, 5, 10]       # K in the thesis; run_retrieval takes --k
RETRIEVED_DEPTHS = [5, 10]   # the depths that actually require a retrieval pass

N_REPEATS = 5             # n in the thesis; the standard repetition budget
# 60 cells at K=5, plus 15 more for "ret" at K=10 = 75 cells
# 75 x 100 x 5 = 37500 generations

SEED = 20260904           # fixes the query sample and the cell order
