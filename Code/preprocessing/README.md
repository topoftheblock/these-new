# preprocessing

Turns an MS MARCO v2.1 release into the two files the experiment needs: a query
set and a knowledge base.

    huggingface.py  download the parquet export and normalise its schema
    fetch.py        download and keep only well-formed rows
    loader.py       read the release, whatever shape it ships in
    filters.py      the selection rules
    corpus.py       assemble the knowledge base
    build.py        command-line entry point

## Run

    python3 -m preprocessing.fetch                                    # download + filter
    python3 -m preprocessing.build --input ../data/ms_marco_v2.1_validation_wellformed.parquet

`fetch` options: `--split train|validation|test`, `--version`, `--shard`, `--force`.
`build` options: `--seed`, `--n-direct`, `--n-multihop`. Defaults from `config.py`.

## Source

`microsoft/ms_marco`, config `v2.1`, read from the Hugging Face parquet export.

The parquet files are read with `pyarrow` rather than through the `datasets`
library. That avoids a heavy dependency, and on this machine `datasets` cannot be
imported at all: the environment has a pydantic / pydantic-core version conflict
that surfaces through `aiohttp`. `pyarrow` reads the same files unaffected.

**Schema difference.** The parquet export does not store passages the way the raw
JSON release does. Instead of a list of passage objects it stores one struct of
parallel arrays:

    passages: struct<is_selected: list<int32>,
                     passage_text: list<string>,
                     url:          list<string>>

`huggingface.normalise()` transposes that back into the per-passage form, so
`filters.py` needs no special case. `loader.load_rows` dispatches on the file
extension, so JSON, JSONL and parquet all work through one entry point.

**What the validation split contains.** 101,093 rows, of which 12,467 (12.3%)
carry a non-empty `wellFormedAnswers`. Among those: 12,062 direct, 386
multi-hop, 19 with no gold passage. Both categories are drawn from that pool.

## The two filters

Applied in order, in `filters.py`.

**1. Well-formed answer required.** Only rows with a non-empty
`wellFormedAnswers` survive. `fetch.py` applies this once, up front, and writes
the survivors to their own parquet file, so every later step operates on rows
that can actually be scored and the count of what survived is recorded rather
than being an invisible side effect of sampling. MS MARCO's well-formed answers are rewritten by
annotators to read as complete sentences without the question attached, which is
what makes them usable as a reference string for free-text correctness. Rows
whose only answer is `"No Answer Present."` carry no reference and are dropped
here too.

The field is stored inconsistently in the release, sometimes as a list and
sometimes as the literal string `"[]"`. `well_formed_answers()` handles both.

**2. Evidence distribution.** Surviving rows are split by how many passages
carry `is_selected == 1`:

| gold passages | category    | what it tests                            |
|---------------|-------------|------------------------------------------|
| exactly one   | `direct`    | using evidence that is directly available |
| two or more   | `multi-hop` | aggregating evidence spread over passages |
| zero          | dropped     | nothing to retrieve                       |

The count comes from the annotation, so a query's category is a property of the
release rather than a judgement made here.

## The knowledge base

`corpus.py` builds it from two sources, both from the release:

- **Gold passages**, from the sampled queries. These are what retrieval has to
  find and what the gate checks for.
- **Distractors**, drawn from rows that were not sampled, taking only passages
  never marked `is_selected` for their own query.

Distractors come from the same collection on purpose. Retrieval difficulty
should come from genuine near-misses, not from off-domain filler that any
encoder separates trivially.

The minimum-length rule in `config.PASSAGE_MIN_WORDS` applies to **distractors
only**. Gold passages are kept at any length: dropping a short one would change
a query's category without changing its label, so a multi-hop query could end up
needing one passage while still being counted and gated as multi-hop.

Passage ids are stable and readable: `"<query_id>-g<n>"` for gold, `"d<n>"` for
distractors. With the same seed a gold passage keeps its id across runs, which
is what lets two retrieval results be compared.

## Output contract

`runs/queries.json`

    {"seed": 20260904,
     "source": "../data/dev_v2.1.json",
     "n_queries": 40,
     "queries": [{"query_id": 1102432,
                  "query": "...",
                  "category": "direct",
                  "answer": "the well-formed answer, used as the reference",
                  "gold_ids": ["1102432-g0"]}, ...]}

`runs/corpus.json`

    {"1102432-g0": "passage text", "d0": "passage text", ...}

## Reading the report

`build.py` prints how many rows it read, how many fell into each category, and
how many each rule dropped. Those counts belong in the thesis: they document
what fraction of the release the query set was drawn from, and a reader can
reproduce them from the same split and seed.

A warning appears when the corpus comes out below the target size in
`config.N_PASSAGES`. Use a larger split rather than lowering the target, since
the distractor density inside the top K is what the target controls.
