# evaluation

Scores the generations and estimates the effect of a rewording.

    judge.py           the language-model judge and its fixed prompts
    metrics/           the four measures
    analysis.py        tau_hat per measure, 95% CIs, Holm, the verdict
    run_scoring.py     generations.jsonl -> scores.jsonl
    run_analysis.py    scores.jsonl -> results.json
    agreement.py       Cohen's kappa of the judge against human labels

## Run

    python3 -m evaluation.run_scoring --program-only    # no judge, no key needed
    python3 -m evaluation.run_scoring                   # with the judge
    python3 -m evaluation.run_analysis

`--program-only` computes correctness and adherence and skips the two judged
measures. It needs no API key, so it is the fast way to check the plumbing.

## The four measures

| measure | how | what it detects |
|---------|-----|-----------------|
| faithfulness | judged | claims the passages do not support |
| relevance | judged + encoder | answers that drift, hedge or pad |
| correctness | program | disagreement with the reference answer |
| adherence | program | rule-checkable requirements of the prompt not met |

Each returns a `MetricResult` carrying `score`, the rate on [0,1] with higher
meaning better. **`score` is what the analysis reads as M(Y).** `breaches`, the
raw count, is still recorded for inspection but nothing downstream sums it: a
raw count grows with answer length and prompts change how much a model writes.

**Faithfulness** has the judge decompose the answer into atomic claims, then
asks it, claim by claim, whether the passages support it. The score is the
supported fraction.

**Relevance** has the judge write three questions the answer would answer well,
then compares them with the question actually asked, by mean cosine in the
retrieval encoder's space; the score is that cosine, and below 0.5 counts as a
breach. A refusal when evidence was supplied scores zero. (The abstention branch
that scored a refusal as *correct* applied to the empty evidence level, which
this design no longer runs.)

**Correctness** checks whether the normalised answer contains the normalised
reference, rather than matching it exactly. MS MARCO's well-formed answers are
complete sentences, so exact match would report verbosity as error. Token F1 and
exact match are kept in `detail`.

**Adherence** is prompt-level strict: one failed requirement fails the whole
generation. Only rule-verifiable requirements are checked, and only those the
baseline's own prompt states:

| baseline | rule checked |
|----------|--------------|
| A | none; adherence is **undefined** and is skipped, not scored as a pass |
| B | every sentence carries one to three `[n]` citation markers |
| C | the answer contains a justification marker (*because*, *since*, ...) |

Grounding is never checked inside adherence, because faithfulness already
measures it. Register, comprehension and concision-without-a-threshold are not
decidable by rule and are recorded in `detail["not_checked"]`.

Because adherence is undefined for baseline A, `tau_hat` for adherence is
reported for baselines B and C only. An inapplicable measure is skipped rather
than counted as satisfied, which would put a free point into the mean.

## The judge

Faithfulness and relevance have no reference string, so a second model scores
them: `config.JUDGE_MODEL` at temperature 0, built by `judge.make_judge()`. It
is a different model from the generator, so the generator never grades its own
output. Since the thesis assumes models are sensitive to their instructions, the
judge cannot be assumed reliable either. Three precautions:

- **Its prompt is fixed.** A module constant, not a parameter.
- **It is blind.** It receives the answer and the context, never the condition
  label, the query id or the evidence level. The format condition is
  recognisable from its surface alone, so a judge error varying with condition
  would bias the estimate.
- **Order is randomised.** `run_scoring.py` shuffles before judging, so position
  in the sequence cannot correlate with condition.

`Judge` wraps any callable mapping a prompt to a string, so a stub can be
substituted in tests without a key.

Agreement with human labelling: `agreement.py sample` draws about 100 scored
outputs, stratified by condition, and writes a labelling sheet with the judge's
verdict withheld. After a person fills in the `human` column, `agreement.py
report` computes Cohen's kappa per condition and measure. **The labelling has
not been done**, so the judged measures currently carry no reliability estimate.

## The analysis

There is **no cost function**. The four measures are never combined. Each is
used in turn as the measured property M of Equation 4.12, so a rewording gets
four effect estimates rather than one pooled score. Combining them would hide
direction: a rewording that improved grounding and damaged relevance equally
would average to no effect.

`run_analysis.py` reports, in order:

1. **Cell means** — the mean of each measure for each condition in each cell, so
   an effect can be read against the level it moves from.
2. **tau_hat** — for each measure and each cell, the effect of each rewording
   against the canonical form of *its own* baseline, with a standard error and
   a 95% confidence interval.
3. **The verdict on H0** — whether any rewording moved any measure after
   correction.

Every comparison is paired at query level, because every query runs in every
condition on a byte-identical context, and pairing removes between-query
variation. The repeats are averaged before pairing: they measure decoding
noise, not variation between queries, and treating them as independent
observations would overstate the sample size fivefold.

**Uncertainty and multiplicity.** `tau_hat` is a sample mean and is almost never
exactly zero even when the true effect is, so a point estimate alone cannot
decide H0. Each estimate carries a 95% CI. H1 is an "at least one" claim over
nine rewordings, so reading nine intervals and declaring H1 because one excluded
zero would report chance as a finding; Holm's method is applied across the nine
rewordings within each measure. Both the raw and adjusted verdicts are recorded.

## Output contract

`runs/scores.jsonl`, one object per generation:

    query_id, category, query_type, condition, evidence_level, depth, cell, repeat
    generator_model, judge_model
    metrics    per measure: breaches, score, applicable, detail

`runs/results.json`:

    cell_means    mean of each measure per condition per cell
    tau_hat       per cell and measure: tau, se, ci, p, p_holm, significant, n
    hypothesis    h0_rejected, how many estimates, and which ones survived
