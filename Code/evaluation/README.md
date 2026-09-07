# evaluation

Scores the generations and computes the reported effect.

    judge.py           the language-model judge and its fixed prompts
    metrics/           the four measures
    cost.py            l(y), the weighted sum of breaches
    analysis.py        empirical risk, tau_hat, paired tests, Holm
    run_scoring.py     generations.jsonl -> scores.jsonl
    run_analysis.py    scores.jsonl -> results.json

## Run

    python3 -m evaluation.run_scoring --program-only    # no model needed
    python3 -m evaluation.run_scoring                   # with the judge
    python3 -m evaluation.run_analysis

`--program-only` computes correctness and adherence and skips the two judged
measures. It loads no model, so it is the fast way to check the plumbing.

## The four measures

| measure | how | what it detects | cost term |
|---------|-----|-----------------|-----------|
| faithfulness | judged | claims the passages do not support | n_1 |
| relevance | judged | answers that drift, hedge or pad | n_2 |
| correctness | program | disagreement with the reference answer | n_3 |
| adherence | program | requirements of the prompt not met | n_4 |

Each returns a `MetricResult` with `breaches` (the count entering the cost) and
`score` (the rate on [0,1], higher better). Both are kept because the cost needs
the count and the results table needs the rate: a raw count grows with answer
length, and prompts change how much a model writes.

**Faithfulness** decomposes the answer into atomic claims and asks the judge
whether the passages support each. n_1 is the number unsupported.

**Relevance** has the judge write questions the answer would answer well, then
compares them with the question actually asked, by cosine in the retrieval
encoder's space. Reusing that encoder avoids introducing a second space.

**Correctness** checks whether the answer contains the reference, rather than
matching it exactly. MS MARCO's well-formed answers are complete sentences, so
exact match would fail on a correct answer phrased differently and report
verbosity as error. Token F1 and exact match are kept in `detail` so a reader
can see how much the containment judgement is carrying.

**Adherence** is prompt-level strict: one failed requirement fails the whole
generation. Only rule-verifiable requirements are checked, which is the design
principle of verifiable instructions.

## Two decisions the thesis leaves open

**Grounding is not counted twice.** The requirement set includes "answer only
from the retrieved passages", and that is exactly what faithfulness measures.
Checking it again inside adherence would count one failure in both n_1 and n_4
and silently weight grounding double in the cost. Adherence therefore checks
`length`, `abstain` and `disagree`, and records the exclusion in
`detail["not_checked"]` rather than leaving it implicit.

`disagree` is only partly rule-checkable: a keyword rule catches an explicit
statement of conflict but will miss a paraphrase, so its failures are more
trustworthy than its passes.

**Faithfulness is inapplicable under the empty evidence level.** With no
passages every claim is unsupported by construction, so scoring it there would
report the absence of context as a failure of grounding. The measure returns
`applicable=False` and contributes nothing, instead of returning a misleading
zero.

That makes raw costs comparable within an evidence level but not across them,
since the empty level sums over fewer terms. The main result compares conditions
inside a level, so this is not a problem for it, but a cost averaged over all
four levels would be meaningless. `n_applicable` is recorded on every row.

## The judge

Faithfulness and relevance have no reference string, so a separate model scores
them. Since the thesis assumes models are sensitive to their instructions, the
judge cannot be assumed reliable either. Three precautions:

- **Its prompt is fixed.** A module constant, not a parameter.
- **It is blind.** It receives the answer and the context, never the condition
  label, the query id or the evidence level. The format condition is
  recognisable from its surface alone, so a judge error varying with condition
  would bias the estimate.
- **Order is randomised.** `run_scoring.py` shuffles before judging, so position
  in the sequence cannot correlate with condition.

`Judge` wraps any callable mapping a prompt to a string, so a stub can be
substituted in tests without loading a model.

Agreement with human labelling is not implemented. It needs about 100 held-out
outputs labelled by hand, and it should be reported per condition rather than
pooled. `sklearn.metrics.cohen_kappa_score` is available for it.

## The cost function

    l(y) = sum_k lambda_k * n_k(y)

**The weights are not fixed by the thesis.** Section 4.9 argues that the four
requirements are not equally serious and that their weight depends on the
deployment setting, but assigns no values. Rather than invent them, `cost.py`
offers two schemes:

- `equal`, the default. Every weight is 1, so the cost is a count of breaches
  and the headline number has no free parameters.
- `grounding`, which doubles faithfulness. Intended as a pre-specified
  sensitivity analysis: if the ordering of conditions survives reweighting, say
  so; if it flips, say that.

Record the scheme with any result. Costs are not comparable across weightings.

## The analysis

`run_analysis.py` reports, in order:

1. **Empirical risk**, the mean cost in each of the 20 cells.
2. **tau_hat**, the cost of each condition relative to S0, computed within each
   evidence level.
3. **The manipulation check**, whether the evidence levels differ at the
   baseline prompt. This establishes that the evidence manipulation moved the
   outcome at all. It is a validity check, not a claim about which evidence is
   better.
4. **The interaction**, whether the prompt matters more under corrupted evidence.

All comparisons are paired at query level, because every query runs in every
condition and pairing removes between-query variation. Repeats are averaged
before pairing: they measure decoding noise, not variation between queries, and
treating them as independent observations would overstate the sample size.

A paired t-test is used where the differences look normal by Shapiro-Wilk, and
Wilcoxon signed-rank otherwise. Which test ran is reported, since the choice is
data-dependent. Constant non-zero differences have no variance for a normality
test and go straight to Wilcoxon.

Each family is Holm-corrected separately. Pooling them would let the
manipulation check inflate the correction on the result of interest.

## Output contract

`runs/scores.jsonl`, one object per generation:

    query_id, category, condition, evidence_level, repeat
    cost                 l(y) under the chosen weights
    n_applicable         how many measures contributed
    excluded             measures that did not apply
    metrics              per measure: breaches, score, applicable, detail

`runs/results.json` holds the empirical risks, the effects with raw and
Holm-adjusted p-values, the manipulation check and the interaction contrasts.
