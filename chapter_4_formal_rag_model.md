# Chapter 4 — Formal Model of the RAG System

## 4.1 Overview and Modelling Objective

This chapter formalizes the RAG system studied in this thesis by combining the probabilistic formulation of retrieval-augmented generation introduced by Lewis et al. (2020) with an explicit representation of the system prompt as a variable of the generation process.

The purpose of the formalization is twofold. First, it specifies the probabilistic relationship between the user query, retrieved documents, and generated output. Second, it provides a formal basis for treating the system prompt as an experimental intervention whose effect on system behavior can be studied while the query, knowledge base, retrieval mechanism, and underlying language model are held constant.

The resulting model can be summarized as:

\[
(q,D)
\rightarrow
p_\eta(z\mid q,D)
\rightarrow
p_\theta(y_i\mid q,z,y_{1:i-1},S)
\rightarrow
p(y\mid q,D,S)
\]

where the system prompt \(S\) enters explicitly at the generation stage.

The central research question can therefore be formulated as:

\[
\boxed{
\text{What changes in }
p_{\mathrm{RAG}}(y\mid q,D,S)
\text{ when }S\text{ is changed?}
}
\]

The formalization follows Lewis et al. (2020) in treating the retrieved document \(z\) as a latent variable and marginalizing over possible retrieved documents. The principal extension introduced in this thesis is to make the system prompt \(S\) an explicit conditioning variable of the generator.

---

## 4.2 Variables and Notation

The following notation is used throughout the formal model.

| Symbol | Definition |
|---|---|
| \(q\) | User query |
| \(D=\{d_1,\ldots,d_N\}\) | Knowledge base |
| \(z\) | A retrieved document or passage |
| \(\mathcal Z_K(q,D)\) | Set of the top-\(K\) retrieved documents |
| \(C\) | Retrieved context consisting of the retrieved documents |
| \(y=(y_1,\ldots,y_T)\) | Generated output sequence |
| \(y_i\) | \(i\)-th output token |
| \(S\) | System prompt |
| \(\eta\) | Parameters of the retriever |
| \(\theta\) | Parameters of the generator/language model |
| \(T\) | Number of output tokens |

The query \(q\) corresponds to the input sequence \(x\) used in Lewis et al. (2020). The notation \(q\) is retained here because it is consistent with the terminology used throughout this thesis.

The knowledge base is represented as:

\[
D=\{d_1,d_2,\ldots,d_N\}.
\]

A retrieved document is denoted by:

\[
z\in D.
\]

The retrieved context can therefore be represented as a collection of the top-\(K\) retrieved documents:

\[
C=\{z_1,\ldots,z_K\}.
\]

The generated answer is represented as a sequence of tokens:

\[
y=(y_1,\ldots,y_T).
\]

Finally, the system prompt is represented as:

\[
S\in\mathcal S,
\]

where \(\mathcal S\) denotes the set of possible system prompts considered by the system.

---

## 4.3 Probabilistic Retrieval

Following Lewis et al. (2020), retrieval is represented probabilistically as:

\[
\boxed{
p_\eta(z\mid q,D)
}
\tag{4.1}
\]

where \(\eta\) denotes the parameters of the retriever.

The expression \(p_\eta(z\mid q,D)\) can be read as:

> The probability assigned by the retriever to document \(z\), given query \(q\) and knowledge base \(D\).

The retriever therefore does not have to make a single deterministic decision about which document is correct. Instead, it assigns different probabilities to candidate documents.

In practice, the full knowledge base may contain a very large number of documents. The RAG formulation therefore considers a top-\(K\) approximation of the documents with the highest retrieval probability:

\[
\boxed{
\mathcal Z_K(q,D)
=
\operatorname{TopK}_{z\in D}
p_\eta(z\mid q,D)
}
\tag{4.2}
\]

The retrieved context can consequently be represented at the system level as:

\[
\boxed{
C=R_K(q,D)
=
\{z_1,\ldots,z_K\}
}
\tag{4.3}
\]

where \(R_K\) denotes the retrieval operation returning the \(K\) highest-ranked documents.

The distinction between \(z\) and \(C\) is important. The variable \(z\) represents an individual candidate document in the probabilistic formulation, whereas \(C\) represents the collection of documents retrieved by the actual system.

Lewis et al. treat \(z\) as a latent variable because the training objective does not directly specify which retrieved document should be responsible for generating the target answer.

---

## 4.4 Prompt-Conditioned Generation

The generator in Lewis et al. is formulated at the token level as:

\[
p_\theta(y_i\mid x,z,y_{1:i-1}).
\]

In the present thesis, this formulation is extended to explicitly represent the system prompt:

\[
\boxed{
p_\theta(y_i\mid q,z,y_{1:i-1},S)
}
\tag{4.4}
\]

where:

- \(q\) is the user query,
- \(z\) is a retrieved document,
- \(y_{1:i-1}\) represents the previously generated tokens,
- \(S\) is the system prompt,
- \(\theta\) represents the fixed parameters of the language model.

The expression can be read as:

> The probability assigned to the next output token \(y_i\), given the user query, retrieved document, previously generated tokens, and system prompt.

Because the generator is autoregressive, the probability of a complete output sequence \(y\) conditional on a particular document \(z\) and system prompt \(S\) is:

\[
\boxed{
p_\theta(y\mid q,z,S)
=
\prod_{i=1}^{T}
p_\theta(y_i\mid q,z,y_{1:i-1},S)
}
\tag{4.5}
\]

This equation connects the token-level language model to the sequence-level output.

The introduction of \(S\) is the principal extension of the original Lewis et al. formulation made by this thesis. Rather than treating the generator simply as \(p_\theta(y_i\mid q,z,y_{1:i-1})\), the model explicitly represents the system prompt as a conditioning variable:

\[
S
\rightarrow
p_\theta(y_i\mid q,z,y_{1:i-1},S).
\]

---

## 4.5 RAG-Sequence

### 4.5.1 Conceptual formulation

RAG-Sequence assumes that one retrieved document is used to condition the generation of the complete output sequence.

For a particular document \(z\), the probability of the complete output is:

\[
p_\theta(y\mid q,z,S)
=
\prod_{i=1}^{T}
p_\theta(y_i\mid q,z,y_{1:i-1},S).
\]

Because the correct document is not known with certainty, the model marginalizes over the retrieved documents.

The resulting RAG-Sequence formulation is:

\[
\boxed{
p_{\mathrm{RAG\text{-}Sequence}}
(y\mid q,D,S)
\approx
\sum_{z\in\mathcal Z_K(q,D)}
p_\eta(z\mid q,D)
p_\theta(y\mid q,z,S)
}
\tag{4.6}
\]

Expanding the sequence probability gives:

\[
\boxed{
p_{\mathrm{RAG\text{-}Sequence}}
(y\mid q,D,S)
\approx
\sum_{z\in\mathcal Z_K(q,D)}
p_\eta(z\mid q,D)
\prod_{i=1}^{T}
p_\theta(y_i\mid q,z,y_{1:i-1},S)
}
\tag{4.7}
\]

The important mathematical structure is therefore:

\[
\boxed{
\sum_z\prod_i
}
\]

The summation over documents occurs **outside** the product over output tokens.

### 4.5.2 Interpretation

RAG-Sequence can be understood as asking:

> How probable is the entire generated answer if document \(z\) is the document that explains the answer?

This calculation is performed for the relevant retrieved documents, with their contributions weighted according to the retriever probabilities.

Thus, the same latent document \(z\) conditions every token in the output sequence.

Conceptually:

```text
Query q
   │
   ▼
Retriever
   │
   ├── z₁ ──► Generator ──► entire output
   │
   ├── z₂ ──► Generator ──► entire output
   │
   └── z₃ ──► Generator ──► entire output
```

The system then marginalizes over these document-level hypotheses.

---

## 4.6 RAG-Token

### 4.6.1 Conceptual formulation

RAG-Token makes a different assumption.

Rather than using the same latent document for the complete output sequence, it marginalizes over the retrieved documents at each generation step.

The formulation is:

\[
\boxed{
p_{\mathrm{RAG\text{-}Token}}
(y\mid q,D,S)
\approx
\prod_{i=1}^{T}
\sum_{z\in\mathcal Z_K(q,D)}
p_\eta(z\mid q,D)
p_\theta(y_i\mid q,z,y_{1:i-1},S)
}
\tag{4.8}
\]

The important mathematical structure is therefore:

\[
\boxed{
\prod_i\sum_z
}
\]

rather than:

\[
\boxed{
\sum_z\prod_i.
}
\]

### 4.6.2 Interpretation

RAG-Token can be understood as asking:

> For each output token, how much does each retrieved document contribute to the probability of generating that token?

This allows different documents to contribute to different parts of the generated answer.

Conceptually:

```text
Query q
   │
   ▼
Retriever
   │
   ├── z₁ ──┐
   ├── z₂ ──┼──► Token y₁
   └── z₃ ──┘

   ├── z₁ ──┐
   ├── z₂ ──┼──► Token y₂
   └── z₃ ──┘

   ├── z₁ ──┐
   ├── z₂ ──┼──► Token y₃
   └── z₃ ──┘
```

Consequently, RAG-Token can effectively use different retrieved documents for different parts of the generated sequence.

---

## 4.7 RAG-Sequence versus RAG-Token

The distinction can be summarized mathematically as:

### RAG-Sequence

\[
\boxed{
\sum_z
\prod_i
}
\]

### RAG-Token

\[
\boxed{
\prod_i
\sum_z
}
\]

This apparently small difference has an important conceptual consequence.

| | RAG-Sequence | RAG-Token |
|---|---|---|
| Document marginalization | Once for the sequence | At every token |
| Document assumption | Same document for complete output | Different documents can contribute to different tokens |
| Mathematical form | \(\sum_z\prod_i\) | \(\prod_i\sum_z\) |
| Flexibility | Lower | Higher |

For RAG-Sequence, the model evaluates document hypotheses at the level of the complete output sequence.

For RAG-Token, the model evaluates document contributions separately at each generation step.

---

## 4.8 Incorporating the System Prompt as a Causal Variable

The main extension introduced by this thesis is the explicit representation of the system prompt \(S\).

Under the proposed model, the prompt enters the generation process:

\[
\boxed{
S
\rightarrow
p_\theta(y_i\mid q,z,y_{1:i-1},S)
\rightarrow
y_i
}
\tag{4.9}
\]

However, the prompt does not enter the retrieval distribution:

\[
\boxed{
p_\eta(z\mid q,D)
}
\tag{4.10}
\]

rather than:

\[
p_\eta(z\mid q,D,S).
\]

This represents the modelling assumption that the system prompt is supplied to the generator but does not influence document retrieval.

Consequently:

\[
\boxed{
S\nrightarrow z
}
\]

while:

\[
\boxed{
S\rightarrow y.
}
\]

The causal structure of the system can therefore be represented as:

```text
                         Knowledge base D
                               │
                               ▼
Query q ───────────────► Retriever
                               │
                     pη(z | q,D)
                               │
                               ▼
                        Retrieved z
                               │
                               │
System prompt S ───────────────┤
                               │
Query q ───────────────────────┤
                               ▼
                         Generator θ
                               │
               pθ(yi | q,z,y1:i−1,S)
                               │
                               ▼
                         Output y
```

The system prompt therefore acts as an intervention variable at the generation stage rather than at the retrieval stage.

---

## 4.9 Prompt Conditions and Experimental Interventions

Let:

\[
S\in\mathcal S
\]

denote the set of system prompts considered in the experiment.

Let:

\[
S_0
\]

denote the baseline system prompt and:

\[
S_1,\ldots,S_J
\]

denote the alternative prompt conditions.

For a particular prompt condition \(S_j\), the output distribution is:

\[
\boxed{
Y^{(j)}
\sim
p_{\mathrm{RAG}}
(y\mid q,D,S_j)
}
\tag{4.11}
\]

where \(Y^{(j)}\) denotes the potential output under prompt condition \(S_j\).

Equivalently, the intervention can be represented using the causal intervention operator:

\[
\boxed{
Y^{(j)}
=
Y\mid do(S=S_j).
}
\tag{4.12}
\]

The causal interpretation is therefore:

> What output would the RAG system produce if the system prompt were set to \(S_j\), while all other relevant system components remained unchanged?

---

## 4.10 Experimental Control Assumptions

For the effect of the system prompt to be interpretable, the following quantities are held constant across prompt conditions.

### Assumption 1 — Fixed query

\[
q=q_0.
\]

The same user query is evaluated under each prompt condition.

### Assumption 2 — Fixed knowledge base

\[
D=D_0.
\]

The same knowledge base is used across conditions.

### Assumption 3 — Prompt-independent retrieval

\[
\boxed{
p_\eta(z\mid q,D,S)
=
p_\eta(z\mid q,D).
}
\tag{4.13}
\]

Under this assumption, changing \(S\) does not change the retrieval distribution.

### Assumption 4 — Fixed retriever

\[
\eta=\eta_0.
\]

The retriever and its parameters remain unchanged.

### Assumption 5 — Fixed generator

\[
\theta=\theta_0.
\]

The underlying language model is not retrained or modified between prompt conditions.

### Assumption 6 — Prompt-dependent generation

Although \(\theta\) remains fixed, the conditional generation distribution may vary with \(S\):

\[
p_{\theta_0}(y_i\mid q,z,y_{1:i-1},S_j)
\neq
p_{\theta_0}(y_i\mid q,z,y_{1:i-1},S_0).
\]

The system prompt therefore changes the conditioning information supplied to the same underlying generator.

### Assumption 7 — Generation randomness

The same configuration may produce different outputs because of decoding randomness. This can be represented by an additional stochastic component:

\[
Y
=
G(q,z,S;\theta,\epsilon),
\]

where \(\epsilon\) represents generation randomness.

---

## 4.11 The System-Level RAG Distribution

The complete prompt-conditioned RAG system can therefore be represented as:

\[
\boxed{
p_{\mathrm{RAG}}
(y\mid q,D,S)
}
\tag{4.14}
\]

For RAG-Sequence:

\[
\boxed{
p_{\mathrm{RAG-S}}
(y\mid q,D,S)
\approx
\sum_{z\in\mathcal Z_K}
p_\eta(z\mid q,D)
p_\theta(y\mid q,z,S)
}
\tag{4.15}
\]

For RAG-Token:

\[
\boxed{
p_{\mathrm{RAG-T}}
(y\mid q,D,S)
\approx
\prod_{i=1}^{T}
\sum_{z\in\mathcal Z_K}
p_\eta(z\mid q,D)
p_\theta(y_i\mid q,z,y_{1:i-1},S)
}
\tag{4.16}
\]

This provides the link between the probabilistic RAG architecture and the causal research question.

The system prompt is an input to the generator, and therefore changing it can change the resulting distribution over generated outputs:

\[
\boxed{
S
\rightarrow
p_{\mathrm{RAG}}(y\mid q,D,S).
}
\tag{4.17}
\]

---

## 4.12 Potential Outcomes and Causal Effects

To quantify the effect of changing the system prompt, define a measurable property of the generated output as:

\[
M(Y).
\]

The metric \(M\) may represent any relevant behavioral property of the system output, such as factuality, instruction adherence, refusal behavior, verbosity, or another predefined behavioral criterion.

For prompt condition \(S_j\), define the expected behavioral outcome as:

\[
E[M(Y^{(j)})].
\]

The causal effect of prompt condition \(S_j\) relative to the baseline \(S_0\) can then be defined as:

\[
\boxed{
\tau_j
=
E[M(Y^{(j)})]
-
E[M(Y^{(0)})].
}
\tag{4.18}
\]

Using causal intervention notation:

\[
\boxed{
\tau_j
=
E[M(Y)\mid do(S=S_j)]
-
E[M(Y)\mid do(S=S_0)].
}
\tag{4.19}
\]

The probabilistic RAG formulation makes the relationship to the underlying system explicit:

\[
\boxed{
\tau_j
=
E_{Y\sim p_{\mathrm{RAG}}(\cdot\mid q,D,S_j)}
[M(Y)]
-
E_{Y\sim p_{\mathrm{RAG}}(\cdot\mid q,D,S_0)}
[M(Y)].
}
\tag{4.20}
\]

This equation connects the RAG probability model directly to the causal estimand of the thesis.

---

## 4.13 Interpretation of the Causal Effect

The causal effect \(\tau_j\) should be interpreted as the difference in expected system behavior resulting from changing the system prompt from the baseline condition \(S_0\) to the alternative condition \(S_j\), under the stated experimental assumptions.

Importantly, this interpretation does not imply that the prompt changes the model parameters:

\[
\theta\neq\theta(S).
\]

Instead:

\[
\boxed{
\theta=\theta_0
}
\]

is held fixed while:

\[
\boxed{
S_0\rightarrow S_j
}
\]

is manipulated.

The prompt changes the conditional distribution of the generator rather than changing the underlying model itself:

\[
p_{\theta_0}(y_i\mid q,z,y_{1:i-1},S_0)
\]

may differ from:

\[
p_{\theta_0}(y_i\mid q,z,y_{1:i-1},S_j).
\]

Thus, the causal intervention is on the **conditioning variable \(S\)** rather than on the model parameters \(\theta\).

---

## 4.14 Relationship to the Original RAG Formulation

The formalization used in this thesis can be understood as an extension of the RAG formulation introduced by Lewis et al. (2020).

The original formulation represents retrieval as:

\[
p_\eta(z\mid x)
\]

and generation as:

\[
p_\theta(y_i\mid x,z,y_{1:i-1}).
\]

The present thesis retains the same retrieval structure while explicitly introducing the system prompt into the generator:

\[
\boxed{
p_\eta(z\mid q,D)
}
\]

and:

\[
\boxed{
p_\theta(y_i\mid q,z,y_{1:i-1},S).
}
\]

The resulting model therefore preserves the distinction between:

1. **retrieval**, governed by \(p_\eta(z\mid q,D)\);
2. **generation**, governed by \(p_\theta(y_i\mid q,z,y_{1:i-1},S)\); and
3. **system behavior**, represented by the resulting distribution \(p_{\mathrm{RAG}}(y\mid q,D,S)\).

The principal addition is the explicit treatment of \(S\) as an experimental variable.

---

## 4.15 Summary of the Formal Model

The complete formal model can be summarized by the following sequence.

### Step 1 — Retrieve documents

\[
\boxed{
p_\eta(z\mid q,D)
}
\]

The retriever determines how probable each candidate document is given the query and knowledge base.

### Step 2 — Condition generation on the retrieved document and system prompt

\[
\boxed{
p_\theta(y_i\mid q,z,y_{1:i-1},S)
}
\]

The generator predicts each output token based on the query, retrieved evidence, previous output, and system prompt.

### Step 3 — Marginalize over retrieved documents

For RAG-Sequence:

\[
\boxed{
p_{\mathrm{RAG-S}}
(y\mid q,D,S)
\approx
\sum_z
p_\eta(z\mid q,D)
p_\theta(y\mid q,z,S)
}
\]

For RAG-Token:

\[
\boxed{
p_{\mathrm{RAG-T}}
(y\mid q,D,S)
\approx
\prod_i
\sum_z
p_\eta(z\mid q,D)
p_\theta(y_i\mid q,z,y_{1:i-1},S)
}
\]

### Step 4 — Treat the system prompt as an intervention

\[
\boxed{
Y^{(j)}
\sim
p_{\mathrm{RAG}}(y\mid q,D,S_j)
}
\]

### Step 5 — Measure the behavioral effect

\[
\boxed{
\tau_j
=
E[M(Y)\mid do(S=S_j)]
-
E[M(Y)\mid do(S=S_0)].
}
\]

The resulting conceptual model is therefore:

```text
                         Knowledge base D
                               │
                               ▼
Query q ───────────────► Retriever
                               │
                     pη(z | q,D)
                               │
                               ▼
                        Retrieved z
                               │
                               │
System prompt S ───────────────┤
                               │
Query q ───────────────────────┤
                               ▼
                         Generator θ
                               │
               pθ(yi | q,z,y1:i−1,S)
                               │
                               ▼
                         Output y
                               │
                               ▼
                       Behavior M(Y)
```

The central causal relationship investigated in this thesis is therefore:

\[
\boxed{
S
\overset{do(\cdot)}{\longrightarrow}
p_{\mathrm{RAG}}(y\mid q,D,S)
\longrightarrow
M(Y)
}
\]

while the retrieval mechanism remains fixed:

\[
\boxed{
p_\eta(z\mid q,D)
}
\]

and does not depend on the system prompt under the assumptions of the experimental design.

This formulation preserves the probabilistic structure of Lewis et al. while making the system prompt an explicit causal variable whose effect on RAG system behavior can be experimentally evaluated.
