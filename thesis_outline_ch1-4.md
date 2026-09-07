# Chapter Plan — Chapters 1–4 (Instruction Robustness in RAG/GraphRAG Agent Pipelines)

Status: **awaiting approval** — no drafting has occurred. This is a planning artifact only.

Style calibration reference: `Bachelor_Thesis_Moritz.pdf` (structure/tone only — subject matter not used).

---

## Style Calibration Findings

- **Paragraphs**: dense, 6–12 sentences, few short paragraphs.
- **Chapter openers**: first paragraph of each chapter recaps the previous chapter and previews the current one.
- **Citation density**: high in Introduction/Literature Review (~1 citation per 1–2 sentences); the formal-modeling chapter is nearly citation-free (own definitions).
- **Equations**: central relationships are boxed (`\boxed{}`); asides go in footnotes, not body text.
- **Literature subsections**: every subsection ends with an explicit synthesis paragraph ("Together, these findings suggest…"), never just sequential summaries.
- **Figures**: one explanatory diagram per major abstract concept.
- **Research questions**: presented as a numbered list (RQ1/1a–c, RQ2/2a–b) at the end of the Approach-equivalent chapter.
- **Formatting**: `scrreprt`, numbered subsections in Intro/Preliminaries/LitReview, unnumbered (`*`) subsections in the modeling chapter — already matches `thesis.tex`.

---

## Chapter 1 — Introduction (~2 pp.)

| Subsection | Status | Content |
|---|---|---|
| 1.1 Background | Drafted | Keep as is. |
| 1.2 Motivation and Research Objective | Drafted | Keep as is; central RQ already stated. |
| **1.3 Notation** *(new)* | To write | Define $q, D, S, C, y, \theta$ and their spaces $\mathcal{Q}, \mathcal{D}, \mathcal{S}, \mathcal{C}, \mathcal{Y}$ once, before Chapter 4 relies on them. Mirrors the reference thesis's dedicated Notation block placed right after the Introduction. |
| 1.4 Thesis Structure | Stub → rewrite | One paragraph previewing all 7 chapters: Preliminaries → Literature Review → Formal Model → Experimental Setup → Results & Discussion → Conclusion. |

---

## Chapter 2 — Preliminaries (~6 pp., currently empty)

### 2.1 Path towards RAG
Short history: closed-book generation → its limits (hallucination, static/frozen knowledge) → open-book/retrieval-augmented approaches → emergence of RAG. Sets up Literature Review Branch 2.

### 2.2 Components of RAG
- 2.2.1 Knowledge Base / Vector Database (chunking, embeddings, indexing)
- 2.2.2 Retriever (similarity search, top-$k$ — prefigures $R(q;D)$ in Ch. 4)
- 2.2.3 Generator / Language Model (decoding, context window)
- 2.2.4 System Prompt (role, instructions, constraints — prefigures $S_\theta$ in Ch. 4)

### 2.3 Use-Case of RAG
Typical applications, why RAG over fine-tuning/long-context alone, known failure modes (motivates Ch. 3 and Ch. 4). Closing paragraph bridges into the literature review.

---

## Chapter 3 — Literature Review (~3 pp.)

### 3.1 Branch of Literature 1 — Prompt Sensitivity & Instruction Following
Content already exists (10 annotated papers: Cheng & Mastropaolo 2026; Mu et al. 2025; Lu et al. 2024; Zhuo et al. 2024; Salinas & Morstatter 2024; Zheng et al. 2024; Ying et al. 2024; Ruiz & Cardona-Betancourt 2026; Hua et al. 2025; Qin et al. 2026). Needs: an opening frame sentence + a closing synthesis paragraph (currently just sequential summaries).

### 3.2 Branch of Literature 2 — RAG Architecture, Evaluation, and Hallucination *(new — literature search completed)*

All five verified via web search (arXiv/ACL/AAAI, not fabricated):

1. **Lewis et al. (2020)** — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 33, 9459–9474. arXiv:2005.11401.
   Introduces RAG: a pretrained neural retriever combined with a pretrained seq2seq generator, outperforming purely parametric models on knowledge-intensive tasks. Directly motivates the $F(q;S,D)=G(q,R(q;D);S)$ decomposition used in this thesis's formal model.

2. **Gao et al. (2023/2024)** — *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997.
   Surveys RAG's evolution through Naive, Advanced, and Modular paradigms, organized around a tripartite retrieval–generation–augmentation structure. Supports treating retrieval and generation as separable functions whose interaction can be studied independently of the system prompt.

3. **Es et al. (2023)** — *Ragas: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217.
   Proposes a reference-free RAG evaluation framework with three metrics: faithfulness, answer relevance, and context relevance. The faithfulness metric closely parallels the evidence-faithfulness measure defined in this thesis and motivates evaluating system-prompt effects independently of retrieval quality.

4. **Chen et al. (2024)** — *Benchmarking Large Language Models in Retrieval-Augmented Generation*. AAAI 38, 17754–17762 (RGB benchmark).
   Shows LLMs tolerate retrieval noise reasonably well but struggle with negative rejection, information integration, and counterfactual robustness. RGB's four testbeds map closely onto this thesis's experimental query categories (direct, multi-hop, conflicting, insufficient-evidence).

5. **Niu et al. (2024)** — *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models*. ACL 2024. arXiv:2401.00396.
   Introduces a manually annotated, word-level hallucination corpus for RAG outputs and benchmarks detection methods across LLMs. Its fine-grained hallucination taxonomy supports the operationalization of this thesis's hallucination-rate metric.

**Closing synthesis paragraph (to draft):** Branch 1 establishes system prompts as a measurable, controllable behavioral mechanism, sensitive even to minor linguistic perturbation. Branch 2 establishes RAG's architecture, evaluation vocabulary, and known failure modes (noise robustness, hallucination, unfaithfulness). Neither branch examines system-prompt perturbation as a controlled experimental variable within a RAG pipeline while retrieval and the underlying model are held fixed — the specific gap this thesis addresses.

### 3.3 (optional) Explicit gap statement
One short subsection stating the gap directly, if you want it separated from the Branch 2 synthesis paragraph rather than folded into it.

---

## Chapter 4 — Formal Model of a RAG System (~5 pp., largely complete)

- **Opening paragraph** *(new)* — bridge from the Ch. 3 gap into "we now formalize…" (currently starts cold at "Let $q \in \mathcal{Q}$…").
- 4.1 Retrieval Function — drafted.
- 4.2 Generation Function — drafted.
- 4.3 Controlled Variables — drafted.
- 4.4 Modeling the System Prompt
  - 4.4.1 Linguistic Realization — drafted.
  - 4.4.2 Variation in System Prompts — drafted.
- **Suggested addition** — one summary diagram ($q \to R \to G \to y$ with $S_\theta$ as an injection point), matching the reference thesis's visual density in its equivalent chapter.

*(Note: Chapter 5 "Experimental Set Up" already contains the six-step experimental design from the exposé — out of scope for this outline, which stops at Chapter 4 per your request.)*
