# rag

A simple retrieval-augmented generation pipeline. Retrieve once, generate once.

This package knows nothing about the experiment. It is a working RAG system you
could point at any corpus, and it is kept that way so that the experimental
manipulation lives entirely in `experiment/` and cannot leak into the model.

    embedder.py    text to vectors
    index.py       the vector store
    retriever.py   stage 1, plus the success check
    prompts.py     assembling the model input
    generator.py   stage 2
    pipeline.py    the two stages wired together

## Use it directly

    from rag import SimpleRAG

    corpus = {"p1": "The tower was completed in 1889.", "p2": "..."}
    rag = SimpleRAG.from_corpus(corpus)
    print(rag.answer("When was the tower completed?", corpus=corpus))

Encoding the corpus happens once, in `from_corpus`. The generator loads lazily,
so a retrieval-only run never pays for the language model.

## Why single-shot

No query rewriting, no reranking, no self-correction, no agent loop. That is a
restriction with a reason rather than a shortcut. In a single-shot pipeline the
system prompt reaches the answer along exactly one path, through the generator.
Add a step where the model decides what to retrieve and the prompt starts
influencing the evidence as well, at which point a difference in the output
cannot be attributed to either cause.

## The stage boundary

`SimpleRAG` exposes three methods:

    retrieve(question)                  stage 1
    generate(question, passages, system) stage 2
    answer(question)                    both, for interactive use

The experiment must use the first two and not the third. `answer()` collapses
the boundary the design depends on, which is fine when you are trying the system
by hand and wrong when you are running the study. `pipeline.py` says so in the
docstring, at the method itself.

Note that `generate()` takes passages as an argument instead of fetching them.
That is what lets the experiment substitute corrupted or empty evidence without
touching the retriever.

## Retrieval success

`RetrievalResult.success` is true when every passage in `gold_ids` appears in
the retrieved set. Two details matter:

- It requires **all** gold passages, not one. A multi-hop query whose answer
  needs two passages is not answerable from one of them, so partial retrieval
  is a failure.
- It returns `None`, not `False`, when `gold_ids` is empty. Unknown is not the
  same as failed, and a caller that treats them alike will silently drop every
  query it has no annotation for.

The retriever computes this but does not act on it. Deciding what to do with a
failure is the experiment's business.

## Embeddings

One encoder for both queries and passages, which is the dual-encoder arrangement
of dense passage retrieval: both land in the same space, so relevance is a dot
product. Mean pooling over the last hidden layer, then L2 normalisation, so an
inner product equals cosine similarity.

## The index

Flat inner product: every query compared against every passage, no
approximation. At this corpus size that is fast, and exactness matters more than
speed because an approximate index would make retrieval depend on how the index
was built rather than on the query alone.

FAISS is used when installed. The numpy path returns the same neighbours,
because a flat FAISS index performs the same exhaustive comparison. Check
`VectorIndex.backend` to see which ran; it is recorded in `runs/retrieval.json`.
