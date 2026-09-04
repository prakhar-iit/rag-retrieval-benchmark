# Tasks

Worked one at a time. Plan and rationale live in `claude/job_search/retrieval_stack_project.md`
in the "Random" Claude project.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Phase 0 — Foundation

- [x] **0.1** Load arXiv ML abstracts (`CShorten/ML-ArXiv-Papers`), inspect fields
  - Fetched via user's own network (huggingface.co unreachable from this environment) and loaded with
    `--local`. Full pool: 117,592 rows, columns `title`/`abstract` (plus two stray `Unnamed:` index
    columns from the HF export, now dropped automatically).
  - Found and filtered: ~127/117,592 abstracts are withdrawal/retraction notices or otherwise
    degenerate (e.g. "This preprint has been withdrawn by the author for revision"), not real
    content. Added a regex filter plus a 15-word minimum in `_sample_and_id`.
- [x] **0.2** Sample 10-20K abstracts, one abstract = one document; freeze sample to `data/` with a fixed seed
  - 20,000 documents sampled (seed=13), frozen to `data/corpus_sample.parquet`. Length distribution:
    min=15, median=165, mean=166.8, max=319 words. All doc_ids unique, zero nulls, zero degenerate
    rows remaining. `tests/test_corpus.py`, 12 passing.
- [x] **0.3** Build the eval set: LLM-generate one research question per sampled abstract for ~300-500 abstracts
  - ⚠️ **Generate questions; do not extract phrases.** Extraction leaks surface tokens to BM25
  - No API key was configured in the working environment, so all 400 questions were composed
    directly (by Claude, reading each abstract) rather than via a scripted LLM call, in batches of
    ~20 documents per round-trip. Selected via `select_docs_for_queries(df, 400, seed=13)` on the
    post-filter 20K corpus.
  - One workshop-proceedings volume was caught among the first 40 docs; added the proceedings/
    front-matter filter to `corpus.py` (see 0.1/0.2) and re-sampled before continuing, so all 400
    final docs are drawn from the filtered corpus.
  - [x] Spot-check 20 by hand (`spot_check(records, n=20, seed=13)`) — all 20 are genuine content
    paraphrases, not title rewrites; reject rate 0/20.
  - Ran an automated title/query word-overlap heuristic (Jaccard-style, 4+ letter tokens minus
    stopwords) as an extra self-check beyond the spot-check: 178/400 queries have >50% overlap with
    their title's words, split almost evenly between the first 200 (90) and last 200 (88) composed.
    Manual inspection at the halfway point found most of this is unavoidable domain-jargon overlap
    on short titles (e.g. "Relative Flatness and Generalization"), not real extraction risk, with a
    minority of genuinely close title-paraphrases. Left as-is per review at the 100-question and
    140-question checkpoints — the rate did not visibly worsen or improve batch to batch, so this
    looks like an inherent property of short, jargon-dense arXiv titles rather than a fixable
    drafting issue.
  - Time actually spent well exceeded the one-hour timebox (400 questions manually composed across
    20 batches); noted here rather than re-scoped mid-flight since the user explicitly chose to push
    through to completion.
- [x] **0.4** Split eval set into fine-tune / held-out slices (must be **disjoint** — see 2b)
  - `split(records, holdout_frac=0.5, seed=13)` — 200 finetune / 200 holdout, disjoint by
    construction (single shuffle-and-partition over unique gold docs). Saved to `data/evalset.jsonl`
    (400 lines, verified round-trip via `load_jsonl`). `tests/test_evalset.py`, 12 passing (31 total
    across `test_metrics.py` + `test_corpus.py` + `test_evalset.py`).
- [x] **0.5** Metrics harness: nDCG@10, Recall@10, MRR, with unit tests on a toy ranking
  - `rss/metrics.py` implemented; `tests/test_metrics.py`, 7 passing.

## Phase 1 — Word2Vec from scratch (in-domain)

- [x] **1.1** Preprocess: tokenize, lowercase, strip punctuation
  - `rss/static_embed.tokenize`: single regex `[a-z][a-z0-9]*` on lowercased text. Tokens must
    start with a letter, so pure numbers are dropped but alphanumeric jargon (`resnet50`, `gpt3`)
    survives. No external tokenizer dependency, consistent with the rest of the repo.
- [x] **1.2** Run `gensim.models.Phrases` for bigrams/trigrams (`reinforcement_learning`, `attention_mechanism`)
  - `rss/static_embed.build_phrases`: two passes (bigrams, then trigrams over the bigram-merged
    sentences). 3,454,074 unigram tokens -> 2,945,107 tokens after merging on the real corpus.
- [x] **1.3** Train `Word2Vec(sg=1, vector_size=300, window=5, min_count=5, negative=10, epochs=10)`
  - `rss/static_embed.train`, run via `scripts/train_word2vec.py`. Vocab: 26,935 terms.
  - ⚠️ **Determinism vs. speed tradeoff, documented in `configs/default.yaml`:** gensim's threaded
    SGD is only bit-exact reproducible at `workers=1`, which takes ~350s for this corpus (10
    epochs) -- too slow for the tooling available while building this (no persistent background
    process across tool calls). Set `workers=4` (~95s train time) instead, which is
    *approximately* deterministic (same seed, but thread scheduling still races) rather than
    bit-exact. `static_embed.train()` itself still defaults to `workers=1` when not overridden by
    config, so unit tests stay bit-exact reproducible.
- [x] **1.4** Sanity check: nearest neighbours of `transformer`, `attention`, `diffusion` (+ `embedding`, `gradient`)
  - All five look sound. `transformer` -> `gnmt`, `xlnet`, `self_attention_mechanism`, `bert`; `attention`
    -> `attention_mechanism`, `multi_head`, `self_attention`; `embedding` -> `embeddings`,
    `distributed_stochastic_neighbor` (t-SNE), `skip_gram`; `gradient` -> `gradients`, `hessian`,
    `natural_gradient_descent`. One corpus-composition observation: `diffusion`'s neighbours are
    mostly diffusion-*process* terms (`diffusion_process`, `indian_buffet`, `markov_jump`), not
    diffusion *models* (image generation) -- this ML-ArXiv-Papers snapshot appears to under-represent
    or predate the diffusion-model boom relative to how the term is used today. Worth remembering
    when interpreting later embedding comparisons that also touch this term.
- [x] **1.5** Document-level vectors: mean pooling **and** IDF-weighted pooling
  - `rss/static_embed.doc_vector(model, tokens, idf=None)` -- one function, both modes (`idf=None`
    -> uniform mean; `idf={...}` -> weighted). `compute_idf` uses sklearn-style smoothed IDF.
- [ ] **1.6** Three-way comparison on the same eval set:
  - [ ] in-domain (arXiv-trained)
  - [ ] out-of-domain (existing OpinRank model from ML-Cookbook) -- **location TBD, asked the user**
  - [ ] pretrained GoogleNews vectors
  - [ ] **Finding to test: does domain beat scale for static embeddings?**

## Phase 2 — Embedding comparison

- [ ] **2.1** BM25 baseline (`rank_bm25`)
- [ ] **2.2** `all-MiniLM-L6-v2` and `all-mpnet-base-v2`
- [ ] **2.3** MRL-capable model (`nomic-embed-text-v1.5` or `Qwen3-Embedding-0.6B`)
- [ ] **2.4** Index in Qdrant (not only FAISS — the claim is *vector DB*)
- [ ] **2.5** **MRL truncation sweep** 768 -> 512 -> 256 -> 128 -> 64
  - [ ] Same sweep on non-MRL `all-mpnet-base-v2` as the control
  - [ ] ⚠️ Renormalise after slicing; assert unit norm in the harness
- [ ] **2.6** Hybrid: reciprocal rank fusion of BM25 + best dense; sweep the weight
- [ ] **2.7** Systems numbers: index build time, index size, p50/p95 latency, $/1M embeddings

### Phase 2b — Fine-tuning
- [ ] **2b.1** Fine-tune a sentence-transformer on the domain pairs with `MultipleNegativesRankingLoss`
- [ ] **2b.2** Evaluate on the **held-out** slice only
- [ ] **2b.3** **Question answered: should you train your own embedding model for RAG?**

### Phase 2c — Reranking
- [ ] **2c.1** Cross-encoder rerank over top-50 (`bge-reranker-base`)
- [ ] **2c.2** nDCG@10 before/after, and added p95 latency

### Phase 2d — Scaling curve
- [ ] **2d.1** Record index build time, size, p95 latency at 5K / 10K / 20K docs
- [ ] **2d.2** Fit and extrapolate to 1M and 100M; write up the arithmetic

## Phase 3 — Topic modelling as diagnostics

- [ ] **3a** Does embedding choice change what you discover? BERTopic across Phase 2 models; NPMI, c_v, topic count, diversity, outlier rate
- [ ] **3b** **Automatic failure taxonomy** — cluster failing queries, LLM-label the clusters
- [ ] **3c** **Corpus coverage gaps** — query-topic density vs document-topic density
- [ ] **3d** *(optional)* Temporal drift — topic distribution over arXiv publication dates

## Phase 4 — RAG layer *(optional, separate weekend)*

- [ ] **4.1** Retrieve top-k -> generate -> LLM-judge answer correctness and groundedness
- [ ] **4.2** Sweep k = 1, 3, 5, 10 across retrieval methods
- [ ] **4.3** **The experiment worth doing: does 256-dim truncation change *answer* quality, not just retrieval quality?**

## Phase 5 — Write-up

- [ ] **5.1** Charts: quality bar, latency-vs-quality scatter, MRL truncation curves, hybrid weight sweep
- [ ] **5.2** Fill every `[TBD]` in the README with measured numbers
- [ ] **5.3** "What surprised me" section
- [ ] **5.4** Limitations section

---

## Scale notes — read-only, not built here

At 10-20K documents none of this bites. Recorded because it is the interview territory, and because
it is why the MRL work matters.

| Corpus | Vectors @ 768d fp32 | + HNSW graph | Fits in RAM? |
|---|---|---|---|
| 1M | ~3 GB | ~4.5-6 GB | comfortably |
| 100M | **~300 GB** | **~450-600 GB** | **no** |

Levers at 100M, from ~300 GB: int8 -> ~75 GB · binary -> ~9.6 GB · MRL 768->256 -> ~100 GB ·
MRL 256 + int8 -> ~25 GB. **At small scale truncation is an optimisation; at 100M it is what makes
the system buildable.**

Other things that break: HNSW build time (hours to days, parallelises badly); re-embedding as a
migration project (100M x ~500 tokens = 50B tokens, ~$1,000 per full re-embed); HNSW deletes leaving
tombstones; ANN recall silently degrading as N grows at fixed `ef_search`; and **filtered search** —
pre-filtering disconnects the HNSW graph, post-filtering can return nothing.
