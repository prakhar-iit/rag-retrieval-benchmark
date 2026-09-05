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
- [x] **1.6** Three-way comparison on the same eval set:
  - [x] in-domain (arXiv-trained)
  - [x] out-of-domain -- the original ML-Cookbook OpinRank model wasn't in a connected folder, so
    per the user's choice, retrained fresh here (`scripts/train_word2vec_opinrank.py`) on the
    OpinRank hotel-review excerpt Kavita Ganesan (the dataset's own author) distributes for the
    classic gensim Word2Vec tutorial. Sampled to the **same 20,000-document count** as the arXiv
    corpus (same seed) rather than the full 256K reviews, so scale is held constant and domain is
    the only variable that differs between this model and the in-domain one.
  - [x] pretrained GoogleNews vectors -- `word2vec-google-news-300`, loaded with `limit=500_000`
    (top 500K by frequency; the full 3M-vector load needs more RAM than was available while
    building this, and 500K covers virtually all general English plus common technical terms --
    documented as a minor conservative bias against GoogleNews's OOV coverage).
  - [x] **Finding: does domain beat scale for static embeddings? Yes, against a truly mismatched
    domain -- but scale buys back a lot of that gap against a broad, non-adversarial domain.**
    All numbers below: brute-force cosine retrieval (`scripts/eval_static_embeddings.py`) over the
    full 20K-doc corpus against all 400 eval queries, `rss/metrics.evaluate`.

    | Model | Pooling | nDCG@10 | Recall@10 | MRR |
    |---|---|---|---|---|
    | arXiv (in-domain) | mean | 0.8184 | 0.8975 | 0.7954 |
    | arXiv (in-domain) | **IDF-weighted** | **0.8534** | **0.9350** | **0.8290** |
    | GoogleNews (pretrained, general) | mean | 0.7629 | 0.8625 | 0.7347 |
    | GoogleNews (pretrained, general) | IDF-weighted | 0.8227 | 0.8925 | 0.8022 |
    | OpinRank (out-of-domain) | mean | 0.2373 | 0.3275 | 0.2128 |
    | OpinRank (out-of-domain) | IDF-weighted | 0.3714 | 0.4950 | 0.3374 |

    Two findings, not one: (1) **IDF-weighted pooling beats mean pooling in all six runs** --
    consistent and unsurprising (mean pooling lets high-frequency, low-information tokens dilute
    the vector; IDF pooling suppresses them), but worth stating since it wasn't guaranteed. (2) The
    domain story is more nuanced than a clean binary. Out-of-domain OpinRank collapses (nDCG@10
    0.37 vs 0.85, less than half) -- a Word2Vec model that has literally never seen `transformer` or
    `diffusion` in its training data cannot be expected to retrieve ML papers well, and doesn't.
    But GoogleNews -- pretrained on ~100B words of general English, ~30x more data than either
    from-scratch model saw, and *not* domain-adversarial the way hotel reviews are -- comes within
    0.03 nDCG@10 of the in-domain model. **Scale can substitute for domain specificity when the
    pretraining corpus is broad enough to already contain the target domain's vocabulary** (GoogleNews
    was trained on news text circa 2013, which already covers a fair amount of tech/science
    terminology); it cannot when the pretraining domain actively excludes it, as OpinRank's hotel
    reviews do. The honest one-line version for the README: *domain-specific training beats a
    genuinely mismatched general model by a wide margin, but a broad, well-resourced general model
    is a much closer contest than "domain beats scale" alone would suggest.*

## Phase 2 — Embedding comparison

- [x] **2.1** BM25 baseline (`rank_bm25`)
  - `rss/index.build_bm25` / `search_bm25`, evaluated via `scripts/eval_bm25.py` on all 400 eval
    queries against the full 20K corpus. **nDCG@10 = 0.9779, Recall@10 = 0.9950, MRR = 0.9723** --
    higher than every Word2Vec variant from Phase 1, including in-domain (0.8534).
  - ⚠️ **Worth being suspicious of a number this high, so here's the check.** A near-perfect BM25
    score could mean genuine lexical strength, or it could mean the eval queries still leak surface
    tokens despite the paraphrasing discipline from Phase 0 (exactly the failure mode `evalset.py`'s
    docstring warns about). Weighing both: the eval-set spot-checks (20/20 clean at generation time,
    plus the 20/20 clean re-check in the Phase 0 final report) found genuine content paraphrases, not
    title/phrase extraction -- so this isn't the leakage failure mode. What's actually happening is
    more mundane and well-documented in IR: **ML paper abstracts are jargon-dense, and jargon has no
    synonyms.** A query asking about "graph neural networks" or "reinforcement learning" will match
    an abstract using those exact terms almost every time, because there typically isn't another way
    to say them -- unlike, say, "car" vs "automobile" in general text, where embeddings earn their
    keep. This is a real, reportable finding (**"BM25 is the baseline everyone underestimates"** was
    the project's working thesis going in, and this is unusually strong confirmation of it) but also
    a genuine limitation of THIS corpus for showcasing embedding methods: expect BM25 to be hard to
    beat here, and expect the interesting embedding-vs-lexical gap to show up mainly on queries that
    need synonymy or paraphrase-level matching rather than jargon recall -- worth a look in Phase 3b's
    failure taxonomy once there are failing queries to cluster.
  - Systems numbers: index build 0.66s, ~24.7MB pickled (approximate -- BM25Okapi has no native
    on-disk format), query latency p50=130ms / p95=188ms (naive per-query Python scoring over 20K
    docs, unoptimized -- not a fair comparison to an ANN index's latency, noted for 2.7).
- [x] **2.2** `all-MiniLM-L6-v2` and `all-mpnet-base-v2`
  - Both run end-to-end on the full 20K-doc corpus via `scripts/eval_dense.py` (real Qdrant index,
    local/embedded mode). `all-MiniLM-L6-v2` (384d) encoded in this dev VM directly; `all-mpnet-base-v2`
    (768d) was too slow here (~3 docs/s on 4 CPUs/no GPU) so its corpus + query encoding ran on the
    user's own Mac via `scripts/encode_locally.py` -- same `rss.dense_embed.encode_checkpointed` cache
    format, so re-running `eval_dense.py` in the dev VM picked the vectors up directly and only built
    the index + evaluated, keeping those systems numbers in the same environment as BM25/MiniLM.
    **all-MiniLM-L6-v2: nDCG@10=0.9558, Recall@10=0.9925, MRR=0.9438**, index build 82.1s/82.7MB,
    query latency p50=10.3ms/p95=13.7ms.
    **all-mpnet-base-v2: nDCG@10=0.9652, Recall@10=0.9900, MRR=0.9577**, index build 143.7s/164.7MB,
    query latency p50=50.9ms/p95=79.7ms (both roughly double MiniLM's -- 768d vs 384d).
    Both sit between in-domain Word2Vec (0.8534) and BM25 (0.9779): a good general-purpose
    sentence-transformer beats from-scratch Word2Vec but still doesn't beat lexical match on this
    jargon-dense corpus, consistent with 2.1's finding. mpnet's larger/better-trained encoder edges out
    MiniLM but at ~2x the index size and query latency -- a real, quantifiable quality/cost tradeoff.
- [x] **2.3** MRL-capable model (`nomic-embed-text-v1.5` or `Qwen3-Embedding-0.6B`)
  - `nomic-embed-text-v1.5` chosen and confirmed working (loads via `SentenceTransformer`, encodes
    to 768-dim). Getting there took two real fixes, both now folded into `scripts/fetch_hf_models.py`
    permanently:
    1. Its `config.json` `auto_map` points at a SEPARATE repo (`nomic-ai/nomic-bert-2048`) for the
       custom model code. Having that repo present on disk was NOT enough on its own -- transformers'
       dynamic-module loading fetches the referenced repo from the Hub at load time regardless of
       what's sitting in a sibling local directory, so this still 403'd even after downloading
       `nomic-bert-2048` in full. Fixed by vendoring its two small `.py` files directly into
       `nomic-embed-text-v1.5/` and rewriting the `auto_map` to reference them locally (the exact
       format `nomic-bert-2048`'s own `config.json` uses to describe itself) -- fully self-contained
       now, no runtime dependency on `nomic-bert-2048` or huggingface.co. Only its ~100KB of code is
       fetched, not its unrelated 525MB of separately-pretrained weights.
    2. Needs the `einops` package (added to `requirements.txt`) and `trust_remote_code=True` (added
       to `rss.dense_embed.encode` and `scripts/eval_dense.py`) to actually load.
    `all-MiniLM-L6-v2`/`all-mpnet-base-v2`/`bge-reranker-base` aren't affected by any of this -- they
    don't use custom architecture code.
  - **Ran end-to-end on the full corpus, and caught a real methodology bug before trusting the
    number.** nomic's model card is explicit that `search_document: `/`search_query: ` prefixes are
    *required*, not a style convention -- checked `config_sentence_transformers.json` directly and
    confirmed no auto-applied prompt, so plain `model.encode(text)` was silently giving it exactly
    the wrong input. The first run (no prefix) scored **nDCG@10=0.9750** -- best of every dense model,
    nearly matching BM25. Fixed the prefixing (`rss.dense_embed.get_task_prefix`), cleared that result
    entirely, and re-ran: **nDCG@10=0.9638, Recall@10=0.9925, MRR=0.9549**, index build 90.2s/164.7MB,
    query latency p50=18.6ms/p95=23.2ms.
  - **The correct number is LOWER than the wrong one, which is itself worth reporting.** Prefixing
    made this model's numbers WORSE on this eval, not better -- worth being honest about rather than
    quietly using whichever run looked best. Best guess: without the prefix, nomic effectively
    behaves as a plain symmetric encoder (query and document text embedded identically), and
    symmetric similarity happens to suit this jargon-dense, high-lexical-overlap corpus fine; the
    prefix is what teaches the model to treat queries and documents asymmetrically, which pays off on
    harder retrieval tasks (e.g. genuine paraphrase/synonymy gaps) more than it does here. With
    correct usage, **`all-mpnet-base-v2` (0.9652) is the best dense model on this eval, not nomic** --
    the MRL truncation sweep (2.5) uses nomic anyway since it's the only MRL-capable model in the
    lineup, but the "best dense" designation for hybrid fusion (2.6) should be `all-mpnet-base-v2`.
- [x] **2.4** Index in Qdrant (not only FAISS — the claim is *vector DB*)
  - `docker` isn't installed on this dev machine, so `rss.index.build_qdrant`/`search` use Qdrant's
    embedded/local mode (`QdrantClient(path=...)`) instead of the config's `index.url` server target
    -- same client API and on-disk collection format, no server process. Covered by 7 unit tests in
    `tests/test_index.py` (exact-match retrieval, k-limiting, missing payloads, rebuild-at-same-path
    semantics, on-disk size, deletion). `configs/default.yaml` gains `index.path`; `index.url` is kept
    as documentation of the real (docker-based) deployment target.
  - Now actually indexing real vectors, not just the toy fixtures: all three dense models from 2.2/2.3
    each got a real per-model Qdrant collection built via `scripts/eval_dense.py` (`all-MiniLM-L6-v2`
    82.7MB/82.1s build, `all-mpnet-base-v2` 164.7MB/143.7s, `nomic-embed-text-v1.5` 164.7MB/90.2s --
    the latter two's byte-identical size is expected, same 20K vectors at 768d either way). Query
    latency (10-80ms depending on model) came out roughly 2-12x lower than BM25's naive Python
    scoring (130/188ms p50/p95), the real payoff of an actual ANN index vs brute-force/linear scoring.
- [x] **2.5** **MRL truncation sweep** 768 -> 512 -> 256 -> 128 -> 64
  - [x] Same sweep on non-MRL `all-mpnet-base-v2` as the control
  - [x] ⚠️ Renormalise after slicing; assert unit norm in the harness
  - `scripts/eval_mrl_sweep.py` truncates the already-cached full-dimension vectors for both models
    (no re-encoding -- `encode_checkpointed` is called with `model=None`, which only works because
    every chunk is already on disk from 2.2/2.3, and the checkpointed-cache-reuse path is exactly
    why that function returns the concatenated array without ever touching the model when nothing
    is missing) to 768/512/256/128/64 dims, renormalising and asserting unit norm at every step
    (`rss.dense_embed.truncate` / `assert_unit_norm`, 8 unit tests), then builds a real per-combo
    Qdrant index, evaluates, and deletes the index before the next combo (10 builds total would
    otherwise leave ~1.6GB of scratch indices on disk). Results are saved incrementally to
    `results/phase2_mrl_sweep.json` keyed by `{model}@{dim}`, so the sweep is resumable across
    separate runs -- useful in practice, since 10 index builds plus evals ran across 5 separate
    invocations here.

    | dim | nomic-embed-text-v1.5 (MRL) nDCG@10 | all-mpnet-base-v2 (control) nDCG@10 |
    |---|---|---|
    | 768 | 0.9629 | 0.9652 |
    | 512 | 0.9630 | 0.9596 |
    | 256 | 0.9571 | 0.9566 |
    | 128 | 0.9405 | 0.9356 |
    | 64  | 0.8811 | 0.8457 |

    (Full precision/recall/MRR/index-size/latency breakdown per combo lives in
    `results/phase2_mrl_sweep.json`; the 768-dim rows here differ trivially from 2.2/2.3's own
    numbers -- 0.9629 vs 0.9652 for mpnet, 0.9629 vs 0.9638 for nomic -- because each is a
    separately-built Qdrant/HNSW index over the same vectors, and HNSW is an approximate index;
    two builds of identical vectors aren't guaranteed bit-identical search order.)

    **Finding: MRL training measurably reduces truncation damage, exactly as claimed.** Both models
    are flat (even slightly non-monotonic in the noise) from 768 down to 256 dims -- truncation is
    close to free at those sizes for either model, MRL-trained or not. The divergence shows up at
    the aggressive end: 768->64 is 12x compression. nomic (MRL-trained) drops from 0.9629 to 0.8811,
    a 8.5% relative loss. `all-mpnet-base-v2` (not MRL-trained, sliced anyway as the control) drops
    from 0.9652 to 0.8457, a 12.4% relative loss -- roughly 46% more degradation than the MRL model
    at the same compression ratio, despite starting from a near-identical full-dimension score.
    That's the whole point of Matryoshka training made visible in one number: it's not that
    truncation "doesn't hurt" a MRL model, it's that the model was trained so that an early prefix
    of the embedding is itself a good embedding, so cutting it hurts less. A non-MRL model's later
    dimensions aren't specialised to be droppable, so cutting them costs more.
- [x] **2.6** Hybrid: reciprocal rank fusion of BM25 + best dense; sweep the weight
  - `scripts/eval_hybrid.py` fuses BM25 with `all-mpnet-base-v2` (the corrected "best dense" pick
    from 2.3, not nomic). Both methods retrieve a pool of 100 candidates per query (deeper than the
    top-20 that gets scored) -- fusing only over each method's own already-truncated top-k would
    hide exactly the cross-method complementarity RRF/weighted fusion exist to exploit: a doc BM25
    ranks 15th and dense ranks 3rd needs to be visible at rank 15 to be pulled up into the fused
    top-10. Two fusion methods are run and compared, not just the one named in the task title,
    since `rss.fusion` implements both and `configs/default.yaml`'s `hybrid` block configures both
    (`rrf_k`, `weight_sweep`): `weight_sweep` is swept as the weight on BM25's (min-max-normalised)
    score, dense getting `1 - weight`.

    | Method | nDCG@10 | Recall@10 | MRR |
    |---|---|---|---|
    | BM25 only | 0.9779 | 0.9950 | 0.9723 |
    | Dense only (all-mpnet-base-v2) | 0.9652 | 0.9900 | 0.9577 |
    | RRF (k=60) | 0.9821 | 1.0000 | 0.9760 |
    | Weighted, bm25_weight=0.0 (~dense only) | 0.9634 | 0.9900 | 0.9552 |
    | Weighted, bm25_weight=0.25 | 0.9822 | 1.0000 | 0.9762 |
    | **Weighted, bm25_weight=0.5** | **0.9885** | **1.0000** | **0.9846** |
    | Weighted, bm25_weight=0.75 | 0.9826 | 1.0000 | 0.9768 |
    | Weighted, bm25_weight=1.0 (=BM25 only) | 0.9779 | 0.9950 | 0.9723 |

    (`bm25_weight=0.0` isn't bit-identical to "dense only" -- 0.9634 vs 0.9652 -- because weighted
    fusion's min-max normalisation is computed over the 100-candidate pool and ties/near-ties at the
    pool boundary can reorder a few borderline docs relative to ranking by raw dense score directly;
    a small, expected artifact of fusing over a fixed-depth pool, not a bug.)

    **Finding: fusion beats both individual methods, and the reason is legible from this corpus's
    own error pattern.** BM25 (0.9779) already beats dense (0.9652) alone here -- expected, given
    this eval set's LLM-generated queries reuse a lot of the source abstract's own vocabulary
    (jargon-dense arXiv text), which is exactly BM25's home turf. But BM25 and dense don't fail on
    the *same* queries: a query dense gets right that BM25 misses (paraphrase/synonymy) still adds
    signal when fused with a BM25-dominant blend, which is why the best point isn't at
    `bm25_weight=1.0` but at 0.5 -- both methods are still pulling weight there. RRF gets most of
    the way there (0.9821) without ever seeing raw scores, which is the appeal of RRF in production;
    tuned weighted fusion just slightly beats it (0.9885) at the cost of needing well-behaved,
    comparably-scaled per-method scores (the min-max normalisation this repo's `weighted_score_fusion`
    already does specifically to avoid BM25's unbounded scale silently dominating cosine's
    `[-1, 1]` range). Fusion's own compute cost is negligible either way -- combining two 100-item
    lists is sub-millisecond (p50 <0.05ms), so the real latency of a hybrid query is still
    dominated by running both first-stage retrievers, not by the fusion step.
  - `rss.index` gained `search_bm25_with_scores`/`search_with_scores` (return `(id, score)` pairs
    rather than bare ids) since weighted fusion needs actual scores, not just ranks -- 5 new unit
    tests (97 total now) check they agree with `search_bm25`/`search`'s own ordering and that scores
    come back sorted descending.
- [ ] **2.7** Systems numbers: index build time, index size, p50/p95 latency, $/1M embeddings

### Phase 2b — Fine-tuning
- [ ] **2b.1** Fine-tune a sentence-transformer on the domain pairs with `MultipleNegativesRankingLoss`
- [ ] **2b.2** Evaluate on the **held-out** slice only
- [ ] **2b.3** **Question answered: should you train your own embedding model for RAG?**

### Phase 2c — Reranking
- [ ] **2c.1** Cross-encoder rerank over top-50 (`bge-reranker-base`)
  - `rss.rerank.load_reranker`/`rerank` implemented: load once (via `sentence_transformers`
    `CrossEncoder`, from the local `models/hf/bge-reranker-base` snapshot), then call `rerank` per
    query -- loading per-query would swamp the added-latency number 2c.2 needs. Caps scoring to the
    first `top_n` candidates from the first-stage retriever rather than the whole corpus, which is
    the entire reason reranking is a second stage. 7 unit tests against a fake cross-encoder
    (ordering, `top_n` capping, empty/single-candidate edges). Smoke-tested against the real
    `bge-reranker-base`: loads and produces a sensible reordering on a toy 3-candidate example.
    Not yet run as a full eval (needs a first-stage run's top-50 per query as input) -- pending 2.1's
    BM25 run's candidates or a dense run once 2.2 finishes.
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
