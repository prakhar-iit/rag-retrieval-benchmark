# rag-retrieval-benchmark

Benchmarking retrieval approaches from BM25 and Word2Vec to modern embeddings, Matryoshka
truncation and hybrid fusion — with topic-level failure diagnostics, on arXiv ML abstracts.

> **Status: in progress.** Scaffold and roadmap are in place; results are not in yet.
> See [TASKS.md](TASKS.md) for what is done and what is next. Numbers below appear as
> `[TBD]` until measured — nothing here is estimated.

## Why

Every team building RAG makes four decisions — **which embedding model, what dimensionality,
hybrid or not, and what `k`** — and most make them by reading a leaderboard. MTEB rank is a poor
selection signal: it averages 50+ heterogeneous datasets, and domain shift usually dominates.

This repo makes those four decisions on one corpus with evidence, and then builds the tooling that
explains where the resulting system fails.

## Two halves

**Measurement** — how good is retrieval, across three decades of approaches?
**Diagnosis** — where and why does it fail?

Most public work does one or the other.

## The stack under test

| Era | Method | Objective it was trained for |
|---|---|---|
| 1994 | BM25 | none — term statistics |
| 2013 | Word2Vec, trained in-domain here | word co-occurrence |
| 2019 | sentence-transformers | contrastive sentence similarity |
| 2024/25 | MRL-capable embeddings | contrastive + front-loaded dimensions |
| — | Hybrid RRF (lexical + dense) | fusion of uncorrelated failure modes |
| — | Cross-encoder reranking | joint query-document scoring |

## Results

All rows use IDF-weighted pooling where pooling applies (see [TASKS.md](TASKS.md) for the
mean-pooling comparison). Word2Vec index size/latency are [TBD] because Phase 1 evaluates with
brute-force cosine similarity over the full corpus, not a real index -- that's Phase 2 (2.4). BM25's
index size/latency are approximate for a different reason: `rank_bm25` has no on-disk index format or
optimized query path, so its numbers below are a pickled-object size and naive per-query Python
scoring latency, not a fair comparison to a real ANN index -- treat them as a floor, not a target.

| Method | nDCG@10 | Recall@10 | MRR | Index size | p95 latency |
|---|---|---|---|---|---|
| BM25 | 0.9779 | 0.9950 | 0.9723 | ~24.7MB* | 188ms* |
| Word2Vec (arXiv, in-domain) | 0.8534 | 0.9350 | 0.8290 | [TBD] | [TBD] |
| Word2Vec (GoogleNews, pretrained) | 0.8227 | 0.8925 | 0.8022 | [TBD] | [TBD] |
| Word2Vec (OpinRank, out-of-domain) | 0.3714 | 0.4950 | 0.3374 | [TBD] | [TBD] |
| sentence-transformers (all-MiniLM-L6-v2, 384d) | 0.9558 | 0.9925 | 0.9438 | 82.7MB | 11.6ms |
| sentence-transformers (all-mpnet-base-v2, 768d) | 0.9652 | 0.9900 | 0.9577 | 164.7MB | 79.7ms |
| MRL model @ 768 (nomic-embed-text-v1.5) | 0.9638 | 0.9925 | 0.9549 | 164.7MB | 23.2ms |
| MRL model @ 256 (nomic-embed-text-v1.5, truncated) | 0.9571 | 0.9875 | 0.9474 | 82.7MB | 9.7ms |
| MRL model @ 64 (nomic-embed-text-v1.5, truncated) | 0.8811 | 0.9500 | 0.8600 | 21.2MB | 4.2ms |
| all-mpnet-base-v2 @ 256 (non-MRL control, truncated) | 0.9566 | 0.9925 | 0.9453 | 82.7MB | 8.7ms |
| all-mpnet-base-v2 @ 64 (non-MRL control, truncated) | 0.8457 | 0.9425 | 0.8156 | 21.2MB | 3.1ms |
| Hybrid RRF (BM25 + all-mpnet-base-v2) | 0.9821 | 1.0000 | 0.9760 | ~189MB† | <1ms‡ |
| **Hybrid weighted, bm25_weight=0.5 (best overall)** | **0.9885** | **1.0000** | **0.9846** | ~189MB† | <1ms‡ |
| + cross-encoder rerank (bge-reranker-base) | 0.9815 | 0.9975 | 0.9760 | (reuses hybrid's indices) | +3038ms† |

† Both first-stage indices (BM25 + the dense Qdrant index) have to exist regardless of fusion
method -- not an extra index, the sum of the BM25 and all-mpnet-base-v2 rows above.
‡ Fusion compute itself (combining two 100-candidate lists) is sub-millisecond; total hybrid query
latency is dominated by running both first-stage retrievers (see their own rows above), not
separately re-measured end-to-end here -- see [TASKS.md](TASKS.md) (2.6).
† Cross-encoder rerank is scored on top of the hybrid weighted-fusion ranking (bm25_weight=0.5,
the best first-stage result from 2.6), not a fresh index -- the added latency is the rerank pass
itself (p50 3038ms, scoring 50 real abstract-length candidates per query), on top of whatever the
first-stage retrievers already cost.

Full 5-point sweep (768/512/256/128/64) for both models is in [TASKS.md](TASKS.md) (2.5) --
the table above shows only the endpoints plus the 256-dim midpoint the config calls out as the
"good enough" compression target, since the point that matters (MRL beats the non-MRL control by
a growing margin as dims shrink) is already visible from four rows without reproducing all ten.

## Method notes that matter

**The evaluation set is LLM-generated, not phrase-extracted.** Extracting phrases from documents to
use as queries leaks surface tokens and hands the comparison to BM25 by construction. Queries here
are natural research questions generated from a sampled abstract; that abstract is the gold document.

**Matryoshka truncation is measured against a non-MRL control, and the divergence shows up late.**
Both models are essentially flat from 768 down to 256 dims -- truncation is close to free there
either way. At 12x compression (768->64) the MRL model (nomic-embed-text-v1.5) loses 8.5% relative
nDCG@10 vs. the non-MRL control's (`all-mpnet-base-v2`, sliced anyway) 12.4% -- confirming MRL is a
training objective that makes early dimensions specifically droppable, not a compression trick that
works on any embedding. Vectors are renormalised after slicing, since cosine similarity assumes
unit norm; full sweep in [TASKS.md](TASKS.md).

**Hybrid fusion beats both of its inputs because BM25 and dense fail on different queries, not
because either is weak.** BM25 alone (0.9779) already beats dense alone (0.9652) on this
jargon-dense, high-lexical-overlap corpus -- BM25's home turf -- so the naive expectation is that
fusion just interpolates between the two, capping out at BM25's score. It doesn't: the best
weighted blend (bm25_weight=0.5) reaches 0.9885, above either input, because dense's occasional
paraphrase/synonymy win still adds signal on top of a BM25-dominant blend. RRF gets most of the
same gain (0.9821) without needing calibrated scores. Full method comparison, including why
RRF's rank-only view slightly undershoots tuned weighted fusion, in [TASKS.md](TASKS.md) (2.6).

**Reranking made this eval WORSE, not better -- and that's a real finding, not a bug.** Reranking
the best hybrid ranking (0.9885 nDCG@10) with `bge-reranker-base` over its top-50 candidates drops
to 0.9815. The baseline is already near-ceiling (recall@10=1.0), so a reranker has more room to
demote an already-correct top result than to improve on it, and a general-purpose cross-encoder
isn't specially tuned to this jargon-dense corpus the way the fusion weight sweep (2.6) is. Caught
via the same discipline as the nomic prefix bug (2.3): a first run of this eval came back
suspicious (bit-identical to plain BM25, not a real hybrid number) because of a stale empty index
being silently reused -- fixed and rerun before trusting the number. Full story in
[TASKS.md](TASKS.md) (2c.2).

**Systems numbers are reported alongside quality.** Index build time, index size, p50/p95 query
latency and cost per 1M embeddings. A model that is 2% better and 5x slower is usually the wrong
production choice, and most comparisons never say so.

## Layout

```
src/rss/          corpus, eval set, embedders, index, metrics, fusion, rerank, topics
configs/          experiment configuration
scripts/          entry points, one per phase
results/          measured outputs (gitignored except .gitkeep)
notebooks/        exploration
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker run -p 6333:6333 qdrant/qdrant     # vector DB for Phase 2
```

## Limitations

Single corpus, single domain. Synthetic evaluation set with LLM-generated queries, spot-checked but
not human-labelled throughout. Corpus is sampled to 10-20K documents for the retrieval sweep, so
none of the failure modes that appear at 1M+ documents are exercised here — see the scaling notes
in [TASKS.md](TASKS.md).

## License

MIT
