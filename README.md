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
| sentence-transformers | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| MRL model @ 768 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| MRL model @ 256 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Hybrid RRF | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| + cross-encoder rerank | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

## Method notes that matter

**The evaluation set is LLM-generated, not phrase-extracted.** Extracting phrases from documents to
use as queries leaks surface tokens and hands the comparison to BM25 by construction. Queries here
are natural research questions generated from a sampled abstract; that abstract is the gold document.

**Matryoshka truncation is measured against a non-MRL control.** Truncating an MRL-trained model
should barely move nDCG; truncating a model that was not trained that way should degrade badly.
Running both is the point — it shows MRL is a training objective, not a compression trick.
Vectors are renormalised after slicing, since cosine similarity assumes unit norm.

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
