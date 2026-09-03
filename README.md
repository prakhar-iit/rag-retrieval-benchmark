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

| Method | nDCG@10 | Recall@10 | MRR | Index size | p95 latency |
|---|---|---|---|---|---|
| BM25 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Word2Vec (arXiv, in-domain) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Word2Vec (out-of-domain) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
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
