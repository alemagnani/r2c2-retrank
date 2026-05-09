# R2C2-Retrank: Implementation Status

> ⚠️ **Correction (post-PR submission):** Earlier sections of this document refer to
> a "≤200 character" passage length cap. That was based on a misread of the
> organizer-delivered `text_snippet` field, which is a 200-char preview.
> The official rules have **no formal passage length limit**; reference runs use
> ~2,400-char passages. We followed the artificial 200-char cap throughout PR
> development and our submitted runs reflect that. To be revisited for any future
> PR re-submission opportunity. AC pipeline does not have this constraint.

## What is implemented

### Corpus & Indexing
- **Passage extraction** — streams both `wikipedia_passages.tar.gz` and `wookieepedia_passages.tar.gz` directly without full extraction; yields 1,231,364 passages
- **BM25S index** (`scripts/build_bm25_index.py`) — sparse matrix BM25+ index over all passages with title prepending; saved to `data/processed/bm25s_index/`
- **FAISS dense index** (`scripts/build_dense_index.py`) — bi-encoder embeddings (`msmarco-MiniLM-L6-cos-v5`) for all 1.2M passages; flat inner-product index saved to `data/processed/faiss_index.bin`

### Retrieval Runs (submission-ready)
- **PO-1: BM25 baseline** (`run_bm25_retrieval.py`) — vanilla BM25 over organizer passages; nDCG@20 = 0.4243
- **PO-2: BM25 + cross-encoder reranker** (`run_reranker.py`) — top-100 BM25 candidates reranked by `ms-marco-MiniLM-L-12-v2`; nDCG@20 = 0.6727
- **PO-3: Dense retrieval** (`run_dense_retrieval.py`) — bi-encoder FAISS nearest-neighbor search; nDCG@20 = 0.4415
- **PO-4: Dense + cross-encoder reranker** — top-100 dense candidates reranked by cross-encoder; nDCG@20 = 0.6282

### Query Expansion Variants (exploratory)
- **HyDE** (`generate_hyde_queries.py` + `--hyde` flag) — LLM generates a hypothetical Wikipedia passage per topic; used as dense/BM25 query instead of raw question; **nDCG@20 = 0.6882 (best single system)**
- **PRF — Pseudo-Relevance Feedback** (`run_prf_retrieval.py`) — free alternative to HyDE; takes top-10 BM25 passages, extracts discriminative expansion terms by TF×IDF weighting, re-retrieves with expanded query; runs in ~2 seconds with no API cost; nDCG@20 = 0.6663 (after reranking)

### Evaluation
- **LLM-as-judge** (`evaluate_runs.py`) — pools passages from all runs, grades each (question, passage) pair 0/1/2 with a specified Claude model; saves per-model cache incrementally so evaluation is resumable and never repeats API calls; computes nDCG@20
- **Full nDCG@20 results (Haiku judge, all 8 runs scored, 6124 judgments)**:

| Run | nDCG@20 |
|-----|---------|
| PO-1 BM25 | 0.4243 |
| PO-2 BM25+reranker | 0.6727 |
| PO-3 Dense | 0.4415 |
| PO-4 Dense+reranker | 0.6282 |
| HyDE top20 | 0.5134 |
| HyDE+reranker | **0.6882** |
| PRF top20 | 0.3789 |
| PRF+reranker | 0.6663 |
| **AutoResearch final** | **0.7237** |

### AutoResearch Loop (COMPLETE — 10 iterations)
- **Failure diagnosis** (`autoresearch.py`) — scores all runs, finds weak topics (nDCG < threshold), asks Claude to classify failure type and propose a reformulated query; taxonomy: `implicit_entity`, `vocabulary_gap`, `multi_hop`, `counting`, `comparison`, `out_of_corpus`
- **10-iteration loop** (`autoresearch_loop.py`) — full orchestrator with:
  - State management: per-topic best passages ratcheted across iterations (`loop_state.json`)
  - Strategy rotation per failure type: `query_reform → wikidata_expand → prf_reform → multi_query → rrf_merge`
  - Wikidata SPARQL for free entity alias expansion (no API cost)
  - RRF fusion of all 8 runs (free, no API cost)
  - Inline LLM judging of new passages; saves incrementally to `qrels_loop.json`
  - Per-iteration run files: `iter_01_run.txt` … `iter_10_run.txt`
  - Final merged run: `data/autoresearch/final_run.txt`
- **Results**: BM25+reranker 0.6727 → **AutoResearch 0.7237** (+0.051), beating all single systems
- **9 of 12 weak topics improved**; biggest gain: topic 0054 (multi_hop) 0.263 → 0.855 (+0.593)
- **3 topics irreducibly failing** (all `out_of_corpus` or unsolvable `multi_hop` — answer not in Wikipedia/Wookieepedia)

### Failure taxonomy (12 diagnosed weak topics on PO-2 baseline):
  - `out_of_corpus`: 8 — answer not in Wikipedia/Wookieepedia (5 recovered by query_reform)
  - `multi_hop`: 2 — requires chaining two facts (1 recovered by multi_query)
  - `implicit_entity`: 2 — indirect references to named entities (partial improvements)

## What remains to be done

### Immediate (before April 17 deadline)
- [ ] **ZIP and submit** — package `alemagnani-PO-[1-4].txt` as ZIP, email to `ntcir19r2c2org@list.waseda.jp`
  - Consider submitting AutoResearch final run as one of the 4 PO runs
- [ ] **Sonnet judge** — run `evaluate_runs.py` with `claude-sonnet-4-6` and `--no-judge` to compute nDCG with Sonnet judgments for inter-annotator comparison

### Research / Paper
- [ ] **Per-failure-type analysis** — measure nDCG improvement by failure type across the 10 autoresearch iterations to validate taxonomy effectiveness
- [ ] **Inter-annotator agreement** — Haiku vs Sonnet judgment comparison using Cohen's kappa or similar
- [ ] **Listwise LLM reranking** — pass top-20 passages together to Claude for holistic ranking instead of pointwise cross-encoder scoring
- [ ] **Entity expansion index** — augment passage text with Wikidata aliases at index time (rather than at query time); would make entity expansion available to all retrieval strategies
