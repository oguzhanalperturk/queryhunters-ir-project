# Query Hunters — Ad-hoc Retrieval System

BM25 baseline, RM3 query expansion, and neural cross-encoder re-ranking for
ad-hoc passage retrieval on the **MS MARCO passage** collection, built with
[PyTerrier](https://pyterrier.readthedocs.io/). The project compares four
configurations — BM25, BM25 + RM3, BM25 + Reranker, and BM25 + RM3 + Reranker —
and ships a Streamlit demo for interactive inspection.

CENG 596 Information Retrieval, Spring 2026.

---

## 1. Project structure

```
queryhunters-ir-project/
├── config.yaml                 # All settings (dataset, retrieval, reranking, demo)
├── find_demo_queries.py        # Helper: finds the best queries for the live demo
├── requirements.txt            # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset / query / qrels loading + query cleaning
│   ├── indexer.py              # Builds the Terrier inverted index (full corpus)
│   ├── retriever.py            # BM25 and BM25 + RM3 pipelines
│   └── reranker.py             # Cross-encoder re-ranker
├── experiments/
│   ├── run_bm25_baseline.py    # BM25 only
│   ├── run_bm25_rm3.py         # BM25 vs BM25 + RM3
│   ├── run_reranker.py         # BM25 vs BM25 + Reranker
│   └── run_full_pipeline.py    # All four systems + saves CSV results
├── ui/
│   └── app.py                  # Streamlit demo
├── indexes/                    # Created on first run (the Terrier index)
└── results/                    # Created on first run (CSV outputs)
```

---

## 2. Requirements

- **Python 3.10+** (developed and tested on Python 3.12)
- **Java 11+** — required by PyTerrier/Terrier. Make sure a JDK is installed
  and `JAVA_HOME` is set; PyTerrier starts the JVM automatically.
- **Disk:** ~15–20 GB free. The MS MARCO passage corpus (~3 GB download) plus
  the Terrier index (a few GB) live under `indexes/`.
- **RAM:** 8 GB minimum. Indexing the full corpus is the most memory-intensive
  step.
- **Internet** on first run: `ir_datasets` downloads the corpus, and the
  cross-encoder model (~80 MB) is fetched from Hugging Face.

A CUDA GPU is **not** required; the cross-encoder runs on CPU (slower, but
fully supported).

---

## 3. Installation

```bash
# from the repository root
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

If you do not have a `requirements.txt`, install the core packages directly:

```bash
pip install python-terrier sentence-transformers ir_datasets pandas pyyaml streamlit
```

---

## 4. Configuration

All settings live in `config.yaml`. The defaults reproduce the results in the
report; the most relevant fields:

| Section | Key | Meaning |
|---|---|---|
| `dataset` | `name` | `msmarco-passage/dev/small` — 6 980 dev queries, all judged. `docs_iter()` still yields the **full ~8.8 M passage corpus**, so the whole collection is indexed. |
| `dataset` | `index_path` | Where the Terrier index is stored (`indexes/msmarco_passage`). |
| `retrieval` | `bm25_num_results` | Size of the BM25/RM3 candidate pool (1000). |
| `query_expansion` | `fb_docs`, `fb_terms`, `original_query_weight` | RM3 feedback documents, expansion terms, and the relevance-model weight (`fb_lambda`). |
| `reranking` | `model_name`, `batch_size`, `rerank_top_k` | Cross-encoder model, batch size, and how many top candidates are re-ranked (100). |
| `evaluation` | `num_queries`, `seed` | How many dev queries to evaluate on (1000), sampled reproducibly. |
| `demo` | `pinned_query_ids` | Query IDs pinned to the top of the demo's selector. |

---

## 5. Running the experiments

> **Run every command from the repository root.** The scripts read
> `config.yaml` from the working directory and import from `src/`.

### First run builds the index (one-time, slow)

The first script you run builds the Terrier index over the full corpus. This
downloads the corpus and indexes ~8.8 M passages — it can take roughly **1–3
hours** and is a one-time cost. Every later run detects the existing index and
reuses it automatically.

### BM25 baseline

```bash
python -m experiments.run_bm25_baseline
```

Builds (or reuses) the index, runs BM25, and prints `RR@10`, `nDCG@10`,
`R@100`. As a correctness check, BM25 should score **RR@10 ≈ 0.17** on this
collection, which is in line with the published literature.

### BM25 + RM3

```bash
python -m experiments.run_bm25_rm3
```

Compares BM25 and BM25 + RM3, including a significance test of RM3 against the
BM25 baseline.

### BM25 + Reranker

```bash
python -m experiments.run_reranker
```

Re-ranks the top-100 BM25 candidates with the cross-encoder and compares
against BM25.

### Full pipeline (all four systems)

```bash
python -m experiments.run_full_pipeline
```

Evaluates BM25, BM25 + RM3, BM25 + Reranker, and BM25 + RM3 + Reranker in one
run, with significance tests, and writes all result CSVs to `results/`. On a
CPU the two re-ranking passes take roughly **1–2 hours** combined; a progress
bar is shown.

---

## 6. Running the demo

The Streamlit demo lets you submit a query and inspect the ranked passages for
each pipeline, with relevance labels and rank-change indicators.

```bash
streamlit run ui/app.py
```

It opens in your browser (usually `http://localhost:8501`). The demo loads the
index built by the experiments, so **run an experiment at least once first** so
that `indexes/msmarco_passage` exists.

**Two query modes** (sidebar):

- **Free Text Query** — type any query. No relevance labels (these queries have
  no qrels).
- **Dev Query with Relevance Labels** — pick a judged dev query. Relevant
  passages are highlighted and Reciprocal Rank@10 is shown. 
  Recommended demo queries configured in `config.yaml`.

**Pipelines** (sidebar): BM25, BM25 + RM3, BM25 + Reranker, BM25 + RM3 +
Reranker, or *Compare All* (side-by-side).

> If you see a long list of `torchvision`-related messages in the console, they
> are harmless: Streamlit's file watcher inspects unrelated `transformers`
> modules. To silence them, run with
> `streamlit run ui/app.py --server.fileWatcherType none`.

### Choosing demo queries

`config.yaml → demo.pinned_query_ids` controls which queries appear (and are
starred) at the top of the dev-query selector. To find strong candidates —
queries where re-ranking lifts the relevant passage from a low BM25 rank to the
top — run, after `run_full_pipeline.py`:

```bash
python find_demo_queries.py
```

It reads the CSVs in `results/`, prints a ranked table of demo candidates, and
saves the full analysis to `results/demo_query_analysis.csv`. Copy the query
IDs you like into `pinned_query_ids` (keep them quoted, e.g. `"299023"`), then
restart Streamlit.

---

## 7. Pipeline overview

1. **BM25** retrieves an initial pool of up to 1000 candidate passages from the
   Terrier inverted index.
2. **RM3** (optional) expands the query via pseudo-relevance feedback and
   re-runs BM25 over the expanded query.
3. **Cross-encoder re-ranking** (optional) re-scores the top-100 candidates by
   feeding each query–passage pair through a neural cross-encoder.
4. Results are evaluated with **RR@10, nDCG@10, and R@100**.

---

## 8. Notes

- The index is built **once** and reused; delete `indexes/msmarco_passage` to
  force a rebuild (e.g. after changing indexing settings).
- Evaluation is averaged over a fixed, reproducible sample of `num_queries`
  dev queries (seed in `config.yaml`). Query sampling does not make retrieval
  easier — the full corpus is always indexed, so every query competes against
  all ~8.8 M passages.
- Raw query text is cleaned before retrieval (special characters that Terrier's
  parser treats as operators are stripped), which prevents query-parsing errors.
