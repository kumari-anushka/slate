# Slate

> **Rank the right candidates, not the loudest keywords.** A two-stage learned candidate–job matching system: bi-encoder retrieval + a fine-tuned cross-encoder reranker, measured against real baselines.

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](#)
[![Live demo](https://img.shields.io/badge/demo-live-blue)](#)

**Live demo:** _add URL (Hugging Face Space)_ · **Writeup:** _add link_

---

## The problem

Matching candidates to jobs is a ranking problem, but most tooling does it badly. Boolean/keyword search over-rewards keyword stuffing and misses semantically-relevant candidates; naive embedding cosine similarity retrieves "topically near" candidates but ranks *fit* poorly because it was never trained on what a good match looks like.

Slate learns to rank. A fast retriever narrows the pool, then a **fine-tuned cross-encoder reranker** — trained on labeled (job, candidate, relevance) data with hard negatives — orders the shortlist by genuine fit.

## Results

Held-out test set, fine-tuned reranker vs. baselines:

| Method | NDCG@10 | MRR | Recall@50 | MAP |
|---|---|---|---|---|
| BM25 (keyword) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Embedding cosine (no fine-tune) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| **Slate (fine-tuned reranker)** | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

**Ablation** (what drives the gain): _e.g. with vs. without hard negatives — tbd._

> _Fill in after `ml/evaluation/run_eval.py`. This table is the project — keep it at the top._

## How it works

```
                     Offline (Colab/Kaggle GPU)
  Public datasets ──▶ label (hard negatives) ──▶ fine-tune cross-encoder ──▶ artifact
                                                                               │
                     Serving (CPU)                                             ▼
  Job query ──▶ Stage 1: bi-encoder retrieve (FAISS, top-K)
            ──▶ Stage 2: cross-encoder rerank (trained)
            ──▶ top-N candidates + score + matched-skill explanation
```

Training is **fully offline** and produces a model artifact; the serving layer just loads it, so the live demo runs on CPU with no GPU required.

## What this demonstrates

Dataset construction and labeling · embeddings + ANN retrieval · **fine-tuning a transformer** · ranking metrics · an offline experiment with baselines and an ablation · a served, explainable model. The modeling counterpart to the orchestration shown in [Trace](#).

## Data

Public / synthetic only — **no proprietary data**. A public resume corpus (with occupation/category labels) plus a job-postings dataset, with graded relevance labels derived from role/skill match. **Hard negatives** are mined from adjacent categories to force the model to learn fit rather than keyword overlap. Splits are **by job** to prevent leakage.

## Tech stack

| Layer | Tech |
|---|---|
| Modeling | PyTorch, Hugging Face Transformers, sentence-transformers |
| Retrieval | FAISS (bi-encoder embeddings) |
| Baselines | rank-bm25, scikit-learn |
| Tracking | MLflow / Weights & Biases |
| Serving | FastAPI |
| Frontend | React + TypeScript + Vite |
| Infra | Docker Compose, GitHub Actions (lint · test · build) |

## Project structure

```
slate/
├── ml/             offline: data prep, labeling, splits, retrieval, training, evaluation
│   ├── training/   fine-tune the cross-encoder reranker
│   └── evaluation/ NDCG/MRR/Recall/MAP + baselines + ablation
├── backend/        FastAPI serving: retrieve → rerank → explain (loads artifact)
├── frontend/       React + Vite: paste a job → ranked slate + explanations
├── docs/           PRD
└── docker-compose.yml
```

## Getting started

**Prerequisites:** Docker + Docker Compose. For training: Python 3.11+ and a GPU (Colab/Kaggle is fine).

**Train (offline):**
```bash
cd ml
python data/prepare.py
python labeling/label.py
python splits/split.py
python retrieval/encode.py && python retrieval/index.py
python training/train_reranker.py --config training/config.yaml
python evaluation/run_eval.py        # writes the comparison table
```

**Serve the demo (uses the trained artifact):**
```bash
cp .env.example .env
docker compose up --build            # backend + frontend
```

Open the frontend (default `http://localhost:5173`), paste a job description, and get a ranked, explained shortlist.

## Roadmap

- Fairness / bias auditing across groups
- Richer learning-to-rank features (structured skills, experience recency) and listwise losses
- Human-relevance feedback loop
- Reuse the reranker as a component in larger retrieval systems

## License

Proprietary — All Rights Reserved

This repository is publicly visible for portfolio and evaluation purposes only.

No permission is granted to use, copy, modify, distribute, sublicense, or create derivative works from this code without explicit written permission from the author.
