# Product Requirements Document — Slate

**Working title:** Slate — Learned Candidate–Job Matching
**Tagline:** *Rank the right candidates, not the loudest keywords.*
**Version:** 1.0 (MVP)
**Owner:** Anushka
**Status:** Draft for build
**Context:** MCA capstone / portfolio project (project 2 of 2). Built solo.
**Portfolio fit:** Slate is the *modeling* counterpart to **Trace**. Where Trace demonstrates AI system orchestration (retrieval, graph reasoning, LLM workflows), Slate demonstrates that you can **train and rigorously evaluate a model**: dataset construction, embeddings, fine-tuning a transformer reranker, ranking metrics, and an offline experiment comparing your model against real baselines. Together the two projects cover orchestration *and* ML.

---

## 1. Problem & vision

Matching candidates to jobs is a ranking problem, but most tooling does it badly. Boolean/keyword search misses semantically-relevant candidates and over-rewards keyword stuffing; naive embedding cosine similarity retrieves "topically near" candidates but ranks fit poorly because it was never trained on what a *good match* actually looks like.

Slate is a **two-stage learned ranking system**: a fast bi-encoder **retriever** narrows a large candidate pool, then a fine-tuned cross-encoder **reranker** — trained on labeled (job, candidate, relevance) data — orders the shortlist by genuine fit. The headline result is an offline evaluation showing the trained reranker beats keyword and naive-embedding baselines on standard ranking metrics.

**This is the part that proves modeling skill:** you build the labeled dataset, fine-tune a transformer, and measure it.

---

## 2. Goals and non-goals

### Goals
- Build a labeled candidate–job relevance dataset from **public / synthetic** data (no proprietary data).
- Implement a **two-stage** pipeline: bi-encoder retrieval + fine-tuned cross-encoder reranking.
- **Fine-tune** the reranker on the labeled data (the core ML deliverable).
- Evaluate with standard ranking metrics (**NDCG@k, MRR, Recall@k, MAP**) against **two baselines**: BM25 (keyword) and raw embedding cosine (no fine-tuning).
- Track every experiment (config, metrics, artifacts) for reproducibility.
- Serve via a FastAPI endpoint with a small React + TS demo UI: paste a job → ranked candidates with scores and a short "why this match" explanation.
- Run via `docker compose up`, with one CI workflow.

### Non-goals (explicitly out of scope for v1.0)
- A full ATS / sourcing product, sourcing integrations, or any Recruiterflow data.
- Handling real PII; only public or synthetic data is used.
- Online / continual learning; training is offline and batch.
- Bias-auditing as a research contribution (acknowledged in §12, but a full fairness study is out of scope for v1.0).
- LLM-generated free-text "match summaries" as the core feature — explanations are lightweight and grounded in matched signals, not the deliverable.

---

## 3. Target users & personas

| Persona | Need | Feature |
|---|---|---|
| Recruiter | "Given this role, who are the top 10 candidates and why?" | Ranked shortlist + explanation |
| Hiring manager | "Is this ranking better than our keyword search?" | Eval comparison / scores |
| ML reviewer / examiner | "Did the model actually learn to rank, and is it measured?" | Evaluation report (§8) |

For v1.0 the "candidate pool" and "jobs" come from public datasets (see §4). The product is demonstrated end to end on that corpus.

---

## 4. Data

### 4.1 Sources (public / synthetic only)
- A public **resume dataset** (e.g. a labeled resume corpus with occupation/category fields, such as those on Kaggle / Hugging Face).
- A public **job-postings dataset** (titles, descriptions, required skills).
- Optionally augment with **synthetic** job/candidate pairs generated to balance categories and create hard cases.

### 4.2 Relevance labeling scheme
Since explicit (job, candidate, relevance) labels rarely exist publicly, derive them with a clear, defensible scheme:
- **Positive (relevant):** candidate whose occupation/category and core skills match the job's role and required skills.
- **Hard negative:** candidate from an adjacent/near-miss category (similar surface terms, wrong fit) — these are what force the model to learn real fit, not keyword overlap.
- **Easy negative:** random candidate from an unrelated category.
- Labels are **graded** where possible (e.g. 2 = strong, 1 = partial, 0 = irrelevant) to support NDCG.

Document the scheme in the report and **manually spot-check** a sample for label quality (report the agreement rate).

### 4.3 Splits
- Train / validation / test split **by job** (no job appears in two splits) to prevent leakage.
- Reserve a held-out test set used **only** for the final §8 comparison.

---

## 5. System architecture

```
        ┌─────────────────────────────────────────────┐
        │  React + TypeScript (Vite) demo UI            │
        │  paste job → ranked candidates + scores + why │
        └───────────────────────┬──────────────────────┘
                                │ REST/JSON
        ┌───────────────────────▼──────────────────────┐
        │  FastAPI serving layer                         │
        │  Stage 1: retrieve (bi-encoder + ANN index)    │
        │  Stage 2: rerank (fine-tuned cross-encoder)    │
        │  + explanation (matched skills/terms)          │
        └───────────────────────┬──────────────────────┘
                                │
        ┌───────────────────────▼──────────────────────┐
        │  Vector index (FAISS or pgvector)             │
        │  Candidate embeddings + metadata               │
        └────────────────────────────────────────────────┘

   Offline (training): data prep → label → fine-tune reranker
                       → evaluate vs baselines → track (MLflow/W&B)
```

The training pipeline is offline and produces a model artifact; the serving layer loads that artifact. Keep them cleanly separated.

---

## 6. Models & pipeline

### 6.1 Stage 1 — Retrieval (recall)
- Encode candidates and the job query with a **bi-encoder** sentence-transformer (e.g. `all-MiniLM-L6-v2` or `bge-small`).
- Index with **FAISS** (or pgvector to reuse Trace's stack) for approximate nearest-neighbor recall.
- Return top-K (e.g. K = 50) candidates per job. Optimize for **Recall@K**, not final ordering.

### 6.2 Stage 2 — Reranking (precision) — **the trained model**
- A **cross-encoder** (e.g. a MiniLM cross-encoder base, or fine-tune `bge-reranker`) that takes (job, candidate) jointly and outputs a relevance score.
- **Fine-tune** on the labeled pairs from §4 (pairwise or listwise ranking loss; LoRA/PEFT acceptable to keep compute light).
- Reorder the Stage-1 shortlist by the learned score → final top-N.

### 6.3 Explanation (lightweight)
- Surface *why* a candidate ranked high: highlight overlapping skills/terms between job and candidate, and the reranker score. Grounded in actual signals — not an LLM essay. Plays to the frontend strength and adds product feel without scope creep.

### 6.4 Baselines (for the comparison)
- **BM25** keyword ranking (`rank-bm25`).
- **Raw embedding cosine** (Stage-1 bi-encoder, no reranking / no fine-tuning).

---

## 7. Functional requirements

- **F1 — Data pipeline:** ingest public datasets, normalize candidate/job text, apply the §4.2 labeling scheme, produce train/val/test splits by job.
- **F2 — Embedding + index:** embed candidates, build the ANN index, support top-K retrieval for a query job.
- **F3 — Training:** fine-tune the cross-encoder reranker; checkpoint best model on validation NDCG; log all runs.
- **F4 — Inference pipeline:** job in → Stage 1 retrieve → Stage 2 rerank → top-N with scores + matched-signal explanation.
- **F5 — Evaluation harness:** compute NDCG@5/@10, MRR, Recall@50, MAP for the reranker and both baselines on the held-out test set; emit a comparison table.
- **F6 — Serving API:** FastAPI `POST /rank` accepting a job description, returning ranked candidates + scores + explanations.
- **F7 — Demo UI:** React + TS page to paste a job and view the ranked slate with scores and matched skills.

---

## 8. Evaluation & quality — the credibility layer

This is the centerpiece of the project's "I can do ML" claim.

- **Metrics:** NDCG@5, NDCG@10, MRR, Recall@50, MAP on the held-out test set.
- **Comparison table:** fine-tuned reranker vs. BM25 vs. raw-embedding cosine. Report the lift.
- **Ablations (at least one):** e.g. with vs. without hard negatives in training, or bi-encoder-only vs. two-stage — to show *which* design choice drives the gain.
- **Error analysis:** inspect failure cases (where the model ranks a poor fit highly) and discuss.
- **Experiment tracking:** MLflow or Weights & Biases logging every run's config, metrics, and model artifact for full reproducibility.

**Success bar:** the fine-tuned two-stage system beats *both* baselines on NDCG@10 and MRR by a clear, reported margin.

---

## 9. Technical stack

| Layer | Choice |
|---|---|
| Modeling | Python, PyTorch, Hugging Face Transformers, sentence-transformers |
| Retrieval index | FAISS (or pgvector to mirror Trace) |
| Baselines | `rank-bm25`, scikit-learn utilities |
| Experiment tracking | MLflow or Weights & Biases |
| Serving | FastAPI |
| Frontend | React + TypeScript + Vite |
| Infra | Docker Compose; GitHub Actions: lint + test + build |
| Testing | pytest (pipeline + metrics), vitest (UI) |
| Compute | Small models + LoRA so training fits on a single modest GPU / free-tier (Colab/Kaggle) |

---

## 10. Non-functional requirements

- **Reproducibility:** training and eval reproducible from a seeded config; tracked runs.
- **Inference latency:** rank a job against a 50-candidate shortlist in under ~2 s on CPU at serving time.
- **Run:** `docker compose up` serves the API + UI against a prebuilt index and shipped model artifact.
- **Footprint:** models small enough to train and serve without specialized hardware.

---

## 11. Milestones (suggested phasing)

| Phase | Deliverable | Done when |
|---|---|---|
| P0 — Skeleton | Repo, env, Docker Compose, CI green | Project scaffolds and runs empty |
| P1 — Data | Datasets ingested, labeling scheme applied, splits by job | Labeled train/val/test sets exist; label spot-check reported |
| P2 — Retrieval | Bi-encoder embeddings + FAISS index + top-K retrieval | Recall@50 measured on test jobs |
| P3 — Baselines | BM25 + raw-embedding cosine evaluated | Baseline metrics table produced |
| P4 — Reranker (core) | Fine-tuned cross-encoder, checkpointed on val NDCG | Reranker beats baselines on val set |
| P5 — Evaluation | Held-out test metrics, ablation, error analysis, tracking | §8 comparison table + ablation done |
| P6 — Serving + UI | FastAPI `/rank` + React demo with explanations | Paste-a-job demo returns a ranked slate live |
| P7 — Polish | Tests, docs, README, demo script, report | Reproducible from clean clone; report-ready |

---

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Weak / noisy labels (no public ground truth) | Clear graded labeling scheme + hard negatives; manual spot-check with reported agreement |
| Small data → overfitting | Start from pretrained models; LoRA/light fine-tune; validate by-job splits; cross-validate |
| "Just embeddings" — no real learning shown | Cross-encoder fine-tuning + ablation isolate the learned gain; baselines make the lift explicit |
| Compute limits | Small models, LoRA, free-tier GPU; cache embeddings |
| Fairness/bias concerns in matching | Acknowledge explicitly; keep features role/skill-based; flag full fairness audit as future work |
| Scope creep into a full product | Non-goals in §2 are binding; the deliverable is a *measured ranking model*, not an ATS |

---

## 13. Future work (deferred)

- Fairness / bias auditing (e.g. demographic parity across groups) as a first-class study.
- Learning-to-rank with richer features (structured skills, experience recency) and listwise losses.
- Human-relevance feedback loop / online learning.
- LLM-generated match rationales grounded in the ranking signals.
- Integration of Slate's reranker as a component inside larger retrieval systems (including Trace).

---

## 14. Success metrics

- Fine-tuned two-stage system beats **both** baselines on NDCG@10 and MRR on the held-out test set, by a reported margin.
- At least one ablation isolates the design choice driving the gain.
- Every experiment tracked and reproducible from config.
- `docker compose up` serves a live paste-a-job demo returning a ranked, explained slate; CI green.


## 15. Repo Structure
```
slate/
├── README.md
├── docker-compose.yml
├── .env.example
├── .github/workflows/ci.yml
├── ml/                          # offline modeling core
│   ├── pyproject.toml
│   ├── data/
│   │   ├── raw/                 # gitignored
│   │   ├── processed/
│   │   └── prepare.py           # ingest + normalize public datasets
│   ├── labeling/label.py        # relevance scheme + hard negatives
│   ├── splits/split.py          # split by job (no leakage)
│   ├── retrieval/
│   │   ├── encode.py            # bi-encoder embeddings
│   │   └── index.py             # FAISS build
│   ├── training/
│   │   ├── dataset.py
│   │   ├── train_reranker.py    # fine-tune cross-encoder (LoRA)
│   │   └── config.yaml
│   ├── evaluation/
│   │   ├── metrics.py           # NDCG / MRR / Recall / MAP
│   │   ├── baselines.py         # BM25, raw cosine
│   │   └── run_eval.py          # comparison table + ablation
│   ├── notebooks/               # Colab / Kaggle training
│   └── artifacts/               # trained model + index (or HF Hub pointer)
├── backend/                     # serving only
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── pipeline.py          # retrieve → rerank → explain
│   │   ├── models/loader.py     # load artifact + index
│   │   ├── api/routes_rank.py
│   │   └── schemas/
│   └── tests/
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── Dockerfile
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api/
│       ├── components/
│       │   ├── JobInput.tsx
│       │   ├── RankedList.tsx
│       │   └── MatchExplain.tsx # matched skills/terms + score
│       └── types/
└── docs/PRD_Slate.md
```