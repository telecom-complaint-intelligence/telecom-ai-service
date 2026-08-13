# telecom-ai-service

AI/ML/NLP core for the Telecom Complaint Intelligence & Automated Resolution Assistant — handles categorization, sentiment analysis, information extraction, priority classification, RAG-based quick fixes, agentic triage, summarization, and the retrain feedback loop. Called by `telecom-backend` via internal HTTP API.

---

## ⚙️ Tech Decisions

To prevent development collision, the following technology standards are locked in for this service:

*   **AI Agents Framework**: **LangGraph** (chosen for its explicit state-graph control over complex workflows and decision-making logic).
*   **Vector Database**: **pgvector** (integrated directly with the existing PostgreSQL database to reduce infrastructure footprint and reuse database components).

---

## 📁 Folder Structure

```
telecom-ai-service/
├── app/
│   ├── api/               # inference endpoints exposed to backend
│   ├── preprocessing/      # shared preprocessing for training & live complaints
│   ├── models/
│   │   ├── categorization/  # DistilBERT/BERT categorization
│   │   └── sentiment/        # DistilBERT Sentiment Analysis (S.A.)
│   ├── extraction/            # Information Extraction (I.E.) — depends on S.A. output
│   ├── aggregator/              # Feature Aggregator (F.A.) — merges BERT+S.A.+I.E.+N.Q.+Duration
│   ├── priority/                  # Low/Medium/High/Critical classification + routing rules
│   ├── rag/                         # RAG → Quick Fix path (Low priority, resolvable via KB)
│   ├── agents/                        # Agentic AI — High priority analysis, decision, ticket raising (LangGraph)
│   ├── summarization/                   # BART ticket/complaint summary generation (generation stage)
│   └── retrain/                           # feedback loop: store resolved data, trigger retrain
├── data/
│   ├── ingestion/          # dataset download/merge scripts (Kaggle sources)
│   ├── preprocessing/        # cleaning, labeling, feature engineering
│   └── schemas/                # column/data contracts
├── training/
│   ├── notebooks/          # experiment notebooks
│   └── scripts/              # reusable training scripts
├── artifacts/               # model weights / MLflow-DVC pointers (not raw large files in git)
├── tests/                    # unit & inference tests
└── .github/
    └── workflows/            # CI config
```

**Rule of thumb — where code goes (mapped to the pipeline flow):**
| Pipeline stage | Folder |
|---|---|
| Dataset preprocessing (shared, both training + live) | `app/preprocessing/` |
| Categorization (DistilBERT/BERT) | `app/models/categorization/` |
| Sentiment analysis (DistilBERT) | `app/models/sentiment/` |
| Information extraction (depends on sentiment) | `app/extraction/` |
| Merging features into historical table | `app/aggregator/` |
| Priority routing (Low/Medium/High/Critical) | `app/priority/` |
| Quick-fix retrieval for Low priority (RAG with pgvector) | `app/rag/` |
| Agentic decision-making for High priority (LangGraph) | `app/agents/` |
| Closed/Fixed ticket summary (BART) | `app/summarization/` |
| Feedback loop: store resolved data, retrain trigger | `app/retrain/` |
| New inference endpoint exposed to backend | `app/api/` |
| Dataset download/merge script | `data/ingestion/` |
| Model training code | `training/scripts/` (promote from `training/notebooks/` once stable) |

**Important routing rule from the flowchart:** Critical-priority complaints bypass AI entirely — no agent involvement, directs straight to technicians. Keep this branch explicit in `app/priority/`, don't let `app/agents/` silently try to handle Critical cases.

---

## Branching Strategy

```
main        → production-ready, protected, deploy-only
  └── dev   → integration branch, all features merge here first
       ├── feature/<yourname>-<short-feature-desc>
       ├── fix/<yourname>-<short-bug-desc>-<issue-number>
       └── chore/<yourname>-<short-task-desc>
```

| Type | Format | Example |
|---|---|---|
| New feature | `feature/<name>-<feature>` | `feature/meera-sentiment-model` |
| Bug fix | `fix/<name>-<feature>-<issue-number>` | `fix/arjun-rag-retriever-31` |
| Refactor / cleanup | `chore/<name>-<task>` | `chore/meera-notebook-cleanup` |
| Hotfix (urgent, off main) | `hotfix/<name>-<issue>` | `hotfix/arjun-model-load-fail` |

**Rules:**
- Never commit directly to `main` or `dev` — always via Pull Request.
- Branch off `dev`, not `main`.
- One branch = one feature/fix.
- **Do not delete branches after merge** — this is a hackathon; the full branch history is part of showcasing individual contribution. Merge via PR, keep the branch.
- Keep PR descriptions detailed (what was built, how tested, linked issue) for the same reason.
- Large model weight files (`artifacts/`) should NOT be committed raw to git — use Git LFS, or push to MLflow/DVC/cloud storage and commit only a pointer/reference file.

### Flow

```
1. git checkout dev
2. git pull origin dev
3. git checkout -b feature/<name>-<feature>
4. ... code, commit ...
5. git push origin feature/<name>-<feature>
6. Open PR: feature/<name>-<feature> → dev
7. Get 1 review approval + CI passing
8. Merge (branch stays, not deleted)
9. Periodically: dev → main (when stable, via PR)
```

---

## 💻 Commands (run from repo root: `telecom-ai-service/`)

First, make sure you have `uv` installed. If you do not have it, install it using:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using Homebrew (macOS)
brew install uv
```

Once `uv` is installed, set up the project:

```bash
# install dependencies (automatically syncs environment)
uv sync

# run local dev server (inference API)
uv run uvicorn app.main:app --reload --port 8001    # http://localhost:8001
                                                    # docs at /docs (Swagger)

# run a training script
uv run python training/scripts/train_categorization.py
uv run python training/scripts/train_sentiment.py

# lint
uv run ruff check app/

# run tests
uv run pytest

# run with coverage
uv run pytest --cov=app
```

> All commands run from the **repo root**. If using Docker: `docker-compose up --build` — check `docker-compose.yml` for the service port (should not collide with `telecom-backend`, typically 8000).

---

## Before You Start Coding (after `git pull`)

- [ ] Confirm you're on the correct branch (`git branch`)
- [ ] Pulled latest `dev`: `git pull origin dev`
- [ ] `uv sync` — dependencies may have changed (new model libs, etc.)
- [ ] `.env` present and up to date (check `.env.example` — vector DB URL, model paths, API keys)
- [ ] Confirm required model artifacts are present locally, or pulled via DVC/MLflow — check `artifacts/README` for the current pointer/version
- [ ] `uv run uvicorn app.main:app --reload --port 8001` — confirm it boots clean, hit `/docs` to sanity-check
- [ ] Check open PRs/issues board — avoid duplicate work on the same pipeline stage (e.g. two people building `priority/` logic)

---

## Before You Push

- [ ] Lint passes (`uv run ruff check app/`)
- [ ] `uv run pytest` — all tests pass, added/updated tests for what changed
- [ ] No raw model weight files or large datasets committed directly — use LFS/DVC/cloud pointer
- [ ] No `print()` / debug leftovers
- [ ] No API keys, vector DB credentials hardcoded or committed — everything through `.env`
- [ ] `.env.example` updated if you added a new required env var
- [ ] If you changed a model's input/output shape, update the matching Pydantic schema AND flag it to backend team (shared contract)
- [ ] Endpoint tested manually via `/docs` (Swagger) at least once
- [ ] Branch up to date with latest `dev` — resolve conflicts locally
- [ ] PR description filled: what changed, why, how tested, linked issue number

---

## Commit Message Convention

```
feat: add BERT categorization model
fix: correct sentiment score not passed to extraction (#31)
chore: clean up unused training notebooks
refactor: split aggregator logic from priority module
docs: update README setup steps
```

Format: `<type>: <short description>` — one line, present tense, no trailing period.
