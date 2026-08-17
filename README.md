# Telecom AI Service - Hybrid Extraction & Technical Complexity Engine

AI/ML/NLP core for the Telecom Complaint Intelligence & Automated Resolution Assistant. Handles complaint categorization, feature extraction using a hybrid ML/LLM pipeline, technical complexity prioritization, and database persistence.

---

## ⚡ Architecture & Pipeline Flow

```
                             POST /complaints
                                    │
                                    ▼
                      TF-IDF + Logistic Regression
                      (app/extraction/ml_predictor.py)
                                    │
                                    ▼
                     Predict technical fields + confidence
                                    │
             ┌──────────────────────┴──────────────────────┐
             │                                             │
    ALL confidence >= 0.75                        ANY confidence < 0.75
             │                                             │
             ▼                                             ▼
    Use ML predictions                           Invoke LLM Fallback
                                             (HuggingFace Llama-3.1-8B)
             │                                             │
             └──────────────────────┬──────────────────────┘
                                    │
                                    ▼
                        Validate & Normalize Output
                      (app/extraction/validator.py)
                                    │
                                    ▼
                       Technical Complexity Engine
                      (app/priority/complexity.py)
                                    │
                                    ▼
                      PostgreSQL Database Storage
                       (app/database/models.py)
```

---

## 📁 Project Structure

```
telecom-ai-service/
├── app/
│   ├── main.py                     # FastAPI app entry point & route registration
│   ├── config.py                   # Environment settings management (Pydantic)
│   │
│   ├── api/
│   │   └── complaint_routes.py     # POST /complaints, GET /complaints API routes
│   │
│   ├── database/
│   │   ├── database.py             # SQLAlchemy engine (PostgreSQL + SQLite fallback)
│   │   └── models.py               # ORM Models: Complaint & TechnicalInformation
│   │
│   ├── models/
│   │   └── complaint_model.py      # Pydantic Request & Response schemas
│   │
│   ├── extraction/
│   │   ├── validator.py            # Enum validation & list normalization
│   │   ├── ml_predictor.py         # TF-IDF + Logistic Regression predictor & regex duration matcher
│   │   ├── llm_extractor.py        # Hugging Face Llama-3.1 API wrapper with JSON cleaning
│   │   ├── hybrid_extractor.py     # Decision router (ML confidence check vs LLM fallback)
│   │   └── service.py              # Pipeline orchestrator
│   │
│   └── priority/
│       └── complexity.py           # Technical complexity calculation engine
│
├── trained_models/
│   ├── train_dummy_model.py        # Model training bootstrap script
│   ├── vectorizer.joblib           # Trained TfidfVectorizer
│   └── field_models.joblib         # Trained LogisticRegression models per field
│
├── tests/
│   └── test_pipeline.py            # Unit & API test suite
│
├── .env.example                    # Template environment settings
├── pyproject.toml                  # Project dependencies
└── README.md
```

---

## 🛠️ Setup & Quickstart

### 1. Installation

Ensure `uv` is installed:
```bash
brew install uv
```

Install dependencies:
```bash
uv sync
```

### 2. Configure Environment

Create `.env` file (copy from `.env.example`):
```bash
cp .env.example .env
```

Set your environment variables in `.env`:
```env
HF_TOKEN=your_hugging_face_token_here
DATABASE_URL=postgresql://postgres:password@localhost:5432/telecom_ai
CONFIDENCE_THRESHOLD=0.75
```

### 3. Model Bootstrapping

Generate initial vectorizer and field models:
```bash
uv run python trained_models/train_dummy_model.py
```

### 4. Run the API Server

Start local server:
```bash
uv run uvicorn app.main:app --reload --port 8000
```
Interactive API documentation will be available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 🧪 Testing & Verification

Run the test suite:
```bash
uv run pytest
```

Run CLI demonstration script:
```bash
uv run python run_demo.py
```

---

## 🛰️ API Endpoints

### `POST /complaints`
Analyzes a telecom complaint, extracts technical attributes using hybrid ML/LLM, computes technical complexity, saves records to PostgreSQL, and returns the response.

**Request**:
```json
{
  "complaint": "Our local fiber cable is damaged, leaving many customers without service since this morning."
}
```

**Response**:
```json
{
  "complaint_id": 1,
  "complaint": "Our local fiber cable is damaged, leaving many customers without service since this morning.",
  "extraction_source": "ml",
  "lowest_confidence": 0.847,
  "technical_information": {
    "component": ["fiber_cable"],
    "failure_type": ["physical_damage"],
    "scope": "multiple_customers",
    "service_impact": "complete_outage",
    "duration_hours": 12.0,
    "occurrence_pattern": "continuous"
  },
  "complexity": "CRITICAL",
  "complexity_score": 88
}
```

### `GET /complaints`
Returns a list of all stored complaint records.

### `GET /complaints/{complaint_id}`
Returns details for a specific complaint record by ID.
