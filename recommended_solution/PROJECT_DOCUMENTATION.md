# CTS-genC Unified Agent Pipeline

## Overview

CTS-genC is an end-to-end multi-agent pipeline designed to automate telecom customer complaint resolution. The system leverages LangGraph, HuggingFace (Qwen models), and Qdrant (vector DB) to triage, diagnose, propose solutions, and intelligently escalate tickets based on complexity and risk.

## Unified Architecture

The system has been refactored from three disconnected folders into a single, cohesive Python project.

```text
CTS-genC/
├── agents/
│   ├── __init__.py
│   ├── solution/         # LOW/MEDIUM Complexity Agent
│   ├── escalation/       # Agent to determine if tickets should be escalated
│   └── high/             # HIGH Complexity Agent (Multi-Agent System)
├── data/                 # Unified dataset (complaints.json, knowledge_base.json)
├── tests/
│   └── inputs/           # Pipeline test JSONs
├── orchestrator.py       # Master pipeline runner
├── requirements.txt      # Unified dependencies
└── .env                  # Unified configuration (API Keys, DB URLs)
```

## Input and Output Expectations

### Expected Input
The application expects a JSON payload containing the customer complaint and its associated metadata. The minimum required fields are:
- `complaint_id`: A unique identifier for the ticket.
- `complaint`: The raw text of the customer's issue.
- `category` / `technical_information`: Metadata used by the agents for routing and context.
- `complexity`: The initial severity level (`LOW`, `MEDIUM`, or `HIGH`).
- `customer_feedback`: (For escalation logic) A string simulating the user's response to the initial solution (e.g., "no", "it worked", "still broken").

### Expected Output
The application outputs a structured JSON response depending on the pipeline execution:
- **For LOW/MEDIUM tickets (Solution Agent):** A JSON object containing `customer_instructions` (step-by-step troubleshooting), `warnings`, and a list of `evidence` articles used.
- **For tickets evaluated by the Escalation Agent:** A JSON object detailing the `Next Severity` and the LLM's `Reasoning` for whether the ticket should be escalated.
- **For HIGH tickets (High Agent):** A comprehensive JSON report containing `Diagnosis`, `Root Cause`, `Priority`, and the `Final Decision` (e.g., ESCALATE, RESOLVE) along with confidence scores.

## Data Flow & State Contracts

The pipeline is driven by `orchestrator.py` which manages the flow between the agents:

1. **Input Stage:** A JSON file containing the complaint and metadata (including simulated `customer_feedback`) is ingested.
2. **Solution Generation (LOW/MEDIUM):** If complexity is `LOW` or `MEDIUM`, the **Solution Agent** retrieves vector context from Qdrant and generates a set of instructions for the customer.
3. **Escalation Validation:** The **Escalation Agent** evaluates the `customer_feedback`, the generated solution, and complaint metadata. Using a guardrail LLM policy, it determines the `next_severity`.
4. **Deep Diagnosis (HIGH):** If the initial complexity was `HIGH`, or if the Escalation Agent escalates the ticket to `HIGH`, the **High Agent** is invoked. The High Agent is a multi-agent system (Diagnosis, Policy, Risk, Planner, Critic) that produces a definitive resolution strategy.

## Agent Details

### 1. Solution Agent (`agents/solution`)
- **Type:** Sequential LangGraph Pipeline
- **Vector DB:** Qdrant (`QDRANT_COLLECTION`)
- **LLM:** HuggingFace `LLM_MODEL` (e.g., `Qwen/Qwen2.5-7B-Instruct`)
- **Function:** Understands basic telecom issues and provides step-by-step customer troubleshooting instructions. It uses a sequence of three specialized sub-agents:
  1. **Analysis Agent:** Parses the raw complaint to extract technical information (domain, component, failure type) and verifies the complexity level.
  2. **Knowledge Retrieval Node:** Takes the parsed technical parameters and queries the Qdrant vector database for relevant historical resolutions and KB articles.
  3. **Solution Synthesis Agent:** Combines the original complaint context with the retrieved knowledge to generate a set of safe, step-by-step troubleshooting instructions specifically for the customer.

### 2. Escalation Agent (`agents/escalation`)
- **Type:** LangGraph Decision Engine
- **LLM:** HuggingFace `LLM_MODEL`
- **Function:** Evaluates customer response to the solution. Implements deterministic policies (e.g., if scope is multiple customers, escalate to HIGH). Includes a `MockChatLLM` fallback if the API quota runs out.

### 3. High Agent (`agents/high`)
- **Type:** Complex LangGraph Multi-Agent Architecture
- **Vector DB:** Integrated Qdrant fallback to JSON
- **LLM:** HuggingFace `QWEN_MODEL` (e.g., `Qwen/Qwen2.5-72B-Instruct`)
- **Function:** Handles critical outages (e.g., physical tower damage). Employs multiple sub-agents to evaluate risk, enforce telecom policy, and propose an executive/engineering action plan. It includes an intelligent fallback mechanism to handle HuggingFace 402 (Payment Required) errors gracefully.

## Setup Instructions

### 1. Virtual Environment & Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables
You must create a file named `.env` and place it at the **root of the project directory** (i.e., `CTS-genC/.env`). The file should contain the following variables:

```env
# HuggingFace authentication for accessing the Qwen LLMs
HF_TOKEN="your_huggingface_token"

# The LLM used by the Solution and Escalation agents
LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"

# The more powerful LLM used by the multi-agent High Agent pipeline
QWEN_MODEL="Qwen/Qwen2.5-72B-Instruct"

# Connection details for the Qdrant Vector Database
QDRANT_URL="your_qdrant_url"
QDRANT_API_KEY="your_qdrant_api_key"
QDRANT_COLLECTION="telecom_knowledge_base"

# The embedding model used for vector search in Qdrant
EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

## Running the Pipeline

You can run the end-to-end integration tests using the orchestrator CLI:

```bash
# Test 1: LOW complaint that gets resolved or stays LOW
PYTHONPATH=. python orchestrator.py tests/inputs/pipeline_low_to_medium.json

# Test 2: MEDIUM complaint that escalates to HIGH
PYTHONPATH=. python orchestrator.py tests/inputs/pipeline_medium_to_high.json

# Test 3: HIGH complaint that goes directly to High Agent
PYTHONPATH=. python orchestrator.py tests/inputs/pipeline_high_direct.json
```

## Storage Strategy (PostgreSQL vs Qdrant)

Once fully integrated with a backend, the application relies on two distinct databases for different purposes:

### 1. PostgreSQL (Relational Database)
PostgreSQL acts as the **source of truth** for transactional state, ticket lifecycles, and relational metadata.
- **Stored Exclusively Here:**
  - `complaint_id`, user/customer IDs, timestamps, and current `status`.
  - The calculated metrics (`complexity_score`, `negativity_score`).
  - The step-by-step state transitions of the ticket (e.g., when it moved from LOW to HIGH).
  - The raw agent outputs (e.g., the JSON blobs generated by the Solution Agent or High Agent) and the final decision generated by the system.
  - User feedback logs ("yes", "no", "still broken").

### 2. Qdrant (Vector Database)
Qdrant is utilized exclusively as a **semantic retrieval engine** for the LLMs, functioning as the system's "Knowledge Base."
- **Stored Exclusively Here:**
  - High-dimensional embeddings representing the text of the Knowledge Base (KB) articles and historical troubleshooting guides.
  - The semantic meaning of past telecom issues and how they were resolved.
  
### 3. Sent to Both
Some fields must exist in both systems to link relational data with semantic embeddings:
- **`knowledge_id` / `article_id`:** The unique ID of a KB article is stored in PostgreSQL (for auditing/linking what evidence was used to solve a ticket) and in Qdrant (as the payload metadata of the vector).
- **Domain/Category Metadata:** Strings like `Internet Performance` or `Account Access` are stored in Postgres as columns for filtering, and also embedded as metadata in Qdrant payloads to narrow down vector searches during the retrieval step.

## Future Roadmap: Backend Integration

The next phase of development involves wrapping this LangGraph architecture into a FastAPI backend connected to a PostgreSQL database (running in Docker). 

### Planned FastAPI Endpoints

For the backend developers, the following REST endpoints should be implemented to connect the AI agents with the database and frontend:

1. `POST /api/v1/complaints/process`
   - **Purpose:** Ingests a new complaint from the frontend/database. It triggers the `Solution Agent` (for LOW/MEDIUM) or the `High Agent` (for HIGH) and saves the generated output to the PostgreSQL database.
2. `POST /api/v1/complaints/{complaint_id}/feedback`
   - **Purpose:** Receives user feedback (e.g., "This didn't work"). It triggers the `Escalation Agent` to evaluate the feedback against the previous solution and updates the severity level in the database.
3. `GET /api/v1/complaints/{complaint_id}`
   - **Purpose:** Fetches the current status, complexity, agent-generated solutions, and escalation history of a specific complaint from the Postgres DB.
4. `GET /api/v1/knowledge-base/sync`
   - **Purpose:** A utility endpoint to trigger the re-embedding of the knowledge base into the Qdrant vector database if new troubleshooting articles are added to Postgres.
