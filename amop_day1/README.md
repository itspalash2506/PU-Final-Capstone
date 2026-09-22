# AMOP — Autonomous Maintenance Operations Platform

AMOP is an AI-assisted maintenance operations platform for turning industrial work-order history and predictive-maintenance telemetry into actionable maintenance decisions. It combines hybrid search, retrieval-augmented generation (RAG), multi-agent orchestration, machine-learning predictions, root-cause analysis, technician routing, and a real-time web interface.

The project is designed for maintenance engineers, reliability teams, plant operators, and technicians who need to investigate equipment issues quickly, reuse historical knowledge, prioritize failures, and decide what action should happen next.

## What problem does it solve?

Industrial maintenance information is often distributed across work orders, technician notes, equipment identifiers, sensor records, and individual staff knowledge. This creates several operational problems:

- **Slow troubleshooting:** Engineers may need to manually search thousands of historical work orders for similar failures.
- **Poor use of historical knowledge:** Useful repair context can remain buried in free-text notes.
- **Keyword-only search limitations:** Exact search misses semantically similar descriptions, while semantic search can miss machine IDs, part numbers, and error codes.
- **Reactive maintenance:** Teams may respond only after a failure instead of identifying elevated risk earlier.
- **Unstructured root-cause investigations:** Incident analysis and corrective actions can vary by person and may not be consistently documented.
- **Manual prioritization and assignment:** Severity, expected resolution time, urgency, and technician selection may require separate manual steps.
- **Limited explainability and observability:** Users need to know which records supported an answer, which agents ran, how long they took, and why a prediction was made.

AMOP addresses these problems through a single query workflow that can retrieve relevant history, analyze likely causes, predict risk and ETA, recommend routing, and present the reasoning process to the user.

## Main features

### 1. Hybrid maintenance knowledge retrieval

The ingestion pipeline in `amop/pipelines/ingest.py` processes CSV work orders and stores them in multiple forms:

1. Normalized work orders are saved to SQLite.
2. A BM25 sparse index is built for lexical matching.
3. Text is embedded and upserted into Qdrant for dense vector search.
4. Models are retrained after ingestion.

The retriever in `backend/rag/retriever.py` combines BM25 and Qdrant results with Reciprocal Rank Fusion. This provides both:

- Strong exact matching for machine IDs, part numbers, and error codes.
- Semantic matching for descriptions such as “motor will not start” and “motor startup failure.”

A cross-encoder reranker can optionally improve the ordering of retrieved results.

### 2. Multi-agent maintenance workflow

`backend/agents/graph.py` defines a LangGraph workflow with specialized nodes:

- **Equipment Agent:** Extracts or infers the equipment identifier.
- **Intent Router:** Classifies the request as history lookup, prediction, root-cause analysis, or a combined request.
- **RAG Agent:** Retrieves relevant historical work orders and generates evidence-backed context.
- **RCA Agent:** Produces structured root-cause analysis.
- **Prediction Agent:** Estimates severity and expected time to resolution.
- **Routing Agent:** Recommends a technician and urgency.
- **Summarizer:** Produces the final user-facing response.

The graph selects the appropriate path instead of running every component for every request. For example, a simple history query can skip RCA, while a combined history-and-prediction request can run retrieval, RCA, prediction, routing, and summarization.

### 3. Failure severity and ETA prediction

The standard prediction pipeline uses XGBoost models trained from work-order text and engineered features. It provides:

- Low, medium, and high severity classification.
- Estimated time-to-resolution prediction.
- Automatic retraining after new work orders are ingested.
- Persisted model bundles using `joblib`.

The current trainer derives labels using feature and keyword-based rules, including synthetic high-severity examples to ensure all severity classes are represented. These predictions should therefore be treated as decision support and validated against real labeled outcomes before production use.

### 4. Predictive maintenance from telemetry

The PdM modules support failure-risk prediction for numeric machine IDs using telemetry-derived features. `backend/prediction/pdm_predictor.py`:

- Builds the latest feature vector for a machine.
- Predicts failure probability for a 24-hour horizon.
- Converts probability into Low, Medium, High, or Critical risk tiers.
- Uses SHAP to identify the five features driving the prediction.
- Returns a structured, explainable result for the API and frontend.

The semantic bridge connects equipment identifiers and maintenance knowledge with predictive-maintenance machine profiles stored in Qdrant.

### 5. Root-cause analysis and corrective guidance

For RCA requests, the platform uses retrieved maintenance history and an LLM-based analysis step to organize:

- Probable cause.
- Contributing factors.
- Supporting evidence.
- Corrective actions.
- Preventive actions.
- Confidence or analysis metadata.

This helps turn unstructured historical records into a repeatable investigation format.

### 6. Technician routing and urgency

The routing agent uses the query context, retrieved history, prediction results, and maintenance information to recommend a technician and urgency level. This supports faster handoff from diagnosis to execution.

### 7. Real-time agent progress and answer streaming

The FastAPI backend exposes both normal and streaming query endpoints:

- `POST /api/v1/query`
- `POST /api/v1/query/stream`

The streaming route uses server-sent events (SSE). The React frontend displays agent completion events, token-by-token answer output, sources, RCA details, routing results, and prediction panels while the workflow is running.

### 8. Feedback, tracing, and evaluation

AMOP stores operational and quality signals in SQLite:

- Query logs.
- Per-agent traces and latency.
- User feedback.
- Retrieved document IDs.
- Evaluation results.

The DeepEval harness measures answer relevance, faithfulness, contextual precision, and contextual recall. Feedback can also update the `effectiveness_score` associated with Qdrant work-order points, enabling future retrieval-quality improvements.

### 9. API security and operational support

The API supports API-key validation, query sanitization, request IDs, structured logging, health checks, CORS configuration, and a global exception handler. Configuration is managed through environment variables and `.env` files.

## Potential impact

If validated with representative production data and integrated into maintenance workflows, AMOP could have the following impacts:

- **Reduced troubleshooting time:** Engineers can find similar work orders and repair history through natural-language questions.
- **Lower mean time to repair:** Faster retrieval, RCA, prioritization, and technician assignment can shorten response cycles.
- **Improved preventive maintenance:** Risk scores can help teams intervene before failures become disruptive.
- **More consistent maintenance decisions:** Structured RCA and routing outputs provide a repeatable operating pattern.
- **Better knowledge retention:** Technician experience encoded in work orders becomes searchable and reusable across shifts and teams.
- **Improved explainability:** Source documents, agent traces, SHAP features, and confidence metadata make recommendations easier to review.
- **Continuous improvement:** Feedback, effectiveness scores, query logs, and DeepEval metrics create a foundation for monitoring and improving the system.
- **Potential safety and cost benefits:** Earlier detection and better prioritization may reduce unplanned downtime, emergency repairs, production disruption, and safety exposure.

These are potential benefits rather than measured outcomes. The repository includes evaluation tooling, but a real impact assessment would require deployment data, baseline metrics, controlled comparisons, and human review.

## Architecture and data flow

```text
CSV work orders
      │
      ▼
parse_csv() → SQLite work_orders
      │       → BM25 index
      │       → local embeddings → Qdrant
      │
      ▼
FastAPI query endpoint
      │
      ▼
Equipment Agent → Intent Router
      │
      ├── RAG → Routing → Summarizer
      ├── Prediction → Summarizer
      ├── RAG → RCA → Summarizer
      └── RAG → RCA → Prediction → Routing → Summarizer
      │
      ▼
Answer + sources + traces + prediction/RCA/routing metadata
      │
      ▼
React/Vite frontend with SSE progress streaming
```

## Repository structure

```text
amop_day1/
├── amop/
│   ├── backend/
│   │   ├── agents/       LangGraph maintenance agents and workflow
│   │   ├── api/          FastAPI application and route handlers
│   │   ├── clients/      OpenRouter, embeddings, and Qdrant clients
│   │   ├── core/         Configuration, logging, and security
│   │   ├── db/           SQLAlchemy models, sessions, and repositories
│   │   ├── prediction/   Work-order and telemetry prediction modules
│   │   ├── rag/          Retrieval, generation, and reranking
│   │   ├── schemas/      Pydantic request and response schemas
│   │   └── requirements.txt
│   ├── evaluation/       DeepEval harness and test cases
│   ├── frontend/         React/Vite user interface
│   ├── models/            BM25 and serialized ML model artifacts
│   ├── pipelines/        Work-order ingestion and PdM training
│   ├── scripts/           Maintenance utilities such as score backfill
│   ├── .env.example      Environment-variable template
│   └── test_*.py         Component and integration-oriented checks
└── how_to_run.MD         Detailed local setup instructions
```

## Technology stack

- **Backend:** Python, FastAPI, Uvicorn
- **Frontend:** React 18, Vite, JavaScript, CSS
- **Agent orchestration:** LangGraph, LangChain
- **LLM access:** OpenRouter through an OpenAI-compatible client
- **Retrieval:** Qdrant, sentence-transformers, BM25, Reciprocal Rank Fusion, optional cross-encoder reranking
- **Persistence:** SQLite and SQLAlchemy
- **Machine learning:** XGBoost, scikit-learn, NumPy, pandas, joblib, SHAP
- **Evaluation:** DeepEval
- **Infrastructure:** Qdrant in Docker; optional CUDA acceleration for embeddings

## Running locally

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker Desktop
- A CUDA-enabled PyTorch installation is recommended for local embedding performance
- An OpenRouter API key for LLM-powered routing, analysis, and summarization

The detailed setup guide is in [`how_to_run.MD`](how_to_run.MD).

### 1. Start Qdrant

```powershell
docker start qdrant
```

If the container does not exist:

```powershell
docker run -d --name qdrant -p 6333:6333 `
  -v qdrant_data:/qdrant/storage qdrant/qdrant:latest
```

### 2. Configure the backend

From `amop_day1/amop`:

```powershell
Copy-Item .env.example .env
```

Set at least:

```env
OPENROUTER_API_KEY=your-openrouter-key
AMOP_API_KEY=amop-dev-secret
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=work_orders
```

### 3. Install Python dependencies and start the API

```powershell
cd amop_day1/amop
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
uvicorn backend.api.main:app --port 8000 --log-level info
```

The API documentation is available at `http://localhost:8000/docs`.

### 4. Start the frontend

In a second terminal:

```powershell
cd amop_day1/amop/frontend
npm install
npm run dev
```

The frontend is available at `http://localhost:5173`.

### 5. Ingest work orders

If Qdrant has no data, upload a compatible CSV through:

```powershell
curl -X POST http://localhost:8000/api/v1/ingest `
  -H "X-API-Key: amop-dev-secret" `
  -F "file=@dataset_files/mwo_dataset_augmented.csv"
```

The expected CSV fields are `mach`, `date_received`, `issue`, `info`, and `tech`.

### 6. Try a query

Use `POST /api/v1/query` or the frontend with a request such as:

```json
{
  "query": "What caused the hydraulic line leak?",
  "machine_id": "A6",
  "top_k": 5
}
```

The dataset convention documented by the project uses machine IDs such as `A1` through `A99` for work orders.

## Evaluation and testing

Run the DeepEval harness from `amop_day1/amop`:

```powershell
python -m evaluation.harness
```

The repository also includes component-oriented checks such as:

```powershell
python test_agents.py
python test_each_component.py
python test_feedback.py
python test_reranker.py
```

Some tests and evaluation paths require Qdrant, model artifacts, telemetry data, and configured API credentials.

## Important limitations and considerations

- The work-order severity and ETA trainer uses synthetic/rule-derived labels; production decisions require real labels and calibration.
- LLM answers and routing recommendations should be reviewed by qualified maintenance personnel.
- PdM predictions depend on the availability and quality of the telemetry dataset and trained model artifact.
- The local setup assumes Qdrant is available and may benefit from GPU acceleration for embeddings.
- API keys and operational secrets must be stored outside source control.
- The repository includes serialized model/index artifacts; retraining and artifact versioning should be formalized for production deployment.
- Impact claims should be validated with baseline measurements such as mean time to diagnose, mean time to repair, downtime, false alarms, and user acceptance.

## Project status

This is a capstone-style prototype demonstrating an end-to-end AI maintenance workflow. It provides a strong foundation for experimentation and evaluation, but production deployment would require additional work around data governance, authentication, model monitoring, reliability, deployment automation, safety review, and validated business metrics.
