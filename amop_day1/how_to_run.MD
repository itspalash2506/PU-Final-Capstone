# AMOP — How To Run

Complete guide to starting all services and running the application locally.

---

## Prerequisites

Make sure these are installed before starting:
- Python 3.11+
- Node.js 20+
- Docker Desktop (running)
- CUDA-enabled PyTorch (for GPU embeddings)

---

## Project Location

```
C:\Final_Capstone_Project\amop_day1\amop\
```

All commands below assume you are in this directory:

```powershell
cd C:\Final_Capstone_Project\amop_day1\amop
```

---

## Step 1 — Start Qdrant (Vector Database)

Qdrant runs in Docker. Start it first before anything else.

```powershell
# Check if already running
docker ps

# If not running, start it
docker start qdrant

# If container doesn't exist, create it fresh
docker run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant:latest

# Verify it's up
docker ps
```

Expected output:
```
CONTAINER ID   IMAGE                  STATUS         PORTS
xxxxxxxxxxxx   qdrant/qdrant:latest   Up X seconds   0.0.0.0:6333->6333/tcp
```

Qdrant dashboard (optional): http://localhost:6333/dashboard

---

## Step 2 — Activate Virtual Environment

```powershell
# Activate venv
C:\Final_Capstone_Project\venv\Scripts\Activate.ps1

# Verify activation — prompt should show (venv)
# (venv) PS C:\Final_Capstone_Project\amop_day1\amop>
```

---

## Step 3 — Verify Qdrant Has Data

```powershell
python -c "
from backend.clients.vector_store import get_qdrant_client
from backend.core.config import get_settings
client = get_qdrant_client()
info = client.get_collection(get_settings().qdrant_collection)
print('Points:', info.points_count)
"
```

Expected: `Points: 19784`

If you see `0` or an error — run the ingestion pipeline (see Step 6).

---

## Step 4 — Start Backend

```powershell
uvicorn backend.api.main:app --port 8000 --log-level info
```

Wait for all these startup messages before using the app:

```
AMOP starting — env=development
SQLite ready
Qdrant collection ready: work_orders
Loading embedding model: BAAI/bge-large-en-v1.5 (device=cuda)
Embedding model ready — dim=1024, device=cuda
BM25 index loaded — vocab: 150 tokens
Retriever ready — mode=hybrid
Application startup complete.
```

**Note:** First startup takes ~10 seconds to load the embedding model onto GPU.
Subsequent startups are faster (model is cached).

API docs available at: http://localhost:8000/docs

---

## Step 5 — Start Frontend

Open a new terminal window:

```powershell
cd C:\Final_Capstone_Project\amop_day1\amop\frontend

# Install dependencies (first time only)
npm install

# Start dev server
npm run dev
```

Frontend available at: http://localhost:5173

---

## Step 6 — Run Ingestion (Only If Qdrant Is Empty)

Only run this if Step 3 shows 0 points. Ingest via the API endpoint (backend must be running):

```powershell
# Ingest the augmented MWO dataset via API
curl -X POST http://localhost:8000/api/v1/ingest `
  -H "X-API-Key: amop-dev-secret" `
  -F "file=@dataset_files/mwo_dataset_augmented.csv"
```

Expected response: `"20000 new work orders (0 duplicates skipped)"`

After ingestion, verify Qdrant point count:
```powershell
python -c "
from backend.clients.vector_store import get_qdrant_client
from backend.core.config import get_settings
client = get_qdrant_client()
info = client.get_collection(get_settings().qdrant_collection)
print('Points:', info.points_count)
"
```

Expected: `Points: 20000`

---

## Step 7 — Test the API

Open http://localhost:8000/docs in your browser.

Use the `/api/v1/query` endpoint with this body:

```json
{
  "query": "What caused the hydraulic line leak?",
  "machine_id": "A6",
  "include_prediction": false,
  "include_routing": true
}
```

**Important:** Machine IDs in the dataset are A1 through A99.
Do NOT use M001, integers, or any other format.

---

## Stopping Everything

```powershell
# Stop backend — press Ctrl+C in the backend terminal

# Stop frontend — press Ctrl+C in the frontend terminal

# Stop Qdrant
docker stop qdrant
```

---

## Restarting After a Reboot

After restarting your PC, run these in order:

```powershell
# 1. Start Docker Desktop (from taskbar or Start menu)
#    Wait for Docker to fully start before continuing

# 2. Start Qdrant
docker start qdrant

# 3. Navigate to project
cd C:\Final_Capstone_Project\amop_day1\amop

# 4. Activate venv
C:\Final_Capstone_Project\venv\Scripts\Activate.ps1

# 5. Start backend
uvicorn backend.api.main:app --port 8000 --log-level info

# 6. Start frontend (new terminal)
cd C:\Final_Capstone_Project\amop_day1\amop\frontend
npm run dev
```

---

## Wipe Qdrant and Re-Ingest (Nuclear Option)

Only do this if you need a completely clean start:

```powershell
# Delete the collection
python -c "
from backend.clients.vector_store import get_qdrant_client
from backend.core.config import get_settings
client = get_qdrant_client()
settings = get_settings()
client.delete_collection(settings.qdrant_collection)
print('Deleted:', settings.qdrant_collection)
"

# Re-ingest
python -m pipelines.ingest_work_orders --csv data/mwo_dataset.csv
```

---

## Health Check

```powershell
# Quick health check via Python
python -c "
from backend.api.routes.health import health_check
import asyncio
result = asyncio.run(health_check())
print(result)
"

# Or via curl after backend is running
curl http://localhost:8000/api/v1/health
```

Expected: `status='healthy'`

---

## Check GPU Is Being Used

```powershell
# Verify CUDA is available
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected:
```
CUDA: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

If CUDA is False, reinstall torch with CUDA support:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Environment Variables

All settings are in `.env` at the project root.
Copy `.env.example` to `.env` if it doesn't exist:

```powershell
Copy-Item .env.example .env
```

Key variables:
```env
OPENROUTER_API_KEY=your-key-from-openrouter.ai
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=work_orders
AMOP_API_KEY=          # leave blank to disable auth
PDM_DATA_DIR=./data/pdm
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `Connection refused port 6333` | Qdrant not running | `docker start qdrant` |
| `Points: 0` | Qdrant empty | Run ingestion (Step 6) |
| `ModuleNotFoundError` | Wrong directory or venv not active | `cd` to project root, activate venv |
| `CUDA: False` | CPU-only torch installed | Reinstall with cu128 (see above) |
| `openrouter_api_key missing` | `.env` not set up | Add `OPENROUTER_API_KEY` to `.env` |
| Backend stuck on startup | PdM data loading | Set `PDM_DATA_DIR=` in `.env` to skip |
| `500 Internal Server Error` | Check backend terminal | Look for traceback in uvicorn output |

---

## Service URLs Summary

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Qdrant Dashboard | http://localhost:6333/dashboard |