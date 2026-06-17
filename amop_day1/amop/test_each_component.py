"""
AMOP Component Test Script
Run from project root: python test_components.py
"""

import os
import sys
import json
import time

# ── make imports work from project root ──────────────────────────────────────
sys.path.insert(0, os.getcwd())

# Load .env manually before importing anything
from dotenv import load_dotenv
load_dotenv()

results = {}

print("=" * 60)
print("AMOP COMPONENT TEST")
print("=" * 60)

# ── Test 1: Config ────────────────────────────────────────────────────────────
print("\n[1] Config loading...")
try:
    from backend.core.config import get_settings
    s = get_settings()
    print(f"    OPENROUTER_MODEL : {s.openrouter_model}")
    print(f"    QDRANT_HOST      : {s.qdrant_host}:{s.qdrant_port}")
    print(f"    SQLITE_PATH      : {s.sqlite_path}")
    print(f"    BM25_INDEX_PATH  : {s.bm25_index_path}")
    print(f"    MODEL_PATH       : {s.model_path}")
    results["config"] = "PASS"
except Exception as e:
    print(f"    FAIL: {e}")
    results["config"] = f"FAIL: {e}"

# ── Test 2: SQLite ────────────────────────────────────────────────────────────
print("\n[2] SQLite connection + schema...")
try:
    from backend.db.session import init_db, SessionLocal
    from backend.db.models import WorkOrder, QueryLog, Feedback
    init_db()
    db = SessionLocal()
    count = db.query(WorkOrder).count()
    db.close()
    print(f"    Work orders in DB : {count}")
    results["sqlite"] = "PASS"
except Exception as e:
    print(f"    FAIL: {e}")
    results["sqlite"] = f"FAIL: {e}"

# ── Test 3: Qdrant ────────────────────────────────────────────────────────────
print("\n[3] Qdrant connection...")
try:
    from backend.clients.vector_store import get_qdrant_client
    from backend.core.config import get_settings
    client = get_qdrant_client()
    collections = client.get_collections()
    names = [c.name for c in collections.collections]
    print(f"    Collections : {names}")
    settings = get_settings()
    if settings.qdrant_collection in names:
        info = client.get_collection(settings.qdrant_collection)
        print(f"    Points in {settings.qdrant_collection}: {info.points_count}")
    results["qdrant"] = "PASS"
except Exception as e:
    print(f"    FAIL: {e}")
    results["qdrant"] = f"FAIL: {e}"

# ── Test 4: Embedding model ───────────────────────────────────────────────────
print("\n[4] Embedding model (sentence-transformers)...")
try:
    from backend.clients.embedding_client import encode
    t0 = time.perf_counter()
    vecs = encode(["hydraulic leak on machine A6", "bearing failure noise"])
    ms = int((time.perf_counter() - t0) * 1000)
    print(f"    Vector dim  : {len(vecs[0])}")
    print(f"    Latency     : {ms}ms for 2 texts")
    results["embeddings"] = "PASS"
except Exception as e:
    print(f"    FAIL: {e}")
    results["embeddings"] = f"FAIL: {e}"

# ── Test 5: BM25 index ────────────────────────────────────────────────────────
print("\n[5] BM25 index...")
try:
    from backend.rag.retriever import _load_bm25
    bundle = _load_bm25()
    if bundle:
        print(f"    Documents indexed : {len(bundle['doc_ids'])}")
        scores = bundle["index"].get_scores("hydraulic leak".split())
        top_score = max(scores)
        print(f"    Top BM25 score for 'hydraulic leak': {top_score:.4f}")
        results["bm25"] = "PASS"
    else:
        print("    NOT BUILT YET — run /api/v1/ingest first")
        results["bm25"] = "NOT_BUILT"
except Exception as e:
    print(f"    FAIL: {e}")
    results["bm25"] = f"FAIL: {e}"

# ── Test 6: Full hybrid retrieval ─────────────────────────────────────────────
print("\n[6] Hybrid retrieval (BM25 + Qdrant + RRF)...")
try:
    from backend.rag.retriever import retrieve
    t0 = time.perf_counter()
    docs = retrieve("hydraulic leak pressure buildup", top_k=3)
    ms = int((time.perf_counter() - t0) * 1000)
    print(f"    Results returned : {len(docs)}")
    print(f"    Latency          : {ms}ms")
    for i, d in enumerate(docs, 1):
        print(f"    [{i}] machine={d['machine_id']} score={d['score']:.4f}")
        print(f"         issue: {d['issue_description'][:60]}...")
    results["retrieval"] = "PASS" if docs else "PASS_BUT_EMPTY"
except Exception as e:
    print(f"    FAIL: {e}")
    results["retrieval"] = f"FAIL: {e}"

# ── Test 7: OpenRouter LLM ────────────────────────────────────────────────────
print("\n[7] OpenRouter LLM (small test call)...")
try:
    from backend.clients.openrouter import get_openrouter_client
    client = get_openrouter_client()
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=get_settings().openrouter_model,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=5,
        temperature=0,
    )
    ms = int((time.perf_counter() - t0) * 1000)
    reply = response.choices[0].message.content.strip()
    print(f"    Response : {reply}")
    print(f"    Latency  : {ms}ms")
    results["llm"] = "PASS"
except Exception as e:
    print(f"    FAIL: {e}")
    results["llm"] = f"FAIL: {e}"

# ── Test 8: Existing prediction model ─────────────────────────────────────────
print("\n[8] Existing prediction model (keyword-based)...")
try:
    from backend.prediction.predictor import predict
    t0 = time.perf_counter()
    result = predict(
        machine_id="A6",
        issue_description="severe hydraulic leak, pressure loss on main line",
        technician_notes="Seal completely failed, replacement needed",
    )
    ms = int((time.perf_counter() - t0) * 1000)
    print(f"    Severity   : {result.get('severity')}")
    print(f"    Confidence : {result.get('severity_confidence')}")
    print(f"    ETA hours  : {result.get('eta_hours')}")
    print(f"    Latency    : {ms}ms")
    results["prediction_existing"] = "PASS"
except Exception as e:
    print(f"    FAIL: {e}")
    results["prediction_existing"] = f"FAIL: {e}"

# ── Test 9: PdM dataset files ─────────────────────────────────────────────────
print("\n[9] PdM dataset files (checking availability)...")
pdm_files = [
    "PdM_telemetry.csv",
    "PdM_failures.csv",
    "PdM_errors.csv",
    "PdM_maint.csv",
    "PdM_machines.csv",
]
pdm_found = []
pdm_missing = []

# Check common locations
search_paths = [
    ".",
    "data",
    "datasets",
    "../datasets",
    "../../datasets",
]

for fname in pdm_files:
    found = False
    for sp in search_paths:
        full = os.path.join(sp, fname)
        if os.path.exists(full):
            size_mb = os.path.getsize(full) / (1024 * 1024)
            print(f"    FOUND  {fname} ({size_mb:.1f} MB) at {full}")
            pdm_found.append(fname)
            found = True
            break
    if not found:
        print(f"    MISSING {fname}")
        pdm_missing.append(fname)

results["pdm_files"] = f"{len(pdm_found)}/5 found"

# ── Test 10: Agent graph ──────────────────────────────────────────────────────
print("\n[10] Agent graph (dry run, no LLM)...")
try:
    from backend.agents.graph import run_agent_graph
    print("    Graph imported successfully")
    print("    NOTE: Full run skipped (would call LLM)")
    print("    To do a full run: uncomment the block below")
    # t0 = time.perf_counter()
    # result = run_agent_graph("what issues has machine A6 had?", top_k=3)
    # ms = int((time.perf_counter() - t0) * 1000)
    # print(f"    Intent   : {result['intent']}")
    # print(f"    Answer   : {result['answer'][:100]}...")
    # print(f"    Latency  : {ms}ms")
    results["agent_graph"] = "IMPORT_OK"
except Exception as e:
    print(f"    FAIL: {e}")
    results["agent_graph"] = f"FAIL: {e}"

# ── Test 11: Feedback loop ────────────────────────────────────────────────────
print("\n[11] Feedback loop (Qdrant effectiveness_score update)...")
try:
    from backend.clients.vector_store import get_qdrant_client
    from backend.core.config import get_settings
    client = get_qdrant_client()
    settings = get_settings()

    # Check if set_payload method exists and collection has effectiveness_score
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in collections:
        # Try to fetch one point and check its payload
        points = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1,
            with_payload=True,
        )[0]
        if points:
            payload = points[0].payload
            has_score = "effectiveness_score" in (payload or {})
            print(f"    effectiveness_score in payload: {has_score}")
            if not has_score:
                print("    MISSING — feedback loop not wired to Qdrant")
                results["feedback_loop"] = "NOT_WIRED"
            else:
                print(f"    Current value: {payload.get('effectiveness_score')}")
                results["feedback_loop"] = "PASS"
        else:
            print("    Collection empty — ingest data first")
            results["feedback_loop"] = "EMPTY"
    else:
        print("    Collection not found — ingest data first")
        results["feedback_loop"] = "NO_COLLECTION"
except Exception as e:
    print(f"    FAIL: {e}")
    results["feedback_loop"] = f"FAIL: {e}"

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for component, status in results.items():
    icon = "✓" if "PASS" in status or "OK" in status else "✗" if "FAIL" in status else "~"
    print(f"  {icon}  {component:<25} {status}")

print("\nPaste the output above back to Claude.")
print("=" * 60)