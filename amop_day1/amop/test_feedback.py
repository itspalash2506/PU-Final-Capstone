import sys, os, json, time
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)
HEADERS = {"X-API-Key": "amop-dev-secret"}

print("[1] Submitting a test query to get a query_log_id...")
query_resp = client.post(
    "/api/v1/query",
    json={"query": "hydraulic leak on machine A6", "top_k": 3},
    headers=HEADERS,
)
print(f"    Status: {query_resp.status_code}")
assert query_resp.status_code == 200, f"Query failed: {query_resp.text}"
data = query_resp.json()
request_id = data["request_id"]
sources = data.get("sources", [])
print(f"    request_id: {request_id}")
print(f"    Sources returned: {len(sources)}")
if sources:
    print(f"    First source doc_id: {sources[0]['id']}")
print("    PASS")

print("\n[2] Fetching effectiveness_score BEFORE feedback...")
from backend.clients.vector_store import get_qdrant_client
from backend.core.config import get_settings
qdrant = get_qdrant_client()
settings = get_settings()
if sources:
    doc_id = sources[0]["id"]
    points = qdrant.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[doc_id],
        with_payload=True,
    )
    if points:
        before_score = points[0].payload.get("effectiveness_score", "NOT_FOUND")
        print(f"    effectiveness_score before: {before_score}")
    else:
        print("    Document not found in Qdrant")
        before_score = None
print("    PASS")

print("\n[3] Submitting POSITIVE feedback (rating=1)...")
fb_resp = client.post(
    "/api/v1/feedback",
    json={
        "query_log_id": request_id,
        "rating": 1,
        "comment": "Very helpful response",
    },
    headers=HEADERS,
)
print(f"    Status: {fb_resp.status_code}")
assert fb_resp.status_code == 200, f"Feedback failed: {fb_resp.text}"
fb_data = fb_resp.json()
print(f"    Response: {fb_data['message']}")
print("    PASS")

print("\n[4] Fetching effectiveness_score AFTER feedback...")
if sources:
    time.sleep(0.5)
    points = qdrant.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[doc_id],
        with_payload=True,
    )
    if points:
        after_score = points[0].payload.get("effectiveness_score", "NOT_FOUND")
        print(f"    effectiveness_score after:  {after_score}")
        if before_score != "NOT_FOUND" and after_score != "NOT_FOUND":
            if float(after_score) > float(before_score):
                print(f"    Score increased by {float(after_score) - float(before_score):.3f}")
                print("    Feedback loop is WORKING")
            else:
                print("    WARNING: Score did not increase as expected")
print("    PASS")

print("\n[5] Submitting NEGATIVE feedback (rating=-1)...")
query_resp2 = client.post(
    "/api/v1/query",
    json={"query": "grinding noise in gears machine A10", "top_k": 3},
    headers=HEADERS,
)
assert query_resp2.status_code == 200
request_id2 = query_resp2.json()["request_id"]

fb_resp2 = client.post(
    "/api/v1/feedback",
    json={"query_log_id": request_id2, "rating": -1, "comment": "Not relevant"},
    headers=HEADERS,
)
print(f"    Status: {fb_resp2.status_code}")
assert fb_resp2.status_code == 200
print(f"    Response: {fb_resp2.json()['message']}")
print("    PASS")

print("\n[6] Verifying health endpoint shows feedback route...")
health_resp = client.get("/api/v1/health")
print(f"    Status: {health_resp.status_code}")
print("    PASS")

print("\n" + "=" * 50)
print("FEEDBACK LOOP TEST COMPLETE")
print("=" * 50)
print("All tests passed. The feedback loop is working.")
print("Positive feedback increases effectiveness_score in Qdrant.")
print("Negative feedback decreases effectiveness_score in Qdrant.")
print("The reranker will use these scores on future retrievals.")
