import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

print('[1] Loading cross-encoder model...')
from backend.rag.reranker import get_reranker
model = get_reranker()
print(f'    Model loaded: {model}')
print('    PASS')

print('[2] Testing rerank() on sample docs...')
from backend.rag.reranker import rerank

sample_docs = [
    {"id": "1", "machine_id": "A6", "issue_description": "hydraulic leak on side B",
     "technician_notes": "seal replacement required", "score": 0.016},
    {"id": "2", "machine_id": "A10", "issue_description": "pressure loss in hydraulic line",
     "technician_notes": "pump inspected", "score": 0.015},
    {"id": "3", "machine_id": "A44", "issue_description": "motor overheating issue",
     "technician_notes": "cooling fan replaced", "score": 0.014},
]

query = "hydraulic leak causing pressure loss"
reranked = rerank(query, sample_docs, top_k=3)

print(f'    Results returned: {len(reranked)}')
for i, d in enumerate(reranked, 1):
    print(f'    [{i}] machine={d["machine_id"]} rerank_score={d["rerank_score"]:.4f}')
    print(f'         issue: {d["issue_description"]}')
print('    PASS')

print('[3] Testing retrieve() with rerank=True...')
from backend.rag.retriever import retrieve
import time
t0 = time.perf_counter()
results = retrieve("hydraulic leak pressure buildup", top_k=3, rerank=True)
ms = int((time.perf_counter() - t0) * 1000)
print(f'    Results: {len(results)}')
print(f'    Latency: {ms}ms')
for i, d in enumerate(results, 1):
    print(f'    [{i}] machine={d["machine_id"]} score={d.get("rerank_score", d["score"]):.4f}')
print('    PASS')

print()
print('All reranker tests passed.')