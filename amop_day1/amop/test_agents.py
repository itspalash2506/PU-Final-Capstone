import sys, os, time
sys.path.insert(0, os.getcwd())
# Force UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from dotenv import load_dotenv
load_dotenv()


def _safe(text: str, limit: int = 120) -> str:
    """Truncate and replace non-ASCII for safe console output."""
    return text[:limit].encode('ascii', errors='replace').decode('ascii')

print("=" * 60)
print("AGENT SYSTEM TEST")
print("=" * 60)

# ── Test 1: Equipment Agent — alpha ID ────────────────────────────────────────
print("\n[1] Equipment Agent — alpha ID extraction (A6)...")
from backend.agents.equipment_agent import equipment_node
from backend.agents.state import AgentState

def make_state(**kwargs):
    defaults = dict(
        query="", machine_id=None, top_k=3, intent="rag",
        rag_result=None, prediction_result=None, rca_result=None,
        final_answer="", sources=[], latency_ms=0, agent_traces=[],
        alpha_machine_id=None, numeric_machine_id=None,
        id_assumption=False, machine_context={}, routing_result=None,
    )
    defaults.update(kwargs)
    return AgentState(**defaults)

state1 = make_state(query="machine A6 has a hydraulic leak")
r1 = equipment_node(state1)
print(f"    alpha_machine_id:   {r1.get('alpha_machine_id')}")
print(f"    numeric_machine_id: {r1.get('numeric_machine_id')}")
print(f"    id_assumption:      {r1.get('id_assumption')}")
print(f"    agent_traces:       {r1.get('agent_traces')}")
assert r1.get('alpha_machine_id') == 'A6', "FAIL: Alpha ID not extracted"
assert r1.get('numeric_machine_id') == 6,  "FAIL: Numeric ID not derived"
assert r1.get('id_assumption') == True,    "FAIL: id_assumption should be True"
assert len(r1.get('agent_traces', [])) > 0,"FAIL: No agent trace recorded"
print("    PASS")

# ── Test 2: Equipment Agent — numeric only ────────────────────────────────────
print("\n[2] Equipment Agent — numeric ID only (machine 14)...")
state2 = make_state(query="what is the failure risk for machine 14")
r2 = equipment_node(state2)
print(f"    alpha_machine_id:   {r2.get('alpha_machine_id')}")
print(f"    numeric_machine_id: {r2.get('numeric_machine_id')}")
assert r2.get('numeric_machine_id') == 14, "FAIL: Numeric ID not extracted"
assert r2.get('alpha_machine_id') is None, "FAIL: Alpha should be None"
assert r2.get('id_assumption') == False,   "FAIL: No assumption for numeric-only"
print("    PASS")

# ── Test 3: Equipment Agent — no ID ───────────────────────────────────────────
print("\n[3] Equipment Agent — no machine ID in query...")
state3 = make_state(query="what are the most common hydraulic failures?")
r3 = equipment_node(state3)
print(f"    alpha_machine_id:   {r3.get('alpha_machine_id')}")
print(f"    numeric_machine_id: {r3.get('numeric_machine_id')}")
assert r3.get('alpha_machine_id') is None,   "FAIL: Alpha should be None"
assert r3.get('numeric_machine_id') is None, "FAIL: Numeric should be None"
trace3 = r3.get('agent_traces', [{}])[-1]
assert trace3.get('status') == 'no_id_found', "FAIL: Wrong status"
print("    PASS")

# ── Test 4: Routing Agent ─────────────────────────────────────────────────────
print("\n[4] Routing Agent — machine A6 technician history...")
from backend.agents.routing_agent import routing_node
state4 = make_state(
    query="hydraulic leak on machine A6",
    alpha_machine_id="A6",
    numeric_machine_id=6,
    id_assumption=True,
    prediction_result={"risk_tier": "High", "failure_probability": 0.65},
)
r4 = routing_node(state4)
routing = r4.get('routing_result')
print(f"    routing_result: {routing}")
assert routing is not None, "FAIL: routing_result is None"
assert "recommended_technician" in routing, "FAIL: Missing recommended_technician"
assert "urgency" in routing, "FAIL: Missing urgency"
assert routing["urgency"] == "urgent", \
    f"FAIL: Expected urgent for High risk, got {routing['urgency']}"
print(f"    Technician : {routing['recommended_technician']}")
print(f"    Urgency    : {routing['urgency']} — {routing['urgency_detail']}")
print(f"    Basis      : {routing['assignment_basis']}")
print("    PASS")

# ── Test 5: Routing Agent — no machine ID ────────────────────────────────────
print("\n[5] Routing Agent — no machine ID (graceful degradation)...")
state5 = make_state(query="what are common failure causes?")
r5 = routing_node(state5)
routing5 = r5.get('routing_result')
assert routing5 is not None, "FAIL: Should return result even with no ID"
print(f"    Result: {routing5}")
print("    PASS")

# ── Test 6: Full pipeline — rag intent ───────────────────────────────────────
print("\n[6] Full agent graph — rag intent with machine A6...")
from fastapi.testclient import TestClient
from backend.api.main import app
client = TestClient(app)
HEADERS = {"X-API-Key": "amop-dev-secret"}

t0 = time.perf_counter()
resp = client.post(
    "/api/v1/query",
    json={"query": "hydraulic leak on machine A6", "top_k": 3},
    headers=HEADERS,
)
ms = int((time.perf_counter() - t0) * 1000)
assert resp.status_code == 200, f"FAIL: {resp.text}"
data = resp.json()
print(f"    Status  : {resp.status_code}")
print(f"    Intent  : {data.get('intent')}")
print(f"    Latency : {ms}ms")
print(f"    Answer  : {_safe(data.get('answer', ''))}...")
traces = data.get('agent_traces', [])
print(f"    Agent traces ({len(traces)}):")
for t in traces:
    name = t.get('agent_name', '?')
    status = t.get('status', '?')
    lat = t.get('latency_ms', '?')
    print(f"      {name:<15} {status:<15} {lat}ms")
routing_in_resp = data.get('routing_result')
print(f"    Routing result in response: {routing_in_resp is not None}")
assert len(traces) >= 3, "FAIL: Expected at least 3 agent traces"
print("    PASS")

# ── Test 7: Full pipeline — predict intent ────────────────────────────────────
print("\n[7] Full agent graph — predict intent with machine 14...")
resp2 = client.post(
    "/api/v1/query",
    json={"query": "what is the failure risk for machine 14", "top_k": 3},
    headers=HEADERS,
)
assert resp2.status_code == 200, f"FAIL: {resp2.text}"
data2 = resp2.json()
print(f"    Intent  : {data2.get('intent')}")
print(f"    Answer  : {_safe(data2.get('answer', ''))}...")
traces2 = data2.get('agent_traces', [])
print(f"    Agent traces ({len(traces2)}):")
for t in traces2:
    print(f"      {t.get('agent_name','?'):<15} {t.get('status','?')}")
print("    PASS")

# ── Test 8: Full pipeline — no machine ID ────────────────────────────────────
print("\n[8] Full agent graph — no machine ID (general query)...")
resp3 = client.post(
    "/api/v1/query",
    json={"query": "what are the most common hydraulic failures?", "top_k": 3},
    headers=HEADERS,
)
assert resp3.status_code == 200, f"FAIL: {resp3.text}"
data3 = resp3.json()
print(f"    Intent  : {data3.get('intent')}")
print(f"    Answer  : {_safe(data3.get('answer', ''))}...")
print("    PASS")

print("\n" + "=" * 60)
print("ALL AGENT TESTS COMPLETE")
print("=" * 60)
