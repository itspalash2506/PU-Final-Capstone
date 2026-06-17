import sys

from dotenv import load_dotenv


sys.path.insert(0, ".")
load_dotenv()


def run_check(label, check_func):
    print(label)
    try:
        check_func()
    except Exception as exc:
        print(f"    FAIL - {exc.__class__.__name__}: {exc}")


def test_pdm_predictor():
    from backend.prediction.pdm_predictor import predict_failure

    result = predict_failure(1)
    if not result:
        print("    FAIL - returned None")
        return

    top_features = result.get("top_features") or []
    top_feature_name = top_features[0].get("name", "N/A") if top_features else "N/A"

    print(f"    Machine 1 failure probability: {result.get('failure_probability')}")
    print(f"    Risk tier: {result.get('risk_tier')}")
    print(f"    Top feature: {top_feature_name}")
    print("    PASS")


def test_semantic_bridge():
    from backend.prediction.pdm_semantic_bridge import build_profiles_if_needed

    built = build_profiles_if_needed()
    print(f"    Profiles built: {built}")
    print("    PASS")


def test_find_similar_pdm_machine():
    from backend.prediction.pdm_semantic_bridge import find_similar_pdm_machine

    matches = find_similar_pdm_machine(
        "hydraulic leak pressure buildup seal failure",
        top_k=1,
    )

    if not matches:
        print("    FAIL - no matches returned")
        return

    match = matches[0]
    risk_data = match.get("risk_data") or {}

    print(f"    Best match: PdM machine {match.get('machine_id')}")
    print(f"    Similarity: {match.get('similarity_score', 0):.4f}")
    print(f"    Match basis: {match.get('match_basis')}")

    if risk_data:
        print(f"    Risk tier: {risk_data.get('risk_tier')}")

    print("    PASS")


if __name__ == "__main__":
    run_check("[1] Testing pdm_predictor...", test_pdm_predictor)
    run_check("[2] Testing semantic bridge...", test_semantic_bridge)
    run_check("[3] Testing find_similar_pdm_machine...", test_find_similar_pdm_machine)
