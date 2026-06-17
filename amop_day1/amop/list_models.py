"""
Run from amop/ to list all currently available FREE models on OpenRouter.
Usage: python list_models.py
"""
import httpx
import os
from dotenv import load_dotenv

load_dotenv(".env")
api_key = os.getenv("OPENROUTER_API_KEY", "")

resp = httpx.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=15,
)
resp.raise_for_status()

models = resp.json().get("data", [])
free = [
    m for m in models
    if m.get("pricing", {}).get("prompt") in ("0", 0, "0.0", 0.0)
]

print(f"\n{'ID':<55} {'Context':>8}")
print("-" * 65)
for m in sorted(free, key=lambda x: x["id"]):
    ctx = m.get("context_length", "?")
    print(f"{m['id']:<55} {str(ctx):>8}")

print(f"\n{len(free)} free models found.")
