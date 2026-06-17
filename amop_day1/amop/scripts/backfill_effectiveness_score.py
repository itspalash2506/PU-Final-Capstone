"""
One-time backfill: adds effectiveness_score=0.5 to all existing Qdrant
work_orders points that are missing this payload field.

Run once: python scripts/backfill_effectiveness_score.py
Safe to run multiple times — skips points that already have the field.
"""
import sys, os
sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
load_dotenv()

from backend.clients.vector_store import get_qdrant_client
from backend.core.config import get_settings

def backfill():
    client = get_qdrant_client()
    settings = get_settings()
    collection = settings.qdrant_collection

    batch_size = 100
    offset = None
    total_updated = 0
    total_skipped = 0

    print(f"Backfilling effectiveness_score in '{collection}'...")

    while True:
        results, next_offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
        )

        if not results:
            break

        to_update = []
        for point in results:
            payload = point.payload or {}
            if "effectiveness_score" not in payload:
                to_update.append(str(point.id))
            else:
                total_skipped += 1

        if to_update:
            client.set_payload(
                collection_name=collection,
                payload={"effectiveness_score": 0.5},
                points=to_update,
            )
            total_updated += len(to_update)

        print(f"  Batch: updated={len(to_update)}, "
              f"skipped={len(results)-len(to_update)}, "
              f"total_updated={total_updated}")

        if next_offset is None:
            break
        offset = next_offset

    print(f"\nBackfill complete.")
    print(f"  Updated : {total_updated} points")
    print(f"  Skipped : {total_skipped} points (already had score)")

if __name__ == "__main__":
    backfill()
