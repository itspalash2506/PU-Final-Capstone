"""
Semantic bridge between work-order text and PdM machine profiles.

The work-order dataset uses alpha machine IDs (e.g. "A6", "M-42") while the
PdM telemetry dataset uses numeric IDs (1-100).  A hard join is impossible
without a mapping table that does not exist.

This module resolves the mismatch by:
  1. Building a natural-language "profile" for each of the 100 PdM machines
     from their telemetry statistics, failure history, and metadata.
  2. Embedding all 100 profiles with the existing sentence-transformer encoder
     and storing them in a dedicated Qdrant collection ("pdm_profiles").
  3. Exposing find_similar_pdm_machine() which embeds an arbitrary work-order
     text and returns the closest PdM machines by cosine similarity.

Because the match is similarity-based (not a lookup), every result carries a
"match_basis": "symptom_similarity" field so callers know exactly what kind
of join was performed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from backend.clients.embedding_client import encode
from backend.clients.vector_store import get_qdrant_client
from backend.prediction.pdm_features import _load_raw_data
from backend.prediction.pdm_predictor import predict_failure

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PDM_PROFILES_COLLECTION = "pdm_profiles"
_EMBEDDING_DIM = 384          # all-MiniLM-L6-v2 output dimension
_N_MACHINES    = 100          # total machines in the PdM dataset


# ── Internal helpers ───────────────────────────────────────────────────────────

def _build_machine_statistics(
    telemetry: pd.DataFrame,
    failures:  pd.DataFrame,
    errors:    pd.DataFrame,
    machines:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute summary statistics for each of the 100 PdM machines.

    Returns a DataFrame indexed by machineID with columns:
        mean_volt, mean_rotate, mean_pressure, mean_vibration
        std_volt, std_rotate, std_pressure, std_vibration
        failure_count, most_common_failure
        avg_error_rate_per_day
        age, model
    """
    # Telemetry statistics per machine
    tel_stats = (
        telemetry
        .groupby("machineID")[["volt", "rotate", "pressure", "vibration"]]
        .agg(["mean", "std"])
    )
    tel_stats.columns = ["_".join(c) for c in tel_stats.columns]

    # Failure statistics
    failure_counts = (
        failures
        .groupby("machineID")
        .size()
        .rename("failure_count")
    )
    most_common_failure = (
        failures
        .groupby("machineID")["failure"]
        .agg(lambda s: s.mode().iloc[0] if len(s) > 0 else "none")
        .rename("most_common_failure")
    )

    # Error rate: total errors / observation days per machine
    telemetry_days = (
        telemetry
        .groupby("machineID")["datetime"]
        .agg(lambda s: (s.max() - s.min()).days + 1)
        .rename("observation_days")
    )
    error_counts = errors.groupby("machineID").size().rename("total_errors")

    error_rate = (error_counts / telemetry_days).rename("avg_error_rate_per_day").fillna(0)

    # Combine everything
    stats = tel_stats.join(failure_counts, how="left")
    stats = stats.join(most_common_failure, how="left")
    stats = stats.join(error_rate, how="left")
    stats = stats.join(machines.set_index("machineID")[["model", "age"]], how="left")

    stats["failure_count"]         = stats["failure_count"].fillna(0).astype(int)
    stats["most_common_failure"]   = stats["most_common_failure"].fillna("none")
    stats["avg_error_rate_per_day"] = stats["avg_error_rate_per_day"].fillna(0.0)

    # Ensure std columns default to 0 where std is NaN (single-point machines)
    for col in stats.columns:
        if "std" in col:
            stats[col] = stats[col].fillna(0.0)

    return stats


def _generate_profile_text(machine_id: int, stats: pd.Series) -> str:
    """
    Convert a row of machine statistics into a natural-language description.

    The description is designed to encode the same physical signals that a
    technician would use when writing a work order — so that symptom-similarity
    search across the two datasets makes semantic sense.

    Example output:
        "Machine 6 (model2, age 15): average vibration 45.2 (std 12.3),
         average pressure 98.1 (std 8.4), voltage mean 170.2 (std 14.1),
         rotation mean 452.3 (std 67.2). Failure history: 3 failures,
         most common: comp2. Error rate: 0.8 errors/day."
    """
    vib_mean = stats.get("vibration_mean", 0.0)
    vib_std  = stats.get("vibration_std",  0.0)
    pre_mean = stats.get("pressure_mean",  0.0)
    pre_std  = stats.get("pressure_std",   0.0)
    vlt_mean = stats.get("volt_mean",      0.0)
    vlt_std  = stats.get("volt_std",       0.0)
    rot_mean = stats.get("rotate_mean",    0.0)
    rot_std  = stats.get("rotate_std",     0.0)

    fail_count   = int(stats.get("failure_count",          0))
    most_common  = str(stats.get("most_common_failure",   "none"))
    error_rate   = float(stats.get("avg_error_rate_per_day", 0.0))
    model_name   = str(stats.get("model", "unknown"))
    age          = int(stats.get("age",   0))

    # Describe vibration variability qualitatively
    if vib_std > 15:
        vib_qualifier = "high variance"
    elif vib_std > 7:
        vib_qualifier = "moderate variance"
    else:
        vib_qualifier = "stable"

    return (
        f"Machine {machine_id} ({model_name}, age {age}): "
        f"average vibration {vib_mean:.1f} ({vib_qualifier}, std {vib_std:.1f}), "
        f"average pressure {pre_mean:.1f} (std {pre_std:.1f}), "
        f"voltage mean {vlt_mean:.1f} (std {vlt_std:.1f}), "
        f"rotation mean {rot_mean:.1f} (std {rot_std:.1f}). "
        f"Failure history: {fail_count} failures, most common: {most_common}. "
        f"Error rate: {error_rate:.2f} errors/day."
    )


def _ensure_pdm_profiles_collection(client: QdrantClient) -> None:
    """
    Create the 'pdm_profiles' Qdrant collection if it does not already exist.

    Uses cosine distance to match the embedding model's training objective.
    Idempotent — safe to call multiple times.
    """
    existing = {c.name for c in client.get_collections().collections}
    if PDM_PROFILES_COLLECTION not in existing:
        client.create_collection(
            collection_name = PDM_PROFILES_COLLECTION,
            vectors_config  = VectorParams(
                size     = _EMBEDDING_DIM,
                distance = Distance.COSINE,
            ),
        )
        logger.info(
            "Created Qdrant collection",
            extra={"collection": PDM_PROFILES_COLLECTION},
        )


# ── Public API ────────────────────────────────────────────────────────────────

def build_and_store_profiles() -> int:
    """
    Build natural-language profiles for all 100 PdM machines, embed them, and
    upsert the resulting vectors into the 'pdm_profiles' Qdrant collection.

    Idempotent: upserting the same point ID (machineID) multiple times is safe —
    Qdrant replaces existing points rather than duplicating them.

    Returns:
        Number of machine profiles successfully stored.
    """
    telemetry, failures, errors, _maint, machines = _load_raw_data()

    logger.info("Building machine profile statistics for 100 machines...")
    stats_df = _build_machine_statistics(telemetry, failures, errors, machines)

    # Generate profile texts for all machines
    machine_ids = sorted(stats_df.index.tolist())
    profile_texts = []
    for machine_id in machine_ids:
        row  = stats_df.loc[machine_id]
        text = _generate_profile_text(machine_id, row)
        profile_texts.append(text)

    # Batch-embed all 100 profiles in one encoder call
    logger.info("Embedding 100 machine profiles...")
    embeddings = encode(profile_texts)  # list[list[float]], length 100

    # Build Qdrant point structs
    client = get_qdrant_client()
    _ensure_pdm_profiles_collection(client)

    points: list[PointStruct] = []
    for i, machine_id in enumerate(machine_ids):
        row = stats_df.loc[machine_id]
        points.append(
            PointStruct(
                id      = int(machine_id),
                vector  = embeddings[i],
                payload = {
                    "machine_id":           int(machine_id),
                    "model":                str(row.get("model", "unknown")),
                    "age":                  int(row.get("age", 0)),
                    "failure_count":        int(row.get("failure_count", 0)),
                    "most_common_failure":  str(row.get("most_common_failure", "none")),
                    "avg_vibration":        round(float(row.get("vibration_mean", 0.0)), 2),
                    "avg_pressure":         round(float(row.get("pressure_mean",  0.0)), 2),
                    "profile_text":         profile_texts[i],
                },
            )
        )

    client.upsert(collection_name=PDM_PROFILES_COLLECTION, points=points)
    logger.info(
        "PdM machine profiles stored in Qdrant",
        extra={"collection": PDM_PROFILES_COLLECTION, "count": len(points)},
    )
    return len(points)


def find_similar_pdm_machine(
    work_order_text: str,
    top_k: int = 1,
) -> list[dict]:
    """
    Find PdM machines whose telemetry profiles most closely match the symptoms
    described in a work-order text.

    The match is based on semantic (embedding) similarity — NOT a hard join on
    machine ID.  Every result carries "match_basis": "symptom_similarity" to
    make this explicit to callers.

    Args:
        work_order_text: Free-text description from a work order (e.g. the
                         issue description or technician notes).
        top_k:           Number of similar machines to return (default 1).

    Returns:
        List of dicts, each containing:
            machine_id       (int)
            similarity_score (float, cosine similarity 0-1)
            profile_text     (str, the stored NL profile)
            risk_data        (dict from predict_failure(), or None if untrained)
            match_basis      ("symptom_similarity")
    """
    if not work_order_text or not work_order_text.strip():
        return []

    query_embedding = encode([work_order_text.strip()])[0]

    client = get_qdrant_client()

    try:
        results = client.query_points(
            collection_name = PDM_PROFILES_COLLECTION,
            query           = query_embedding,
            limit           = top_k,
            with_payload    = True,
        )
        scored_points = results.points
    except Exception as exc:
        logger.warning(
            "Qdrant search on pdm_profiles failed",
            extra={"error": str(exc)},
        )
        return []

    output: list[dict] = []
    for point in scored_points:
        machine_id   = int(point.payload.get("machine_id", 0))
        risk_data    = None

        try:
            risk_data = predict_failure(machine_id)
        except Exception as exc:
            logger.warning(
                "predict_failure failed for similar machine",
                extra={"machine_id": machine_id, "error": str(exc)},
            )

        output.append({
            "machine_id":       machine_id,
            "similarity_score": round(float(point.score), 4),
            "profile_text":     str(point.payload.get("profile_text", "")),
            "risk_data":        risk_data,
            "match_basis":      "symptom_similarity",
        })

    return output


def build_profiles_if_needed() -> bool:
    """
    Check whether the 'pdm_profiles' collection already has 100 points.

    If it does, return False (no work needed).
    If not (collection missing or fewer than 100 points), call
    build_and_store_profiles() and return True.

    Called at backend startup so profiles are always ready for similarity
    search without requiring a manual training step.

    Returns:
        True if profiles were (re-)built, False if they were already present.
    """
    client = get_qdrant_client()

    try:
        existing = {c.name for c in client.get_collections().collections}
        if PDM_PROFILES_COLLECTION in existing:
            info = client.get_collection(PDM_PROFILES_COLLECTION)
            if (info.points_count or 0) >= _N_MACHINES:
                logger.info(
                    "PdM profiles already present — skipping rebuild",
                    extra={
                        "collection":   PDM_PROFILES_COLLECTION,
                        "points_count": info.points_count,
                    },
                )
                return False
    except Exception as exc:
        logger.warning(
            "Could not query pdm_profiles collection count — will rebuild",
            extra={"error": str(exc)},
        )

    build_and_store_profiles()
    return True
