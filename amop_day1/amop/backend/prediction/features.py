"""
Feature engineering for work order severity and ETA prediction.

All features are interpretable keyword/numeric indicators — no TF-IDF vectorizer
needed, which keeps the saved model bundle self-contained.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

# ── Severity keyword sets ─────────────────────────────────────────────────────

_HIGH_KW = {
    "fire", "explosion", "emergency", "critical", "safety hazard", "unsafe",
    "severe", "major breakdown", "complete failure", "total failure", "urgent",
    "immediate", "fault", "catastrophic", "imminent", "collapse",
    # Real maintenance events that indicate production-stopping failures
    "failure", "failed", "seized", "seized up", "machine down", "system down",
    "production stopped", "production halt", "shut down", "shutdown",
    "broken shaft", "burned", "burnt", "smoke", "overloaded",
}
_MEDIUM_KW = {
    "leak", "pressure", "overheating", "overheat", "vibration", "vibrating",
    "malfunction", "error", "broken", "damaged", "damage", "unusual", "noise",
    "not working", "worn", "stuck", "jam", "misalign", "loose", "cracked", "crack",
    "intermittent", "slow", "reduced", "erratic", "inconsistent",
}
_LOW_KW = {
    "routine", "inspection", "scheduled", "preventive", "preventative",
    "maintenance", "service", "check", "calibrat", "lubrication", "adjust",
    "minor", "normal", "review", "cleaning", "replace filter", "oil change",
}

# ── Component keyword sets ────────────────────────────────────────────────────

_HYDRAULIC_KW = {"hydraulic", "oil", "fluid", "coolant", "pump"}
_ELECTRICAL_KW = {"electri", "wiring", "wire", "short circuit", "overload", "fuse", "relay", "sensor"}
_MECHANICAL_KW = {"gear", "bearing", "shaft", "motor", "belt", "chain", "conveyor", "pulley", "roller"}
_THERMAL_KW = {"heat", "hot", "cool", "temperatur", "overheat", "thermal"}
_NOISE_KW = {"noise", "vibrat", "rattle", "squeak", "grind", "knock", "bang"}

FEATURE_NAMES = [
    "issue_len", "notes_len", "total_words", "has_notes",
    "has_high_kw", "has_medium_kw", "has_low_kw",
    "has_leak", "has_pressure", "has_hydraulic",
    "has_electrical", "has_mechanical",
    "has_thermal", "has_noise",
    "day_of_week", "month",
    "machine_prefix_ord",
]


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(kw in text for kw in keywords)


def extract_features(
    issue: str,
    notes: Optional[str],
    machine_id: str = "",
    created_at: Optional[datetime] = None,
) -> list[float]:
    """
    Return a fixed-length feature vector in FEATURE_NAMES order.
    """
    issue_lo = issue.lower()
    notes_lo = (notes or "").lower()
    combined = f"{issue_lo} {notes_lo}"

    dt = created_at or datetime.utcnow()
    machine_prefix = machine_id[0].upper() if machine_id else "A"
    prefix_ord = max(0, ord(machine_prefix) - ord("A"))

    vec = [
        float(len(issue)),
        float(len(notes or "")),
        float(len(combined.split())),
        float(bool(notes and len(notes.strip()) > 15)),
        float(_contains_any(issue_lo, _HIGH_KW)),
        float(_contains_any(issue_lo, _MEDIUM_KW)),
        float(_contains_any(issue_lo, _LOW_KW)),
        float("leak" in combined),
        float("pressure" in combined or "psi" in combined),
        float(_contains_any(combined, _HYDRAULIC_KW)),
        float(_contains_any(combined, _ELECTRICAL_KW)),
        float(_contains_any(combined, _MECHANICAL_KW)),
        float(_contains_any(combined, _THERMAL_KW)),
        float(_contains_any(combined, _NOISE_KW)),
        float(dt.weekday()),
        float(dt.month),
        float(prefix_ord),
    ]
    return vec


def active_feature_names(vec: list[float]) -> list[str]:
    """Return names of features that are non-zero (i.e. the active indicators)."""
    return [name for name, val in zip(FEATURE_NAMES, vec) if val > 0.0]


# ── Label generation (synthetic) ─────────────────────────────────────────────

def assign_severity(issue: str) -> str:
    """Derive severity label from issue description text."""
    lo = issue.lower()
    if _contains_any(lo, _HIGH_KW):
        return "high"
    if _contains_any(lo, _MEDIUM_KW):
        return "medium"
    return "low"


def assign_eta(severity: str, total_words: int) -> float:
    """Deterministic ETA estimate (hours) from severity + description complexity."""
    base = {"high": 24.0, "medium": 6.0, "low": 1.0}[severity]
    if total_words > 20:
        base *= 1.3
    elif total_words < 5:
        base *= 0.7
    return round(base, 1)
