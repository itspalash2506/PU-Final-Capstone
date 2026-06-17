"""
Feature engineering for the Azure PdM telemetry dataset.

Loads 5 CSV files, builds a time-series feature matrix with strict backward-looking
rolling windows to prevent data leakage, and provides per-machine feature extraction
for online inference.

Temporal split: data before TRAIN_CUTOFF = train, data from TRAIN_CUTOFF onward = test.
Random splits are never used — this is time-series data and the ordering matters.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Named constant so callers and trainer can reference the same cutoff without
# repeating a magic string.
TRAIN_CUTOFF = "2015-10-01"

# Derive paths relative to this file so they work regardless of cwd.
# This file is at backend/prediction/pdm_features.py
# parents[2] is the project root (amop/).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "dataset_files"

TELEMETRY_CSV = _DATA_DIR / "PdM_telemetry.csv"
FAILURES_CSV  = _DATA_DIR / "PdM_failures.csv"
ERRORS_CSV    = _DATA_DIR / "PdM_errors.csv"
MAINT_CSV     = _DATA_DIR / "PdM_maint.csv"
MACHINES_CSV  = _DATA_DIR / "PdM_machines.csv"

# Ordered list of feature columns — must match the order used during training.
FEATURE_COLS = [
    "volt_mean_24h",
    "volt_std_24h",
    "rotate_mean_24h",
    "rotate_std_24h",
    "pressure_mean_24h",
    "pressure_std_24h",
    "pressure_max_7d",
    "vibration_mean_24h",
    "vibration_std_24h",
    "vibration_std_7d",
    "volt_trend_24h",
    "vibration_trend_48h",
    "error_count_24h",
    "error_count_7d",
    "days_since_last_maint",
    "maint_count_30d",
    "machine_age",
    "machine_model",
]

TARGET_COL = "failure_within_24h"

# Label-encode machine model strings so XGBoost sees integers.
_MODEL_LABEL_MAP = {"model1": 0, "model2": 1, "model3": 2, "model4": 3}


# ── Data loading (cached for lifetime of process) ─────────────────────────────

@lru_cache(maxsize=1)
def _load_raw_data() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Load all five PdM CSV files and return them as DataFrames.

    Results are cached with lru_cache so subsequent calls (e.g. from
    get_latest_features during inference) pay no I/O cost.

    Returns:
        (telemetry, failures, errors, maint, machines)

    Raises:
        FileNotFoundError: with the exact expected path if any CSV is missing.
    """
    for path in [TELEMETRY_CSV, FAILURES_CSV, ERRORS_CSV, MAINT_CSV, MACHINES_CSV]:
        if not path.exists():
            raise FileNotFoundError(
                f"PdM dataset file not found.\n"
                f"  Expected: {path}\n"
                f"  Please place the Azure PdM CSV files in: {_DATA_DIR}"
            )

    logger.info("Loading PdM telemetry (~876K rows) — this happens once...")
    telemetry = pd.read_csv(TELEMETRY_CSV, parse_dates=["datetime"])
    failures  = pd.read_csv(FAILURES_CSV,  parse_dates=["datetime"])
    errors    = pd.read_csv(ERRORS_CSV,    parse_dates=["datetime"])
    maint     = pd.read_csv(MAINT_CSV,     parse_dates=["datetime"])
    machines  = pd.read_csv(MACHINES_CSV)

    logger.info(
        "PdM raw data loaded",
        extra={
            "telemetry_rows": len(telemetry),
            "failure_events": len(failures),
            "error_events":   len(errors),
            "maint_events":   len(maint),
            "machines":       len(machines),
        },
    )
    return telemetry, failures, errors, maint, machines


# ── Pre-processing helpers ────────────────────────────────────────────────────

def _resample_events_hourly(
    df: pd.DataFrame, count_col: str
) -> pd.DataFrame:
    """
    Floor event timestamps to the hour and count events per (machineID, hour).

    This converts a sparse event log (one row per event) into an hourly count
    series that can be left-joined onto the hourly telemetry DataFrame.

    Args:
        df:        DataFrame with columns [machineID, datetime, ...]
        count_col: Name for the output count column.

    Returns:
        DataFrame with columns [machineID, datetime, <count_col>].
    """
    tmp = df[["machineID", "datetime"]].copy()
    tmp["datetime"] = tmp["datetime"].dt.floor("h")
    return (
        tmp.groupby(["machineID", "datetime"])
        .size()
        .reset_index(name=count_col)
    )


def _create_failure_labels(
    tel: pd.DataFrame, failures: pd.DataFrame
) -> np.ndarray:
    """
    Create binary label: 1 if a failure occurs in the window (T, T+24h] for
    the same machine, 0 otherwise.

    Implementation uses a "mark-backwards" strategy: for each failure event
    at time T_f, mark every telemetry row whose timestamp T satisfies
    T_f - 24h < T <= T_f as a positive.  This is mathematically equivalent
    to asking "does a failure happen within the next 24 hours?" at time T,
    while keeping all feature computation strictly backward-looking.

    This is the ONLY forward-looking operation in the pipeline.

    Args:
        tel:      Full telemetry DataFrame (sorted by machineID, datetime).
        failures: Failure events DataFrame with columns [machineID, datetime].

    Returns:
        int8 numpy array of length len(tel), values in {0, 1}.
    """
    labels       = np.zeros(len(tel), dtype=np.int8)
    tel_machines = tel["machineID"].values
    tel_times    = tel["datetime"].values.astype("datetime64[ns]")
    window_ns    = np.timedelta64(24, "h").astype("timedelta64[ns]")

    for machine_id, machine_failures in failures.groupby("machineID"):
        machine_indices = np.where(tel_machines == machine_id)[0]
        if len(machine_indices) == 0:
            continue

        machine_times = tel_times[machine_indices]

        for f_time in machine_failures["datetime"].values.astype("datetime64[ns]"):
            # Mark rows where T is in (f_time - 24h, f_time]
            in_window = (machine_times > f_time - window_ns) & (machine_times <= f_time)
            labels[machine_indices[in_window]] = 1

    return labels


def _compute_days_since_last_maint(
    tel: pd.DataFrame, maint: pd.DataFrame
) -> np.ndarray:
    """
    For each telemetry row at time T, compute the number of days elapsed since
    the most recent maintenance event STRICTLY before T for the same machine.

    Uses pd.merge_asof for an efficient O(n log n) vectorised solution rather
    than a Python loop per row.

    Rows with no prior maintenance are assigned 365.0 (one year).

    Args:
        tel:   Telemetry DataFrame sorted by machineID then datetime, with a
               clean 0..N-1 integer index.
        maint: Maintenance events DataFrame with columns [machineID, datetime].

    Returns:
        float64 numpy array of length len(tel) with days-since-last-maint.
    """
    # Preserve original row order via an explicit index column so we can
    # restore it after merge_asof re-sorts by the 'on' key (datetime).
    tel_asof = tel[["machineID", "datetime"]].copy()
    tel_asof.index.name = "orig_idx"
    tel_asof = tel_asof.reset_index()  # orig_idx becomes a regular column

    maint_sorted = maint[["machineID", "datetime"]].sort_values("datetime")

    merged = pd.merge_asof(
        tel_asof.sort_values("datetime"),
        maint_sorted.rename(columns={"datetime": "last_maint_time"}),
        left_on="datetime",
        right_on="last_maint_time",
        by="machineID",
        direction="backward",
        allow_exact_matches=False,  # strictly before T
    )

    # Re-order to match tel's original row order.
    merged = merged.set_index("orig_idx").sort_index()

    days = (tel["datetime"] - merged["last_maint_time"]).dt.total_seconds() / 86400
    return days.fillna(365.0).values


def _compute_rolling_for_machine(
    machine_tel: pd.DataFrame,
    machine_errors_hourly: pd.DataFrame,
    machine_maint_hourly: pd.DataFrame,
    machine_age: int,
    machine_model_encoded: int,
) -> pd.DataFrame:
    """
    Compute all backward-looking rolling features for a single machine's telemetry.

    Using time-based rolling ('24h', '168h', '720h') with a DatetimeIndex ensures
    the windows are always measured in real time even if a few hours are missing
    from the telemetry — no row-count assumptions.

    All windows are RIGHT-CLOSED by default in pandas time-based rolling, meaning
    they include observations in [T - window, T].  This is what we want: the
    feature at time T summarises PAST data only.

    Args:
        machine_tel:           Telemetry rows for this machine (all raw columns).
        machine_errors_hourly: Hourly error counts for this machine.
        machine_maint_hourly:  Hourly maintenance counts for this machine.
        machine_age:           Age in years from PdM_machines.csv.
        machine_model_encoded: Integer label (0-3) for the machine model.

    Returns:
        DataFrame with all FEATURE_COLS computed (no target label yet).
    """
    g = machine_tel.sort_values("datetime").set_index("datetime")

    # ── Merge hourly event counts ──────────────────────────────────────────────
    if len(machine_errors_hourly) > 0:
        err_idx = machine_errors_hourly.set_index("datetime")["error_count"]
        g["error_count"] = g.index.map(err_idx).fillna(0)
    else:
        g["error_count"] = 0.0

    if len(machine_maint_hourly) > 0:
        mnt_idx = machine_maint_hourly.set_index("datetime")["maint_count"]
        g["maint_count"] = g.index.map(mnt_idx).fillna(0)
    else:
        g["maint_count"] = 0.0

    # ── 24-hour rolling statistics (backward-only) ─────────────────────────────
    g["volt_mean_24h"]      = g["volt"].rolling("24h",  min_periods=1).mean()
    g["volt_std_24h"]       = g["volt"].rolling("24h",  min_periods=1).std().fillna(0)
    g["rotate_mean_24h"]    = g["rotate"].rolling("24h", min_periods=1).mean()
    g["rotate_std_24h"]     = g["rotate"].rolling("24h", min_periods=1).std().fillna(0)
    g["pressure_mean_24h"]  = g["pressure"].rolling("24h", min_periods=1).mean()
    g["pressure_std_24h"]   = g["pressure"].rolling("24h", min_periods=1).std().fillna(0)
    g["vibration_mean_24h"] = g["vibration"].rolling("24h", min_periods=1).mean()
    g["vibration_std_24h"]  = g["vibration"].rolling("24h", min_periods=1).std().fillna(0)

    # ── 7-day rolling statistics (backward-only, 168h = 7 * 24) ──────────────
    g["pressure_max_7d"]  = g["pressure"].rolling("168h", min_periods=1).max()
    g["vibration_std_7d"] = g["vibration"].rolling("168h", min_periods=1).std().fillna(0)

    # ── Trend features: difference from N rows ago ────────────────────────────
    # shift(24) on hourly data gives the value 24 hours ago.
    # First 24 / 48 rows per machine will be NaN → filled with 0.
    g["volt_trend_24h"]       = (g["volt"]      - g["volt"].shift(24)).fillna(0)
    g["vibration_trend_48h"]  = (g["vibration"] - g["vibration"].shift(48)).fillna(0)

    # ── Event rolling counts (backward-only) ──────────────────────────────────
    g["error_count_24h"] = g["error_count"].rolling("24h",  min_periods=1).sum()
    g["error_count_7d"]  = g["error_count"].rolling("168h", min_periods=1).sum()

    # ── Maintenance rolling count (30 days = 720 hours) ───────────────────────
    g["maint_count_30d"] = g["maint_count"].rolling("720h", min_periods=1).sum()

    # ── Static machine features ────────────────────────────────────────────────
    g["machine_age"]   = float(machine_age)
    g["machine_model"] = float(machine_model_encoded)

    return g.reset_index()


# ── Public API ────────────────────────────────────────────────────────────────

def build_feature_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the complete PdM feature matrix and apply a strict temporal train/test
    split at TRAIN_CUTOFF = "2015-10-01".

    All feature windows look backward only. The failure label is the only
    forward-looking computation (what happens in the next 24h).

    Prints a summary including row counts, positive rate, and NaN verification.

    Returns:
        (train_df, test_df) — each contains FEATURE_COLS + TARGET_COL columns.
        train_df: rows with datetime < TRAIN_CUTOFF
        test_df:  rows with datetime >= TRAIN_CUTOFF
    """
    telemetry, failures, errors, maint, machines = _load_raw_data()

    # Pre-process event logs to hourly counts (used in per-machine rolling).
    errors_hourly = _resample_events_hourly(errors, "error_count")
    maint_hourly  = _resample_events_hourly(maint,  "maint_count")

    # Prepare machine metadata lookup.
    machines_enc = machines.copy()
    machines_enc["machine_model_enc"] = (
        machines_enc["model"].map(_MODEL_LABEL_MAP).fillna(0).astype(int)
    )
    machine_meta = machines_enc.set_index("machineID")[["age", "machine_model_enc"]]

    # ── Compute rolling features per machine ───────────────────────────────────
    logger.info("Computing rolling features for 100 machines...")
    all_parts: list[pd.DataFrame] = []

    for machine_id in sorted(telemetry["machineID"].unique()):
        m_tel   = telemetry[telemetry["machineID"] == machine_id].copy()
        m_err   = errors_hourly[errors_hourly["machineID"] == machine_id]
        m_maint = maint_hourly[maint_hourly["machineID"] == machine_id]

        meta = machine_meta.loc[machine_id] if machine_id in machine_meta.index else None
        age   = int(meta["age"])               if meta is not None else 0
        model = int(meta["machine_model_enc"]) if meta is not None else 0

        part = _compute_rolling_for_machine(m_tel, m_err, m_maint, age, model)
        part["machineID"] = machine_id
        all_parts.append(part)

    df = pd.concat(all_parts, ignore_index=True)
    df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    # ── Compute days_since_last_maint (vectorised across all machines) ─────────
    logger.info("Computing days_since_last_maint via merge_asof...")
    df["days_since_last_maint"] = _compute_days_since_last_maint(df, maint)

    # ── Add failure labels (ONLY forward-looking operation) ───────────────────
    logger.info("Generating failure_within_24h labels...")
    df[TARGET_COL] = _create_failure_labels(df, failures)

    # ── Final NaN cleanup ──────────────────────────────────────────────────────
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # ── Temporal split — NEVER random for time-series data ────────────────────
    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    train_df = df[df["datetime"] < cutoff][FEATURE_COLS + [TARGET_COL]].reset_index(drop=True)
    test_df  = df[df["datetime"] >= cutoff][FEATURE_COLS + [TARGET_COL]].reset_index(drop=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    total_rows    = len(df)
    positive_rate = df[TARGET_COL].mean() * 100
    train_nans    = train_df[FEATURE_COLS].isna().sum().sum()
    test_nans     = test_df[FEATURE_COLS].isna().sum().sum()

    print(f"\n{'='*60}")
    print(f"PdM Feature Matrix Summary")
    print(f"{'='*60}")
    print(f"Total rows:        {total_rows:,}")
    print(f"Positive rate:     {positive_rate:.2f}%  (failure_within_24h=1)")
    print(f"Train size:        {len(train_df):,}  (datetime < {TRAIN_CUTOFF})")
    print(f"Test size:         {len(test_df):,}  (datetime >= {TRAIN_CUTOFF})")
    print(f"Features:          {len(FEATURE_COLS)}")
    print(f"NaN in train set:  {train_nans}  (target: 0)")
    print(f"NaN in test set:   {test_nans}  (target: 0)")
    print(f"{'='*60}\n")

    return train_df, test_df


def get_latest_features(machine_id: int) -> Optional[dict]:
    """
    Build features for a single machine using its full telemetry history,
    then return the feature values for the most recent observation only.

    Called by pdm_predictor.predict_failure() for online inference.
    Raw data is cached by _load_raw_data(), so only the first call pays I/O.

    Args:
        machine_id: Integer machine identifier (1-100 in the PdM dataset).

    Returns:
        Dict mapping feature_name -> float for the latest row, or None if the
        machine is not found in the dataset.
    """
    telemetry, _failures, errors, maint, machines = _load_raw_data()

    m_tel = telemetry[telemetry["machineID"] == machine_id].copy()
    if len(m_tel) == 0:
        logger.warning("Machine not found in telemetry", extra={"machine_id": machine_id})
        return None

    m_info = machines[machines["machineID"] == machine_id]
    if len(m_info) == 0:
        logger.warning("Machine not found in machines table", extra={"machine_id": machine_id})
        return None

    m_errors = errors[errors["machineID"] == machine_id]
    m_maint  = maint[maint["machineID"] == machine_id]

    m_err_hourly   = _resample_events_hourly(m_errors, "error_count") if len(m_errors) > 0 \
                     else pd.DataFrame(columns=["machineID", "datetime", "error_count"])
    m_maint_hourly = _resample_events_hourly(m_maint, "maint_count") if len(m_maint) > 0 \
                     else pd.DataFrame(columns=["machineID", "datetime", "maint_count"])

    row      = m_info.iloc[0]
    age      = int(row["age"])
    model    = int(_MODEL_LABEL_MAP.get(str(row["model"]), 0))

    features_df = _compute_rolling_for_machine(m_tel, m_err_hourly, m_maint_hourly, age, model)
    features_df = features_df.sort_values("datetime").reset_index(drop=True)
    features_df["machineID"] = machine_id

    # Compute days_since_last_maint for this machine.
    features_df["days_since_last_maint"] = _compute_days_since_last_maint(features_df, m_maint)

    if len(features_df) == 0:
        return None

    latest = features_df.iloc[-1]

    return {col: float(latest.get(col, 0.0)) for col in FEATURE_COLS}
