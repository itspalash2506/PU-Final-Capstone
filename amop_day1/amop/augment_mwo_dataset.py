"""
MWO Dataset Augmentation Script
================================
Augments the existing 20,000-row MWO dataset with 6 synthetic fields:
  machine_type, severity, component_affected, repair_success,
  repair_duration_hours, plant

Approach: rule-based derivation from real issue text + technician notes.
Original data is never modified — new columns are added alongside.

Run from project root:
  python augment_mwo_dataset.py

Output: dataset_files/mwo_dataset_augmented.csv
"""

import pandas as pd
import numpy as np
import hashlib
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = Path("dataset_files/mwo_dataset.csv")
OUTPUT_PATH = Path("dataset_files/mwo_dataset_augmented.csv")

# NOTE: Run this script from your project root:
# C:\Final_Capstone_Project\amop_day1\amop\
# python augment_mwo_dataset.py

# ── Random seed for reproducibility ──────────────────────────────────────────
RNG = np.random.default_rng(seed=42)

# ══════════════════════════════════════════════════════════════════════════════
# ISSUE → MACHINE TYPE MAPPING
# All 24 unique issues mapped to machine types.
# Mix approach: keyword-derive where clear, consistent assignment as fallback.
# ══════════════════════════════════════════════════════════════════════════════

ISSUE_TO_MACHINE_TYPE = {
    # Hydraulic system issues
    "Leak in side B hydraulic line":                    "hydraulic_system",
    "Hydraulic hose on side B is leaking":              "hydraulic_system",
    "Saw attachment is leaking hydraulics":             "hydraulic_system",
    "Hydraulic fluid leak at saw attachment":           "hydraulic_system",
    "Hydraulic line leak on side B":                    "hydraulic_system",
    "Leak detected in hydraulic line of saw attachment":"hydraulic_system",
    "Turret A and B are leaking":                       "hydraulic_system",
    "Turret sections A & B showing leaks":              "hydraulic_system",
    "Leaks found at turrets A and B":                   "hydraulic_system",

    # Spindle / CNC issues
    "Spindle indexing error with overfeed":             "cnc_spindle",
    "Spindle fails to index, overfeeding":              "cnc_spindle",
    "Milling spindle requires repair":                  "cnc_spindle",
    "Repair work needed on milling spindle":            "cnc_spindle",
    "Maintenance on milling spindle needed":            "cnc_spindle",
    "Indexing issue causing overfeed on spindle":       "cnc_spindle",

    # Mechanical / gear / saw attachment issues
    "Gears on the saw attachment are too tight and grinding": "mechanical_drive",
    "Grinding noise from tight gears on saw attachment":      "mechanical_drive",
    "Saw attachment gears are binding and grinding":          "mechanical_drive",

    # Electrical / power issues
    "Machine is unpowered":                             "electrical_system",
    "Power supply failure detected":                    "electrical_system",
    "Unit lost electrical power":                       "electrical_system",

    # Pneumatic / accumulator issues
    "Accumulator inspection and charging required":     "pneumatic_system",
    "Inspect and recharge accumulators":                "pneumatic_system",
    "Accumulators need checking and charging":          "pneumatic_system",
}

# ══════════════════════════════════════════════════════════════════════════════
# ISSUE → COMPONENT AFFECTED
# ══════════════════════════════════════════════════════════════════════════════

ISSUE_TO_COMPONENT = {
    "Leak in side B hydraulic line":                    "hydraulic_line",
    "Hydraulic hose on side B is leaking":              "hydraulic_hose",
    "Saw attachment is leaking hydraulics":             "saw_attachment_hydraulics",
    "Hydraulic fluid leak at saw attachment":           "saw_attachment_hydraulics",
    "Hydraulic line leak on side B":                    "hydraulic_line",
    "Leak detected in hydraulic line of saw attachment":"hydraulic_line",
    "Turret A and B are leaking":                       "turret_seals",
    "Turret sections A & B showing leaks":              "turret_seals",
    "Leaks found at turrets A and B":                   "turret_seals",
    "Spindle indexing error with overfeed":             "spindle_indexer",
    "Spindle fails to index, overfeeding":              "spindle_indexer",
    "Milling spindle requires repair":                  "milling_spindle",
    "Repair work needed on milling spindle":            "milling_spindle",
    "Maintenance on milling spindle needed":            "milling_spindle",
    "Indexing issue causing overfeed on spindle":       "spindle_indexer",
    "Gears on the saw attachment are too tight and grinding": "saw_attachment_gears",
    "Grinding noise from tight gears on saw attachment":      "saw_attachment_gears",
    "Saw attachment gears are binding and grinding":          "saw_attachment_gears",
    "Machine is unpowered":                             "power_supply",
    "Power supply failure detected":                    "power_supply",
    "Unit lost electrical power":                       "electrical_control_unit",
    "Accumulator inspection and charging required":     "accumulator",
    "Inspect and recharge accumulators":                "accumulator",
    "Accumulators need checking and charging":          "accumulator",
}

# ══════════════════════════════════════════════════════════════════════════════
# SEVERITY RULES
# Base: features.py keyword classifier logic
# Override: richer rules where we are confident
# ══════════════════════════════════════════════════════════════════════════════

# Base severity from issue keywords (consistent with features.py)
ISSUE_BASE_SEVERITY = {
    # High severity — machine stopped or safety risk
    "Machine is unpowered":                             "high",
    "Power supply failure detected":                    "high",
    "Unit lost electrical power":                       "high",
    "Spindle indexing error with overfeed":             "high",
    "Spindle fails to index, overfeeding":              "high",
    "Indexing issue causing overfeed on spindle":       "high",

    # Medium severity — functional impairment, needs attention soon
    "Leak in side B hydraulic line":                    "medium",
    "Hydraulic hose on side B is leaking":              "medium",
    "Hydraulic line leak on side B":                    "medium",
    "Turret A and B are leaking":                       "medium",
    "Turret sections A & B showing leaks":              "medium",
    "Leaks found at turrets A and B":                   "medium",
    "Saw attachment is leaking hydraulics":             "medium",
    "Hydraulic fluid leak at saw attachment":           "medium",
    "Leak detected in hydraulic line of saw attachment":"medium",
    "Milling spindle requires repair":                  "medium",
    "Repair work needed on milling spindle":            "medium",
    "Gears on the saw attachment are too tight and grinding": "medium",
    "Grinding noise from tight gears on saw attachment":      "medium",
    "Saw attachment gears are binding and grinding":          "medium",

    # Low severity — scheduled maintenance, inspection
    "Accumulator inspection and charging required":     "low",
    "Inspect and recharge accumulators":                "low",
    "Accumulators need checking and charging":          "low",
    "Maintenance on milling spindle needed":            "low",
}

# Override: boost severity based on technician notes content
# If notes mention urgent keywords, upgrade severity
SEVERITY_UPGRADE_KEYWORDS = [
    "emergency", "critical", "urgent", "immediate", "failed completely",
    "production stopped", "shutdown", "safety", "fire", "explosion",
    "complete failure", "cannot operate"
]
SEVERITY_DOWNGRADE_KEYWORDS = [
    "routine", "scheduled", "minor", "slight", "small", "inspection only",
    "no damage", "preventive"
]

# ══════════════════════════════════════════════════════════════════════════════
# REPAIR DURATION by severity and component type (hours)
# Based on realistic manufacturing maintenance times
# ══════════════════════════════════════════════════════════════════════════════

REPAIR_DURATION_PARAMS = {
    # (severity, machine_type) → (min_hours, max_hours, mean_hours)
    ("high",   "electrical_system"):  (2.0,  8.0,  4.0),
    ("high",   "cnc_spindle"):        (4.0, 12.0,  7.0),
    ("high",   "hydraulic_system"):   (3.0,  8.0,  5.0),
    ("high",   "mechanical_drive"):   (3.0, 10.0,  6.0),
    ("high",   "pneumatic_system"):   (2.0,  6.0,  3.5),
    ("medium", "hydraulic_system"):   (1.5,  5.0,  3.0),
    ("medium", "cnc_spindle"):        (2.0,  6.0,  4.0),
    ("medium", "mechanical_drive"):   (1.5,  5.0,  3.0),
    ("medium", "electrical_system"):  (1.0,  4.0,  2.5),
    ("medium", "pneumatic_system"):   (1.0,  3.0,  2.0),
    ("low",    "pneumatic_system"):   (0.5,  2.0,  1.0),
    ("low",    "cnc_spindle"):        (1.0,  3.0,  1.5),
    ("low",    "hydraulic_system"):   (0.5,  2.0,  1.0),
    ("low",    "mechanical_drive"):   (0.5,  2.0,  1.0),
    ("low",    "electrical_system"):  (0.5,  1.5,  1.0),
}

# ══════════════════════════════════════════════════════════════════════════════
# REPAIR SUCCESS RATES by severity and technician
# Higher-experienced techs have higher success rates
# michele_williams appears most (10k times) → most experienced
# ══════════════════════════════════════════════════════════════════════════════

TECH_EXPERIENCE_LEVEL = {
    "michele_williams": "senior",    # 10,060 appearances
    "angie_henderson":  "senior",    # 3,405
    "gina_moore":       "senior",    # 3,398
    "dylan_miller":     "mid",       # 3,398
    "nathan_maldonado": "mid",       # 3,309
    "ethan_adams":      "mid",       # 3,289
    "cristian_santos":  "junior",    # 3,233
}

SUCCESS_RATE_BY_SEVERITY_EXPERIENCE = {
    ("low",    "senior"):  0.97,
    ("low",    "mid"):     0.93,
    ("low",    "junior"):  0.90,
    ("medium", "senior"):  0.91,
    ("medium", "mid"):     0.85,
    ("medium", "junior"):  0.78,
    ("high",   "senior"):  0.83,
    ("high",   "mid"):     0.74,
    ("high",   "junior"):  0.65,
}

# ══════════════════════════════════════════════════════════════════════════════
# PLANT ASSIGNMENT
# Since all machines have similar issue distributions, we cluster by
# the RATIO of spindle vs electrical vs pneumatic issues per machine.
# This produces 5 meaningful clusters based on actual (subtle) differences.
# ══════════════════════════════════════════════════════════════════════════════

def assign_plants(df: pd.DataFrame) -> dict[str, str]:
    """
    Cluster 99 machines into 5 plants based on issue category ratios.
    Uses K-means-style assignment on 3 ratio features.
    Returns dict: machine_id → plant_name
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    def get_category(issue):
        issue_lower = issue.lower()
        if any(w in issue_lower for w in ['spindle', 'index', 'milling', 'overfeed']):
            return 'spindle'
        elif any(w in issue_lower for w in ['power', 'electrical', 'unpowered', 'unit lost']):
            return 'electrical'
        elif any(w in issue_lower for w in ['accumulator', 'charging', 'recharge']):
            return 'pneumatic'
        elif any(w in issue_lower for w in ['gear', 'saw', 'grinding', 'binding']):
            return 'mechanical'
        else:
            return 'hydraulic'

    df_temp = df.copy()
    df_temp['category'] = df_temp['issue'].apply(get_category)

    # Build ratio matrix per machine
    ratios = (
        df_temp.groupby('mach')['category']
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    # Use spindle, electrical, pneumatic ratios for clustering
    features = ['spindle', 'electrical', 'pneumatic']
    for f in features:
        if f not in ratios.columns:
            ratios[f] = 0.0

    X = ratios[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means with 5 clusters → 5 plants
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)

    plant_names = ["Plant_Alpha", "Plant_Beta", "Plant_Gamma", "Plant_Delta", "Plant_Epsilon"]
    machine_to_plant = {
        mach: plant_names[label]
        for mach, label in zip(ratios.index, cluster_labels)
    }

    print("Plant assignment cluster sizes:")
    from collections import Counter
    counts = Counter(machine_to_plant.values())
    for plant, count in sorted(counts.items()):
        print(f"  {plant}: {count} machines")

    return machine_to_plant

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def derive_severity(issue: str, info: str) -> str:
    """
    Start with base severity from issue mapping.
    Override based on technician notes keywords.
    """
    base = ISSUE_BASE_SEVERITY.get(issue, "medium")

    if not info or info.strip() == "":
        return base

    info_lower = info.lower()

    # Check for upgrade keywords
    if any(kw in info_lower for kw in SEVERITY_UPGRADE_KEYWORDS):
        if base == "low":
            return "medium"
        elif base == "medium":
            return "high"

    # Check for downgrade keywords
    if any(kw in info_lower for kw in SEVERITY_DOWNGRADE_KEYWORDS):
        if base == "high":
            return "medium"
        elif base == "medium":
            return "low"

    return base


def derive_repair_duration(severity: str, machine_type: str) -> float:
    """
    Sample repair duration from a realistic distribution.
    Uses triangular distribution bounded by min/max with given mean.
    """
    key = (severity, machine_type)
    if key not in REPAIR_DURATION_PARAMS:
        # Fallback defaults
        defaults = {"high": (2.0, 8.0, 5.0), "medium": (1.0, 5.0, 2.5), "low": (0.5, 2.0, 1.0)}
        min_h, max_h, mean_h = defaults.get(severity, (1.0, 4.0, 2.0))
    else:
        min_h, max_h, mean_h = REPAIR_DURATION_PARAMS[key]

    # Triangular distribution: left=min, right=max, mode=mean
    sample = float(RNG.triangular(left=min_h, mode=mean_h, right=max_h))
    return round(sample, 1)


def derive_repair_success(severity: str, primary_tech: str) -> bool:
    """
    Probabilistic repair success based on severity and technician experience.
    """
    experience = TECH_EXPERIENCE_LEVEL.get(primary_tech, "mid")
    success_rate = SUCCESS_RATE_BY_SEVERITY_EXPERIENCE.get(
        (severity, experience), 0.80
    )
    return bool(RNG.random() < success_rate)


def get_primary_tech(tech_field: str) -> str:
    """Extract first technician name from comma-separated tech field."""
    if not tech_field or not str(tech_field).strip():
        return "unknown"
    parts = [t.strip() for t in str(tech_field).split(",") if t.strip()]
    return parts[0] if parts else "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def augment(df: pd.DataFrame, machine_to_plant: dict) -> pd.DataFrame:
    """Add 6 synthetic columns to the dataframe."""
    result = df.copy()

    machine_type_list     = []
    severity_list         = []
    component_list        = []
    repair_success_list   = []
    repair_duration_list  = []
    plant_list            = []

    for _, row in result.iterrows():
        issue      = str(row.get("issue", "")).strip()
        info       = str(row.get("info", "")).strip()
        mach       = str(row.get("mach", "")).strip()
        tech_field = str(row.get("tech", "")).strip()

        # Machine type
        machine_type = ISSUE_TO_MACHINE_TYPE.get(issue)
        if not machine_type:
            # Fallback: keyword scan on issue text
            issue_lower = issue.lower()
            if any(w in issue_lower for w in ["hydraulic", "leak", "hose", "turret"]):
                machine_type = "hydraulic_system"
            elif any(w in issue_lower for w in ["spindle", "index", "milling"]):
                machine_type = "cnc_spindle"
            elif any(w in issue_lower for w in ["gear", "saw", "grinding"]):
                machine_type = "mechanical_drive"
            elif any(w in issue_lower for w in ["power", "electrical", "unpowered"]):
                machine_type = "electrical_system"
            elif any(w in issue_lower for w in ["accumulator", "charging"]):
                machine_type = "pneumatic_system"
            else:
                machine_type = "general_equipment"

        # Component affected
        component = ISSUE_TO_COMPONENT.get(issue, "unknown_component")

        # Severity (base + override)
        severity = derive_severity(issue, info)

        # Primary technician for success rate
        primary_tech = get_primary_tech(tech_field)

        # Repair success
        repair_success = derive_repair_success(severity, primary_tech)

        # Repair duration
        repair_duration = derive_repair_duration(severity, machine_type)

        # Plant
        plant = machine_to_plant.get(mach, "Plant_Alpha")

        machine_type_list.append(machine_type)
        severity_list.append(severity)
        component_list.append(component)
        repair_success_list.append(repair_success)
        repair_duration_list.append(repair_duration)
        plant_list.append(plant)

    result["machine_type"]          = machine_type_list
    result["severity"]              = severity_list
    result["component_affected"]    = component_list
    result["repair_success"]        = repair_success_list
    result["repair_duration_hours"] = repair_duration_list
    result["plant"]                 = plant_list

    return result


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("AUGMENTATION VALIDATION")
    print("=" * 60)
    print(f"Total rows       : {len(df):,}")
    print(f"Columns          : {df.columns.tolist()}")
    print(f"Null values      : {df.isnull().sum().sum()}")
    print()

    print("machine_type distribution:")
    print(df["machine_type"].value_counts().to_string())
    print()

    print("severity distribution:")
    print(df["severity"].value_counts().to_string())
    print()

    print("component_affected distribution:")
    print(df["component_affected"].value_counts().to_string())
    print()

    print("repair_success rate:")
    rate = df["repair_success"].mean()
    print(f"  {rate:.1%} success ({df['repair_success'].sum():,} / {len(df):,})")
    print()

    print("repair_duration_hours stats:")
    print(df["repair_duration_hours"].describe().round(2).to_string())
    print()

    print("plant distribution:")
    print(df["plant"].value_counts().to_string())
    print()

    print("severity by machine_type:")
    print(pd.crosstab(df["machine_type"], df["severity"]).to_string())
    print()

    print("repair_success by severity:")
    print(pd.crosstab(df["severity"], df["repair_success"]).to_string())
    print()

    print("Sample rows (5):")
    cols = ["mach", "issue", "machine_type", "severity",
            "component_affected", "repair_success", "repair_duration_hours", "plant"]
    print(df[cols].sample(5, random_state=42).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    t0 = time.perf_counter()

    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df):,} rows")

    print("\nAssigning plants via K-means clustering on issue ratios...")
    machine_to_plant = assign_plants(df)

    print("\nAugmenting dataset...")
    df_aug = augment(df, machine_to_plant)

    validate(df_aug)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_aug.to_csv(OUTPUT_PATH, index=False)

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Rows    : {len(df_aug):,}")
    print(f"Columns : {len(df_aug.columns)}")
    print(f"Time    : {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print("\nNext step: copy this file to your project and run it:")
    print("  python augment_mwo_dataset.py")
    print("Then re-ingest with the augmented CSV via /api/v1/ingest")
