"""
Sample DeepEval test cases for the AMOP RAG pipeline.

Each test case has:
    id               — unique identifier
    input            — technician's question (sent to RAG)
    actual_output    — answer that the RAG pipeline would generate
                       (populated at eval-time; empty here as placeholder)
    expected_output  — ground-truth answer (used by contextual_recall metric)
    retrieval_context — list of text chunks the retriever should surface

The `actual_output` fields are left empty intentionally — the eval route
calls the RAG pipeline live and fills them in before running DeepEval.
When run offline (python -m evaluation.harness), fill actual_output manually.
"""

SAMPLE_TEST_CASES: list[dict] = [
    {
        "id": "tc_hydraulic_pump_failure",
        "input": "What caused the hydraulic pump failures on machine A7?",
        "actual_output": "",
        "expected_output": (
            "Hydraulic pump failures on machine A7 were primarily caused by "
            "seal degradation due to operating above rated pressure, combined "
            "with delayed preventive maintenance intervals."
        ),
        "retrieval_context": [
            "Machine A7 | Issue: Hydraulic pump failure | Notes: Seal worn out, pressure relief valve set too high. "
            "Replaced seal and adjusted pressure setting.",
            "Machine A7 | Issue: Pump not building pressure | Notes: Found cracked pump housing. "
            "Suspect prolonged over-pressure operation. Replaced pump assembly.",
            "Machine A7 | Issue: Oil leak from pump | Notes: Shaft seal blown. "
            "Pump overheating due to low oil level. Flushed system and replaced seal.",
        ],
    },
    {
        "id": "tc_conveyor_belt_slippage",
        "input": "What are the most common issues reported for conveyor belt machines?",
        "actual_output": "",
        "expected_output": (
            "Common conveyor belt issues include belt slippage caused by worn drive drums, "
            "misalignment from off-center loading, and belt tears from foreign object ingestion."
        ),
        "retrieval_context": [
            "Machine B12 | Issue: Belt slipping | Notes: Drive drum worn smooth. Replaced drum and tensioned belt.",
            "Machine C3 | Issue: Belt misaligned | Notes: Off-center load distribution. Realigned idlers and adjusted loading chute.",
            "Machine B8 | Issue: Belt tear | Notes: Metal scrap ingested from upstream. "
            "Installed magnetic separator. Repaired belt with vulcanized patch.",
            "Machine B12 | Issue: Belt tracking off | Notes: Head pulley lagging worn. "
            "Re-lagged pulley and adjusted tracking.",
        ],
    },
    {
        "id": "tc_electrical_fault_m42",
        "input": "What electrical faults has machine M42 experienced?",
        "actual_output": "",
        "expected_output": (
            "Machine M42 has experienced motor overheating due to blocked cooling vents, "
            "VFD faults caused by loose wiring in the control cabinet, and contactor failures "
            "from excessive switching cycles."
        ),
        "retrieval_context": [
            "Machine M42 | Issue: Motor overheating | Notes: Cooling fins blocked by dust buildup. "
            "Cleaned fins, checked thermal protection relay settings.",
            "Machine M42 | Issue: VFD fault code E04 | Notes: Loose terminal on L2 input. "
            "Tightened connections, reset drive, tested under load.",
            "Machine M42 | Issue: Main contactor chattering | Notes: Coil voltage fluctuation from upstream. "
            "Replaced contactor, added surge suppressor.",
        ],
    },
    {
        "id": "tc_bearing_diagnosis",
        "input": "How do technicians typically diagnose bearing failures?",
        "actual_output": "",
        "expected_output": (
            "Technicians diagnose bearing failures through vibration analysis detecting high-frequency "
            "peaks, thermal imaging showing hot spots, and physical inspection revealing pitting, "
            "spalling, or lubricant breakdown."
        ),
        "retrieval_context": [
            "Machine D5 | Issue: Unusual vibration on main shaft | Notes: Vibration analysis showed "
            "bearing defect frequency at 3× shaft speed. Replaced bearing, regreased.",
            "Machine D9 | Issue: Bearing running hot | Notes: Thermal camera showed 85°C at outer race. "
            "Grease degraded, contaminated with metal particles. Flushed and repacked.",
            "Machine E1 | Issue: Bearing noise | Notes: Metallic grinding sound. Disassembled — found "
            "spalling on inner race. Replaced bearing set, checked alignment.",
            "Machine D5 | Issue: Shaft seal leaking oil | Notes: Bearing overheating caused seal "
            "failure. Root cause: insufficient lubrication interval.",
        ],
    },
    {
        "id": "tc_root_cause_pump_vibration",
        "input": "What is the root cause of recurring pump vibration issues across multiple machines?",
        "actual_output": "",
        "expected_output": (
            "The root cause of recurring pump vibration is cavitation caused by insufficient "
            "inlet pressure, combined with misalignment between pump and motor shafts that "
            "was not corrected during previous maintenance events."
        ),
        "retrieval_context": [
            "Machine F2 | Issue: Pump vibrating excessively | Notes: Cavitation evident — "
            "inlet strainer blocked. Cleaned strainer, rechecked NPSH margin.",
            "Machine F7 | Issue: Pump vibration and noise | Notes: Shaft misalignment 0.8mm "
            "above tolerance. Laser aligned coupling.",
            "Machine G1 | Issue: Pump bearing failure recurring | Notes: Third bearing replacement "
            "in 6 months. Misalignment and cavitation both present. "
            "Redesigned inlet piping to increase NPSH.",
            "Machine F2 | Issue: Pump vibration returning | Notes: Inlet strainer blocked again "
            "within 2 months. Installed auto-cleaning filter.",
        ],
    },
]


def get_test_case_by_id(case_id: str) -> dict | None:
    return next((tc for tc in SAMPLE_TEST_CASES if tc["id"] == case_id), None)
