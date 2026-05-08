"""
Gold-standard examples for process knowledge extraction.

These serve multiple purposes:
1. Few-shot examples for LLM prompting
2. Test fixtures for the extraction pipeline
3. Reference for what correct extraction looks like
4. Training data seeds for the teacher-student pipeline

Each example pairs a source text with the expected extraction result.
"""

from datetime import datetime
from backend.schemas.process_knowledge.entities import (
    Tool, Material, PPE, Worker, Step, Procedure,
    ProcessParameter, QualityRequirement
)
from backend.schemas.process_knowledge.relations import (
    StepOrder, ToolRequirement, StepExecution,
    QualityCheck, QualificationRecord, ProcedureExecution,
    StepFeedback
)


# ============================================================================
# LEVEL 1 — Flat entity extraction
# ============================================================================

LEVEL_1_TEXT = """
Für diese Schweißaufgabe verwenden wir das MIG-Schweißgerät Fronius TPS 400i. 
Als Zusatzwerkstoff kommt der Schweißdraht ER70S-6 mit 1,0mm Durchmesser zum Einsatz. 
Das Schutzgas ist eine Mischung aus 82% Argon und 18% CO2.
"""

LEVEL_1_EXPECTED_ENTITIES = [
    Tool(
        name="Fronius TPS 400i",
        tool_type="welding",
        manufacturer="Fronius",
        model="TPS 400i",
        source_text="MIG-Schweißgerät Fronius TPS 400i"
    ),
    Material(
        name="ER70S-6 Schweißdraht",
        material_type="filler",
        specification="ER70S-6",
        dimensions="1,0mm Durchmesser",
        source_text="Schweißdraht ER70S-6 mit 1,0mm Durchmesser"
    ),
    Material(
        name="Schutzgas Argon/CO2 82/18",
        material_type="shielding gas",
        specification="82% Argon, 18% CO2",
        source_text="Schutzgas ist eine Mischung aus 82% Argon und 18% CO2"
    ),
]


# ============================================================================
# LEVEL 2 — Binary relations
# ============================================================================

LEVEL_2_TEXT = """
Die Schweißprozedur WPS-2024-042 für die MIG-Schweißung von S235 T-Stößen 
besteht aus vier Schritten: Zuerst wird das Werkstück vorbereitet, dann erfolgt 
das Heften, anschließend die Wurzellage, und zum Schluss die Decklage.
"""

LEVEL_2_EXPECTED_PROCEDURE = Procedure(
    name="MIG-Schweißung S235 T-Stoß",
    procedure_id="WPS-2024-042",
    domain="welding",
    status="active",
    source_text="Schweißprozedur WPS-2024-042 für die MIG-Schweißung von S235 T-Stößen"
)

LEVEL_2_EXPECTED_STEPS = [
    Step(name="Werkstück vorbereiten", step_number=1, step_type="preparation"),
    Step(name="Heften", step_number=2, step_type="execution"),
    Step(name="Wurzellage", step_number=3, step_type="execution", is_critical=True),
    Step(name="Decklage", step_number=4, step_type="execution"),
]

LEVEL_2_EXPECTED_ORDER = [
    StepOrder(
        before=LEVEL_2_EXPECTED_STEPS[0], 
        after=LEVEL_2_EXPECTED_STEPS[1],
        source_text="Zuerst wird das Werkstück vorbereitet, dann erfolgt das Heften"
    ),
    StepOrder(
        before=LEVEL_2_EXPECTED_STEPS[1], 
        after=LEVEL_2_EXPECTED_STEPS[2],
        source_text="dann erfolgt das Heften, anschließend die Wurzellage"
    ),
    StepOrder(
        before=LEVEL_2_EXPECTED_STEPS[2], 
        after=LEVEL_2_EXPECTED_STEPS[3],
        source_text="anschließend die Wurzellage, und zum Schluss die Decklage"
    ),
]


# ============================================================================
# LEVEL 3 — Relations with own attributes
# ============================================================================

LEVEL_3_TEXT = """
Für die Wurzellage wird das MIG-Schweißgerät Fronius TPS 400i benötigt. 
Der Schweißstrom soll auf 180 Ampere eingestellt werden, die Spannung auf 
24 Volt, und die Drahtvorschubgeschwindigkeit auf 8 m/min. 
Wichtig: Die Kontaktspitze muss auf 1,0mm Drahtdurchmesser eingestellt sein.
"""

LEVEL_3_EXPECTED = ToolRequirement(
    step=Step(name="Wurzellage", step_number=3, step_type="execution", is_critical=True),
    tool=Tool(name="Fronius TPS 400i", tool_type="welding", manufacturer="Fronius"),
    is_mandatory=True,
    configuration_notes="Kontaktspitze auf 1,0mm Drahtdurchmesser einstellen",
    parameters=[
        ProcessParameter(
            name="Schweißstrom", parameter_type="current", 
            unit="A", nominal_value=180.0
        ),
        ProcessParameter(
            name="Spannung", parameter_type="voltage", 
            unit="V", nominal_value=24.0
        ),
        ProcessParameter(
            name="Drahtvorschubgeschwindigkeit", parameter_type="speed", 
            unit="m/min", nominal_value=8.0
        ),
    ],
    source_text="Für die Wurzellage wird das MIG-Schweißgerät Fronius TPS 400i benötigt"
)


# ============================================================================
# LEVEL 4 — N-ary relation extraction
# ============================================================================

LEVEL_4_TEXT = """
Am 11.10.2024 um 14:30 hat der Schweißer Hans Müller die Wurzellage 
durchgeführt. Er hat das Fronius TPS 400i verwendet, allerdings den 
Schweißstrom auf 185A eingestellt statt der vorgeschriebenen 180A. 
Die Spannung lag bei 24V. Der Schritt hat 12 Minuten gedauert und 
wurde erfolgreich abgeschlossen, trotz der Abweichung beim Strom.
"""

LEVEL_4_EXPECTED = StepExecution(
    step=Step(name="Wurzellage", step_number=3, step_type="execution", is_critical=True),
    executed_by=Worker(name="Hans Müller", role="operator", expertise_level="skilled"),
    tools_used=[Tool(name="Fronius TPS 400i", tool_type="welding", manufacturer="Fronius")],
    started_at=datetime(2024, 10, 11, 14, 30),
    actual_duration_seconds=720,
    status="completed",
    observed_parameters=[
        ProcessParameter(name="Schweißstrom", parameter_type="current", unit="A", nominal_value=185.0),
        ProcessParameter(name="Spannung", parameter_type="voltage", unit="V", nominal_value=24.0),
    ],
    deviation_from_spec="Schweißstrom auf 185A statt vorgeschriebener 180A eingestellt",
    source_text="Am 11.10.2024 um 14:30 hat der Schweißer Hans Müller die Wurzellage durchgeführt"
)


# ============================================================================
# LEVEL 5 — Deeply nested extraction
# ============================================================================

LEVEL_5_TEXT = """
Am 11.10.2024 wurde die Schweißprozedur WPS-2024-042 (MIG-Schweißung S235 T-Stoß) 
durchgeführt. Das Team bestand aus Schweißer Hans Müller und Qualitätsprüferin 
Maria Schmidt. Schweißfachingenieur Dr. Weber hatte die Aufsicht.

Hans hat zuerst das Werkstück vorbereitet (10 Minuten), dann geheftet (5 Minuten). 
Bei der Wurzellage (12 Minuten) hat er den Strom auf 185A gesetzt statt der 
vorgeschriebenen 180A. Er hat nachgefragt, ob das in Ordnung sei. Dr. Weber hat 
bestätigt, dass 185A noch im Toleranzbereich liegt.

Die Decklage lief ohne Probleme (15 Minuten).

Maria hat anschließend die Schweißnaht nach EN ISO 5817 Bewertungsgruppe B 
geprüft. Ergebnis: akzeptiert, geringfügige Porosität festgestellt, keine 
Nacharbeit erforderlich.

Gesamtergebnis: abgenommen mit Anmerkungen.
"""

LEVEL_5_EXPECTED = ProcedureExecution(
    procedure=Procedure(
        name="MIG-Schweißung S235 T-Stoß",
        procedure_id="WPS-2024-042",
        domain="welding",
        status="active"
    ),
    executed_by=[
        Worker(name="Hans Müller", role="operator", expertise_level="skilled"),
        Worker(name="Maria Schmidt", role="quality_inspector"),
    ],
    supervised_by=Worker(name="Dr. Weber", role="welding_engineer"),
    started_at=datetime(2024, 10, 11),
    overall_status="completed_with_deviations",
    overall_result="accepted_with_conditions",
    procedure_version=None,
    step_executions=[
        StepExecution(
            step=Step(name="Werkstück vorbereiten", step_number=1, step_type="preparation"),
            executed_by=Worker(name="Hans Müller", role="operator"),
            actual_duration_seconds=600,
            status="completed",
        ),
        StepExecution(
            step=Step(name="Heften", step_number=2, step_type="execution"),
            executed_by=Worker(name="Hans Müller", role="operator"),
            actual_duration_seconds=300,
            status="completed",
        ),
        StepExecution(
            step=Step(name="Wurzellage", step_number=3, step_type="execution", is_critical=True),
            executed_by=Worker(name="Hans Müller", role="operator"),
            actual_duration_seconds=720,
            status="completed",
            observed_parameters=[
                ProcessParameter(name="Schweißstrom", parameter_type="current", unit="A", nominal_value=185.0),
            ],
            deviation_from_spec="Strom auf 185A statt 180A",
        ),
        StepExecution(
            step=Step(name="Decklage", step_number=4, step_type="execution"),
            executed_by=Worker(name="Hans Müller", role="operator"),
            actual_duration_seconds=900,
            status="completed",
        ),
    ],
    quality_checks=[
        QualityCheck(
            inspector=Worker(name="Maria Schmidt", role="quality_inspector"),
            step_execution=StepExecution(
                step=Step(name="Wurzellage", step_number=3),
                executed_by=Worker(name="Hans Müller", role="operator"),
                status="completed",
            ),
            requirement=QualityRequirement(
                name="Sichtprüfung Schweißnaht",
                requirement_type="visual",
                acceptance_criteria="EN ISO 5817 Bewertungsgruppe B",
                applicable_norm="EN ISO 5817",
            ),
            result="pass",
            findings="Geringfügige Porosität festgestellt, keine Nacharbeit erforderlich",
            corrective_action_required=False,
        ),
    ],
    feedback=[
        StepFeedback(
            step_execution=StepExecution(
                step=Step(name="Wurzellage", step_number=3),
                executed_by=Worker(name="Hans Müller", role="operator"),
                status="completed",
            ),
            reported_by=Worker(name="Hans Müller", role="operator"),
            feedback_type="question",
            content="Ist 185A statt 180A in Ordnung?",
            severity="info",
            resolved=True,
            resolution="Dr. Weber bestätigt: 185A liegt noch im Toleranzbereich",
            resolved_by=Worker(name="Dr. Weber", role="welding_engineer"),
        ),
    ],
    source_text="Am 11.10.2024 wurde die Schweißprozedur WPS-2024-042 durchgeführt"
)


# ============================================================================
# ALL EXAMPLES — for easy iteration in tests and training
# ============================================================================

ALL_EXAMPLES = [
    {"level": 1, "text": LEVEL_1_TEXT, "expected": LEVEL_1_EXPECTED_ENTITIES},
    {"level": 2, "text": LEVEL_2_TEXT, "expected": {
        "procedure": LEVEL_2_EXPECTED_PROCEDURE,
        "steps": LEVEL_2_EXPECTED_STEPS,
        "step_orders": LEVEL_2_EXPECTED_ORDER,
    }},
    {"level": 3, "text": LEVEL_3_TEXT, "expected": LEVEL_3_EXPECTED},
    {"level": 4, "text": LEVEL_4_TEXT, "expected": LEVEL_4_EXPECTED},
    {"level": 5, "text": LEVEL_5_TEXT, "expected": LEVEL_5_EXPECTED},
]