"""DermAbench Clinical Query Suite.

Defines the 11 clinical query types, templates, and dynamic query generation
mechanisms for case evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ClinicalQueryType(str, Enum):
    DIAGNOSIS = "diagnosis"
    DIFFERENTIAL = "differential"
    TRIAGE = "triage"
    CODING = "coding"
    MANAGEMENT = "management"
    HISTORY_ANALYSIS = "history_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    DERMOSCOPIC = "dermoscopic"
    COMPARATIVE = "comparative"
    TREATMENT = "treatment"
    PATIENT_COMMUNICATION = "patient_communication"


@dataclass
class QueryTemplate:
    query_type: ClinicalQueryType
    template: str
    auto_scorable: bool
    scoring_method: str
    target_dimensions: list[int]
    description: str


QUERY_TEMPLATES: dict[ClinicalQueryType, QueryTemplate] = {
    ClinicalQueryType.DIAGNOSIS: QueryTemplate(
        query_type=ClinicalQueryType.DIAGNOSIS,
        template="What is the primary diagnosis for this skin lesion/condition? Provide the diagnosis name only.",
        auto_scorable=True,
        scoring_method="exact_match",
        target_dimensions=[1],
        description="Primary diagnostic accuracy check.",
    ),
    ClinicalQueryType.DIFFERENTIAL: QueryTemplate(
        query_type=ClinicalQueryType.DIFFERENTIAL,
        template="Provide a ranked list of the top 3 differential diagnoses for this skin condition. Format as a list of names.",
        auto_scorable=True,
        scoring_method="rank_and_jaccard",
        target_dimensions=[4],
        description="Differential diagnosis list quality and ranking evaluation.",
    ),
    ClinicalQueryType.TRIAGE: QueryTemplate(
        query_type=ClinicalQueryType.TRIAGE,
        template="Does this patient require an urgent referral or immediate biopsy? Answer with 'Yes' or 'No' and explain why.",
        auto_scorable=True,
        scoring_method="triage_sensitivity",
        target_dimensions=[7],
        description="Triage sensitivity and urgency evaluation.",
    ),
    ClinicalQueryType.CODING: QueryTemplate(
        query_type=ClinicalQueryType.CODING,
        template="Provide the most appropriate ICD-10 code and SNOMED-CT concept ID for the primary diagnosis: '{diagnosis}'.",
        auto_scorable=True,
        scoring_method="exact_match",
        target_dimensions=[3],
        description="Standard coding terminology mapping evaluation.",
    ),
    ClinicalQueryType.MANAGEMENT: QueryTemplate(
        query_type=ClinicalQueryType.MANAGEMENT,
        template="What is the recommended management plan for this condition? Choose one: 'biopsy', 'monitor', or 'reassure'.",
        auto_scorable=True,
        scoring_method="exact_match",
        target_dimensions=[7],
        description="Appropriateness of management recommendations.",
    ),
    ClinicalQueryType.HISTORY_ANALYSIS: QueryTemplate(
        query_type=ClinicalQueryType.HISTORY_ANALYSIS,
        template="Analyze the patient's clinical history: '{clinical_history}'. What are the key history features or risk factors present? List them.",
        auto_scorable=True,
        scoring_method="recall",
        target_dimensions=[2],
        description="Extraction of clinical history features and integration with visual analysis.",
    ),
    ClinicalQueryType.RISK_ASSESSMENT: QueryTemplate(
        query_type=ClinicalQueryType.RISK_ASSESSMENT,
        template="Assess the malignancy risk for a patient presenting with {clinical_history}. Is the risk high or low? Provide clinical justification.",
        auto_scorable=True,
        scoring_method="triage_sensitivity",
        target_dimensions=[7],
        description="Detailed malignancy risk assessment.",
    ),
    ClinicalQueryType.DERMOSCOPIC: QueryTemplate(
        query_type=ClinicalQueryType.DERMOSCOPIC,
        template="Identify and describe key dermoscopic structures or morphology features visible in the lesion image.",
        auto_scorable=False,
        scoring_method="rubric",
        target_dimensions=[1, 2],
        description="Identification of key dermoscopic visual structures.",
    ),
    ClinicalQueryType.COMPARATIVE: QueryTemplate(
        query_type=ClinicalQueryType.COMPARATIVE,
        template="Compare the likelihood of '{diagnosis}' vs. '{differential_fallback}' for this patient. Which is more probable and why?",
        auto_scorable=True,
        scoring_method="exact_match",
        target_dimensions=[4],
        description="Comparative likelihood analysis.",
    ),
    ClinicalQueryType.TREATMENT: QueryTemplate(
        query_type=ClinicalQueryType.TREATMENT,
        template="Propose a first-line treatment plan for a confirmed diagnosis of '{diagnosis}'. Include non-pharmacological advice if applicable.",
        auto_scorable=False,
        scoring_method="rubric",
        target_dimensions=[2],
        description="Appropriateness and safety of the proposed treatment plan.",
    ),
    ClinicalQueryType.PATIENT_COMMUNICATION: QueryTemplate(
        query_type=ClinicalQueryType.PATIENT_COMMUNICATION,
        template="Draft a communication letter or script explaining the diagnosis of '{diagnosis}' to the patient in layman terms.",
        auto_scorable=False,
        scoring_method="llm_judge",
        target_dimensions=[2],
        description="Translating clinical jargon to patient-friendly communication.",
    ),
}


class QuerySuiteGenerator:
    """Generates concrete queries for a given case based on templates."""

    def __init__(self, query_types: list[ClinicalQueryType] | None = None) -> None:
        self.query_types = query_types or self.auto_scorable_types()

    @classmethod
    def auto_scorable_types(cls) -> list[ClinicalQueryType]:
        """Returns all auto-scorable query types."""
        return [t for t, templ in QUERY_TEMPLATES.items() if templ.auto_scorable]

    @classmethod
    def all_types(cls) -> list[ClinicalQueryType]:
        """Returns all 11 clinical query types."""
        return list(ClinicalQueryType)

    def generate_queries(self, case: dict[str, Any]) -> list[dict[str, Any]]:
        """Generates concrete queries using templates and case metadata."""
        queries = []
        gt = case.get("ground_truth", {})
        diagnosis = gt.get("diagnosis_label", "unknown skin condition")
        clinical_history = case.get("clinical_history", "")
        if not clinical_history:
            metadata = case.get("patient_context", {})
            age = metadata.get("age", "")
            sex = metadata.get("sex", "")
            loc = metadata.get("location", "")
            clinical_history = f"{age}yo {sex}"
            if loc:
                clinical_history += f" with lesion on {loc}"

        ref_diffs = gt.get("reference_differential", [])
        diff_fallback = ref_diffs[1] if len(ref_diffs) > 1 else "other benign lesions"

        for qtype in self.query_types:
            template_obj = QUERY_TEMPLATES.get(qtype)
            if not template_obj:
                continue

            query_text = template_obj.template.format(
                diagnosis=diagnosis,
                clinical_history=clinical_history,
                differential_fallback=diff_fallback,
            )

            queries.append({
                "query_type": qtype.value,
                "query": query_text,
                "auto_scorable": template_obj.auto_scorable,
                "scoring_method": template_obj.scoring_method,
                "target_dimensions": template_obj.target_dimensions,
            })

        return queries


def extend_case_with_queries(
    case: dict[str, Any], query_types: list[ClinicalQueryType] | None = None
) -> dict[str, Any]:
    """Extends a case dictionary with a generated query suite."""
    generator = QuerySuiteGenerator(query_types)
    case_copy = dict(case)
    case_copy["query_suite"] = generator.generate_queries(case_copy)
    return case_copy
