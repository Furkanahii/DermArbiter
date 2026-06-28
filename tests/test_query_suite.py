"""Tests for the query_suite module."""

from __future__ import annotations

import pytest

from dermarbiter.evaluation.query_suite import (
    ClinicalQueryType,
    QuerySuiteGenerator,
    QUERY_TEMPLATES,
    extend_case_with_queries,
)


def test_clinical_query_type_enum() -> None:
    """Verify all 11 enum members are defined."""
    assert len(ClinicalQueryType) == 11
    assert ClinicalQueryType.DIAGNOSIS == "diagnosis"
    assert ClinicalQueryType.DIFFERENTIAL == "differential"
    assert ClinicalQueryType.TRIAGE == "triage"
    assert ClinicalQueryType.CODING == "coding"
    assert ClinicalQueryType.MANAGEMENT == "management"
    assert ClinicalQueryType.HISTORY_ANALYSIS == "history_analysis"
    assert ClinicalQueryType.RISK_ASSESSMENT == "risk_assessment"
    assert ClinicalQueryType.DERMOSCOPIC == "dermoscopic"
    assert ClinicalQueryType.COMPARATIVE == "comparative"
    assert ClinicalQueryType.TREATMENT == "treatment"
    assert ClinicalQueryType.PATIENT_COMMUNICATION == "patient_communication"


def test_query_templates() -> None:
    """Verify templates map correctly and have required fields."""
    for qtype in ClinicalQueryType:
        template = QUERY_TEMPLATES.get(qtype)
        assert template is not None
        assert template.query_type == qtype
        assert isinstance(template.template, str)
        assert isinstance(template.auto_scorable, bool)
        assert isinstance(template.scoring_method, str)
        assert isinstance(template.target_dimensions, list)
        assert len(template.target_dimensions) > 0


def test_generator_default_types() -> None:
    """Verify generator defaults to auto-scorable types."""
    generator = QuerySuiteGenerator()
    assert len(generator.query_types) == len(QuerySuiteGenerator.auto_scorable_types())
    for qtype in generator.query_types:
        assert QUERY_TEMPLATES[qtype].auto_scorable is True


def test_generator_custom_types() -> None:
    """Verify generator accepts custom query types."""
    custom_types = [ClinicalQueryType.DIAGNOSIS, ClinicalQueryType.TREATMENT]
    generator = QuerySuiteGenerator(custom_types)
    assert generator.query_types == custom_types


def test_generate_queries_with_complete_case() -> None:
    """Verify dynamic query text formatting with complete case context."""
    case = {
        "case_id": "TEST-001",
        "clinical_history": "50yo female with growing dark spot on shoulder",
        "ground_truth": {
            "diagnosis_label": "melanoma",
            "reference_differential": ["melanoma", "atypical nevus", "seborrheic keratosis"],
        },
    }
    generator = QuerySuiteGenerator(QuerySuiteGenerator.all_types())
    queries = generator.generate_queries(case)
    
    assert len(queries) == 11
    
    # Check string replacements
    diag_query = next(q for q in queries if q["query_type"] == "diagnosis")
    assert "What is the primary diagnosis" in diag_query["query"]
    
    coding_query = next(q for q in queries if q["query_type"] == "coding")
    assert "melanoma" in coding_query["query"]
    
    comp_query = next(q for q in queries if q["query_type"] == "comparative")
    assert "melanoma" in comp_query["query"]
    assert "atypical nevus" in comp_query["query"]
    
    treatment_query = next(q for q in queries if q["query_type"] == "treatment")
    assert "melanoma" in treatment_query["query"]


def test_generate_queries_fallback_history() -> None:
    """Verify fallback history formatting if clinical_history is missing."""
    case = {
        "case_id": "TEST-002",
        "patient_context": {
            "age": 35,
            "sex": "male",
            "location": "back",
        },
        "ground_truth": {
            "diagnosis_label": "basal cell carcinoma",
        },
    }
    generator = QuerySuiteGenerator([ClinicalQueryType.HISTORY_ANALYSIS])
    queries = generator.generate_queries(case)
    assert len(queries) == 1
    assert "35yo male with lesion on back" in queries[0]["query"]


def test_extend_case_with_queries() -> None:
    """Verify extend helper adds query_suite field to case copy."""
    case = {
        "case_id": "TEST-003",
        "clinical_history": "25yo female",
        "ground_truth": {
            "diagnosis_label": "eczema",
        },
    }
    extended = extend_case_with_queries(case, [ClinicalQueryType.DIAGNOSIS])
    assert "query_suite" in extended
    assert len(extended["query_suite"]) == 1
    assert extended["query_suite"][0]["query_type"] == "diagnosis"
    # Verify original is untouched
    assert "query_suite" not in case
