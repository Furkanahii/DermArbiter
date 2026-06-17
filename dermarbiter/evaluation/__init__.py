"""Evaluation module for DermArbiter.

This module provides benchmark runners, metric computations, and fairness
analysis for dermatological diagnosis evaluation:
- Benchmark harnesses for HAM10000, Derm7pt, SkinCon, SkinCap, Fitzpatrick17k
- Classification and captioning metrics
- Fairness metrics: equalized odds, demographic parity
- Uncertainty calibration analysis
- DermAbench: 8-dimension multi-agent evaluation harness
- Standard clinical code mappings (ICD-10, SNOMED-CT)
"""

from dermarbiter.evaluation.ablation import AblationAnalyzer, VariantStats
from dermarbiter.evaluation.benchmark_runner import BenchmarkRunner, DatasetLoader
from dermarbiter.evaluation.derm_codes import (
    all_classes,
    default_management,
    icd10_for,
    is_malignant,
    normalize_to_class,
    reference_record,
    snomed_for,
)
from dermarbiter.evaluation.dermabench import (
    DermAbenchScorer,
    state_to_dermabench_prediction,
)
from dermarbiter.evaluation.fairness_analyzer import FairnessAnalyzer, GroupMetrics
from dermarbiter.evaluation.metrics import MetricsCalculator

__all__ = [
    "AblationAnalyzer",
    "BenchmarkRunner",
    "DatasetLoader",
    "DermAbenchScorer",
    "FairnessAnalyzer",
    "GroupMetrics",
    "MetricsCalculator",
    "VariantStats",
    # derm_codes utilities
    "all_classes",
    "default_management",
    "icd10_for",
    "is_malignant",
    "normalize_to_class",
    "reference_record",
    "snomed_for",
    # dermabench bridge
    "state_to_dermabench_prediction",
]

