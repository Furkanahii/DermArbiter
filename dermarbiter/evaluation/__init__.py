"""Evaluation module for DermArbiter.

This module provides benchmark runners, metric computations, and fairness
analysis for dermatological diagnosis evaluation:
- Benchmark harnesses for HAM10000, Derm7pt, SkinCon, SkinCap, Fitzpatrick17k
- Classification and captioning metrics
- Fairness metrics: equalized odds, demographic parity
- Uncertainty calibration analysis
"""

from dermarbiter.evaluation.ablation import AblationAnalyzer, VariantStats
from dermarbiter.evaluation.benchmark_runner import BenchmarkRunner, DatasetLoader
from dermarbiter.evaluation.fairness_analyzer import FairnessAnalyzer, GroupMetrics
from dermarbiter.evaluation.metrics import MetricsCalculator

__all__ = [
    "AblationAnalyzer",
    "BenchmarkRunner",
    "DatasetLoader",
    "FairnessAnalyzer",
    "GroupMetrics",
    "MetricsCalculator",
    "VariantStats",
]
