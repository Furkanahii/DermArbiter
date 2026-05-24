"""DermArbiter Experiments — Benchmarking, Analysis & Ablation Pipeline.

Provides:
    • ``BenchmarkRunner``  — run the full pipeline on JSONL test cases
    • ``ResultsAnalyzer``  — compute accuracy, F1, calibration, and efficiency metrics
    • ``AblationRunner``   — systematic ablation over agents, tools, and rounds
"""

from dermarbiter.experiments.runner import BenchmarkRunner
from dermarbiter.experiments.analyze import ResultsAnalyzer
from dermarbiter.experiments.ablation import AblationRunner

__all__ = [
    "BenchmarkRunner",
    "ResultsAnalyzer",
    "AblationRunner",
]
