"""DermArbiter Experiments — Benchmarking, Analysis & Ablation Pipeline.

Provides:
    • ``ExperimentRunner`` — run the full pipeline on JSONL test cases
    • ``ResultsAnalyzer``  — compute accuracy, F1, calibration, and efficiency metrics
    • ``AblationRunner``   — systematic ablation over agents, tools, and rounds
    • ``BaselineRunner``   — single-LLM baseline (no debate)
    • ``FullPanelRunner``  — full debate panel evaluation
"""

from dermarbiter.experiments.runner import ExperimentRunner, BenchmarkRunner
from dermarbiter.experiments.analyze import ResultsAnalyzer
from dermarbiter.experiments.ablation import AblationRunner
from dermarbiter.experiments.baseline import BaselineRunner
from dermarbiter.experiments.full_panel import FullPanelRunner

__all__ = [
    "ExperimentRunner",
    "BenchmarkRunner",
    "ResultsAnalyzer",
    "AblationRunner",
    "BaselineRunner",
    "FullPanelRunner",
]
