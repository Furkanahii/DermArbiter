"""DermArbiter Evaluation — BenchmarkRunner.

Orchestrates end-to-end evaluation of the DermArbiter pipeline across
standardised dermatology benchmarks defined in ``configs/benchmarks.yaml``:

    • HAM10000   — 7-class skin lesion classification
    • Derm7pt    — 2-class dermoscopic analysis
    • SkinCon    — concept detection
    • Fitzpatrick17k — 114-class classification + fairness evaluation

Each benchmark is loaded, run through the orchestrator (mock or real),
and results are saved as JSONL for downstream metric computation via
:class:`dermarbiter.evaluation.metrics.MetricsCalculator`.

Usage::

    from dermarbiter.evaluation.benchmark_runner import BenchmarkRunner

    runner = BenchmarkRunner(config_dir="configs/", mock=True)
    results = runner.run_benchmark("ham10000")
    runner.run_all()
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from dermarbiter.core.blackboard import BlackboardState
from dermarbiter.core.orchestrator import DermArbiterOrchestrator
from dermarbiter.tools.base_tool import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

class DatasetLoader:
    """Load benchmark datasets from various formats.

    Supported formats:
    - JSONL: one JSON object per line
    - CSV:   comma-separated with header row
    - Directory: folder of images with metadata file
    """

    @staticmethod
    def load_jsonl(path: str | Path, max_cases: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load cases from a JSONL file.

        Expected fields per line:
            case_id, image_path, query, ground_truth_label,
            patient_context (optional), fitzpatrick_type (optional)
        """
        cases: List[Dict[str, Any]] = []
        path = Path(path)
        if not path.exists():
            logger.warning("Dataset file not found: %s", path)
            return cases

        with open(path, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSON on line %d: %s", line_num, exc)
                    continue
                cases.append(obj)
                if max_cases is not None and len(cases) >= max_cases:
                    break
        return cases

    @staticmethod
    def load_csv(path: str | Path, max_cases: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load cases from a CSV file with header."""
        import csv

        cases: List[Dict[str, Any]] = []
        path = Path(path)
        if not path.exists():
            logger.warning("Dataset file not found: %s", path)
            return cases

        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cases.append(dict(row))
                if max_cases is not None and len(cases) >= max_cases:
                    break
        return cases

    @staticmethod
    def load_ham10000(
        data_dir: str | Path,
        split: str = "test",
        max_cases: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load HAM10000 dataset.

        Expects:
            data_dir/
                images/           (ISIC_XXXXXXX.jpg)
                HAM10000_metadata.csv
                    columns: lesion_id, image_id, dx, dx_type, age, sex, localization
        """
        data_dir = Path(data_dir)
        metadata_path = data_dir / "HAM10000_metadata.csv"

        if not metadata_path.exists():
            # Fallback: try JSONL
            jsonl_path = data_dir / f"{split}.jsonl"
            if jsonl_path.exists():
                return DatasetLoader.load_jsonl(jsonl_path, max_cases)
            logger.warning("HAM10000 metadata not found at %s", metadata_path)
            return []

        import csv

        cases: List[Dict[str, Any]] = []
        with open(metadata_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                image_id = row.get("image_id", "")
                image_path = str(data_dir / "images" / f"{image_id}.jpg")

                case = {
                    "case_id": image_id,
                    "image_path": image_path,
                    "query": "What is the diagnosis for this skin lesion?",
                    "ground_truth_label": row.get("dx", "").strip().lower(),
                    "patient_context": {
                        "age": row.get("age", ""),
                        "sex": row.get("sex", ""),
                        "localization": row.get("localization", ""),
                    },
                }
                cases.append(case)
                if max_cases is not None and len(cases) >= max_cases:
                    break
        return cases

    @staticmethod
    def load_fitzpatrick17k(
        data_dir: str | Path,
        split: str = "test",
        max_cases: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load Fitzpatrick17k dataset.

        Expects:
            data_dir/
                images/
                fitzpatrick17k.csv
                    columns: hasher, label, fitzpatrick, three_partition_label, nine_partition_label, url
        """
        data_dir = Path(data_dir)
        metadata_path = data_dir / "fitzpatrick17k.csv"

        if not metadata_path.exists():
            jsonl_path = data_dir / f"{split}.jsonl"
            if jsonl_path.exists():
                return DatasetLoader.load_jsonl(jsonl_path, max_cases)
            logger.warning("Fitzpatrick17k metadata not found at %s", metadata_path)
            return []

        import csv

        # The published CSV uses `fitzpatrick_scale` (self-reported) and
        # `fitzpatrick_centaur` (annotator-verified). Earlier scrapes shipped a
        # single `fitzpatrick` column. Probe each in order; -1 (unknown) maps
        # to empty so fairness subgroup analysis can drop these rows.
        fitz_map = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI"}

        def _resolve_fitz(row: Dict[str, str]) -> str:
            for key in ("fitzpatrick_scale", "fitzpatrick_centaur", "fitzpatrick"):
                raw = (row.get(key) or "").strip()
                if not raw or raw == "-1":
                    continue
                return fitz_map.get(raw, raw)
            return ""

        cases: List[Dict[str, Any]] = []
        with open(metadata_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                hasher = row.get("md5hash", row.get("hasher", ""))
                image_path = str(data_dir / "images" / f"{hasher}.jpg")
                fitzpatrick_type = _resolve_fitz(row)

                case = {
                    "case_id": hasher,
                    "image_path": image_path,
                    "query": "What is the diagnosis for this skin condition?",
                    "ground_truth_label": row.get("label", row.get("three_partition_label", "")).strip().lower(),
                    "fitzpatrick_type": fitzpatrick_type,
                    "patient_context": {
                        "fitzpatrick_type": fitzpatrick_type,
                    },
                }
                cases.append(case)
                if max_cases is not None and len(cases) >= max_cases:
                    break
        return cases

    @staticmethod
    def load_generic(
        data_dir: str | Path,
        split: str = "test",
        max_cases: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fallback loader: tries JSONL, then CSV in the data directory."""
        data_dir = Path(data_dir)
        # Try JSONL first
        for ext in [".jsonl", ".json"]:
            path = data_dir / f"{split}{ext}"
            if path.exists():
                return DatasetLoader.load_jsonl(path, max_cases)
        # Try CSV
        for fname in [f"{split}.csv", "metadata.csv", "test.csv"]:
            path = data_dir / fname
            if path.exists():
                return DatasetLoader.load_csv(path, max_cases)
        logger.warning("No dataset found in %s", data_dir)
        return []


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Run DermArbiter evaluation across configured benchmarks.

    Parameters
    ----------
    config_dir : str or Path
        Path to the configuration directory containing YAML files.
    output_dir : str or Path
        Directory for writing result JSONL files.
    mock : bool
        If True, use mock agents/tools for CPU-only testing.
    max_cases : int, optional
        Limit the number of cases per benchmark.
    """

    # Maps benchmark names to specialised loaders
    _LOADERS = {
        "ham10000": DatasetLoader.load_ham10000,
        "fitzpatrick17k": DatasetLoader.load_fitzpatrick17k,
    }

    def __init__(
        self,
        config_dir: str | Path = "configs/",
        output_dir: str | Path = "results/",
        mock: bool = False,
        max_cases: Optional[int] = None,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.mock = mock
        self.max_cases = max_cases

        # Load benchmark definitions
        self._benchmarks = self._load_benchmark_config()

        # Pipeline components (lazily initialised)
        self._agents: Optional[Dict[str, Any]] = None
        self._tool_registry: Optional[ToolRegistry] = None
        self._orchestrator: Optional[DermArbiterOrchestrator] = None

    # ----- Config ----------------------------------------------------------

    def _load_benchmark_config(self) -> Dict[str, Dict[str, Any]]:
        """Load benchmark definitions from benchmarks.yaml."""
        config_path = self.config_dir / "benchmarks.yaml"
        if not config_path.exists():
            logger.warning("benchmarks.yaml not found at %s", config_path)
            return {}

        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        return raw.get("benchmarks", {})

    @property
    def available_benchmarks(self) -> List[str]:
        """Names of all configured benchmarks."""
        return list(self._benchmarks.keys())

    # ----- Setup -----------------------------------------------------------

    def _setup_pipeline(self) -> None:
        """Initialise agents, tools, and orchestrator."""
        if self._orchestrator is not None:
            return

        if self.mock:
            from dermarbiter.core.mock_factory import create_mock_agents, create_mock_registry

            self._agents = create_mock_agents()
            self._tool_registry = create_mock_registry()
        else:
            raise NotImplementedError(
                "Non-mock mode requires live LLM backends. "
                "Use mock=True for CPU-only testing."
            )

        self._orchestrator = DermArbiterOrchestrator(
            agents=self._agents,
            tool_registry=self._tool_registry,
        )

    # ----- Dataset loading -------------------------------------------------

    def _load_dataset(
        self,
        benchmark_name: str,
        benchmark_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Load dataset for a given benchmark."""
        data_dir = benchmark_config.get("data_dir", f"data/{benchmark_name}/")

        # Use specialised loader if available
        loader = self._LOADERS.get(benchmark_name)
        if loader is not None:
            return loader(
                data_dir=data_dir,
                split=benchmark_config.get("split", "test"),
                max_cases=self.max_cases,
            )

        # Fallback to generic loader
        return DatasetLoader.load_generic(
            data_dir=data_dir,
            split=benchmark_config.get("split", "test"),
            max_cases=self.max_cases,
        )

    # ----- Single case execution -------------------------------------------

    def _run_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single case through the pipeline and return result dict."""
        assert self._orchestrator is not None

        initial_state = BlackboardState(
            case_id=case.get("case_id", "UNKNOWN"),
            query=case.get("query", ""),
            image_path=case.get("image_path"),
            patient_context=case.get("patient_context", {}),
        )

        t0 = time.monotonic()
        final_state = self._orchestrator.run(initial_state)
        latency_ms = (time.monotonic() - t0) * 1000.0

        predicted = (
            final_state.final_diagnosis[0]
            if final_state.final_diagnosis
            else ""
        )

        # Count distinct debate rounds
        debate_rounds = set()
        for turn in final_state.debate_log:
            debate_rounds.add(turn.round_num)

        result = {
            "case_id": case.get("case_id", "UNKNOWN"),
            "predicted": predicted,
            "ground_truth": case.get("ground_truth_label", ""),
            "final_diagnosis": list(final_state.final_diagnosis),
            "consensus_score": final_state.consensus_score,
            "early_exit": final_state.early_exit,
            "num_debate_rounds": len(debate_rounds),
            "total_tokens": final_state.total_tokens,
            "total_tool_calls": final_state.total_tool_calls,
            "latency_ms": round(latency_ms, 2),
        }

        # Preserve fairness-relevant metadata
        if "fitzpatrick_type" in case:
            result["fitzpatrick_type"] = case["fitzpatrick_type"]

        return result

    # ----- Benchmark execution ---------------------------------------------

    def run_benchmark(
        self,
        benchmark_name: str,
        output_path: Optional[str | Path] = None,
    ) -> List[Dict[str, Any]]:
        """Run evaluation on a single benchmark.

        Parameters
        ----------
        benchmark_name : str
            Name of the benchmark (must be in benchmarks.yaml).
        output_path : str or Path, optional
            Custom output path.  Defaults to ``output_dir/benchmark_name.jsonl``.

        Returns
        -------
        list of dict
            Per-case result records.
        """
        if benchmark_name not in self._benchmarks:
            raise ValueError(
                f"Unknown benchmark '{benchmark_name}'. "
                f"Available: {self.available_benchmarks}"
            )

        bench_config = self._benchmarks[benchmark_name]
        logger.info("Starting benchmark: %s (%s)", bench_config.get("name", benchmark_name), benchmark_name)

        # Setup pipeline
        self._setup_pipeline()

        # Load dataset
        cases = self._load_dataset(benchmark_name, bench_config)
        if not cases:
            logger.warning("No cases loaded for benchmark %s", benchmark_name)
            return []

        logger.info("Loaded %d cases for %s", len(cases), benchmark_name)

        # Determine output path
        if output_path is None:
            output_path = self.output_dir / f"{benchmark_name}.jsonl"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run cases
        results: List[Dict[str, Any]] = []
        n_errors = 0

        with open(output_path, "w", encoding="utf-8") as fh:
            for idx, case in enumerate(cases, 1):
                case_id = case.get("case_id", f"case_{idx}")
                logger.info(
                    "[%d/%d] %s — case %s",
                    idx, len(cases), benchmark_name, case_id,
                )

                try:
                    result = self._run_case(case)
                except Exception as exc:
                    logger.error(
                        "Case %s failed: %s", case_id, exc, exc_info=True,
                    )
                    result = {
                        "case_id": case_id,
                        "predicted": "",
                        "ground_truth": case.get("ground_truth_label", ""),
                        "final_diagnosis": [],
                        "consensus_score": 0.0,
                        "early_exit": False,
                        "num_debate_rounds": 0,
                        "total_tokens": 0,
                        "total_tool_calls": 0,
                        "latency_ms": 0.0,
                        "error": str(exc),
                    }
                    n_errors += 1

                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.append(result)

        logger.info(
            "Benchmark %s complete: %d cases, %d errors, results → %s",
            benchmark_name, len(results), n_errors, output_path,
        )
        return results

    def run_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Run all configured benchmarks.

        Returns
        -------
        dict
            Mapping benchmark_name → list of result records.
        """
        all_results: Dict[str, List[Dict[str, Any]]] = {}
        for name in self.available_benchmarks:
            try:
                all_results[name] = self.run_benchmark(name)
            except Exception as exc:
                logger.error("Benchmark %s failed: %s", name, exc, exc_info=True)
                all_results[name] = []
        return all_results

    # ----- Summary ---------------------------------------------------------

    def print_summary(self, results: Dict[str, List[Dict[str, Any]]]) -> None:
        """Print a brief summary of benchmark results."""
        sep = "═" * 64
        print(f"\n{sep}")
        print("  DermArbiter Benchmark Summary")
        print(sep)

        for name, records in results.items():
            if not records:
                print(f"  {name:20s}  ⚠ No results")
                continue
            n = len(records)
            correct = sum(
                1 for r in records
                if r.get("predicted", "").strip().lower() == r.get("ground_truth", "").strip().lower()
            )
            acc = correct / n if n > 0 else 0.0
            errors = sum(1 for r in records if "error" in r)
            print(f"  {name:20s}  n={n:5d}  acc={acc:.4f}  errors={errors}")

        print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="DermArbiter Benchmark Runner")
    parser.add_argument("--config", default="configs/", help="Config directory path.")
    parser.add_argument("--output", default="results/", help="Output directory.")
    parser.add_argument("--benchmark", default=None, help="Run specific benchmark (default: all).")
    parser.add_argument("--mock", action="store_true", help="Use mock pipeline.")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit cases per benchmark.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    runner = BenchmarkRunner(
        config_dir=args.config,
        output_dir=args.output,
        mock=args.mock,
        max_cases=args.max_cases,
    )

    if args.benchmark:
        runner.run_benchmark(args.benchmark)
    else:
        results = runner.run_all()
        runner.print_summary(results)


if __name__ == "__main__":
    main()
