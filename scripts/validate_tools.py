#!/usr/bin/env python3
"""DermArbiter — Tool Validation & Smoke Test Script

Attempts to import and instantiate each of the 9 diagnostic tools,
runs a quick smoke test on available tools, and prints a formatted
status report.

Usage:
    python scripts/validate_tools.py
    python scripts/validate_tools.py --smoke-test
    python scripts/validate_tools.py --verbose
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Tool definitions
# ═══════════════════════════════════════════════════════════════════════════

# Each entry: (module_path, class_name, tool_name, description)
TOOL_DEFINITIONS = [
    (
        "dermarbiter.tools.panderm_tool",
        "PanDermClassifier",
        "panderm_classifier",
        "PanDerm universal dermatology foundation model classifier",
    ),
    (
        "dermarbiter.tools.make_tool",
        "MAKEAnnotator",
        "make_annotator",
        "MAKE multi-attribute knowledge extraction (ABCDE)",
    ),
    (
        "dermarbiter.tools.dermogpt_tool",
        "DermoGPTVQA",
        "dermogpt_vqa",
        "DermoGPT dermatology visual question answering",
    ),
    (
        "dermarbiter.tools.medgemma_tool",
        "MedGemmaVQA",
        "general_vqa",
        "MedGemma general medical VQA (second opinion)",
    ),
    (
        "dermarbiter.tools.guideline_rag",
        "GuidelineRAG",
        "guideline_rag",
        "RAG over clinical guidelines (BAD, AAD, etc.)",
    ),
    (
        "dermarbiter.tools.case_rag",
        "CaseRAG",
        "case_rag",
        "RAG over historical case database",
    ),
    (
        "dermarbiter.tools.ontology_graph",
        "OntologyGraph",
        "ontology_graph",
        "ICD-10, SNOMED-CT, DermLex ontology graph",
    ),
    (
        "dermarbiter.tools.fairness_probe",
        "FairnessProbe",
        "fairness_probe",
        "Fitzpatrick skin tone fairness evaluation",
    ),
    (
        "dermarbiter.tools.uncertainty_probe",
        "UncertaintyProbe",
        "uncertainty_probe",
        "Epistemic/aleatoric uncertainty quantification",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Validation logic
# ═══════════════════════════════════════════════════════════════════════════


def validate_tool_import(module_path: str, class_name: str) -> tuple[bool, str, Any]:
    """
    Try to import a tool module and retrieve its class.

    Returns:
        (success, message, class_or_none)
    """
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return True, "importable", cls
    except ImportError as e:
        return False, f"ImportError: {e}", None
    except AttributeError:
        return False, f"Class '{class_name}' not found in {module_path}", None
    except Exception as e:
        return False, f"Error: {e}", None


def validate_tool_instantiation(cls: Any) -> tuple[bool, str, Any]:
    """
    Try to instantiate a tool class.

    Returns:
        (success, message, instance_or_none)
    """
    try:
        instance = cls()
        return True, "instantiable", instance
    except Exception as e:
        return False, f"InstantiationError: {e}", None


def run_smoke_test(instance: Any, tool_name: str) -> tuple[bool, str, float]:
    """
    Run a minimal smoke test on an instantiated tool.

    Returns:
        (success, message, elapsed_ms)
    """
    try:
        t0 = time.time()
        output = instance.run(
            image_path=None,
            query="Smoke test: assess lesion",
        )
        elapsed_ms = (time.time() - t0) * 1000

        # Validate output structure
        if hasattr(output, "tool_name") and hasattr(output, "confidence"):
            return True, f"OK (conf={output.confidence:.2f})", elapsed_ms
        else:
            return False, "Invalid output structure", elapsed_ms
    except NotImplementedError:
        return False, "Not implemented", 0.0
    except Exception as e:
        return False, f"RuntimeError: {str(e)[:60]}", 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Registry validation
# ═══════════════════════════════════════════════════════════════════════════


def validate_tool_registry() -> tuple[bool, str]:
    """Validate that the ToolRegistry can be imported and used."""
    try:
        from dermarbiter.tools.base_tool import ToolRegistry
        registry = ToolRegistry()
        assert len(registry) == 0
        return True, "ToolRegistry OK"
    except Exception as e:
        return False, f"ToolRegistry error: {e}"


def validate_base_tool() -> tuple[bool, str]:
    """Validate that BaseTool ABC can be imported."""
    try:
        from dermarbiter.tools.base_tool import BaseTool, ToolOutput
        assert BaseTool is not None
        assert ToolOutput is not None
        return True, "BaseTool & ToolOutput OK"
    except Exception as e:
        return False, f"BaseTool error: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════


def print_header() -> None:
    print("\n╔" + "═" * 78 + "╗")
    print("║" + "  DERMARBITER — TOOL VALIDATION REPORT".center(78) + "║")
    print("╚" + "═" * 78 + "╝")


def print_infrastructure(results: list[tuple[str, bool, str]]) -> None:
    print("\n  ── Infrastructure " + "─" * 58)
    for name, ok, msg in results:
        icon = "✓" if ok else "✗"
        print(f"    {icon}  {name:<30} {msg}")


def print_tool_table(
    rows: list[dict],
    show_smoke: bool = False,
) -> None:
    # Header
    print("\n  ── Tool Status " + "─" * 61)
    if show_smoke:
        print(
            f"  {'#':<3} {'Tool Name':<22} {'Import':<10} {'Init':<10} "
            f"{'Smoke Test':<18} {'Time':>8}"
        )
        print("  " + "─" * 78)
    else:
        print(f"  {'#':<3} {'Tool Name':<22} {'Import':<10} {'Init':<10} {'Description':<36}")
        print("  " + "─" * 78)

    for i, row in enumerate(rows, 1):
        import_icon = "✓" if row["import_ok"] else "✗"
        init_icon   = "✓" if row["init_ok"]   else "✗" if row["import_ok"] else "-"

        if show_smoke:
            smoke_icon = "✓" if row.get("smoke_ok") else "✗" if row.get("smoke_ran") else "-"
            smoke_msg  = row.get("smoke_msg", "")[:16]
            smoke_time = row.get("smoke_ms", 0)
            time_str = f"{smoke_time:.0f}ms" if smoke_time > 0 else "-"
            print(
                f"  {i:<3} {row['name']:<22} {import_icon:<10} {init_icon:<10} "
                f"{smoke_icon} {smoke_msg:<15} {time_str:>8}"
            )
        else:
            desc = row["description"][:36]
            print(f"  {i:<3} {row['name']:<22} {import_icon:<10} {init_icon:<10} {desc:<36}")


def print_summary(rows: list[dict], show_smoke: bool = False) -> None:
    total = len(rows)
    importable = sum(1 for r in rows if r["import_ok"])
    instantiable = sum(1 for r in rows if r["init_ok"])

    print("\n  ── Summary " + "─" * 65)
    print(f"    Total tools:       {total}")
    print(f"    Importable:        {importable}/{total}")
    print(f"    Instantiable:      {instantiable}/{total}")

    if show_smoke:
        smoke_pass = sum(1 for r in rows if r.get("smoke_ok"))
        smoke_ran  = sum(1 for r in rows if r.get("smoke_ran"))
        print(f"    Smoke tests run:   {smoke_ran}/{total}")
        print(f"    Smoke tests pass:  {smoke_pass}/{smoke_ran if smoke_ran else 1}")

    # List failing tools
    failing = [r["name"] for r in rows if not r["import_ok"]]
    if failing:
        print(f"\n    ⚠ Unavailable tools: {', '.join(failing)}")
        print("      Install missing dependencies or check configuration.")
    else:
        print("\n    ✓ All tools importable!")

    print("  " + "─" * 78)


def print_errors(rows: list[dict]) -> None:
    errors = [r for r in rows if r.get("error_detail")]
    if not errors:
        return

    print("\n  ── Error Details " + "─" * 59)
    for row in errors:
        print(f"\n    {row['name']}:")
        print(f"      {row['error_detail']}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DermArbiter — Tool Validation & Smoke Test",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run smoke tests on instantiable tools.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed error tracebacks.",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Save results to JSON file.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    print_header()

    # Infrastructure checks
    infra_results = []
    bt_ok, bt_msg = validate_base_tool()
    infra_results.append(("BaseTool & ToolOutput", bt_ok, bt_msg))
    tr_ok, tr_msg = validate_tool_registry()
    infra_results.append(("ToolRegistry", tr_ok, tr_msg))
    print_infrastructure(infra_results)

    if not bt_ok:
        print("\n  ✗ Cannot proceed: BaseTool import failed.")
        sys.exit(1)

    # Validate each tool
    rows = []
    for module_path, class_name, tool_name, description in TOOL_DEFINITIONS:
        row = {
            "name": tool_name,
            "class_name": class_name,
            "module": module_path,
            "description": description,
            "import_ok": False,
            "init_ok": False,
            "smoke_ran": False,
            "smoke_ok": False,
            "smoke_msg": "",
            "smoke_ms": 0.0,
            "error_detail": "",
        }

        # Import
        imp_ok, imp_msg, cls = validate_tool_import(module_path, class_name)
        row["import_ok"] = imp_ok
        if not imp_ok:
            row["error_detail"] = imp_msg
            rows.append(row)
            continue

        # Instantiation
        init_ok, init_msg, instance = validate_tool_instantiation(cls)
        row["init_ok"] = init_ok
        if not init_ok:
            row["error_detail"] = init_msg
            rows.append(row)
            continue

        # Smoke test (optional)
        if args.smoke_test and instance is not None:
            row["smoke_ran"] = True
            smoke_ok, smoke_msg, smoke_ms = run_smoke_test(instance, tool_name)
            row["smoke_ok"] = smoke_ok
            row["smoke_msg"] = smoke_msg
            row["smoke_ms"] = smoke_ms
            if not smoke_ok:
                row["error_detail"] = smoke_msg

        rows.append(row)

    # Display
    print_tool_table(rows, show_smoke=args.smoke_test)
    print_summary(rows, show_smoke=args.smoke_test)

    if args.verbose:
        print_errors(rows)

    # JSON export
    if args.json:
        import json
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=str)
        print(f"\n  Results saved to: {json_path}")

    # Exit code based on import success
    all_ok = all(r["import_ok"] for r in rows)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
