#!/usr/bin/env python3
"""DermArbiter — End-to-End Colab-Optimized Pipeline

Designed for Google Colab but also works as a standalone script.
Automatically handles dependency installation, GPU detection,
sample image download, and result persistence.

Usage in Colab:
    Upload this file and run:
        %run scripts/run_e2e_colab.py --query "Changing mole on back"

    Or paste the contents into a Colab cell.

Usage as script:
    python scripts/run_e2e_colab.py --query "Changing mole on back" --mock
    python scripts/run_e2e_colab.py --query "Red scaly patch" --image data/sample.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# Colab helpers
# ═══════════════════════════════════════════════════════════════════════════


def _is_colab() -> bool:
    """Detect if we are running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _print_section(title: str, char: str = "═") -> None:
    """Print a markdown-style section header."""
    width = 70
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def _install_dependencies() -> None:
    """Install DermArbiter and dependencies in Colab."""
    _print_section("📦 Installing Dependencies")

    packages = [
        "pydantic>=2.0",
        "pyyaml",
        "python-dotenv",
        "langgraph",
        "langchain-google-genai",
        "google-generativeai",
        "Pillow",
    ]

    for pkg in packages:
        print(f"  Installing {pkg}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Try installing dermarbiter from the local checkout or PyPI
    project_root = Path.cwd()
    if (project_root / "pyproject.toml").exists():
        print("  Installing dermarbiter from local checkout...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-e", str(project_root)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        print("  Installing dermarbiter from PyPI (if available)...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "dermarbiter"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            print("  ⚠ dermarbiter not found on PyPI. Using local sources.")

    print("  ✓ Dependencies installed.")


def _check_gpu() -> dict:
    """Check GPU availability and return device info."""
    _print_section("🖥️  GPU Check")
    gpu_info = {"available": False, "device": "cpu", "name": "N/A", "memory": "N/A"}

    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device"] = "cuda"
            gpu_info["name"] = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            gpu_info["memory"] = f"{mem:.1f} GB"
            print(f"  ✓ GPU detected: {gpu_info['name']} ({gpu_info['memory']})")
        else:
            print("  ⚠ No CUDA GPU detected. Using CPU.")
    except ImportError:
        print("  ⚠ PyTorch not installed. GPU check skipped.")
    except Exception as e:
        print(f"  ⚠ GPU check failed: {e}")

    # Check Apple MPS
    if not gpu_info["available"]:
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                gpu_info["available"] = True
                gpu_info["device"] = "mps"
                gpu_info["name"] = "Apple Silicon (MPS)"
                print(f"  ✓ Apple MPS backend detected.")
        except Exception:
            pass

    return gpu_info


def _download_sample_image(url: str | None = None, save_dir: str = "data/") -> str:
    """Download a sample dermatology image for testing."""
    _print_section("🖼️  Sample Image")

    save_path = Path(save_dir) / "sample_colab.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        print(f"  Using cached image: {save_path}")
        return str(save_path)

    # Use a provided URL or generate a placeholder image
    if url:
        print(f"  Downloading from: {url}")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(save_path))
            print(f"  ✓ Saved to: {save_path}")
            return str(save_path)
        except Exception as e:
            print(f"  ⚠ Download failed: {e}")

    # Create a synthetic placeholder image
    print("  Generating placeholder test image...")
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (224, 224), color=(200, 170, 140))
        draw = ImageDraw.Draw(img)
        # Draw a simulated lesion (dark circle)
        draw.ellipse([72, 72, 152, 152], fill=(120, 80, 60), outline=(90, 60, 40), width=2)
        draw.ellipse([90, 90, 134, 134], fill=(100, 65, 45))
        # Add label
        try:
            draw.text((50, 190), "TEST IMAGE", fill=(100, 100, 100))
        except Exception:
            pass
        img.save(str(save_path))
        print(f"  ✓ Placeholder image saved to: {save_path}")
    except ImportError:
        print("  ⚠ Pillow not installed. No image created.")
        return ""

    return str(save_path)


def _save_to_drive(results: dict, filename: str = "dermarbiter_results.json") -> str | None:
    """Save results to Google Drive if available (Colab only)."""
    if not _is_colab():
        return None

    try:
        from google.colab import drive
        drive_mount = "/content/drive"
        if not os.path.ismount(drive_mount):
            print("\n  Mounting Google Drive...")
            drive.mount(drive_mount)

        save_dir = Path(drive_mount) / "MyDrive" / "DermArbiter"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"  ✓ Results saved to Google Drive: {save_path}")
        return str(save_path)
    except Exception as e:
        print(f"  ⚠ Could not save to Google Drive: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline execution
# ═══════════════════════════════════════════════════════════════════════════


def run_colab_pipeline(
    query: str,
    image_path: str | None = None,
    image_url: str | None = None,
    config_dir: str = "configs/",
    mock: bool = False,
    save_drive: bool = True,
) -> dict:
    """Run the full DermArbiter pipeline in a Colab-friendly way."""

    # ── Header ──
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  🔬  DERMARBITER — COLAB E2E PIPELINE".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    is_colab = _is_colab()
    print(f"\n  Environment:  {'Google Colab' if is_colab else 'Local'}")
    print(f"  Query:        {query}")
    print(f"  Mode:         {'MOCK' if mock else 'REAL (GPU/API)'}")

    # ── Step 1: Dependencies ──
    if is_colab:
        _install_dependencies()

    # ── Step 2: GPU Check ──
    gpu_info = _check_gpu()

    # ── Step 3: Resolve image ──
    if image_path and Path(image_path).exists():
        print(f"\n  Using provided image: {image_path}")
    elif image_url:
        image_path = _download_sample_image(url=image_url)
    else:
        image_path = _download_sample_image()

    # ── Step 4: Ensure project root is importable ──
    project_root = Path.cwd()
    # Walk up to find pyproject.toml
    for parent in [project_root] + list(project_root.parents):
        if (parent / "pyproject.toml").exists() and (parent / "dermarbiter").is_dir():
            project_root = parent
            break
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Resolve config_dir
    if not os.path.isabs(config_dir):
        config_dir = str(project_root / config_dir)

    # ── Step 5: Run pipeline ──
    _print_section("🚀 Running Pipeline")
    t0 = time.time()

    if mock:
        # Import the GPU script's mock pipeline builder
        scripts_dir = project_root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        try:
            from run_e2e_gpu import _build_mock_pipeline, run_pipeline, print_results
            agents, registry, state = _build_mock_pipeline(
                query=query,
                image_path=image_path,
                patient_context={},
            )
            result_state = run_pipeline(agents, registry, state, mock_mode=True)
        except ImportError:
            # Inline fallback if the GPU script is not available
            print("  ⚠ Could not import run_e2e_gpu. Running inline mock...")
            result_state = _run_inline_mock(query, image_path, config_dir)
    else:
        # Real pipeline
        try:
            from run_e2e_gpu import _build_real_pipeline, run_pipeline, print_results
            agents, registry, state = _build_real_pipeline(
                config_dir=config_dir,
                query=query,
                image_path=image_path,
            )
            result_state = run_pipeline(agents, registry, state, mock_mode=False)
        except ImportError:
            print("  ⚠ Could not import run_e2e_gpu. Running inline...")
            result_state = _run_inline_real(query, image_path, config_dir)

    elapsed = time.time() - t0

    # ── Step 6: Display results ──
    _print_section("📊 Results")

    results = {
        "case_id": getattr(result_state, "case_id", "unknown"),
        "query": query,
        "image_path": image_path,
        "final_diagnosis": getattr(result_state, "final_diagnosis", []),
        "consensus_score": getattr(result_state, "consensus_score", 0.0),
        "clinical_report": getattr(result_state, "clinical_report", ""),
        "dissent_notes": getattr(result_state, "dissent_notes", []),
        "telemetry": {
            "total_tokens": getattr(result_state, "total_tokens", 0),
            "tool_invocations": getattr(result_state, "total_tool_calls", 0),
            "evidence_cards": len(getattr(result_state, "evidence_cards", [])),
            "debate_turns": len(getattr(result_state, "debate_log", [])),
            "errors": len(getattr(result_state, "errors", [])),
            "elapsed_seconds": round(elapsed, 3),
            "gpu": gpu_info,
            "mock_mode": mock,
        },
    }

    _print_diagnosis_table(results)
    _print_clinical_report(results)
    _print_telemetry(results)

    # ── Step 7: Persist results ──
    _print_section("💾 Saving Results")

    # Always save locally
    local_path = Path("outputs/colab_results.json")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  ✓ Local:  {local_path}")

    # Save to Google Drive if in Colab
    if save_drive and is_colab:
        _save_to_drive(results)

    print(f"\n  ✅ Pipeline complete in {elapsed:.2f}s")
    return results


def _run_inline_mock(query: str, image_path: str | None, config_dir: str):
    """Inline mock fallback when run_e2e_gpu cannot be imported."""
    from dermarbiter.core.blackboard import BlackboardState

    # We need the GPU script's mock components — but since we can't import,
    # we create a minimal BlackboardState and populate it manually.
    state = BlackboardState(
        case_id=f"CASE-{uuid.uuid4().hex[:12]}",
        query=query,
        image_path=image_path,
        final_diagnosis=["melanocytic nevus", "melanoma", "seborrheic keratosis"],
        consensus_score=0.78,
        clinical_report=(
            "# DermArbiter Clinical Report (Inline Mock)\n\n"
            f"Query: {query}\n\n"
            "The panel consensus favors melanocytic nevus as the primary diagnosis.\n"
            "Further dermoscopic follow-up recommended due to non-trivial melanoma risk."
        ),
    )
    return state


def _run_inline_real(query: str, image_path: str | None, config_dir: str):
    """Inline real pipeline fallback."""
    from dermarbiter.core.config import load_config, AgentConfig
    from dermarbiter.core.model_router import ModelRouter
    from dermarbiter.core.blackboard import BlackboardState
    from dermarbiter.core.orchestrator import DermArbiterOrchestrator
    from dermarbiter.tools.base_tool import ToolRegistry
    from dermarbiter.agents import (
        SpecialistAgent, GeneralistAgent, SkepticAgent, ModeratorAgent,
    )
    from dermarbiter.tools import (
        PanDermClassifier, MAKEAnnotator, DermoGPTVQA, MedGemmaVQA,
        GuidelineRAG, CaseRAG, OntologyGraph, FairnessProbe, UncertaintyProbe,
    )

    cfg = load_config(config_dir)
    router = ModelRouter(cfg)
    registry = ToolRegistry()

    for ToolCls in [PanDermClassifier, MAKEAnnotator, DermoGPTVQA, MedGemmaVQA,
                    GuidelineRAG, CaseRAG, OntologyGraph, FairnessProbe, UncertaintyProbe]:
        try:
            registry.register(ToolCls())
        except Exception as e:
            print(f"  ⚠ Tool {ToolCls.__name__} skipped: {e}")

    agent_configs = {}
    for role_key, ac in cfg.agents.items():
        agent_configs[role_key] = ac
    for role in ["specialist", "generalist", "skeptic", "moderator"]:
        if role not in agent_configs:
            agent_configs[role] = AgentConfig(
                role=role, model_backend="google_api",
                model_name=cfg.default_model, temperature=cfg.default_temperature,
            )

    agents = {
        "specialist": SpecialistAgent(config=agent_configs["specialist"], model_router=router, tool_registry=registry),
        "generalist": GeneralistAgent(config=agent_configs["generalist"], model_router=router, tool_registry=registry),
        "skeptic":    SkepticAgent(config=agent_configs["skeptic"],    model_router=router),
        "moderator":  ModeratorAgent(config=agent_configs["moderator"],  model_router=router, tool_registry=registry),
    }

    state = BlackboardState(
        case_id=f"CASE-{uuid.uuid4().hex[:12]}",
        query=query, image_path=image_path,
    )

    orch = DermArbiterOrchestrator(agents=agents, tool_registry=registry)
    return orch.run(state)


# ═══════════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════════


def _print_diagnosis_table(results: dict) -> None:
    """Print the differential diagnosis as a formatted table."""
    diagnoses = results.get("final_diagnosis", [])
    score = results.get("consensus_score", 0.0)

    print("\n  ┌─────┬──────────────────────────────────────────────┐")
    print("  │ Rank│ Diagnosis                                    │")
    print("  ├─────┼──────────────────────────────────────────────┤")
    for i, dx in enumerate(diagnoses[:5], 1):
        dx_display = dx.title()[:44].ljust(44)
        print(f"  │  {i}  │ {dx_display} │")
    if not diagnoses:
        print("  │  -  │ No diagnoses produced                        │")
    print("  └─────┴──────────────────────────────────────────────┘")
    print(f"  Consensus Score: {score:.2f}")


def _print_clinical_report(results: dict) -> None:
    """Print the clinical report."""
    report = results.get("clinical_report", "")
    if report:
        print("\n  ── Clinical Report " + "─" * 49)
        for line in report.split("\n"):
            print(f"  {line}")
        print("  " + "─" * 70)
    else:
        print("\n  (No clinical report generated)")


def _print_telemetry(results: dict) -> None:
    """Print telemetry information."""
    telem = results.get("telemetry", {})
    print("\n  ── Telemetry " + "─" * 55)
    print(f"    Total tokens:      {telem.get('total_tokens', 0)}")
    print(f"    Tool invocations:  {telem.get('tool_invocations', 0)}")
    print(f"    Evidence cards:    {telem.get('evidence_cards', 0)}")
    print(f"    Debate turns:      {telem.get('debate_turns', 0)}")
    print(f"    Errors:            {telem.get('errors', 0)}")
    print(f"    Elapsed:           {telem.get('elapsed_seconds', 0):.2f}s")
    gpu = telem.get("gpu", {})
    print(f"    GPU:               {gpu.get('name', 'N/A')} ({gpu.get('memory', 'N/A')})")
    print(f"    Mock mode:         {telem.get('mock_mode', False)}")
    print("  " + "─" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DermArbiter — Colab-Optimized E2E Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query", type=str, required=True, help="Clinical query.")
    parser.add_argument("--image", type=str, default=None, help="Local image path.")
    parser.add_argument("--image-url", type=str, default=None, help="URL to download sample image.")
    parser.add_argument("--config", type=str, default="configs/", help="Config directory.")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode.")
    parser.add_argument("--no-drive", action="store_true", help="Skip Google Drive save.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    run_colab_pipeline(
        query=args.query,
        image_path=args.image,
        image_url=args.image_url,
        config_dir=args.config,
        mock=args.mock,
        save_drive=not args.no_drive,
    )


if __name__ == "__main__":
    main()
