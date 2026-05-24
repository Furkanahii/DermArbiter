#!/usr/bin/env python3
"""DermArbiter — Colab Setup Script.

Installs all dependencies and prepares the environment for the
tool validation pipeline on Google Colab T4.

Usage (first cell of Colab notebook):
    !git clone https://github.com/<user>/DermArbiter.git
    %cd DermArbiter
    !python scripts/setup_colab.py

Or manually:
    !pip install -e ".[dev]"
    !python scripts/setup_colab.py --check-only
"""

from __future__ import annotations

import os
import subprocess
import sys


def run(cmd: str) -> int:
    """Run a shell command and return the exit code."""
    print(f"\n{'─'*50}")
    print(f"  $ {cmd}")
    print(f"{'─'*50}")
    return subprocess.call(cmd, shell=True)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                        help="Only check dependencies, don't install.")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token for gated models.")
    args = parser.parse_args()

    print("═" * 50)
    print("  DermArbiter — Colab Environment Setup")
    print("═" * 50)

    # Step 1: Check GPU
    print("\n📊 GPU Check:")
    run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'No GPU detected'")

    if args.check_only:
        print("\n🔍 Checking existing packages...")
        run(f"{sys.executable} -c \"import torch; print(f'PyTorch {{torch.__version__}}, CUDA {{torch.cuda.is_available()}}')\"")
        run(f"{sys.executable} -c \"import timm; print(f'timm {{timm.__version__}}')\"")
        run(f"{sys.executable} -c \"import open_clip; print(f'open_clip {{open_clip.__version__}}')\"")
        run(f"{sys.executable} -c \"import transformers; print(f'transformers {{transformers.__version__}}')\"")
        run(f"{sys.executable} -c \"import chromadb; print(f'chromadb {{chromadb.__version__}}')\"")
        run(f"{sys.executable} -c \"import bitsandbytes; print(f'bitsandbytes {{bitsandbytes.__version__}}')\"")
        return

    # Step 2: Install project in dev mode
    print("\n📦 Installing DermArbiter...")
    run(f"{sys.executable} -m pip install -e '.[dev]' --quiet")

    # Step 3: Additional Colab-specific packages
    print("\n📦 Installing Colab-specific packages...")
    colab_packages = [
        "bitsandbytes>=0.43.0",     # 4-bit quantisation
        "accelerate>=0.30.0",       # device_map="auto"
        "flash-attn>=2.5.0",        # Faster attention (optional)
    ]

    for pkg in colab_packages:
        pkg_name = pkg.split(">=")[0].split("==")[0]
        try:
            __import__(pkg_name.replace("-", "_"))
            print(f"  ✅ {pkg_name} already installed")
        except ImportError:
            print(f"  📥 Installing {pkg}...")
            run(f"{sys.executable} -m pip install '{pkg}' --quiet")

    # Step 4: Set HF_TOKEN
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        print(f"\n🔑 HF_TOKEN set from --hf-token argument")
    elif os.environ.get("HF_TOKEN"):
        print(f"\n🔑 HF_TOKEN already set in environment")
    else:
        print("\n⚠️  HF_TOKEN not set. Gated models (DermoGPT, MedGemma) will be skipped.")
        print("   Set via: export HF_TOKEN='hf_...' or --hf-token argument")
        print("   Or in Colab: from google.colab import userdata; os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')")

    # Step 5: Create required directories
    print("\n📁 Creating directory structure...")
    dirs = ["weights/", "data/", "results/", "data/chroma_cases", "data/chroma_guidelines"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  ✅ {d}")

    # Step 6: Final verification
    print("\n🔬 Final verification:")
    run(f"{sys.executable} -c \"\
import torch; \
print(f'  PyTorch: {{torch.__version__}}'); \
print(f'  CUDA:    {{torch.cuda.is_available()}}'); \
print(f'  GPU:     {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \\\"N/A\\\"}}'); \
print(f'  VRAM:    {{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}} GB' if torch.cuda.is_available() else '  VRAM:    N/A')\"")

    print("\n" + "═" * 50)
    print("  ✅ Setup complete!")
    print()
    print("  Next steps:")
    print("    1. Run tool validation:")
    print("       !python scripts/validate_tools_colab.py")
    print()
    print("    2. Run mock benchmark:")
    print("       !python -m dermarbiter.evaluation.benchmark_runner \\")
    print("           --mock --max-cases 10 --output results/")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    main()
