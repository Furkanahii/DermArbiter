"""
01_panderm_test.py — Colab Test Script for PanDerm Classifier

Run this script in Google Colab (T4 GPU) to verify the PanDerm
tool wrapper works end-to-end with a real model checkpoint.

Usage (Colab):
    1. Upload this script or clone the DermArbiter repo
    2. Install dependencies: !pip install -e ".[dev]"
    3. Download PanDerm weights (see instructions below)
    4. Run: !python notebooks/01_panderm_test.py

Usage (Local):
    python notebooks/01_panderm_test.py [--image PATH] [--weights PATH]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Setup — add project root to path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_environment() -> dict[str, bool]:
    """Check that required packages are available."""
    checks: dict[str, bool] = {}

    try:
        import torch
        checks["torch"] = True
        checks["cuda_available"] = torch.cuda.is_available()
        if checks["cuda_available"]:
            checks["gpu_name"] = torch.cuda.get_device_name(0)
            checks["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1e9, 1
            )
    except ImportError:
        checks["torch"] = False

    try:
        import timm
        checks["timm"] = True
        checks["timm_version"] = timm.__version__
    except ImportError:
        checks["timm"] = False

    try:
        from PIL import Image
        checks["pillow"] = True
    except ImportError:
        checks["pillow"] = False

    try:
        import torchvision
        checks["torchvision"] = True
    except ImportError:
        checks["torchvision"] = False

    return checks


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# 1. Test: PanDerm tool interface
# ---------------------------------------------------------------------------

def test_interface() -> bool:
    """Verify PanDerm implements BaseTool correctly."""
    print_section("Test 1: Interface Compliance")

    from dermarbiter.tools.base_tool import BaseTool, ToolRegistry
    from dermarbiter.tools.panderm_tool import PanDermClassifier

    tool = PanDermClassifier()

    checks = {
        "Is BaseTool subclass": isinstance(tool, BaseTool),
        "Has name": tool.name == "panderm_classifier",
        "Has description": len(tool.description) > 20,
        "Not loaded on init": tool._loaded is False,
        "Can register": True,
    }

    # Test registry
    registry = ToolRegistry()
    registry.register(tool)
    checks["In registry"] = "panderm_classifier" in registry

    # Test schema
    schema = tool.to_schema()
    checks["Schema valid"] = (
        "name" in schema and "parameters" in schema
    )

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")

    all_passed = all(checks.values())
    print(f"\n  {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    return all_passed


# ---------------------------------------------------------------------------
# 2. Test: Input validation
# ---------------------------------------------------------------------------

def test_validation() -> bool:
    """Test input validation edge cases."""
    print_section("Test 2: Input Validation")

    from dermarbiter.tools.panderm_tool import PanDermClassifier

    tool = PanDermClassifier()

    checks = {
        "None image → False": tool.validate_input(None) is False,
        "Missing file → False": tool.validate_input("/no/file.jpg") is False,
        "No extension → False": tool.validate_input("/no/file") is False,
    }

    # Test run() with invalid input
    output = tool.run(image_path=None)
    checks["None → error output"] = output.confidence == 0.0

    output = tool.run(image_path="/no/such/file.jpg")
    checks["Missing → error output"] = "error" in output.result

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")

    all_passed = all(checks.values())
    print(f"\n  {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    return all_passed


# ---------------------------------------------------------------------------
# 3. Test: Inference with mock model (CPU, no weights needed)
# ---------------------------------------------------------------------------

def test_mock_inference() -> bool:
    """Test inference pipeline with a fake model on CPU."""
    print_section("Test 3: Mock Inference (CPU)")

    import torch
    import torch.nn as nn
    from PIL import Image

    from dermarbiter.tools.base_tool import ToolOutput
    from dermarbiter.tools.panderm_tool import DERMATOLOGY_CLASSES, PanDermClassifier

    tool = PanDermClassifier(device="cpu")

    # Inject fake model
    class FakeEncoder(nn.Module):
        def forward_features(self, x):
            return torch.randn(x.shape[0], 1024)

    tool._model = FakeEncoder()
    tool._head = nn.Linear(1024, 7)
    tool._transform = tool._build_transform()
    tool._device = torch.device("cpu")
    tool._loaded = True

    # Create test image
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (300, 300), color=(150, 100, 80))
        img.save(f.name)
        test_image = f.name

    t0 = time.perf_counter()
    output = tool.run(image_path=test_image)
    elapsed = (time.perf_counter() - t0) * 1000

    checks = {
        "Returns ToolOutput": isinstance(output, ToolOutput),
        "Correct tool_name": output.tool_name == "panderm_classifier",
        "Has predictions": "predictions" in output.result,
        "7 predictions": len(output.result["predictions"]) == 7,
        "Confidence in [0,1]": 0.0 <= output.confidence <= 1.0,
        "Probs sum ≈ 1": abs(
            sum(output.result["all_probabilities"].values()) - 1.0
        ) < 0.01,
        "Labels from known set": all(
            p["disease"] in DERMATOLOGY_CLASSES
            for p in output.result["predictions"]
        ),
        "Has metadata": "model" in output.metadata,
        "Has raw_text": len(output.raw_text) > 0,
        "Has timestamp": len(output.timestamp) > 0,
    }

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")

    print(f"\n  ⏱  Inference time: {elapsed:.1f} ms")
    print(f"  📊 Top prediction: {output.result['predictions'][0]}")
    print(f"  🎯 Confidence: {output.confidence:.4f}")
    print(f"  📝 Summary: {output.raw_text}")

    # Cleanup
    Path(test_image).unlink(missing_ok=True)

    all_passed = all(checks.values())
    print(f"\n  {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    return all_passed


# ---------------------------------------------------------------------------
# 4. Test: Real model inference (requires checkpoint)
# ---------------------------------------------------------------------------

def test_real_inference(
    image_path: str | None = None,
    weights_path: str = "weights/panderm.pth",
    head_path: str = "weights/panderm_head.pth",
) -> bool:
    """Test with real PanDerm weights (if available)."""
    print_section("Test 4: Real Model Inference")

    weights = Path(weights_path)
    if not weights.exists():
        print("  ⏭  SKIPPED — PanDerm weights not found at:")
        print(f"     {weights_path}")
        print("     Download from: https://github.com/SiyuanYan1/PanDerm")
        return True  # Not a failure

    from dermarbiter.tools.panderm_tool import PanDermClassifier

    tool = PanDermClassifier(
        model_path=weights_path,
        head_path=head_path,
    )

    # Use provided image or create a test one
    if image_path and Path(image_path).exists():
        test_image = image_path
    else:
        from PIL import Image
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (400, 400), color=(180, 130, 90))
            img.save(f.name)
            test_image = f.name

    print(f"  📸 Image: {test_image}")
    print(f"  🏋️ Weights: {weights_path}")

    t0 = time.perf_counter()
    output = tool.run(image_path=test_image)
    elapsed = (time.perf_counter() - t0) * 1000

    if "error" in output.result:
        print(f"  ❌ Inference error: {output.result['error']}")
        return False

    print(f"\n  ⏱  Total time (load + infer): {elapsed:.1f} ms")
    print(f"  🏥 Predictions:")
    for pred in output.result["predictions"]:
        bar = "█" * int(pred["probability"] * 40)
        print(f"     {pred['disease']:30s} {pred['probability']:.4f} {bar}")
    print(f"  🎯 Confidence: {output.confidence:.4f}")
    print(f"  📝 {output.raw_text}")

    tool.unload()
    print("  🧹 Model unloaded")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PanDerm Tool Test Script")
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    parser.add_argument("--weights", type=str, default="weights/panderm.pth")
    parser.add_argument("--head", type=str, default="weights/panderm_head.pth")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  DermArbiter — PanDerm Tool Test Suite")
    print("=" * 60)

    # Environment check
    env = check_environment()
    print_section("Environment")
    for k, v in env.items():
        print(f"  {k}: {v}")

    # Run tests
    results = {
        "Interface": test_interface(),
        "Validation": test_validation(),
        "Mock Inference": test_mock_inference(),
        "Real Inference": test_real_inference(
            args.image, args.weights, args.head
        ),
    }

    # Summary
    print_section("SUMMARY")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name}")

    all_ok = all(results.values())
    print(f"\n  {'🎉 All tests passed!' if all_ok else '💥 Some tests failed!'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
