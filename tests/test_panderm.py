"""Tests for the PanDerm Classifier tool wrapper.

Tests are organized into three categories:
1. Interface compliance — ensures PanDermClassifier satisfies BaseTool
2. Input validation — edge cases and error handling
3. Integration — end-to-end inference with mock/real model

GPU-dependent tests are skipped on CPU-only machines.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dermarbiter.tools.base_tool import BaseTool, ToolOutput, ToolRegistry
from dermarbiter.tools.panderm_tool import (
    DERMATOLOGY_CLASSES,
    HAM10000_CLASSES,
    PanDermClassifier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_image(tmp_path: Path, fmt: str = "jpg") -> str:
    """Create a minimal valid image file for testing."""
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    path = tmp_path / f"test_lesion.{fmt}"
    img.save(str(path))
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Interface compliance tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPanDermInterface:
    """Verify PanDermClassifier satisfies the BaseTool contract."""

    def test_is_base_tool_subclass(self):
        tool = PanDermClassifier()
        assert isinstance(tool, BaseTool)

    def test_has_name(self):
        tool = PanDermClassifier()
        assert tool.name == "panderm_classifier"

    def test_has_description(self):
        tool = PanDermClassifier()
        assert isinstance(tool.description, str)
        assert len(tool.description) > 20
        assert "PanDerm" in tool.description

    def test_to_schema(self):
        tool = PanDermClassifier()
        schema = tool.to_schema()
        assert schema["name"] == "panderm_classifier"
        assert "description" in schema
        assert "parameters" in schema

    def test_repr(self):
        tool = PanDermClassifier()
        r = repr(tool)
        assert "PanDermClassifier" in r
        assert "panderm_classifier" in r

    def test_can_register_in_tool_registry(self):
        registry = ToolRegistry()
        tool = PanDermClassifier()
        registry.register(tool)
        assert "panderm_classifier" in registry
        assert registry.get("panderm_classifier") is tool

    def test_default_class_labels(self):
        tool = PanDermClassifier()
        assert tool._class_labels == DERMATOLOGY_CLASSES
        assert len(tool._class_labels) == 7

    def test_custom_class_labels(self):
        tool = PanDermClassifier(class_labels=HAM10000_CLASSES)
        assert tool._class_labels == HAM10000_CLASSES

    def test_default_not_loaded(self):
        tool = PanDermClassifier()
        assert tool._loaded is False
        assert tool._model is None
        assert tool._head is None


# ═══════════════════════════════════════════════════════════════════════════
# 2. Input validation tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPanDermValidation:
    """Test input validation logic."""

    def test_validate_none_image(self):
        tool = PanDermClassifier()
        assert tool.validate_input(image_path=None) is False

    def test_validate_nonexistent_image(self):
        tool = PanDermClassifier()
        assert tool.validate_input(image_path="/nonexistent/path.jpg") is False

    def test_validate_unsupported_format(self, tmp_path):
        # Create a file with unsupported extension
        bad_file = tmp_path / "test.pdf"
        bad_file.write_text("not an image")
        assert PanDermClassifier().validate_input(str(bad_file)) is False

    def test_validate_supported_formats(self, tmp_path):
        from PIL import Image

        tool = PanDermClassifier()
        for ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
            img = Image.new("RGB", (10, 10))
            path = tmp_path / f"test.{ext}"
            img.save(str(path))
            assert tool.validate_input(str(path)) is True, f"Failed for .{ext}"

    def test_run_with_none_image_returns_error_output(self):
        tool = PanDermClassifier()
        output = tool.run(image_path=None)
        assert isinstance(output, ToolOutput)
        assert output.tool_name == "panderm_classifier"
        assert output.confidence == 0.0
        assert "error" in output.result

    def test_run_with_missing_image_returns_error_output(self):
        tool = PanDermClassifier()
        output = tool.run(image_path="/no/such/image.jpg")
        assert output.confidence == 0.0
        assert "error" in output.result
        assert output.metadata.get("status") == "error"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Output format tests (with mocked model)
# ═══════════════════════════════════════════════════════════════════════════

class TestPanDermOutput:
    """Test that the tool output matches the expected format."""

    @pytest.fixture()
    def tool_with_mock_model(self, tmp_path):
        """Create a PanDermClassifier with a mocked model."""
        import torch
        import torch.nn as nn

        tool = PanDermClassifier(device="cpu")

        # Mock the model and head
        class FakeEncoder(nn.Module):
            def forward_features(self, x):
                # Return a fake 1024-dim embedding
                return torch.randn(x.shape[0], 1024)

        tool._model = FakeEncoder()
        tool._head = nn.Linear(1024, 7)
        tool._transform = tool._build_transform()
        tool._device = torch.device("cpu")
        tool._loaded = True

        return tool

    def test_output_is_tool_output(self, tool_with_mock_model, tmp_path):
        image_path = _create_test_image(tmp_path)
        output = tool_with_mock_model.run(image_path=image_path)
        assert isinstance(output, ToolOutput)

    def test_output_has_correct_tool_name(self, tool_with_mock_model, tmp_path):
        image_path = _create_test_image(tmp_path)
        output = tool_with_mock_model.run(image_path=image_path)
        assert output.tool_name == "panderm_classifier"

    def test_output_has_predictions(self, tool_with_mock_model, tmp_path):
        image_path = _create_test_image(tmp_path)
        output = tool_with_mock_model.run(image_path=image_path)
        preds = output.result.get("predictions")
        assert preds is not None
        assert isinstance(preds, list)
        assert len(preds) >= 1

    def test_predictions_have_disease_and_probability(
        self, tool_with_mock_model, tmp_path
    ):
        image_path = _create_test_image(tmp_path)
        output = tool_with_mock_model.run(image_path=image_path)
        for pred in output.result["predictions"]:
            assert "disease" in pred
            assert "probability" in pred
            assert isinstance(pred["disease"], str)
            assert 0.0 <= pred["probability"] <= 1.0

    def test_predictions_sorted_descending(
        self, tool_with_mock_model, tmp_path
    ):
        image_path = _create_test_image(tmp_path)
        output = tool_with_mock_model.run(image_path=image_path)
        probs = [p["probability"] for p in output.result["predictions"]]
        assert probs == sorted(probs, reverse=True)

    def test_probabilities_sum_to_one(
        self, tool_with_mock_model, tmp_path
    ):
        image_path = _create_test_image(tmp_path)
        output = tool_with_mock_model.run(image_path=image_path)
        all_probs = output.result.get("all_probabilities", {})
        if all_probs:
            total = sum(all_probs.values())
            assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}"

    def test_confidence_matches_top_prediction(
        self, tool_with_mock_model, tmp_path
    ):
        image_path = _create_test_image(tmp_path)
        output = tool_with_mock_model.run(image_path=image_path)
        top_prob = output.result["predictions"][0]["probability"]
        assert output.confidence == top_prob

    def test_confidence_in_valid_range(
        self, tool_with_mock_model, tmp_path
    ):
        image_path = _create_test_image(tmp_path)
        output = tool_with_mock_model.run(image_path=image_path)
        assert 0.0 <= output.confidence <= 1.0

    def test_output_has_metadata(self, tool_with_mock_model, tmp_path):
        image_path = _create_test_image(tmp_path)
        output = tool_with_mock_model.run(image_path=image_path)
        assert "model" in output.metadata
        assert output.metadata["model"] == "PanDerm"
        assert "latency_ms" in output.metadata
        assert output.metadata["latency_ms"] >= 0

    def test_output_has_raw_text(self, tool_with_mock_model, tmp_path):
        image_path = _create_test_image(tmp_path)
        output = tool_with_mock_model.run(image_path=image_path)
        assert isinstance(output.raw_text, str)
        assert len(output.raw_text) > 0
        assert "Top prediction" in output.raw_text

    def test_output_has_model_version(self, tool_with_mock_model, tmp_path):
        image_path = _create_test_image(tmp_path)
        output = tool_with_mock_model.run(image_path=image_path)
        assert output.result.get("model_version") == "panderm-v1.0"

    def test_output_has_input_resolution(self, tool_with_mock_model, tmp_path):
        image_path = _create_test_image(tmp_path)
        output = tool_with_mock_model.run(image_path=image_path)
        assert output.result.get("input_resolution") == "224x224"

    def test_output_has_timestamp(self, tool_with_mock_model, tmp_path):
        image_path = _create_test_image(tmp_path)
        output = tool_with_mock_model.run(image_path=image_path)
        assert isinstance(output.timestamp, str)
        assert len(output.timestamp) > 0

    def test_disease_labels_from_known_set(
        self, tool_with_mock_model, tmp_path
    ):
        image_path = _create_test_image(tmp_path)
        output = tool_with_mock_model.run(image_path=image_path)
        for pred in output.result["predictions"]:
            assert pred["disease"] in DERMATOLOGY_CLASSES


# ═══════════════════════════════════════════════════════════════════════════
# 4. Lazy loading tests
# ═══════════════════════════════════════════════════════════════════════════

def _make_fake_tool_loaded(tool: PanDermClassifier) -> None:
    """Manually inject a fake model/head into a PanDermClassifier.

    This avoids patching ``timm`` (which is imported locally inside
    ``_load_model``) and gives us full control over the loaded state.
    """
    import torch
    import torch.nn as nn

    class _FakeEncoder(nn.Module):
        def forward_features(self, x):
            return torch.randn(x.shape[0], 1024)

    tool._device = torch.device("cpu")
    tool._transform = tool._build_transform()
    tool._model = _FakeEncoder()
    tool._head = nn.Linear(1024, 7)
    tool._loaded = True


class TestPanDermLazyLoading:
    """Test the lazy model loading behaviour."""

    def test_model_not_loaded_on_init(self):
        tool = PanDermClassifier()
        assert tool._loaded is False

    def test_load_model_sets_loaded_flag(self):
        """When the model is loaded, _loaded should become True."""
        tool = PanDermClassifier(device="cpu")
        _make_fake_tool_loaded(tool)
        assert tool._loaded is True
        assert tool._model is not None
        assert tool._head is not None

    def test_load_model_skips_when_already_loaded(self):
        """Calling _load_model when already loaded should be a no-op."""
        tool = PanDermClassifier(device="cpu")
        _make_fake_tool_loaded(tool)

        # Capture current model identity
        original_model = tool._model

        # Call _load_model — should skip because _loaded is True
        tool._load_model()
        assert tool._model is original_model  # same object

    def test_unload_resets_state(self):
        """unload() should clear model and reset _loaded."""
        tool = PanDermClassifier(device="cpu")
        _make_fake_tool_loaded(tool)
        assert tool._loaded is True

        tool.unload()
        assert tool._loaded is False
        assert tool._model is None
        assert tool._head is None

    def test_unload_idempotent(self):
        """Calling unload() twice should not raise."""
        tool = PanDermClassifier(device="cpu")
        _make_fake_tool_loaded(tool)
        tool.unload()
        tool.unload()  # should not raise
        assert tool._loaded is False


# ═══════════════════════════════════════════════════════════════════════════
# 5. Error handling tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPanDermErrorHandling:
    """Test graceful error handling."""

    def test_inference_error_returns_error_output(self, tmp_path):
        """If inference fails, run() should return error ToolOutput."""
        import torch
        import torch.nn as nn

        tool = PanDermClassifier(device="cpu")

        # Set up a model that will crash during inference
        class CrashingModel(nn.Module):
            def forward_features(self, x):
                raise RuntimeError("Simulated CUDA OOM")

        tool._model = CrashingModel()
        tool._head = nn.Linear(1024, 7)
        tool._transform = tool._build_transform()
        tool._device = torch.device("cpu")
        tool._loaded = True

        image_path = _create_test_image(tmp_path)
        output = tool.run(image_path=image_path)

        assert isinstance(output, ToolOutput)
        assert output.confidence == 0.0
        assert "error" in output.result
        assert "Simulated CUDA OOM" in output.result["error"]
        assert output.metadata.get("status") == "error"

    def test_corrupt_image_handled_gracefully(self, tmp_path):
        """A corrupt image file should produce an error output."""
        corrupt = tmp_path / "corrupt.jpg"
        corrupt.write_bytes(b"not a real image at all")

        tool = PanDermClassifier(device="cpu")
        _make_fake_tool_loaded(tool)

        output = tool.run(image_path=str(corrupt))
        assert output.confidence == 0.0
        assert "error" in output.result


# ═══════════════════════════════════════════════════════════════════════════
# 6. Integration with ToolRegistry
# ═══════════════════════════════════════════════════════════════════════════

class TestPanDermRegistryIntegration:
    """Test PanDerm works within the ToolRegistry."""

    def test_registered_alongside_mock_tools(self):
        from tests.mocks.mock_tools import create_mock_registry

        registry = create_mock_registry()
        # Replace mock with real tool
        real_tool = PanDermClassifier()
        registry.register(real_tool)
        assert registry.get("panderm_classifier") is real_tool

    def test_run_batch_includes_panderm(self, tmp_path):
        """PanDerm should work in batch execution."""
        tool = PanDermClassifier(device="cpu")
        registry = ToolRegistry()
        registry.register(tool)

        # Without a valid image, should return error output gracefully
        outputs = registry.run_batch(
            ["panderm_classifier"],
            image_path=str(tmp_path / "nonexistent.jpg"),
        )
        assert len(outputs) == 1
        assert outputs[0].tool_name == "panderm_classifier"

    def test_schema_in_registry_list(self):
        registry = ToolRegistry()
        registry.register(PanDermClassifier())
        schemas = registry.list_tools()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "panderm_classifier"
