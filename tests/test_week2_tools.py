"""Tests for Week 2 tool wrappers: MAKE, DermoGPT-RL, MedGemma-4B.

Tests use mock models injected directly to avoid needing GPU or API keys.
Each tool is tested for:
- Interface compliance (BaseTool contract)
- Input validation
- Output format (matching mock tool expectations)
- Lazy loading behavior
- Error handling
- ToolRegistry integration
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import importlib.util
_has_torchvision = importlib.util.find_spec("torchvision") is not None
_skip_no_torchvision = pytest.mark.skipif(
    not _has_torchvision, reason="torchvision not installed"
)

from dermarbiter.tools.base_tool import BaseTool, ToolOutput, ToolRegistry
from dermarbiter.tools.dermogpt_tool import DermoGPTVQA
from dermarbiter.tools.make_tool import (
    DERMOSCOPIC_CONCEPTS,
    MAKEAnnotator,
)
from dermarbiter.tools.medgemma_tool import MedGemmaVQA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_image(tmp_path: Path, fmt: str = "jpg") -> str:
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    path = tmp_path / f"test_lesion.{fmt}"
    img.save(str(path))
    return str(path)


def _make_fake_make_loaded(tool: MAKEAnnotator) -> None:
    """Inject a fake CLIP model into MAKEAnnotator."""
    num_concepts = len(tool._concepts)

    class FakeCLIP(nn.Module):
        def encode_image(self, x):
            return torch.randn(x.shape[0], 768)

        def encode_text(self, x):
            return torch.randn(x.shape[0], 768)

    tool._device = torch.device("cpu")
    tool._model = FakeCLIP()

    # Fake text features (pre-normalised)
    text_feats = torch.randn(num_concepts, 768)
    text_feats /= text_feats.norm(dim=-1, keepdim=True)
    tool._text_features = text_feats

    # Fake preprocess: just resize and tensor
    try:
        from torchvision import transforms
        tool._preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    except ImportError:
        # Fallback: minimal PIL-based transform
        def _fallback_preprocess(img):
            img = img.resize((224, 224))
            import numpy as np
            arr = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)
        tool._preprocess = _fallback_preprocess

    tool._loaded = True


def _make_fake_vqa_loaded(tool: DermoGPTVQA | MedGemmaVQA) -> None:
    """Inject a fake VQA model into DermoGPT or MedGemma."""
    tool._device = torch.device("cpu")

    class FakeBatchEncoding(dict):
        """Dict subclass with .to() for compatibility with tool code."""

        def to(self, device):
            return self

    class FakeProcessor:
        def __call__(self, text=None, images=None, return_tensors=None, **kw):
            return FakeBatchEncoding(
                input_ids=torch.tensor([[1, 2, 3]]),
            )

        def decode(self, token_ids, skip_special_tokens=True):
            return (
                "The lesion shows irregular borders and color variegation. "
                "Features consistent with melanoma. ABCDE criteria suggest "
                "asymmetry and border irregularity. Recommend excisional "
                "biopsy for histopathological confirmation."
            )

        def apply_chat_template(self, messages, tokenize=False, **kw):
            # New MedGemma code uses tokenize=True, return_dict=True
            # and expects a dict-like BatchEncoding, not a string.
            if tokenize:
                return FakeBatchEncoding(
                    input_ids=torch.tensor([[1, 2, 3]]),
                )
            return "User: test query"

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Linear(1, 1)  # need at least one param

        def generate(self, input_ids=None, **kwargs):
            return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        @property
        def device(self):
            return torch.device("cpu")

    tool._processor = FakeProcessor()
    tool._model = FakeModel()
    tool._loaded = True


# ═══════════════════════════════════════════════════════════════════════════
# MAKE Annotator Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMAKEInterface:
    """MAKE annotator BaseTool compliance."""

    def test_is_base_tool(self):
        assert isinstance(MAKEAnnotator(), BaseTool)

    def test_name(self):
        assert MAKEAnnotator().name == "make_annotator"

    def test_description(self):
        desc = MAKEAnnotator().description
        assert "MAKE" in desc
        assert len(desc) > 20

    def test_to_schema(self):
        schema = MAKEAnnotator().to_schema()
        assert schema["name"] == "make_annotator"
        assert "parameters" in schema

    def test_repr(self):
        assert "make_annotator" in repr(MAKEAnnotator())

    def test_default_concepts(self):
        tool = MAKEAnnotator()
        assert tool._concepts == DERMOSCOPIC_CONCEPTS
        assert len(tool._concepts) == 12

    def test_custom_concepts(self):
        custom = ["concept_a", "concept_b"]
        tool = MAKEAnnotator(concepts=custom)
        assert tool._concepts == custom

    def test_not_loaded_on_init(self):
        tool = MAKEAnnotator()
        assert tool._loaded is False

    def test_registry_integration(self):
        registry = ToolRegistry()
        registry.register(MAKEAnnotator())
        assert "make_annotator" in registry


class TestMAKEValidation:
    """MAKE input validation."""

    def test_none_image(self):
        assert MAKEAnnotator().validate_input(None) is False

    def test_missing_image(self):
        assert MAKEAnnotator().validate_input("/no/file.jpg") is False

    def test_bad_format(self, tmp_path):
        f = tmp_path / "file.pdf"
        f.write_text("x")
        assert MAKEAnnotator().validate_input(str(f)) is False

    def test_valid_image(self, tmp_path):
        img = _create_test_image(tmp_path)
        assert MAKEAnnotator().validate_input(img) is True

    def test_run_invalid_returns_error(self):
        output = MAKEAnnotator().run(image_path=None)
        assert output.confidence == 0.0
        assert "error" in output.result


@_skip_no_torchvision
class TestMAKEOutput:
    """MAKE output format with mock model."""

    @pytest.fixture()
    def loaded_tool(self):
        tool = MAKEAnnotator(device="cpu")
        _make_fake_make_loaded(tool)
        return tool

    def test_output_type(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert isinstance(output, ToolOutput)

    def test_tool_name(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert output.tool_name == "make_annotator"

    def test_has_concepts(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        concepts = output.result.get("concepts")
        assert concepts is not None
        assert len(concepts) == 12

    def test_concepts_have_score(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        for c in output.result["concepts"]:
            assert "concept" in c
            assert "score" in c
            assert 0.0 <= c["score"] <= 1.0

    def test_concepts_sorted_descending(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        scores = [c["score"] for c in output.result["concepts"]]
        assert scores == sorted(scores, reverse=True)

    def test_confidence_in_range(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert 0.0 <= output.confidence <= 1.0

    def test_has_metadata(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert output.metadata["model"] == "MAKE"
        assert "latency_ms" in output.metadata

    def test_has_raw_text(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert len(output.raw_text) > 0
        assert "concept" in output.raw_text.lower() or "dermoscopic" in output.raw_text.lower()

    def test_has_model_version(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert "model_version" in output.result


@_skip_no_torchvision
class TestMAKELifecycle:
    """MAKE lazy loading and unloading."""

    def test_not_loaded_initially(self):
        assert MAKEAnnotator()._loaded is False

    def test_loaded_after_inject(self):
        tool = MAKEAnnotator(device="cpu")
        _make_fake_make_loaded(tool)
        assert tool._loaded is True

    def test_skip_reload(self):
        tool = MAKEAnnotator(device="cpu")
        _make_fake_make_loaded(tool)
        original = tool._model
        tool._load_model()  # should skip
        assert tool._model is original

    def test_unload(self):
        tool = MAKEAnnotator(device="cpu")
        _make_fake_make_loaded(tool)
        tool.unload()
        assert tool._loaded is False
        assert tool._model is None


# ═══════════════════════════════════════════════════════════════════════════
# DermoGPT-RL Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDermoGPTInterface:
    """DermoGPT BaseTool compliance."""

    def test_is_base_tool(self):
        assert isinstance(DermoGPTVQA(), BaseTool)

    def test_name(self):
        assert DermoGPTVQA().name == "dermogpt_vqa"

    def test_description(self):
        assert "DermoGPT" in DermoGPTVQA().description

    def test_to_schema(self):
        assert DermoGPTVQA().to_schema()["name"] == "dermogpt_vqa"

    def test_not_loaded(self):
        assert DermoGPTVQA()._loaded is False

    def test_registry(self):
        registry = ToolRegistry()
        registry.register(DermoGPTVQA())
        assert "dermogpt_vqa" in registry


class TestDermoGPTValidation:
    """DermoGPT input validation."""

    def test_none_image_valid(self):
        # VQA can work text-only
        assert DermoGPTVQA().validate_input(None) is True

    def test_missing_image_invalid(self):
        assert DermoGPTVQA().validate_input("/no/file.jpg") is False

    def test_run_invalid_image(self):
        output = DermoGPTVQA().run(image_path="/no/file.jpg")
        assert output.confidence == 0.0
        assert "error" in output.result


class TestDermoGPTOutput:
    """DermoGPT output format with mock model."""

    @pytest.fixture()
    def loaded_tool(self):
        tool = DermoGPTVQA(device="cpu")
        _make_fake_vqa_loaded(tool)
        return tool

    def test_output_type(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert isinstance(output, ToolOutput)

    def test_tool_name(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert output.tool_name == "dermogpt_vqa"

    def test_has_answer(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert "answer" in output.result
        assert isinstance(output.result["answer"], str)
        assert len(output.result["answer"]) > 0

    def test_has_question(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path), query="test?")
        assert output.result["question"] == "test?"

    def test_default_question(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert "diagnosis" in output.result["question"].lower()

    def test_confidence_in_range(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert 0.0 <= output.confidence <= 1.0

    def test_has_metadata(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert output.metadata["model"] == "DermoGPT-RL"
        assert "latency_ms" in output.metadata

    def test_has_raw_text(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert "DermoGPT" in output.raw_text

    def test_text_only_works(self, loaded_tool):
        output = loaded_tool.run(query="What is melanoma?")
        assert isinstance(output, ToolOutput)
        assert output.confidence > 0


class TestDermoGPTConfidence:
    """DermoGPT confidence estimation heuristic."""

    def test_empty_answer(self):
        assert DermoGPTVQA._estimate_confidence("") == 0.3

    def test_short_answer(self):
        c = DermoGPTVQA._estimate_confidence("melanoma likely")
        assert 0.5 <= c <= 0.8

    def test_detailed_answer(self):
        c = DermoGPTVQA._estimate_confidence(
            "The dermoscopic features suggest melanoma with atypical pigment "
            "network. Asymmetry of structure and ABCDE criteria point to "
            "malignant melanoma. Recommend excisional biopsy for diagnosis."
        )
        assert c >= 0.6  # should get bonuses for length + keywords

    def test_capped_at_one(self):
        long_clinical = (
            "melanoma carcinoma biopsy dermoscopy diagnosis differential "
            "malignant benign ABCDE pigment asymmetry " * 10
        )
        assert DermoGPTVQA._estimate_confidence(long_clinical) <= 1.0


class TestDermoGPTLifecycle:
    """DermoGPT lazy loading."""

    def test_skip_reload(self):
        tool = DermoGPTVQA(device="cpu")
        _make_fake_vqa_loaded(tool)
        original = tool._model
        tool._load_model()
        assert tool._model is original

    def test_unload(self):
        tool = DermoGPTVQA(device="cpu")
        _make_fake_vqa_loaded(tool)
        tool.unload()
        assert tool._loaded is False
        assert tool._model is None
        assert tool._processor is None


# ═══════════════════════════════════════════════════════════════════════════
# MedGemma-4B Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMedGemmaInterface:
    """MedGemma BaseTool compliance."""

    def test_is_base_tool(self):
        assert isinstance(MedGemmaVQA(), BaseTool)

    def test_name(self):
        assert MedGemmaVQA().name == "general_vqa"

    def test_description(self):
        assert "MedGemma" in MedGemmaVQA().description

    def test_to_schema(self):
        assert MedGemmaVQA().to_schema()["name"] == "general_vqa"

    def test_not_loaded(self):
        assert MedGemmaVQA()._loaded is False

    def test_registry(self):
        registry = ToolRegistry()
        registry.register(MedGemmaVQA())
        assert "general_vqa" in registry


class TestMedGemmaValidation:
    """MedGemma input validation."""

    def test_none_image_valid(self):
        assert MedGemmaVQA().validate_input(None) is True

    def test_missing_image_invalid(self):
        assert MedGemmaVQA().validate_input("/no/file.jpg") is False

    def test_run_invalid_image(self):
        output = MedGemmaVQA().run(image_path="/no/file.jpg")
        assert output.confidence == 0.0


class TestMedGemmaOutput:
    """MedGemma output format with mock model."""

    @pytest.fixture()
    def loaded_tool(self):
        tool = MedGemmaVQA(device="cpu")
        _make_fake_vqa_loaded(tool)
        return tool

    def test_output_type(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert isinstance(output, ToolOutput)

    def test_tool_name(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert output.tool_name == "general_vqa"

    def test_has_answer(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert "answer" in output.result
        assert len(output.result["answer"]) > 0

    def test_has_question(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path), query="test?")
        assert output.result["question"] == "test?"

    def test_confidence_in_range(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert 0.0 <= output.confidence <= 1.0

    def test_has_metadata(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert output.metadata["model"] == "MedGemma-4B"

    def test_has_raw_text(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert "General VQA" in output.raw_text

    def test_text_only_works(self, loaded_tool):
        output = loaded_tool.run(query="What are signs of melanoma?")
        assert isinstance(output, ToolOutput)
        assert output.confidence > 0

    def test_model_version(self, loaded_tool, tmp_path):
        output = loaded_tool.run(_create_test_image(tmp_path))
        assert output.result["model_version"] == "medgemma-4b-v1"


class TestMedGemmaConfidence:
    """MedGemma confidence estimation."""

    def test_empty(self):
        assert MedGemmaVQA._estimate_confidence("") == 0.25

    def test_baseline_lower_than_dermogpt(self):
        text = "melanoma likely based on features"
        mg = MedGemmaVQA._estimate_confidence(text)
        dg = DermoGPTVQA._estimate_confidence(text)
        assert mg <= dg  # generalist should be less confident


class TestMedGemmaLifecycle:
    """MedGemma lazy loading."""

    def test_skip_reload(self):
        tool = MedGemmaVQA(device="cpu")
        _make_fake_vqa_loaded(tool)
        original = tool._model
        tool._load_model()
        assert tool._model is original

    def test_unload(self):
        tool = MedGemmaVQA(device="cpu")
        _make_fake_vqa_loaded(tool)
        tool.unload()
        assert tool._loaded is False


# ═══════════════════════════════════════════════════════════════════════════
# Cross-tool integration tests
# ═══════════════════════════════════════════════════════════════════════════

class TestWeek2RegistryIntegration:
    """All 4 tools (PanDerm + Week 2) work together in a registry."""

    def test_all_four_tools_register(self):
        from dermarbiter.tools.panderm_tool import PanDermClassifier

        registry = ToolRegistry()
        registry.register(PanDermClassifier())
        registry.register(MAKEAnnotator())
        registry.register(DermoGPTVQA())
        registry.register(MedGemmaVQA())
        assert len(registry) == 4
        assert set(registry.tool_names) == {
            "panderm_classifier",
            "make_annotator",
            "dermogpt_vqa",
            "general_vqa",
        }

    def test_all_schemas_valid(self):
        from dermarbiter.tools.panderm_tool import PanDermClassifier

        registry = ToolRegistry()
        for tool in [PanDermClassifier(), MAKEAnnotator(), DermoGPTVQA(), MedGemmaVQA()]:
            registry.register(tool)

        schemas = registry.list_tools()
        assert len(schemas) == 4
        for s in schemas:
            assert "name" in s
            assert "description" in s
            assert "parameters" in s

    def test_replace_mock_with_real(self):
        """Real tools should cleanly replace mocks in registry."""
        from tests.mocks.mock_tools import create_mock_registry

        registry = create_mock_registry()
        assert len(registry) == 9  # all mocks

        # Replace 4 tools with real wrappers
        from dermarbiter.tools.panderm_tool import PanDermClassifier

        registry.register(PanDermClassifier())
        registry.register(MAKEAnnotator())
        registry.register(DermoGPTVQA())
        registry.register(MedGemmaVQA())

        # Should still have 9 (mocks replaced, not duplicated)
        # Actually mock has 'general_vqa' not 'medgemma_vqa', so names match
        assert len(registry) == 9
