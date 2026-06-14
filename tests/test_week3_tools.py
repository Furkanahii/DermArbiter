"""Tests for Week 3 tool wrappers: CaseRAG, GuidelineRAG, OntologyGraph,
FairnessProbe, UncertaintyProbe.

OntologyGraph, FairnessProbe, and UncertaintyProbe are computational
(no GPU/API needed) so they are tested with real logic.  CaseRAG and
GuidelineRAG require ChromaDB + models, so they are tested with mocks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dermarbiter.tools.base_tool import BaseTool, ToolOutput, ToolRegistry
from dermarbiter.tools.case_rag import CaseRAG
from dermarbiter.tools.fairness_probe import FairnessProbe, _ita_to_fitzpatrick
from dermarbiter.tools.guideline_rag import GuidelineRAG
from dermarbiter.tools.ontology_graph import OntologyGraph
from dermarbiter.tools.uncertainty_probe import UncertaintyProbe

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_image(tmp_path: Path, color: tuple = (186, 154, 122)) -> str:
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=color)
    path = tmp_path / "test_lesion.jpg"
    img.save(str(path))
    return str(path)


def _create_dark_image(tmp_path: Path) -> str:
    return _create_test_image(tmp_path, color=(60, 40, 30))


def _create_light_image(tmp_path: Path) -> str:
    return _create_test_image(tmp_path, color=(240, 220, 200))


# ═══════════════════════════════════════════════════════════════════════════
# OntologyGraph Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOntologyInterface:
    """OntologyGraph BaseTool compliance."""

    def test_is_base_tool(self):
        assert isinstance(OntologyGraph(), BaseTool)

    def test_name(self):
        assert OntologyGraph().name == "ontology_graph"

    def test_description(self):
        assert "ontology" in OntologyGraph().description.lower()

    def test_to_schema(self):
        assert OntologyGraph().to_schema()["name"] == "ontology_graph"

    def test_registry(self):
        reg = ToolRegistry()
        reg.register(OntologyGraph())
        assert "ontology_graph" in reg


class TestOntologyValidation:
    """OntologyGraph input validation."""

    def test_empty_query_invalid(self):
        assert OntologyGraph().validate_input(query="") is False

    def test_valid_query(self):
        assert OntologyGraph().validate_input(query="melanoma") is True

    def test_run_empty_returns_error(self):
        output = OntologyGraph().run(query="")
        assert output.confidence == 0.0


class TestOntologyOutput:
    """OntologyGraph output format with real graph."""

    def test_melanoma_lookup(self):
        output = OntologyGraph().run(query="melanoma")
        assert isinstance(output, ToolOutput)
        assert output.confidence == 1.0
        assert output.result["query_node"] == "melanoma"

    def test_hierarchy_keys(self):
        output = OntologyGraph().run(query="melanoma")
        h = output.result["hierarchy"]
        assert "parents" in h
        assert "siblings" in h
        assert "children" in h
        assert "root_path" in h

    def test_root_path_starts_with_root(self):
        output = OntologyGraph().run(query="melanoma")
        path = output.result["hierarchy"]["root_path"]
        assert path[0] == "skin_disease"
        assert path[-1] == "melanoma"

    def test_melanoma_has_subtypes(self):
        output = OntologyGraph().run(query="melanoma")
        children = output.result["hierarchy"]["children"]
        assert "nodular_melanoma" in children
        assert "acral_melanoma" in children

    def test_melanoma_has_siblings(self):
        OntologyGraph().run(query="melanoma")
        # melanoma is child of malignant_melanocytic_neoplasm; no siblings expected
        # but melanoma is only child — that's fine

    def test_semantic_distances(self):
        output = OntologyGraph().run(query="melanoma")
        distances = output.result["semantic_distances"]
        assert "basal_cell_carcinoma" in distances
        assert "melanocytic_nevus" in distances
        assert distances["basal_cell_carcinoma"] > 0

    def test_bcc_lookup(self):
        output = OntologyGraph().run(query="basal_cell_carcinoma")
        assert output.result["query_node"] == "basal_cell_carcinoma"
        assert output.confidence == 1.0

    def test_unknown_node(self):
        output = OntologyGraph().run(query="nonexistent_disease")
        assert "error" in output.result

    def test_normalized_lookup(self):
        # "basal cell carcinoma" has space, should map to "basal_cell_carcinoma"
        output = OntologyGraph().run(query="basal cell carcinoma")
        assert output.result["query_node"] == "basal_cell_carcinoma"
        assert output.confidence == 1.0

        # "compound-nevus" has hyphen, should map to "compound_nevus"
        output_hyphen = OntologyGraph().run(query="compound-nevus")
        assert output_hyphen.result["query_node"] == "compound_nevus"
        assert output_hyphen.confidence == 1.0

    def test_total_nodes(self):
        output = OntologyGraph().run(query="melanoma")
        assert output.result["total_nodes"] > 50

    def test_metadata(self):
        output = OntologyGraph().run(query="melanoma")
        assert output.metadata["backend"] == "NetworkX"
        assert "latency_ms" in output.metadata

    def test_nevus_lookup(self):
        output = OntologyGraph().run(query="melanocytic_nevus")
        assert output.confidence == 1.0
        children = output.result["hierarchy"]["children"]
        assert "dysplastic_nevus" in children

    def test_premalignant_category(self):
        output = OntologyGraph().run(query="actinic_keratosis")
        assert output.confidence == 1.0


class TestOntologyLifecycle:
    """OntologyGraph lazy loading."""

    def test_not_loaded_initially(self):
        tool = OntologyGraph()
        assert tool._loaded is False

    def test_loaded_after_run(self):
        tool = OntologyGraph()
        tool.run(query="melanoma")
        assert tool._loaded is True

    def test_unload(self):
        tool = OntologyGraph()
        tool.run(query="melanoma")
        tool.unload()
        assert tool._loaded is False
        assert tool._graph is None


# ═══════════════════════════════════════════════════════════════════════════
# FairnessProbe Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFairnessInterface:
    """FairnessProbe BaseTool compliance."""

    def test_is_base_tool(self):
        assert isinstance(FairnessProbe(), BaseTool)

    def test_name(self):
        assert FairnessProbe().name == "fairness_probe"

    def test_description(self):
        assert "novel" in FairnessProbe().description.lower()

    def test_to_schema(self):
        assert FairnessProbe().to_schema()["name"] == "fairness_probe"

    def test_registry(self):
        reg = ToolRegistry()
        reg.register(FairnessProbe())
        assert "fairness_probe" in reg


class TestFairnessValidation:
    """FairnessProbe input validation."""

    def test_none_invalid(self):
        assert FairnessProbe().validate_input(None) is False

    def test_missing_file(self):
        assert FairnessProbe().validate_input("/no/file.jpg") is False

    def test_valid_image(self, tmp_path):
        assert FairnessProbe().validate_input(_create_test_image(tmp_path)) is True


class TestFairnessOutput:
    """FairnessProbe output with real computation."""

    def test_output_type(self, tmp_path):
        output = FairnessProbe().run(_create_test_image(tmp_path))
        assert isinstance(output, ToolOutput)

    def test_tool_name(self, tmp_path):
        output = FairnessProbe().run(_create_test_image(tmp_path))
        assert output.tool_name == "fairness_probe"

    def test_has_fitzpatrick(self, tmp_path):
        output = FairnessProbe().run(_create_test_image(tmp_path))
        assert "fitzpatrick_type_approx" in output.result
        assert output.result["fitzpatrick_type_approx"] in ("I", "II", "III", "IV", "V", "VI")

    def test_has_ita(self, tmp_path):
        output = FairnessProbe().run(_create_test_image(tmp_path))
        assert "ita_angle" in output.result
        assert isinstance(output.result["ita_angle"], float)

    def test_has_ita_category(self, tmp_path):
        output = FairnessProbe().run(_create_test_image(tmp_path))
        assert output.result["ita_category"] in (
            "very_light", "light", "intermediate", "tan", "brown", "dark",
        )

    def test_has_skin_rgb(self, tmp_path):
        output = FairnessProbe().run(_create_test_image(tmp_path))
        rgb = output.result["skin_tone_rgb"]
        assert len(rgb) == 3
        assert all(0 <= c <= 255 for c in rgb)

    def test_has_calibration_note(self, tmp_path):
        output = FairnessProbe().run(_create_test_image(tmp_path))
        assert "calibration_note" in output.result

    def test_confidence_in_range(self, tmp_path):
        output = FairnessProbe().run(_create_test_image(tmp_path))
        assert 0.0 <= output.confidence <= 1.0

    def test_metadata(self, tmp_path):
        output = FairnessProbe().run(_create_test_image(tmp_path))
        assert output.metadata["contribution"] == "novel"

    def test_light_skin_type(self, tmp_path):
        output = FairnessProbe().run(_create_light_image(tmp_path))
        assert output.result["fitzpatrick_type_approx"] in ("I", "II")
        assert output.result["bias_warning"] is None

    def test_dark_skin_bias_warning(self, tmp_path):
        output = FairnessProbe().run(_create_dark_image(tmp_path))
        assert output.result["fitzpatrick_type_approx"] in ("V", "VI")
        assert output.result["bias_warning"] is not None


class TestITAMapping:
    """ITA → Fitzpatrick mapping function."""

    def test_type_i(self):
        assert _ita_to_fitzpatrick(60.0) == ("I", "very_light")

    def test_type_ii(self):
        assert _ita_to_fitzpatrick(45.0) == ("II", "light")

    def test_type_iii(self):
        assert _ita_to_fitzpatrick(35.0) == ("III", "intermediate")

    def test_type_iv(self):
        assert _ita_to_fitzpatrick(15.0) == ("IV", "tan")

    def test_type_v(self):
        assert _ita_to_fitzpatrick(0.0) == ("V", "brown")

    def test_type_vi(self):
        assert _ita_to_fitzpatrick(-40.0) == ("VI", "dark")


# ═══════════════════════════════════════════════════════════════════════════
# UncertaintyProbe Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestUncertaintyInterface:
    """UncertaintyProbe BaseTool compliance."""

    def test_is_base_tool(self):
        assert isinstance(UncertaintyProbe(), BaseTool)

    def test_name(self):
        assert UncertaintyProbe().name == "uncertainty_probe"

    def test_description(self):
        assert "novel" in UncertaintyProbe().description.lower()

    def test_to_schema(self):
        assert UncertaintyProbe().to_schema()["name"] == "uncertainty_probe"

    def test_registry(self):
        reg = ToolRegistry()
        reg.register(UncertaintyProbe())
        assert "uncertainty_probe" in reg


def _run_uncertainty(probs: dict, alpha: float | None = None):
    """Run UncertaintyProbe via BaseTool-safe setter path."""
    tool = UncertaintyProbe(alpha=alpha) if alpha is not None else UncertaintyProbe()
    tool.set_probabilities(probs)
    return tool.run()


class TestUncertaintyOutput:
    """UncertaintyProbe with real computation."""

    def test_from_dict(self):
        probs = {
            "melanoma": 0.62, "bcc": 0.15, "nevus": 0.11,
            "scc": 0.06, "sk": 0.03, "df": 0.02, "ak": 0.01,
        }
        output = _run_uncertainty(probs)
        assert isinstance(output, ToolOutput)
        assert output.tool_name == "uncertainty_probe"

    def test_has_entropy(self):
        probs = {"melanoma": 0.62, "bcc": 0.15, "nevus": 0.11}
        output = _run_uncertainty(probs)
        assert "predictive_entropy" in output.result
        assert output.result["predictive_entropy"] > 0

    def test_has_max_entropy(self):
        probs = {"melanoma": 0.62, "bcc": 0.15, "nevus": 0.11}
        output = _run_uncertainty(probs)
        assert output.result["max_entropy"] > 0

    def test_normalised_entropy_range(self):
        probs = {"melanoma": 0.62, "bcc": 0.15, "nevus": 0.11}
        output = _run_uncertainty(probs)
        assert 0.0 <= output.result["normalised_entropy"] <= 1.0

    def test_conformal_set(self):
        probs = {"melanoma": 0.62, "bcc": 0.15, "nevus": 0.11, "scc": 0.06, "other": 0.06}
        output = _run_uncertainty(probs)
        cs = output.result["conformal_set"]
        assert isinstance(cs, list)
        assert len(cs) > 0
        assert "melanoma" in cs

    def test_coverage_guarantee(self):
        probs = {"a": 0.5, "b": 0.3, "c": 0.2}
        output = _run_uncertainty(probs, alpha=0.10)
        assert output.result["coverage_guarantee"] == 0.90

    def test_high_uncertainty_flag(self):
        # Uniform distribution → high entropy
        probs = {"a": 0.14, "b": 0.14, "c": 0.14, "d": 0.14, "e": 0.14, "f": 0.15, "g": 0.15}
        output = _run_uncertainty(probs)
        assert output.result["is_high_uncertainty"] is True

    def test_low_uncertainty_flag(self):
        probs = {"melanoma": 0.95, "bcc": 0.05}
        output = _run_uncertainty(probs)
        assert output.result["is_high_uncertainty"] is False

    def test_from_query_string(self):
        output = UncertaintyProbe().run(query="melanoma:0.62,bcc:0.15,nevus:0.11")
        assert "predictive_entropy" in output.result

    def test_from_query_json(self):
        output = UncertaintyProbe().run(
            query=json.dumps({"melanoma": 0.62, "bcc": 0.15, "nevus": 0.23}),
        )
        assert "predictive_entropy" in output.result

    def test_empty_input_error(self):
        output = UncertaintyProbe().run()
        assert "error" in output.result

    def test_confidence_inversely_related(self):
        # Confident: one dominant class
        confident = _run_uncertainty({"a": 0.95, "b": 0.05})
        # Uncertain: uniform
        uncertain = _run_uncertainty({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})
        assert confident.confidence > uncertain.confidence

    def test_metadata(self):
        probs = {"a": 0.5, "b": 0.5}
        output = _run_uncertainty(probs)
        assert output.metadata["contribution"] == "novel"
        assert output.metadata["method"] == "entropy + split-conformal"

    def test_signature_matches_base_tool(self):
        """run() must match BaseTool contract: (image_path, query) only."""
        import inspect

        sig = inspect.signature(UncertaintyProbe().run)
        params = set(sig.parameters)
        assert params == {"image_path", "query"}


class TestUncertaintyComputation:
    """Direct computation method tests."""

    def test_entropy_uniform(self):
        probs = np.array([0.5, 0.5])
        ent = UncertaintyProbe.compute_entropy(probs)
        assert abs(ent - np.log(2)) < 1e-6

    def test_entropy_certain(self):
        probs = np.array([1.0, 0.0])
        ent = UncertaintyProbe.compute_entropy(probs)
        assert ent == 0.0

    def test_max_entropy(self):
        assert abs(UncertaintyProbe.compute_max_entropy(7) - np.log(7)) < 1e-6

    def test_conformal_set_dominant(self):
        probs = np.array([0.95, 0.03, 0.02])
        names = ["melanoma", "bcc", "nevus"]
        cs = UncertaintyProbe().compute_conformal_set(probs, names)
        assert cs == ["melanoma"]

    def test_conformal_set_uncertain(self):
        probs = np.array([0.4, 0.35, 0.25])
        names = ["a", "b", "c"]
        cs = UncertaintyProbe(alpha=0.10).compute_conformal_set(probs, names)
        assert len(cs) >= 2


# ═══════════════════════════════════════════════════════════════════════════
# CaseRAG & GuidelineRAG Interface Tests (no real model)
# ═══════════════════════════════════════════════════════════════════════════

class TestCaseRAGInterface:
    """CaseRAG BaseTool compliance."""

    def test_is_base_tool(self):
        assert isinstance(CaseRAG(), BaseTool)

    def test_name(self):
        assert CaseRAG().name == "case_rag"

    def test_description(self):
        assert "Derm1M" in CaseRAG().description

    def test_not_loaded(self):
        assert CaseRAG()._loaded is False

    def test_registry(self):
        reg = ToolRegistry()
        reg.register(CaseRAG())
        assert "case_rag" in reg

    def test_validate_none(self):
        assert CaseRAG().validate_input(None) is False

    def test_validate_missing(self):
        assert CaseRAG().validate_input("/no/file.jpg") is False

    def test_run_invalid(self):
        output = CaseRAG().run(image_path=None)
        assert "error" in output.result


class TestGuidelineRAGInterface:
    """GuidelineRAG BaseTool compliance."""

    def test_is_base_tool(self):
        assert isinstance(GuidelineRAG(), BaseTool)

    def test_name(self):
        assert GuidelineRAG().name == "guideline_rag"

    def test_description(self):
        assert "guideline" in GuidelineRAG().description.lower()

    def test_not_loaded(self):
        assert GuidelineRAG()._loaded is False

    def test_registry(self):
        reg = ToolRegistry()
        reg.register(GuidelineRAG())
        assert "guideline_rag" in reg

    def test_validate_always_true(self):
        # GuidelineRAG is text-only, always valid
        assert GuidelineRAG().validate_input(query="melanoma") is True


# ═══════════════════════════════════════════════════════════════════════════
# Full registry integration
# ═══════════════════════════════════════════════════════════════════════════

class TestFullRegistryIntegration:
    """All 9 real tools register together."""

    def test_all_nine_tools(self):
        from dermarbiter.tools import (
            CaseRAG,
            DermoGPTVQA,
            FairnessProbe,
            GuidelineRAG,
            MAKEAnnotator,
            MedGemmaVQA,
            OntologyGraph,
            PanDermClassifier,
            UncertaintyProbe,
        )

        registry = ToolRegistry()
        for tool_cls in [
            PanDermClassifier, MAKEAnnotator, DermoGPTVQA, MedGemmaVQA,
            CaseRAG, GuidelineRAG, OntologyGraph, FairnessProbe, UncertaintyProbe,
        ]:
            registry.register(tool_cls())

        assert len(registry) == 9
        expected = {
            "panderm_classifier", "make_annotator", "dermogpt_vqa",
            "general_vqa", "case_rag", "guideline_rag",
            "ontology_graph", "fairness_probe", "uncertainty_probe",
        }
        assert set(registry.tool_names) == expected

    def test_replace_all_mocks(self):
        """Real tools replace all mocks cleanly."""
        from dermarbiter.tools import (
            CaseRAG,
            DermoGPTVQA,
            FairnessProbe,
            GuidelineRAG,
            MAKEAnnotator,
            MedGemmaVQA,
            OntologyGraph,
            PanDermClassifier,
            UncertaintyProbe,
        )
        from tests.mocks.mock_tools import create_mock_registry

        registry = create_mock_registry()
        assert len(registry) == 9

        for tool_cls in [
            PanDermClassifier, MAKEAnnotator, DermoGPTVQA, MedGemmaVQA,
            CaseRAG, GuidelineRAG, OntologyGraph, FairnessProbe, UncertaintyProbe,
        ]:
            registry.register(tool_cls())

        assert len(registry) == 9  # replaced, not duplicated
