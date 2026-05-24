"""Unit tests for DermArbiter tools — BaseTool, ToolRegistry, and mock tools.

Tests cover:
  • BaseTool abstract interface enforcement
  • ToolRegistry registration, lookup, listing, and batch execution
  • Mock tool output format and content validation
"""

from __future__ import annotations

import pytest

from dermarbiter.tools.base_tool import BaseTool, ToolOutput, ToolRegistry
from tests.mocks.mock_tools import (
    MockCaseRAG,
    MockDermoGPTVQA,
    MockFairnessProbe,
    MockGeneralVQA,
    MockGuidelineRAG,
    MockMAKEAnnotator,
    MockOntology,
    MockPanDermClassifier,
    MockUncertaintyProbe,
    create_mock_registry,
)


# ═══════════════════════════════════════════════════════════════════════════
# BaseTool interface
# ═══════════════════════════════════════════════════════════════════════════

class TestBaseToolInterface:
    """Tests for the BaseTool abstract base class."""

    def test_base_tool_interface(self):
        """BaseTool cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTool()  # type: ignore[abstract]

    def test_concrete_tool_has_name(self):
        """A concrete tool subclass must have a name property."""
        tool = MockPanDermClassifier()
        assert isinstance(tool.name, str)
        assert len(tool.name) > 0

    def test_concrete_tool_has_description(self):
        """A concrete tool subclass must have a description property."""
        tool = MockPanDermClassifier()
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0

    def test_concrete_tool_run_returns_tool_output(self):
        """run() must return a ToolOutput instance."""
        tool = MockPanDermClassifier()
        output = tool.run()
        assert isinstance(output, ToolOutput)

    def test_to_schema_returns_dict(self):
        """to_schema() should return a JSON-serialisable dict."""
        tool = MockPanDermClassifier()
        schema = tool.to_schema()
        assert isinstance(schema, dict)
        assert schema["name"] == "panderm_classifier"
        assert "description" in schema
        assert "parameters" in schema

    def test_repr(self):
        """__repr__ should include the tool name."""
        tool = MockPanDermClassifier()
        assert "panderm_classifier" in repr(tool)


# ═══════════════════════════════════════════════════════════════════════════
# ToolRegistry — register
# ═══════════════════════════════════════════════════════════════════════════

class TestToolRegistryRegister:
    """Tests for ToolRegistry.register()."""

    def test_tool_registry_register(self):
        """Registering a tool should make it retrievable."""
        registry = ToolRegistry()
        tool = MockPanDermClassifier()
        registry.register(tool)

        assert "panderm_classifier" in registry
        assert len(registry) == 1

    def test_register_multiple_tools(self):
        """Multiple tools should coexist in the registry."""
        registry = ToolRegistry()
        registry.register(MockPanDermClassifier())
        registry.register(MockMAKEAnnotator())
        registry.register(MockCaseRAG())

        assert len(registry) == 3

    def test_register_overwrites_duplicate(self):
        """Registering a tool with the same name should overwrite."""
        registry = ToolRegistry()
        tool1 = MockPanDermClassifier()
        tool2 = MockPanDermClassifier()

        registry.register(tool1)
        registry.register(tool2)

        assert len(registry) == 1

    def test_register_rejects_non_tool(self):
        """Registering a non-BaseTool should raise TypeError."""
        registry = ToolRegistry()
        with pytest.raises(TypeError):
            registry.register("not a tool")  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
# ToolRegistry — get
# ═══════════════════════════════════════════════════════════════════════════

class TestToolRegistryGet:
    """Tests for ToolRegistry.get()."""

    def test_tool_registry_get(self, mock_tool_registry: ToolRegistry):
        """get() should return the correct tool."""
        tool = mock_tool_registry.get("panderm_classifier")
        assert tool.name == "panderm_classifier"

    def test_get_nonexistent_raises_key_error(
        self, mock_tool_registry: ToolRegistry
    ):
        """get() should raise KeyError for unknown tools."""
        with pytest.raises(KeyError, match="nonexistent"):
            mock_tool_registry.get("nonexistent")

    def test_get_all_nine_tools(self, mock_tool_registry: ToolRegistry):
        """All 9 mock tools should be retrievable."""
        expected_names = [
            "panderm_classifier",
            "make_annotator",
            "dermogpt_vqa",
            "general_vqa",
            "case_rag",
            "guideline_rag",
            "ontology_graph",
            "fairness_probe",
            "uncertainty_probe",
        ]
        for name in expected_names:
            tool = mock_tool_registry.get(name)
            assert tool.name == name


# ═══════════════════════════════════════════════════════════════════════════
# ToolRegistry — list_tools
# ═══════════════════════════════════════════════════════════════════════════

class TestToolRegistryList:
    """Tests for ToolRegistry.list_tools()."""

    def test_tool_registry_list(self, mock_tool_registry: ToolRegistry):
        """list_tools() should return schemas for all registered tools."""
        schemas = mock_tool_registry.list_tools()
        assert isinstance(schemas, list)
        assert len(schemas) == 9

    def test_list_tools_schema_format(self, mock_tool_registry: ToolRegistry):
        """Each schema should have name, description, and parameters."""
        schemas = mock_tool_registry.list_tools()
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema

    def test_tool_names_property(self, mock_tool_registry: ToolRegistry):
        """tool_names should return a sorted list of tool names."""
        names = mock_tool_registry.tool_names
        assert names == sorted(names)
        assert len(names) == 9


# ═══════════════════════════════════════════════════════════════════════════
# Mock tool output format — PanDerm
# ═══════════════════════════════════════════════════════════════════════════

class TestMockPanDermOutputFormat:
    """Tests for MockPanDermClassifier output structure."""

    def test_mock_panderm_output_format(self):
        """PanDerm output should contain predictions list."""
        tool = MockPanDermClassifier()
        output = tool.run()

        assert output.tool_name == "panderm_classifier"
        assert "predictions" in output.result
        assert isinstance(output.result["predictions"], list)
        assert len(output.result["predictions"]) >= 5

    def test_panderm_predictions_have_disease_and_probability(self):
        """Each prediction should have disease and probability keys."""
        tool = MockPanDermClassifier()
        output = tool.run()

        for pred in output.result["predictions"]:
            assert "disease" in pred
            assert "probability" in pred
            assert 0.0 <= pred["probability"] <= 1.0

    def test_panderm_probabilities_sum_approximately_one(self):
        """Disease probabilities should sum to ~1.0."""
        tool = MockPanDermClassifier()
        output = tool.run()

        total = sum(p["probability"] for p in output.result["predictions"])
        assert abs(total - 1.0) < 0.01

    def test_panderm_top_prediction_is_melanoma(self):
        """The mock's top prediction should be melanoma."""
        tool = MockPanDermClassifier()
        output = tool.run()

        top = output.result["predictions"][0]
        assert top["disease"] == "melanoma"

    def test_panderm_confidence_matches_top_probability(self):
        """Output confidence should match the top prediction probability."""
        tool = MockPanDermClassifier()
        output = tool.run()

        top_prob = output.result["predictions"][0]["probability"]
        assert output.confidence == top_prob

    def test_panderm_has_metadata(self):
        """PanDerm output should have model metadata."""
        tool = MockPanDermClassifier()
        output = tool.run()

        assert "model" in output.metadata
        assert output.metadata["model"] == "PanDerm"


# ═══════════════════════════════════════════════════════════════════════════
# Mock tool output format — RAG tools
# ═══════════════════════════════════════════════════════════════════════════

class TestMockRAGOutputFormat:
    """Tests for CaseRAG and GuidelineRAG output structures."""

    def test_mock_case_rag_output_format(self):
        """CaseRAG should return similar_cases with distances."""
        tool = MockCaseRAG()
        output = tool.run()

        assert output.tool_name == "case_rag"
        assert "similar_cases" in output.result
        cases = output.result["similar_cases"]
        assert isinstance(cases, list)
        assert len(cases) >= 3

    def test_case_rag_cases_have_required_fields(self):
        """Each similar case should have case_id, diagnosis, and distance."""
        tool = MockCaseRAG()
        output = tool.run()

        for case in output.result["similar_cases"]:
            assert "case_id" in case
            assert "diagnosis" in case
            assert "distance" in case
            assert isinstance(case["distance"], (int, float))

    def test_case_rag_distances_sorted(self):
        """Similar cases should be sorted by ascending distance."""
        tool = MockCaseRAG()
        output = tool.run()

        distances = [c["distance"] for c in output.result["similar_cases"]]
        assert distances == sorted(distances)

    def test_mock_guideline_rag_output_format(self):
        """GuidelineRAG should return chunks with source and text."""
        tool = MockGuidelineRAG()
        output = tool.run()

        assert output.tool_name == "guideline_rag"
        assert "chunks" in output.result
        chunks = output.result["chunks"]
        assert isinstance(chunks, list)
        assert len(chunks) >= 2

    def test_guideline_chunks_have_required_fields(self):
        """Each chunk should have source, text, and relevance_score."""
        tool = MockGuidelineRAG()
        output = tool.run()

        for chunk in output.result["chunks"]:
            assert "source" in chunk
            assert "text" in chunk
            assert "relevance_score" in chunk
            assert 0.0 <= chunk["relevance_score"] <= 1.0

    def test_guideline_chunks_sorted_by_relevance(self):
        """Guideline chunks should be sorted by descending relevance."""
        tool = MockGuidelineRAG()
        output = tool.run()

        scores = [c["relevance_score"] for c in output.result["chunks"]]
        assert scores == sorted(scores, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
# Mock tool output format — other tools
# ═══════════════════════════════════════════════════════════════════════════

class TestMockOtherToolOutputFormats:
    """Tests for MAKE, DermoGPT, GeneralVQA, Ontology, Fairness, Uncertainty."""

    def test_make_annotator_has_concepts(self):
        """MAKE should return a list of dermoscopic concepts with scores."""
        output = MockMAKEAnnotator().run()
        assert "concepts" in output.result
        for concept in output.result["concepts"]:
            assert "concept" in concept
            assert "score" in concept

    def test_dermogpt_vqa_has_answer(self):
        """DermoGPT VQA should return question and answer."""
        output = MockDermoGPTVQA().run(query="What is this lesion?")
        assert "answer" in output.result
        assert "question" in output.result

    def test_general_vqa_has_answer(self):
        """General VQA should return question and answer."""
        output = MockGeneralVQA().run()
        assert "answer" in output.result

    def test_ontology_has_hierarchy(self):
        """Ontology should return hierarchy with parents/siblings."""
        output = MockOntology().run(query="melanoma")
        assert "hierarchy" in output.result
        hierarchy = output.result["hierarchy"]
        assert "parents" in hierarchy
        assert "siblings" in hierarchy

    def test_fairness_probe_has_fitzpatrick(self):
        """Fairness probe should return Fitzpatrick type and ITA."""
        output = MockFairnessProbe().run()
        assert "fitzpatrick_type" in output.result
        assert "ita_angle" in output.result

    def test_uncertainty_probe_has_entropy(self):
        """Uncertainty probe should return entropy and conformal set."""
        output = MockUncertaintyProbe().run()
        assert "predictive_entropy" in output.result
        assert "conformal_set" in output.result
        assert isinstance(output.result["conformal_set"], list)


# ═══════════════════════════════════════════════════════════════════════════
# ToolRegistry — run_batch
# ═══════════════════════════════════════════════════════════════════════════

class TestToolRegistryRunBatch:
    """Tests for ToolRegistry.run_batch()."""

    def test_tool_registry_run_batch(self, mock_tool_registry: ToolRegistry):
        """run_batch() should return outputs for all requested tools."""
        outputs = mock_tool_registry.run_batch(
            tool_names=["panderm_classifier", "make_annotator", "case_rag"],
            query="classify lesion",
        )

        assert len(outputs) == 3
        assert all(isinstance(o, ToolOutput) for o in outputs)

    def test_run_batch_preserves_order(self, mock_tool_registry: ToolRegistry):
        """Outputs should match the order of requested tool names."""
        names = ["case_rag", "panderm_classifier", "guideline_rag"]
        outputs = mock_tool_registry.run_batch(tool_names=names)

        assert [o.tool_name for o in outputs] == names

    def test_run_batch_handles_unknown_tool_gracefully(
        self, mock_tool_registry: ToolRegistry
    ):
        """Unknown tools should produce error outputs, not crash."""
        outputs = mock_tool_registry.run_batch(
            tool_names=["panderm_classifier", "nonexistent_tool"]
        )

        assert len(outputs) == 2
        assert outputs[0].tool_name == "panderm_classifier"
        assert outputs[0].confidence > 0.0

        assert outputs[1].tool_name == "nonexistent_tool"
        assert outputs[1].confidence == 0.0
        assert "error" in outputs[1].result

    def test_run_batch_empty_list(self, mock_tool_registry: ToolRegistry):
        """An empty tool list should return an empty output list."""
        outputs = mock_tool_registry.run_batch(tool_names=[])
        assert outputs == []

    def test_run_batch_all_nine_tools(self, mock_tool_registry: ToolRegistry):
        """Running all 9 tools should produce 9 outputs."""
        all_names = mock_tool_registry.tool_names
        outputs = mock_tool_registry.run_batch(tool_names=all_names)

        assert len(outputs) == 9
        assert all(isinstance(o, ToolOutput) for o in outputs)
        assert all(o.confidence >= 0.0 for o in outputs)
