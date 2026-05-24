"""Mock tool implementations for DermArbiter testing.

Provides deterministic, realistic mock versions of all 9 tools
so that agents and the debate protocol can be tested without GPU
hardware or API keys.

Usage::

    from tests.mocks.mock_tools import create_mock_registry
    registry = create_mock_registry()
    output = registry.get("panderm_classifier").run(query="classify lesion")
"""

from __future__ import annotations

from dermarbiter.tools.base_tool import BaseTool, ToolOutput, ToolRegistry


# ═══════════════════════════════════════════════════════════════════════════
# 1. MockPanDermClassifier — disease classification probabilities
# ═══════════════════════════════════════════════════════════════════════════

class MockPanDermClassifier(BaseTool):
    """Mock PanDerm vision foundation model — returns realistic disease
    probability distribution for a melanocytic lesion."""

    @property
    def name(self) -> str:
        return "panderm_classifier"

    @property
    def description(self) -> str:
        return (
            "PanDerm foundation model for skin lesion classification. "
            "Returns top-k disease probabilities from dermoscopic images."
        )

    def run(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> ToolOutput:
        return ToolOutput(
            tool_name=self.name,
            result={
                "predictions": [
                    {"disease": "melanoma", "probability": 0.62},
                    {"disease": "basal_cell_carcinoma", "probability": 0.15},
                    {"disease": "melanocytic_nevus", "probability": 0.11},
                    {"disease": "squamous_cell_carcinoma", "probability": 0.06},
                    {"disease": "seborrheic_keratosis", "probability": 0.03},
                    {"disease": "dermatofibroma", "probability": 0.02},
                    {"disease": "actinic_keratosis", "probability": 0.01},
                ],
                "model_version": "panderm-v1.0",
                "input_resolution": "224x224",
            },
            confidence=0.62,
            raw_text=(
                "Top prediction: melanoma (62%), followed by BCC (15%) "
                "and melanocytic nevus (11%)."
            ),
            metadata={
                "model": "PanDerm",
                "source": "Nature Medicine 2025",
                "latency_ms": 142,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. MockMAKEAnnotator — dermoscopic concept annotations
# ═══════════════════════════════════════════════════════════════════════════

class MockMAKEAnnotator(BaseTool):
    """Mock MAKE (CLIP-based) zero-shot dermoscopic concept annotator."""

    @property
    def name(self) -> str:
        return "make_annotator"

    @property
    def description(self) -> str:
        return (
            "MAKE concept annotator — extracts dermoscopic features "
            "(pigment network, globules, streaks, blue-white veil, etc.) "
            "using CLIP-based zero-shot classification."
        )

    def run(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> ToolOutput:
        return ToolOutput(
            tool_name=self.name,
            result={
                "concepts": [
                    {"concept": "atypical_pigment_network", "score": 0.87},
                    {"concept": "blue_white_veil", "score": 0.74},
                    {"concept": "irregular_dots_globules", "score": 0.69},
                    {"concept": "regression_structures", "score": 0.52},
                    {"concept": "streaks", "score": 0.41},
                    {"concept": "regular_pigment_network", "score": 0.12},
                    {"concept": "milia_like_cysts", "score": 0.05},
                ],
                "model_version": "make-clip-v1",
            },
            confidence=0.87,
            raw_text=(
                "Key dermoscopic concepts: atypical pigment network (0.87), "
                "blue-white veil (0.74), irregular dots/globules (0.69). "
                "Pattern consistent with melanocytic neoplasm."
            ),
            metadata={
                "model": "MAKE",
                "source": "CLIP-based zero-shot",
                "latency_ms": 98,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. MockDermoGPTVQA — dermatology-specific VQA
# ═══════════════════════════════════════════════════════════════════════════

class MockDermoGPTVQA(BaseTool):
    """Mock DermoGPT-RL visual question answering."""

    @property
    def name(self) -> str:
        return "dermogpt_vqa"

    @property
    def description(self) -> str:
        return (
            "DermoGPT-RL — dermatology-specialised VQA model fine-tuned "
            "with SFT + MAVIC reinforcement learning."
        )

    def run(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> ToolOutput:
        effective_query = query or "What is the most likely diagnosis?"
        return ToolOutput(
            tool_name=self.name,
            result={
                "question": effective_query,
                "answer": (
                    "The lesion shows asymmetric borders, irregular pigmentation, "
                    "and blue-white veil, which are characteristic features of "
                    "melanoma. The ABCDE criteria suggest a high index of "
                    "suspicion for malignant melanoma. Recommend excisional "
                    "biopsy for histopathological confirmation."
                ),
                "model_version": "dermogpt-rl-v1",
            },
            confidence=0.78,
            raw_text=(
                "DermoGPT assessment: features consistent with melanoma "
                "(asymmetry, irregular pigmentation, blue-white veil). "
                "Recommends excisional biopsy."
            ),
            metadata={
                "model": "DermoGPT-RL",
                "source": "HuggingFace (gated)",
                "latency_ms": 310,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. MockGeneralVQA — general-purpose medical VQA
# ═══════════════════════════════════════════════════════════════════════════

class MockGeneralVQA(BaseTool):
    """Mock MedGemma-4B general-purpose medical VQA."""

    @property
    def name(self) -> str:
        return "general_vqa"

    @property
    def description(self) -> str:
        return (
            "MedGemma-4B — general medical VQA providing a second, "
            "non-specialist opinion on dermatoscopic images."
        )

    def run(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> ToolOutput:
        effective_query = query or "Describe this skin lesion."
        return ToolOutput(
            tool_name=self.name,
            result={
                "question": effective_query,
                "answer": (
                    "This image shows a pigmented skin lesion with irregular "
                    "borders and color variegation. The lesion exhibits multiple "
                    "shades of brown and black with areas of regression. "
                    "Differential includes melanoma, dysplastic nevus, and "
                    "pigmented basal cell carcinoma."
                ),
                "model_version": "medgemma-4b-v1",
            },
            confidence=0.65,
            raw_text=(
                "General VQA: pigmented lesion with irregular borders and "
                "color variegation. Differential: melanoma, dysplastic nevus, "
                "pigmented BCC."
            ),
            metadata={
                "model": "MedGemma-4B",
                "source": "HuggingFace (gated)",
                "latency_ms": 420,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. MockCaseRAG — similar-case retrieval
# ═══════════════════════════════════════════════════════════════════════════

class MockCaseRAG(BaseTool):
    """Mock case-based retrieval from Derm1M + DermLIP + ChromaDB."""

    @property
    def name(self) -> str:
        return "case_rag"

    @property
    def description(self) -> str:
        return (
            "Case RAG — retrieves visually similar cases from Derm1M "
            "(413K+ dermatology cases) using DermLIP embeddings."
        )

    def run(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> ToolOutput:
        return ToolOutput(
            tool_name=self.name,
            result={
                "similar_cases": [
                    {
                        "case_id": "derm1m_042187",
                        "diagnosis": "melanoma",
                        "distance": 0.12,
                        "location": "upper back",
                        "age": 54,
                    },
                    {
                        "case_id": "derm1m_118934",
                        "diagnosis": "melanoma",
                        "distance": 0.18,
                        "location": "left shoulder",
                        "age": 47,
                    },
                    {
                        "case_id": "derm1m_073621",
                        "diagnosis": "dysplastic_nevus",
                        "distance": 0.24,
                        "location": "trunk",
                        "age": 32,
                    },
                    {
                        "case_id": "derm1m_201455",
                        "diagnosis": "melanoma_in_situ",
                        "distance": 0.27,
                        "location": "right thigh",
                        "age": 61,
                    },
                    {
                        "case_id": "derm1m_089002",
                        "diagnosis": "basal_cell_carcinoma",
                        "distance": 0.35,
                        "location": "face",
                        "age": 68,
                    },
                ],
                "encoder": "DermLIP-ViT-B/16",
                "index_size": 413_000,
            },
            confidence=0.82,
            raw_text=(
                "Top 5 similar cases: 3/5 melanoma (d=0.12, 0.18, 0.27), "
                "1 dysplastic nevus (d=0.24), 1 BCC (d=0.35)."
            ),
            metadata={
                "model": "DermLIP",
                "database": "Derm1M",
                "retrieval_k": 5,
                "latency_ms": 67,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# 6. MockGuidelineRAG — clinical guideline retrieval
# ═══════════════════════════════════════════════════════════════════════════

class MockGuidelineRAG(BaseTool):
    """Mock guideline RAG from DermNet + Mayo Clinic guidelines."""

    @property
    def name(self) -> str:
        return "guideline_rag"

    @property
    def description(self) -> str:
        return (
            "Guideline RAG — retrieves relevant clinical guideline "
            "chunks from DermNet NZ and Mayo Clinic via ChromaDB."
        )

    def run(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> ToolOutput:
        return ToolOutput(
            tool_name=self.name,
            result={
                "chunks": [
                    {
                        "source": "DermNet NZ — Melanoma",
                        "text": (
                            "Melanoma should be suspected in any changing mole, "
                            "new pigmented lesion in an adult, or lesion with "
                            "ABCDE features. Dermoscopic features include "
                            "atypical network, blue-white veil, irregular "
                            "streaks, and regression structures."
                        ),
                        "relevance_score": 0.94,
                    },
                    {
                        "source": "Mayo Clinic — Melanoma Diagnosis",
                        "text": (
                            "Excisional biopsy is the gold standard for "
                            "melanoma diagnosis. Shave biopsies should be "
                            "avoided for suspected melanoma to ensure "
                            "accurate Breslow thickness measurement."
                        ),
                        "relevance_score": 0.88,
                    },
                    {
                        "source": "DermNet NZ — Dermoscopy",
                        "text": (
                            "The revised 7-point checklist for dermoscopy "
                            "assigns major criteria (atypical network, "
                            "blue-white veil, atypical vascular pattern) "
                            "2 points each, and minor criteria 1 point each. "
                            "Score ≥ 3 suggests melanoma."
                        ),
                        "relevance_score": 0.82,
                    },
                ],
                "retrieval_method": "ChromaDB + sentence-transformers",
            },
            confidence=0.91,
            raw_text=(
                "Guidelines support melanoma suspicion: ABCDE criteria, "
                "7-point checklist (score ≥ 3), excisional biopsy recommended."
            ),
            metadata={
                "sources": ["DermNet NZ", "Mayo Clinic"],
                "chunks_retrieved": 3,
                "latency_ms": 45,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# 7. MockOntology — disease hierarchy graph
# ═══════════════════════════════════════════════════════════════════════════

class MockOntology(BaseTool):
    """Mock skin disease ontology graph (NetworkX-based)."""

    @property
    def name(self) -> str:
        return "ontology_graph"

    @property
    def description(self) -> str:
        return (
            "Skin disease ontology graph — provides hierarchical "
            "relationships, parent/child nodes, and semantic distance "
            "between diagnoses."
        )

    def run(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> ToolOutput:
        target = query or "melanoma"
        return ToolOutput(
            tool_name=self.name,
            result={
                "query_node": target,
                "hierarchy": {
                    "parents": ["malignant_melanocytic_neoplasm", "skin_cancer"],
                    "siblings": [
                        "nodular_melanoma",
                        "superficial_spreading_melanoma",
                        "lentigo_maligna",
                        "acral_melanoma",
                    ],
                    "children": [],
                    "root_path": [
                        "skin_disease",
                        "neoplasm",
                        "malignant_neoplasm",
                        "skin_cancer",
                        "malignant_melanocytic_neoplasm",
                        "melanoma",
                    ],
                },
                "semantic_distances": {
                    "melanocytic_nevus": 3,
                    "basal_cell_carcinoma": 2,
                    "squamous_cell_carcinoma": 2,
                    "seborrheic_keratosis": 4,
                },
                "total_nodes": 247,
            },
            confidence=1.0,
            raw_text=(
                f"Ontology lookup for '{target}': parent = "
                "malignant_melanocytic_neoplasm → skin_cancer. "
                "4 melanoma subtypes. Semantic distance to BCC = 2."
            ),
            metadata={
                "backend": "NetworkX",
                "ontology_version": "derm-ontology-v1",
                "latency_ms": 3,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# 8. MockFairnessProbe — Fitzpatrick type + ITA estimation
# ═══════════════════════════════════════════════════════════════════════════

class MockFairnessProbe(BaseTool):
    """Mock fairness probe — estimates Fitzpatrick skin type and ITA angle."""

    @property
    def name(self) -> str:
        return "fairness_probe"

    @property
    def description(self) -> str:
        return (
            "Fairness Probe (★ novel contribution) — estimates Fitzpatrick "
            "skin type and Individual Typology Angle (ITA) from dermoscopic "
            "images for bias-aware diagnosis."
        )

    def run(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> ToolOutput:
        return ToolOutput(
            tool_name=self.name,
            result={
                "fitzpatrick_type": "III",
                "fitzpatrick_confidence": 0.81,
                "ita_angle": 34.7,
                "ita_category": "intermediate",
                "skin_tone_rgb": [186, 154, 122],
                "bias_warning": None,
                "calibration_note": (
                    "Classifier performance may vary for Fitzpatrick "
                    "types V–VI. Monitor per-subgroup metrics."
                ),
            },
            confidence=0.81,
            raw_text=(
                "Fitzpatrick type III (confidence 0.81), ITA = 34.7° "
                "(intermediate). No bias warning for this skin type."
            ),
            metadata={
                "model": "ITA-estimator-v1",
                "contribution": "novel",
                "latency_ms": 55,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# 9. MockUncertaintyProbe — entropy + conformal prediction interval
# ═══════════════════════════════════════════════════════════════════════════

class MockUncertaintyProbe(BaseTool):
    """Mock uncertainty probe — returns entropy and conformal interval."""

    @property
    def name(self) -> str:
        return "uncertainty_probe"

    @property
    def description(self) -> str:
        return (
            "Uncertainty Probe (★ novel contribution) — computes "
            "predictive entropy and conformal prediction set for "
            "calibrated uncertainty quantification."
        )

    def run(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> ToolOutput:
        return ToolOutput(
            tool_name=self.name,
            result={
                "predictive_entropy": 1.23,
                "max_entropy": 2.81,
                "normalised_entropy": 0.44,
                "conformal_set": [
                    "melanoma",
                    "basal_cell_carcinoma",
                ],
                "conformal_alpha": 0.10,
                "coverage_guarantee": 0.90,
                "calibration_ece": 0.034,
                "is_high_uncertainty": False,
            },
            confidence=0.72,
            raw_text=(
                "Normalised entropy = 0.44 (moderate). Conformal set at "
                "α=0.10: {melanoma, BCC} — 90% coverage guarantee. "
                "ECE = 0.034 (well-calibrated)."
            ),
            metadata={
                "method": "entropy + split-conformal",
                "contribution": "novel",
                "calibration_set_size": 1500,
                "latency_ms": 28,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# Registry factory
# ═══════════════════════════════════════════════════════════════════════════

def create_mock_registry() -> ToolRegistry:
    """Create a ``ToolRegistry`` pre-populated with all 9 mock tools.

    Returns:
        A fully populated ``ToolRegistry`` ready for testing.
    """
    registry = ToolRegistry()
    mock_tools: list[BaseTool] = [
        MockPanDermClassifier(),
        MockMAKEAnnotator(),
        MockDermoGPTVQA(),
        MockGeneralVQA(),
        MockCaseRAG(),
        MockGuidelineRAG(),
        MockOntology(),
        MockFairnessProbe(),
        MockUncertaintyProbe(),
    ]
    for tool in mock_tools:
        registry.register(tool)
    return registry
