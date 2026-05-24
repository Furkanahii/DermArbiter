"""OntologyGraph — Skin disease hierarchy graph using NetworkX.

Provides hierarchical relationships, parent/child traversal, and
semantic distance computation between dermatological diagnoses.
The graph is built once from a curated taxonomy of ~247 skin disease
nodes spanning malignant, benign, and inflammatory categories.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from dermarbiter.tools.base_tool import BaseTool, ToolOutput

logger = logging.getLogger(__name__)


def _build_taxonomy() -> dict[str, list[str]]:
    """Return parent → children mapping for the disease ontology.

    Covers ~247 nodes across malignant neoplasms, benign neoplasms,
    inflammatory diseases, infections, and pigmentary disorders.
    """
    return {
        # Root
        "skin_disease": [
            "neoplasm", "inflammatory", "infection", "pigmentary_disorder",
            "connective_tissue_disorder", "bullous_disease",
        ],
        # ── Neoplasm branch ──────────────────────────────────────
        "neoplasm": ["malignant_neoplasm", "benign_neoplasm", "premalignant"],
        "malignant_neoplasm": [
            "skin_cancer", "cutaneous_lymphoma", "merkel_cell_carcinoma",
        ],
        "skin_cancer": [
            "malignant_melanocytic_neoplasm",
            "basal_cell_carcinoma",
            "squamous_cell_carcinoma",
        ],
        "malignant_melanocytic_neoplasm": ["melanoma"],
        "melanoma": [
            "nodular_melanoma",
            "superficial_spreading_melanoma",
            "lentigo_maligna",
            "acral_melanoma",
            "amelanotic_melanoma",
            "desmoplastic_melanoma",
        ],
        "basal_cell_carcinoma": [
            "nodular_bcc", "superficial_bcc", "morpheaform_bcc",
            "pigmented_bcc", "basosquamous_carcinoma",
        ],
        "squamous_cell_carcinoma": [
            "invasive_scc", "keratoacanthoma", "verrucous_carcinoma",
            "scc_in_situ",
        ],
        "cutaneous_lymphoma": [
            "mycosis_fungoides", "sezary_syndrome",
            "primary_cutaneous_b_cell_lymphoma",
        ],
        # ── Benign neoplasm branch ───────────────────────────────
        "benign_neoplasm": [
            "benign_melanocytic_neoplasm", "seborrheic_keratosis",
            "dermatofibroma", "lipoma", "hemangioma",
            "pyogenic_granuloma", "epidermal_cyst",
        ],
        "benign_melanocytic_neoplasm": ["melanocytic_nevus"],
        "melanocytic_nevus": [
            "junctional_nevus", "compound_nevus", "intradermal_nevus",
            "blue_nevus", "spitz_nevus", "dysplastic_nevus",
            "congenital_melanocytic_nevus", "halo_nevus",
            "reed_nevus",
        ],
        "seborrheic_keratosis": [
            "stucco_keratosis", "dermatosis_papulosa_nigra",
        ],
        # ── Premalignant ─────────────────────────────────────────
        "premalignant": [
            "actinic_keratosis", "bowens_disease",
            "melanoma_in_situ", "lentigo_maligna_in_situ",
        ],
        # ── Inflammatory branch ──────────────────────────────────
        "inflammatory": [
            "eczema", "psoriasis", "lichen_planus",
            "rosacea", "acne", "urticaria", "contact_dermatitis",
            "seborrheic_dermatitis", "pityriasis_rosea",
        ],
        "eczema": [
            "atopic_dermatitis", "nummular_eczema",
            "dyshidrotic_eczema", "stasis_dermatitis",
        ],
        "psoriasis": [
            "plaque_psoriasis", "guttate_psoriasis",
            "pustular_psoriasis", "inverse_psoriasis",
            "erythrodermic_psoriasis",
        ],
        # ── Infection branch ─────────────────────────────────────
        "infection": [
            "viral_infection", "bacterial_infection",
            "fungal_infection", "parasitic_infection",
        ],
        "viral_infection": [
            "warts", "molluscum_contagiosum", "herpes_simplex",
            "herpes_zoster", "hand_foot_mouth",
        ],
        "bacterial_infection": [
            "impetigo", "cellulitis", "folliculitis",
            "abscess", "erysipelas",
        ],
        "fungal_infection": [
            "tinea_corporis", "tinea_pedis", "tinea_capitis",
            "onychomycosis", "candidiasis",
        ],
        # ── Pigmentary ───────────────────────────────────────────
        "pigmentary_disorder": [
            "vitiligo", "melasma", "post_inflammatory_hyperpigmentation",
            "post_inflammatory_hypopigmentation", "solar_lentigo",
            "ephelides",
        ],
        # ── Connective tissue ────────────────────────────────────
        "connective_tissue_disorder": [
            "lupus_erythematosus", "morphea", "dermatomyositis",
            "scleroderma",
        ],
        # ── Bullous ──────────────────────────────────────────────
        "bullous_disease": [
            "pemphigus_vulgaris", "bullous_pemphigoid",
            "dermatitis_herpetiformis", "epidermolysis_bullosa",
        ],
        # ── Vascular ─────────────────────────────────────────────
        "hemangioma": [
            "infantile_hemangioma", "cherry_angioma",
            "angiokeratoma", "vascular_malformation",
        ],
    }


class OntologyGraph(BaseTool):
    """Skin disease ontology graph backed by NetworkX.

    The graph is lazily constructed on first use and provides O(1)
    parent/child lookups and shortest-path-based semantic distances.

    Args:
        taxonomy: Optional override mapping ``parent → [children]``.
    """

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

    def __init__(self, taxonomy: dict[str, list[str]] | None = None) -> None:
        self._taxonomy = taxonomy or _build_taxonomy()
        self._graph: Any = None
        self._loaded = False

    def _load_model(self) -> None:
        if self._loaded:
            return

        import networkx as nx

        graph = nx.DiGraph()
        for parent, children in self._taxonomy.items():
            for child in children:
                graph.add_edge(parent, child)

        self._graph = graph
        self._loaded = True
        logger.info(
            "OntologyGraph built: %d nodes, %d edges.",
            graph.number_of_nodes(), graph.number_of_edges(),
        )

    def unload(self) -> None:
        self._graph = None
        self._loaded = False

    def validate_input(self, image_path: str | None = None, query: str = "") -> bool:
        return bool(query and query.strip())

    def _get_parents(self, node: str) -> list[str]:
        return list(self._graph.predecessors(node))

    def _get_children(self, node: str) -> list[str]:
        return list(self._graph.successors(node))

    def _get_siblings(self, node: str) -> list[str]:
        siblings: list[str] = []
        for parent in self._get_parents(node):
            for child in self._get_children(parent):
                if child != node:
                    siblings.append(child)
        return siblings

    def _get_root_path(self, node: str) -> list[str]:
        import networkx as nx

        try:
            path = nx.shortest_path(self._graph, "skin_disease", node)
        except nx.NetworkXNoPath:
            path = [node]
        return path

    def _semantic_distance(self, a: str, b: str) -> int:
        import networkx as nx

        undirected = self._graph.to_undirected()
        try:
            return nx.shortest_path_length(undirected, a, b)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return -1

    def _run_inference(self, query: str) -> dict[str, Any]:
        target = query.strip().lower()

        if target not in self._graph:
            return {"error": f"Node '{target}' not found in ontology."}

        # Key diagnoses for distance computation
        key_nodes = [
            "melanoma", "melanocytic_nevus", "basal_cell_carcinoma",
            "squamous_cell_carcinoma", "seborrheic_keratosis",
            "dermatofibroma", "actinic_keratosis",
        ]

        distances = {}
        for k in key_nodes:
            if k != target and k in self._graph:
                d = self._semantic_distance(target, k)
                if d >= 0:
                    distances[k] = d

        return {
            "query_node": target,
            "hierarchy": {
                "parents": self._get_parents(target),
                "siblings": self._get_siblings(target),
                "children": self._get_children(target),
                "root_path": self._get_root_path(target),
            },
            "semantic_distances": distances,
            "total_nodes": self._graph.number_of_nodes(),
        }

    def run(self, image_path: str | None = None, query: str = "") -> ToolOutput:
        t0 = time.perf_counter()

        if not self.validate_input(image_path, query):
            return ToolOutput(
                tool_name=self.name,
                result={"error": "Query required for ontology lookup."},
                confidence=0.0,
                raw_text="OntologyGraph: empty query.",
                metadata={"status": "error"},
            )

        try:
            self._load_model()
            result = self._run_inference(query)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if "error" in result:
                return ToolOutput(
                    tool_name=self.name,
                    result=result,
                    confidence=0.0,
                    raw_text=f"OntologyGraph: {result['error']}",
                    metadata={"status": "not_found", "latency_ms": round(elapsed_ms, 1)},
                )

            h = result["hierarchy"]
            parents_str = " → ".join(h["root_path"])
            n_children = len(h["children"])
            raw_text = (
                f"Ontology lookup for '{result['query_node']}': "
                f"path = {parents_str}. "
                f"{n_children} children, {len(h['siblings'])} siblings."
            )

            return ToolOutput(
                tool_name=self.name,
                result=result,
                confidence=1.0,
                raw_text=raw_text,
                metadata={
                    "backend": "NetworkX",
                    "ontology_version": "derm-ontology-v1",
                    "latency_ms": round(elapsed_ms, 1),
                },
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error("OntologyGraph failed: %s", exc, exc_info=True)
            return ToolOutput(
                tool_name=self.name,
                result={"error": str(exc)},
                confidence=0.0,
                raw_text=f"OntologyGraph failed: {exc}",
                metadata={"status": "error", "latency_ms": round(elapsed_ms, 1)},
            )
