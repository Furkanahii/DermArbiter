"""Tool wrappers for the DermArbiter tool pool.

This module provides integrations with external models and knowledge sources:
- PanDerm: Universal dermatology foundation model
- MAKE: Multi-attribute knowledge extraction
- DermGPT: Dermatology visual question answering
- MedGemma: General medical VQA (second opinion)
- CaseRAG / GuidelineRAG: Retrieval-augmented generation
- OntologyGraph: ICD-10, SNOMED-CT, DermLex hierarchy
- FairnessProbe: Fitzpatrick skin tone fairness evaluation
- UncertaintyProbe: Prediction uncertainty quantification
"""

from dermarbiter.tools.base_tool import BaseTool, ToolOutput, ToolRegistry
from dermarbiter.tools.case_rag import CaseRAG
from dermarbiter.tools.dermogpt_tool import DermoGPTVQA
from dermarbiter.tools.fairness_probe import FairnessProbe
from dermarbiter.tools.guideline_rag import GuidelineRAG
from dermarbiter.tools.make_tool import MAKEAnnotator
from dermarbiter.tools.medgemma_tool import MedGemmaVQA
from dermarbiter.tools.ontology_graph import OntologyGraph
from dermarbiter.tools.panderm_tool import PanDermClassifier
from dermarbiter.tools.uncertainty_probe import UncertaintyProbe

__all__ = [
    "BaseTool",
    "ToolOutput",
    "ToolRegistry",
    "CaseRAG",
    "DermoGPTVQA",
    "FairnessProbe",
    "GuidelineRAG",
    "MAKEAnnotator",
    "MedGemmaVQA",
    "OntologyGraph",
    "PanDermClassifier",
    "UncertaintyProbe",
]
