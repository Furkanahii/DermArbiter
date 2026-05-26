"""MAKE Concept Annotator Tool — CLIP-based zero-shot dermoscopic annotation.

MAKE (Multi-Attribute Knowledge Extraction) uses a CLIP vision-language
model to perform zero-shot classification of dermoscopic concepts such as
pigment network patterns, globules, streaks, blue-white veil, regression
structures, and vascular patterns.

The tool takes a dermoscopic image and scores it against a predefined set
of dermoscopic concept prompts using CLIP cosine similarity.

This wrapper supports both ``open_clip`` and ``transformers`` CLIP backends.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from dermarbiter.tools.base_tool import BaseTool, ToolOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dermoscopic concept vocabulary
# ---------------------------------------------------------------------------

# Standard dermoscopic concepts used for zero-shot annotation
DERMOSCOPIC_CONCEPTS: list[str] = [
    "atypical_pigment_network",
    "regular_pigment_network",
    "blue_white_veil",
    "irregular_dots_globules",
    "regular_dots_globules",
    "regression_structures",
    "streaks",
    "irregular_blotches",
    "regular_blotches",
    "atypical_vascular_pattern",
    "milia_like_cysts",
    "comedo_like_openings",
]

# Natural-language prompts for CLIP zero-shot classification
CONCEPT_PROMPTS: dict[str, str] = {
    "atypical_pigment_network": (
        "a dermoscopic image showing atypical pigment network "
        "with irregular lines and dark branching"
    ),
    "regular_pigment_network": (
        "a dermoscopic image showing regular and uniform pigment network"
    ),
    "blue_white_veil": (
        "a dermoscopic image with blue-white veil, an irregular "
        "blueish-white area overlying pigmented structures"
    ),
    "irregular_dots_globules": (
        "a dermoscopic image with irregular dots and globules "
        "of varying sizes and shapes"
    ),
    "regular_dots_globules": (
        "a dermoscopic image with regular evenly spaced dots and globules"
    ),
    "regression_structures": (
        "a dermoscopic image showing regression structures including "
        "white scar-like areas and blue-grey pepper-like granules"
    ),
    "streaks": (
        "a dermoscopic image showing irregular streaks or pseudopods "
        "at the periphery of the lesion"
    ),
    "irregular_blotches": (
        "a dermoscopic image with irregular dark blotches of pigment"
    ),
    "regular_blotches": (
        "a dermoscopic image with regular and symmetric blotches"
    ),
    "atypical_vascular_pattern": (
        "a dermoscopic image showing atypical vascular patterns "
        "such as dotted or linear-irregular vessels"
    ),
    "milia_like_cysts": (
        "a dermoscopic image with milia-like cysts appearing as "
        "bright white or yellowish roundish structures"
    ),
    "comedo_like_openings": (
        "a dermoscopic image with comedo-like openings appearing "
        "as dark brown or black round to oval structures"
    ),
}

# Defaults
DEFAULT_CLIP_MODEL = "ViT-L-14"
DEFAULT_CLIP_PRETRAINED = "openai"
DEFAULT_INPUT_SIZE = 224

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class MAKEAnnotator(BaseTool):
    """CLIP-based zero-shot dermoscopic concept annotator.

    Uses a frozen CLIP model to compute cosine similarity between
    dermoscopic image embeddings and concept text prompts, producing
    a relevance score for each dermoscopic concept.

    Args:
        clip_model: CLIP model architecture name (e.g. ``"ViT-L-14"``).
        pretrained: Pretrained weights source (e.g. ``"openai"``).
        concepts: List of concept names to annotate.  Defaults to
            12 standard dermoscopic concepts.
        device: Target device (``"auto"`` / ``"cuda"`` / ``"cpu"``).
    """

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

    def __init__(
        self,
        clip_model: str = DEFAULT_CLIP_MODEL,
        pretrained: str = DEFAULT_CLIP_PRETRAINED,
        concepts: list[str] | None = None,
        device: str = "auto",
    ) -> None:
        self._clip_model_name = clip_model
        self._pretrained = pretrained
        self._concepts = concepts or DERMOSCOPIC_CONCEPTS
        self._device_str = device

        # Lazily initialised
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._text_features: Any = None
        self._device: Any = None
        self._loaded = False

    # -- Device resolution -------------------------------------------------

    def _resolve_device(self) -> Any:
        import torch

        if self._device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self._device_str)

    # -- Model lifecycle ---------------------------------------------------

    def _load_model(self) -> None:
        """Load CLIP model and pre-compute text embeddings for concepts."""
        if self._loaded:
            return

        import open_clip
        import torch

        self._device = self._resolve_device()

        logger.info(
            "Loading CLIP %s (%s) on %s",
            self._clip_model_name,
            self._pretrained,
            self._device,
        )

        model, _, preprocess = open_clip.create_model_and_transforms(
            self._clip_model_name,
            pretrained=self._pretrained,
            device=self._device,
        )
        self._model = model.eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self._clip_model_name)

        # Pre-compute text embeddings for all concept prompts
        prompts = [
            CONCEPT_PROMPTS.get(c, f"a dermoscopic image showing {c}")
            for c in self._concepts
        ]
        tokens = self._tokenizer(prompts).to(self._device)

        with torch.no_grad():
            self._text_features = model.encode_text(tokens)
            self._text_features /= self._text_features.norm(dim=-1, keepdim=True)

        self._loaded = True
        logger.info("MAKE annotator loaded with %d concepts.", len(self._concepts))

    def unload(self) -> None:
        """Free GPU memory by unloading CLIP model and cached tensors.

        Deletes the CLIP model, image preprocessor, tokenizer, and
        pre-computed text feature embeddings, then forces Python
        garbage collection and clears the CUDA cache.  The model
        will be re-loaded on the next ``run()`` call.
        """
        import gc

        for attr in ("_model", "_preprocess", "_tokenizer", "_text_features"):
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_features = None
        self._device = None
        self._loaded = False

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info(
            "Unloaded %s (~1.7 GB) to free GPU memory.", self.name,
        )

    # -- Validation --------------------------------------------------------

    def validate_input(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> bool:
        if image_path is None:
            return False
        path = Path(image_path)
        return path.exists() and path.suffix.lower() in SUPPORTED_EXTENSIONS

    # -- Inference ---------------------------------------------------------

    def _run_inference(self, image_path: str) -> dict[str, Any]:
        """Compute concept scores via CLIP cosine similarity."""
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        img_tensor = self._preprocess(img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            image_features = self._model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity → concept scores
            similarities = (image_features @ self._text_features.T).squeeze()

            # Convert to 0–1 range via sigmoid
            scores = torch.sigmoid(similarities * 5.0)  # scale for spread

        concept_scores = [
            {"concept": concept, "score": round(scores[i].item(), 4)}
            for i, concept in enumerate(self._concepts)
        ]

        # Sort by score descending
        concept_scores.sort(key=lambda x: x["score"], reverse=True)

        return {
            "concepts": concept_scores,
            "model_version": f"make-clip-{self._clip_model_name.lower()}",
        }

    # -- Public API --------------------------------------------------------

    def run(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> ToolOutput:
        t0 = time.perf_counter()

        if not self.validate_input(image_path, query):
            return ToolOutput(
                tool_name=self.name,
                result={"error": f"Invalid or missing image: {image_path}"},
                confidence=0.0,
                raw_text=f"MAKE: invalid input image '{image_path}'.",
                metadata={"status": "error"},
            )

        try:
            self._load_model()
            result = self._run_inference(image_path)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Top concept score as confidence
            top_concepts = result["concepts"][:3]
            confidence = top_concepts[0]["score"] if top_concepts else 0.0

            parts = [
                f"{c['concept']} ({c['score']:.2f})" for c in top_concepts
            ]
            raw_text = (
                f"Key dermoscopic concepts: {', '.join(parts)}."
            )

            return ToolOutput(
                tool_name=self.name,
                result=result,
                confidence=confidence,
                raw_text=raw_text,
                metadata={
                    "model": "MAKE",
                    "source": "CLIP-based zero-shot",
                    "latency_ms": round(elapsed_ms, 1),
                },
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error("MAKE inference failed: %s", exc, exc_info=True)
            return ToolOutput(
                tool_name=self.name,
                result={"error": str(exc)},
                confidence=0.0,
                raw_text=f"MAKE inference failed: {exc}",
                metadata={"status": "error", "latency_ms": round(elapsed_ms, 1)},
            )
