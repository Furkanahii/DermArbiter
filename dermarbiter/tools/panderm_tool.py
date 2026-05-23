"""PanDerm Classifier Tool — Wrapper for the PanDerm foundation model.

PanDerm (Yan et al., Nature Medicine 2025) is a universal dermatology
foundation model pre-trained on ~2M dermoscopic images using masked
latent modelling and CLIP feature alignment.  The core encoder is a
**ViT-Large** (``vit_large_patch16_224``) producing 1024-dim feature
embeddings.

Since PanDerm is a **feature extractor** (not a classifier), this
wrapper adds a linear classification head on top of the frozen encoder.
The head must be fine-tuned on a labelled dataset (e.g. HAM10000) and
saved alongside the encoder checkpoint.

Reference:
    Yan et al. "A Universal Dermatology Foundation Model",
    Nature Medicine, 2025.
    GitHub: https://github.com/SiyuanYan1/PanDerm

This wrapper loads the frozen PanDerm ViT-Large checkpoint via ``timm``
and runs single-image inference, returning a ``ToolOutput`` with top-k
disease probabilities.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from dermarbiter.tools.base_tool import BaseTool, ToolOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------

# HAM10000 canonical 7-class labels (alphabetical by abbreviation)
HAM10000_CLASSES: list[str] = [
    "actinic_keratosis",       # akiec
    "basal_cell_carcinoma",    # bcc
    "benign_keratosis",        # bkl
    "dermatofibroma",          # df
    "melanoma",                # mel
    "melanocytic_nevus",       # nv
    "vascular_lesion",         # vasc
]

# Extended labels matching mock tool output format
DERMATOLOGY_CLASSES: list[str] = [
    "melanoma",
    "basal_cell_carcinoma",
    "melanocytic_nevus",
    "squamous_cell_carcinoma",
    "seborrheic_keratosis",
    "dermatofibroma",
    "actinic_keratosis",
]

# ImageNet normalisation constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Defaults
DEFAULT_INPUT_SIZE = 224
DEFAULT_EMBEDDING_DIM = 1024  # ViT-Large
DEFAULT_MODEL_PATH = "weights/panderm.pth"
DEFAULT_HEAD_PATH = "weights/panderm_head.pth"

# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class PanDermClassifier(BaseTool):
    """PanDerm foundation model for skin lesion classification.

    Architecture:
        - Encoder: ViT-Large/16 (``vit_large_patch16_224``) from timm
        - Head: Linear classifier (1024 → num_classes)
        - Both are frozen at inference time

    The encoder checkpoint is loaded from ``model_path`` and the
    classification head from ``head_path``.  If neither is found, a
    pretrained ImageNet ViT is used as a development fallback.

    Args:
        model_path: Path to the PanDerm encoder ``.pth`` checkpoint.
        head_path: Path to the classification head ``.pth`` file.
        class_labels: Ordered list of disease class labels.
        input_size: Expected input resolution (square).
        device: Target device (``"auto"``, ``"cuda"``, ``"cpu"``, ``"mps"``).
        top_k: Number of top predictions to include in output.
    """

    # -- BaseTool interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "panderm_classifier"

    @property
    def description(self) -> str:
        return (
            "PanDerm foundation model for skin lesion classification. "
            "Returns top-k disease probabilities from dermoscopic images. "
            "(Yan et al., Nature Medicine 2025)"
        )

    # -- Constructor -------------------------------------------------------

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        head_path: str = DEFAULT_HEAD_PATH,
        class_labels: list[str] | None = None,
        input_size: int = DEFAULT_INPUT_SIZE,
        device: str = "auto",
        top_k: int = 7,
    ) -> None:
        self._model_path = model_path
        self._head_path = head_path
        self._class_labels = class_labels or DERMATOLOGY_CLASSES
        self._input_size = input_size
        self._device_str = device
        self._top_k = min(top_k, len(self._class_labels))

        # Lazily initialised
        self._model: Any = None
        self._head: Any = None
        self._transform: Any = None
        self._device: Any = None
        self._loaded = False

    # -- Model lifecycle ---------------------------------------------------

    def _resolve_device(self) -> Any:
        """Resolve the target device string to a ``torch.device``."""
        import torch

        if self._device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self._device_str)

    def _build_transform(self) -> Any:
        """Build the image preprocessing pipeline.

        PanDerm uses standard ImageNet preprocessing:
            Resize(224) → CenterCrop(224) → ToTensor → Normalize
        """
        from torchvision import transforms

        return transforms.Compose([
            transforms.Resize(
                (self._input_size, self._input_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(self._input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _load_model(self) -> None:
        """Load the PanDerm ViT-Large encoder and classification head.

        The model is loaded once and kept on-device until ``unload()``
        is called.  Loading strategy:

        1. If ``model_path`` exists → load PanDerm encoder checkpoint
        2. Otherwise → fallback to pretrained ImageNet ViT-Large
        3. If ``head_path`` exists → load trained classification head
        4. Otherwise → initialise a random linear head (dev only)
        """
        if self._loaded:
            return

        import timm
        import torch
        import torch.nn as nn

        self._device = self._resolve_device()
        self._transform = self._build_transform()

        encoder_path = Path(self._model_path)
        head_path = Path(self._head_path)

        # -- Load encoder --------------------------------------------------
        if encoder_path.exists():
            logger.info(
                "Loading PanDerm encoder from %s on %s",
                encoder_path,
                self._device,
            )
            # PanDerm uses ViT-Large/16
            self._model = timm.create_model(
                "vit_large_patch16_224",
                pretrained=False,
                num_classes=0,  # remove default head → feature extractor
            )

            checkpoint = torch.load(
                encoder_path,
                map_location=self._device,
                weights_only=False,
            )

            # Handle various checkpoint formats
            if isinstance(checkpoint, dict):
                state_dict = (
                    checkpoint.get("model")
                    or checkpoint.get("state_dict")
                    or checkpoint.get("model_state_dict")
                    or checkpoint
                )
            else:
                state_dict = checkpoint

            # Strip 'module.' prefix from DataParallel checkpoints
            state_dict = {
                k.replace("module.", ""): v
                for k, v in state_dict.items()
            }

            # Filter out classification head weights if present
            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("head.") and not k.startswith("fc.")
            }

            self._model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(
                "PanDerm checkpoint not found at %s — using pretrained "
                "ImageNet ViT-Large as fallback (NOT for production).",
                encoder_path,
            )
            self._model = timm.create_model(
                "vit_large_patch16_224",
                pretrained=True,
                num_classes=0,
            )

        # -- Load classification head --------------------------------------
        num_classes = len(self._class_labels)

        if head_path.exists():
            logger.info("Loading classification head from %s", head_path)
            self._head = nn.Linear(DEFAULT_EMBEDDING_DIM, num_classes)
            head_state = torch.load(
                head_path,
                map_location=self._device,
                weights_only=True,
            )
            self._head.load_state_dict(head_state)
        else:
            logger.warning(
                "Classification head not found at %s — using random "
                "initialisation (NOT for production).",
                head_path,
            )
            self._head = nn.Linear(DEFAULT_EMBEDDING_DIM, num_classes)

        # Move to device and freeze
        self._model = self._model.to(self._device)
        self._head = self._head.to(self._device)
        self._model.eval()
        self._head.eval()
        self._loaded = True

        param_count = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "PanDerm loaded: %s encoder params, %d classes, device=%s",
            f"{param_count:,}",
            num_classes,
            self._device,
        )

    def unload(self) -> None:
        """Release the model from GPU memory."""
        if self._model is not None or self._head is not None:
            import torch

            del self._model
            del self._head
            self._model = None
            self._head = None
            self._loaded = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("PanDerm model unloaded, GPU cache cleared.")

    # -- Input validation --------------------------------------------------

    def validate_input(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> bool:
        """Validate that the image path exists and is a supported format."""
        if image_path is None:
            return False

        path = Path(image_path)
        if not path.exists():
            logger.warning("Image not found: %s", image_path)
            return False

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(
                "Unsupported image format: %s (supported: %s)",
                path.suffix,
                SUPPORTED_EXTENSIONS,
            )
            return False

        return True

    # -- Core inference ----------------------------------------------------

    def _preprocess(self, image_path: str) -> Any:
        """Load and preprocess a dermoscopic image for inference.

        Returns:
            ``torch.Tensor`` of shape ``(1, 3, 224, 224)`` on device.
        """
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        tensor = self._transform(img)                  # (3, H, W)
        return tensor.unsqueeze(0).to(self._device)    # (1, 3, H, W)

    def _run_inference(self, image_path: str) -> dict[str, Any]:
        """Run encoder + head and return structured predictions.

        Pipeline:
            image → preprocess → ViT encoder → 1024-d features
            → linear head → softmax → top-k predictions
        """
        import torch

        tensor = self._preprocess(image_path)

        with torch.no_grad():
            features = self._model.forward_features(tensor)  # (1, 1024)

            # Handle both pooled (1, D) and sequence (1, N+1, D) outputs
            if features.dim() == 3:
                features = features[:, 0, :]  # use [CLS] token

            logits = self._head(features)                     # (1, C)
            probs = torch.softmax(logits, dim=-1).squeeze()   # (C,)

        # Sort by descending probability
        top_probs, top_indices = torch.topk(probs, k=self._top_k)

        predictions = [
            {
                "disease": self._class_labels[idx.item()],
                "probability": round(prob.item(), 4),
            }
            for prob, idx in zip(top_probs, top_indices, strict=True)
        ]

        return {
            "predictions": predictions,
            "model_version": "panderm-v1.0",
            "input_resolution": f"{self._input_size}x{self._input_size}",
            "num_classes": len(self._class_labels),
            "embedding_dim": DEFAULT_EMBEDDING_DIM,
            "all_probabilities": {
                self._class_labels[i]: round(probs[i].item(), 4)
                for i in range(len(self._class_labels))
            },
        }

    # -- Public API --------------------------------------------------------

    def run(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> ToolOutput:
        """Classify a dermoscopic image using the PanDerm model.

        Args:
            image_path: Path to the dermoscopic image (required).
            query:      Ignored for classification; included for API
                        consistency.

        Returns:
            ``ToolOutput`` with disease probabilities and confidence.
        """
        t0 = time.perf_counter()

        # Validate input
        if not self.validate_input(image_path, query):
            return ToolOutput(
                tool_name=self.name,
                result={"error": f"Invalid or missing image: {image_path}"},
                confidence=0.0,
                raw_text=f"PanDerm: invalid input image '{image_path}'.",
                metadata={"status": "error"},
            )

        try:
            # Lazy load model
            self._load_model()

            # Run inference
            result = self._run_inference(image_path)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Top prediction confidence
            top_pred = result["predictions"][0]
            confidence = top_pred["probability"]

            # Build human-readable summary
            top3 = result["predictions"][:3]
            parts = [
                f"{p['disease']} ({p['probability']:.0%})" for p in top3
            ]
            raw_text = f"Top prediction: {parts[0]}"
            if len(parts) > 1:
                raw_text += f", followed by {parts[1]}"
            if len(parts) > 2:
                raw_text += f" and {parts[2]}"
            raw_text += "."

            return ToolOutput(
                tool_name=self.name,
                result=result,
                confidence=confidence,
                raw_text=raw_text,
                metadata={
                    "model": "PanDerm",
                    "source": "Nature Medicine 2025",
                    "latency_ms": round(elapsed_ms, 1),
                    "device": str(self._device),
                    "architecture": "vit_large_patch16_224",
                },
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error("PanDerm inference failed: %s", exc, exc_info=True)
            return ToolOutput(
                tool_name=self.name,
                result={"error": str(exc)},
                confidence=0.0,
                raw_text=f"PanDerm inference failed: {exc}",
                metadata={
                    "status": "error",
                    "latency_ms": round(elapsed_ms, 1),
                },
            )
