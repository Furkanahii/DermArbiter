"""CaseRAG — Similar case retrieval using DermLIP embeddings + ChromaDB.

Retrieves visually similar dermatology cases from the Derm1M database
(413K+ cases) by computing DermLIP image embeddings and performing
nearest-neighbour search via ChromaDB.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from dermarbiter.tools.base_tool import BaseTool, ToolOutput

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

DEFAULT_CHROMA_DIR = "data/chroma_cases"
DEFAULT_COLLECTION = "derm1m_cases"
DEFAULT_CLIP_MODEL = "hf-hub:redlessone/DermLIP_ViT-B-16"
DEFAULT_TOP_K = 5


class CaseRAG(BaseTool):
    """Case-based retrieval from Derm1M using DermLIP embeddings.

    Args:
        chroma_persist_dir: Path to ChromaDB persistence directory.
        collection_name: Name of the ChromaDB collection.
        clip_model: ``open_clip`` model identifier for image embedding.
        top_k: Number of similar cases to retrieve.
        device: Target device.
    """

    @property
    def name(self) -> str:
        return "case_rag"

    @property
    def description(self) -> str:
        return (
            "Case RAG — retrieves visually similar cases from Derm1M "
            "(413K+ dermatology cases) using DermLIP embeddings."
        )

    def __init__(
        self,
        chroma_persist_dir: str = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        clip_model: str = DEFAULT_CLIP_MODEL,
        top_k: int = DEFAULT_TOP_K,
        device: str = "auto",
    ) -> None:
        self._chroma_dir = chroma_persist_dir
        self._collection_name = collection_name
        self._clip_model_name = clip_model
        self._top_k = top_k
        self._device_str = device

        self._model: Any = None
        self._preprocess: Any = None
        self._collection: Any = None
        self._device: Any = None
        self._loaded = False

    def _resolve_device(self) -> Any:
        import torch

        if self._device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self._device_str)

    def _load_model(self) -> None:
        if self._loaded:
            return

        import chromadb
        import open_clip

        self._device = self._resolve_device()

        logger.info("Loading DermLIP encoder %s", self._clip_model_name)
        model, _, preprocess = open_clip.create_model_and_transforms(
            self._clip_model_name, device=self._device,
        )
        self._model = model.eval()
        self._preprocess = preprocess

        logger.info("Connecting to ChromaDB at %s", self._chroma_dir)
        client = chromadb.PersistentClient(path=self._chroma_dir)
        self._collection = client.get_or_create_collection(self._collection_name)

        self._loaded = True
        logger.info(
            "CaseRAG loaded: collection '%s' (%d items).",
            self._collection_name,
            self._collection.count(),
        )

    def unload(self) -> None:
        """Free GPU memory by unloading DermLIP encoder and references.

        Deletes the CLIP model, image preprocessor, and ChromaDB
        collection reference, then forces Python garbage collection
        and clears the CUDA cache.  The model will be re-loaded on
        the next ``run()`` call.
        """
        import gc

        for attr in ("_model", "_preprocess"):
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
        self._model = None
        self._preprocess = None
        self._collection = None
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
            "Unloaded %s to free GPU memory.", self.name,
        )

    def validate_input(self, image_path: str | None = None, query: str = "") -> bool:
        if image_path is None:
            return False
        path = Path(image_path)
        return path.exists() and path.suffix.lower() in SUPPORTED_EXTENSIONS

    def _embed_image(self, image_path: str) -> list[float]:
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        tensor = self._preprocess(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze().cpu().tolist()

    def _run_inference(self, image_path: str) -> dict[str, Any]:
        embedding = self._embed_image(image_path)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=self._top_k,
            include=["metadatas", "distances"],
        )

        similar_cases = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for i, case_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            similar_cases.append({
                "case_id": case_id,
                "diagnosis": meta.get("diagnosis", "unknown"),
                "distance": round(distances[i], 4) if i < len(distances) else 0.0,
                "location": meta.get("location", "unknown"),
                "age": meta.get("age", 0),
            })

        return {
            "similar_cases": similar_cases,
            "encoder": "DermLIP-ViT-B/16",
            "index_size": self._collection.count(),
        }

    def run(self, image_path: str | None = None, query: str = "") -> ToolOutput:
        t0 = time.perf_counter()

        if not self.validate_input(image_path, query):
            return ToolOutput(
                tool_name=self.name,
                result={"error": f"Invalid or missing image: {image_path}"},
                confidence=0.0,
                raw_text=f"CaseRAG: invalid input '{image_path}'.",
                metadata={"status": "error"},
            )

        try:
            self._load_model()
            result = self._run_inference(image_path)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            cases = result["similar_cases"]
            if cases:
                avg_dist = sum(c["distance"] for c in cases) / len(cases)
                confidence = round(max(0.0, min(1.0, 1.0 - avg_dist)), 2)
                top3 = cases[:3]
                diags = [f"{c['diagnosis']} (d={c['distance']:.2f})" for c in top3]
                raw_text = f"Top similar cases: {', '.join(diags)}."
            else:
                confidence = 0.0
                raw_text = "No similar cases found."

            return ToolOutput(
                tool_name=self.name,
                result=result,
                confidence=confidence,
                raw_text=raw_text,
                metadata={
                    "model": "DermLIP",
                    "database": "Derm1M",
                    "retrieval_k": self._top_k,
                    "latency_ms": round(elapsed_ms, 1),
                },
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error("CaseRAG failed: %s", exc, exc_info=True)
            return ToolOutput(
                tool_name=self.name,
                result={"error": str(exc)},
                confidence=0.0,
                raw_text=f"CaseRAG failed: {exc}",
                metadata={"status": "error", "latency_ms": round(elapsed_ms, 1)},
            )
