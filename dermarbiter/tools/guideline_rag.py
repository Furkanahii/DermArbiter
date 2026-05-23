"""GuidelineRAG — Clinical guideline retrieval via ChromaDB.

Retrieves relevant clinical guideline chunks from dermatology knowledge
bases (DermNet NZ, Mayo Clinic, etc.) using sentence-transformer
embeddings and ChromaDB vector search.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from dermarbiter.tools.base_tool import BaseTool, ToolOutput

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_DIR = "data/chroma_guidelines"
DEFAULT_COLLECTION = "clinical_guidelines"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
DEFAULT_QUERY = "melanoma diagnosis dermoscopy"


class GuidelineRAG(BaseTool):
    """Clinical guideline retrieval via ChromaDB + sentence-transformers.

    Args:
        chroma_persist_dir: Path to ChromaDB persistence directory.
        collection_name: ChromaDB collection name.
        embedding_model: Sentence-transformer model for text embedding.
        top_k: Number of guideline chunks to retrieve.
    """

    @property
    def name(self) -> str:
        return "guideline_rag"

    @property
    def description(self) -> str:
        return (
            "Guideline RAG — retrieves relevant clinical guideline "
            "chunks from DermNet NZ and Mayo Clinic via ChromaDB."
        )

    def __init__(
        self,
        chroma_persist_dir: str = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._chroma_dir = chroma_persist_dir
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model
        self._top_k = top_k

        self._embedder: Any = None
        self._collection: Any = None
        self._loaded = False

    def _load_model(self) -> None:
        if self._loaded:
            return

        import chromadb
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s", self._embedding_model_name)
        self._embedder = SentenceTransformer(self._embedding_model_name)

        logger.info("Connecting to ChromaDB at %s", self._chroma_dir)
        client = chromadb.PersistentClient(path=self._chroma_dir)
        self._collection = client.get_or_create_collection(self._collection_name)

        self._loaded = True
        logger.info(
            "GuidelineRAG loaded: collection '%s' (%d chunks).",
            self._collection_name,
            self._collection.count(),
        )

    def unload(self) -> None:
        if self._embedder is not None:
            del self._embedder
            self._embedder = None
            self._collection = None
            self._loaded = False
            logger.info("GuidelineRAG unloaded.")

    def validate_input(self, image_path: str | None = None, query: str = "") -> bool:
        # GuidelineRAG is text-only — always valid if we have a query
        return True

    def _run_inference(self, query: str) -> dict[str, Any]:
        effective_query = query or DEFAULT_QUERY
        embedding = self._embedder.encode(effective_query).tolist()

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=self._top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for i, doc in enumerate(docs):
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else 1.0
            relevance = round(max(0.0, 1.0 - dist), 4)
            chunks.append({
                "source": meta.get("source", "Unknown"),
                "text": doc,
                "relevance_score": relevance,
            })

        # Sort by relevance descending
        chunks.sort(key=lambda c: c["relevance_score"], reverse=True)

        sources = list({c["source"] for c in chunks})

        return {
            "chunks": chunks,
            "retrieval_method": "ChromaDB + sentence-transformers",
            "_sources": sources,
        }

    def run(self, image_path: str | None = None, query: str = "") -> ToolOutput:
        t0 = time.perf_counter()

        if not self.validate_input(image_path, query):
            return ToolOutput(
                tool_name=self.name,
                result={"error": "Invalid input"},
                confidence=0.0,
                raw_text="GuidelineRAG: invalid input.",
                metadata={"status": "error"},
            )

        try:
            self._load_model()
            result = self._run_inference(query)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            chunks = result["chunks"]
            sources = result.pop("_sources", [])

            if chunks:
                confidence = chunks[0]["relevance_score"]
                top_texts = [c["source"] for c in chunks[:3]]
                raw_text = f"Guidelines from: {', '.join(top_texts)}."
            else:
                confidence = 0.0
                raw_text = "No relevant guidelines found."

            return ToolOutput(
                tool_name=self.name,
                result=result,
                confidence=confidence,
                raw_text=raw_text,
                metadata={
                    "sources": sources,
                    "chunks_retrieved": len(chunks),
                    "latency_ms": round(elapsed_ms, 1),
                },
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error("GuidelineRAG failed: %s", exc, exc_info=True)
            return ToolOutput(
                tool_name=self.name,
                result={"error": str(exc)},
                confidence=0.0,
                raw_text=f"GuidelineRAG failed: {exc}",
                metadata={"status": "error", "latency_ms": round(elapsed_ms, 1)},
            )
