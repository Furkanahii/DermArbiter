"""Tests for ``scripts/build_guideline_rag_index.py``.

Pure-Python tests for the chunking and JSONL-loading logic (no embedding,
no network, no ChromaDB) plus one optional end-to-end test that exercises
the full ingest + ``GuidelineRAG.run`` round-trip when sentence-transformers
and chromadb are importable.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Add scripts/ to sys.path so we can import the ingest module by file.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import build_guideline_rag_index as ingest  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# chunk_text — sentence-aware splitting with overlap
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkText:
    def test_empty_text_returns_empty_list(self):
        assert ingest.chunk_text("") == []
        assert ingest.chunk_text("   ") == []

    def test_short_text_returns_single_chunk(self):
        text = "Melanoma is malignant. Diagnosis requires biopsy."
        chunks = ingest.chunk_text(text, chunk_size=1024, chunk_overlap=128)
        assert chunks == [text]

    def test_long_text_splits_into_multiple_chunks(self):
        # Build a long text from many short sentences.
        sentences = [f"Sentence number {i} about dermatology." for i in range(50)]
        text = " ".join(sentences)
        chunks = ingest.chunk_text(text, chunk_size=200, chunk_overlap=40)
        assert len(chunks) > 1
        # Every chunk respects (approximately) the size budget.
        for c in chunks:
            assert len(c) <= 200 + 40, f"Chunk too long: {len(c)} > {200 + 40}"

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            ingest.chunk_text("text", chunk_size=0)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            ingest.chunk_text("text", chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError):
            ingest.chunk_text("text", chunk_size=100, chunk_overlap=-1)

    def test_oversized_single_sentence_is_force_split(self):
        # One sentence longer than chunk_size triggers the hard-window fallback.
        long_sent = "A" * 500 + "."
        chunks = ingest.chunk_text(long_sent, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 100


# ─────────────────────────────────────────────────────────────────────────────
# chunk_document — deterministic IDs from (source, text)
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkDocument:
    def test_returns_chunks_with_ids_and_metadata(self):
        doc = ingest.GuidelineDocument(
            source="DermNet:Test",
            text="Short clinical fact about acne.",
            url="https://example.com/acne",
            title="Acne Overview",
            topic="acne",
        )
        chunks = ingest.chunk_document(doc, chunk_size=1024, chunk_overlap=128)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.chunk_id and len(c.chunk_id) == 16
        assert c.text == doc.text
        assert c.metadata["source"] == "DermNet:Test"
        assert c.metadata["url"] == "https://example.com/acne"
        assert c.metadata["topic"] == "acne"

    def test_ids_are_deterministic(self):
        doc = ingest.GuidelineDocument(source="X", text="One sentence.")
        a = ingest.chunk_document(doc, 1024, 128)[0].chunk_id
        b = ingest.chunk_document(doc, 1024, 128)[0].chunk_id
        assert a == b

    def test_different_source_yields_different_id_for_same_text(self):
        d1 = ingest.GuidelineDocument(source="A", text="Same body text here.")
        d2 = ingest.GuidelineDocument(source="B", text="Same body text here.")
        id1 = ingest.chunk_document(d1, 1024, 128)[0].chunk_id
        id2 = ingest.chunk_document(d2, 1024, 128)[0].chunk_id
        assert id1 != id2


# ─────────────────────────────────────────────────────────────────────────────
# load_local_jsonl — schema validation and error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadLocalJsonl:
    def test_loads_valid_jsonl(self, tmp_path: Path):
        p = tmp_path / "g.jsonl"
        p.write_text(
            json.dumps({"source": "S1", "text": "First doc.", "topic": "t1"}) + "\n"
            + json.dumps({"source": "S2", "text": "Second doc.", "title": "Doc 2"}) + "\n",
            encoding="utf-8",
        )
        docs = ingest.load_local_jsonl(p)
        assert len(docs) == 2
        assert docs[0].source == "S1"
        assert docs[1].title == "Doc 2"

    def test_skips_lines_missing_required_fields(self, tmp_path: Path):
        p = tmp_path / "g.jsonl"
        p.write_text(
            json.dumps({"source": "OK", "text": "Has both."}) + "\n"
            + json.dumps({"text": "Missing source."}) + "\n"
            + json.dumps({"source": "Missing text."}) + "\n",
            encoding="utf-8",
        )
        docs = ingest.load_local_jsonl(p)
        assert len(docs) == 1
        assert docs[0].source == "OK"

    def test_skips_malformed_json_lines(self, tmp_path: Path):
        p = tmp_path / "g.jsonl"
        p.write_text(
            json.dumps({"source": "OK", "text": "Valid."}) + "\n"
            + "{not json}\n",
            encoding="utf-8",
        )
        docs = ingest.load_local_jsonl(p)
        assert len(docs) == 1

    def test_raises_when_file_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            ingest.load_local_jsonl(tmp_path / "nonexistent.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# read_urls_file — flat-list and {urls: [...]} forms, JSON + YAML
# ─────────────────────────────────────────────────────────────────────────────

class TestReadUrlsFile:
    def test_reads_flat_json_list(self, tmp_path: Path):
        p = tmp_path / "urls.json"
        p.write_text(json.dumps(["https://a.com", "https://b.com"]), encoding="utf-8")
        urls = ingest.read_urls_file(p)
        assert urls == ["https://a.com", "https://b.com"]

    def test_reads_dict_form_json(self, tmp_path: Path):
        p = tmp_path / "urls.json"
        p.write_text(json.dumps({"urls": ["https://x.com"]}), encoding="utf-8")
        assert ingest.read_urls_file(p) == ["https://x.com"]

    def test_reads_yaml(self, tmp_path: Path):
        pytest.importorskip("yaml")
        p = tmp_path / "urls.yaml"
        p.write_text("- https://a.com\n- https://b.com\n", encoding="utf-8")
        assert ingest.read_urls_file(p) == ["https://a.com", "https://b.com"]

    def test_rejects_non_list_payload(self, tmp_path: Path):
        p = tmp_path / "urls.json"
        p.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
        with pytest.raises(ValueError):
            ingest.read_urls_file(p)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end test (skipped if optional deps missing)
# ─────────────────────────────────────────────────────────────────────────────

_HAS_EMBED_STACK = (
    importlib.util.find_spec("sentence_transformers") is not None
    and importlib.util.find_spec("chromadb") is not None
)


@pytest.mark.skipif(not _HAS_EMBED_STACK, reason="sentence-transformers or chromadb not installed")
class TestEndToEnd:
    def test_full_ingest_then_query(self, tmp_path: Path):
        """Build a tiny index in a tmp dir and query it via the real GuidelineRAG."""
        # 1. Tiny seed.
        seed = tmp_path / "seed.jsonl"
        seed.write_text(
            json.dumps({
                "source": "Test:Melanoma",
                "text": (
                    "Melanoma is a malignant tumour of melanocytes. "
                    "Diagnosis uses the ABCDE rule and dermoscopy. "
                    "Treatment is wide local excision and sentinel node biopsy."
                ),
                "topic": "melanoma",
            }) + "\n"
            + json.dumps({
                "source": "Test:Psoriasis",
                "text": (
                    "Psoriasis is an immune-mediated disease driven by IL-17 and TNF. "
                    "Biologic therapies have transformed treatment of severe disease."
                ),
                "topic": "psoriasis",
            }) + "\n",
            encoding="utf-8",
        )

        persist = tmp_path / "chroma"

        # 2. Run the ingest pipeline programmatically.
        argv = [
            "--source", "local",
            "--input", str(seed),
            "--persist-dir", str(persist),
            "--collection", "test_guidelines",
            "--reset",
        ]
        exit_code = ingest.main(argv)
        assert exit_code == 0

        # 3. Query the resulting index via the production tool.
        from dermarbiter.tools.guideline_rag import GuidelineRAG
        rag = GuidelineRAG(
            chroma_persist_dir=str(persist),
            collection_name="test_guidelines",
            top_k=2,
        )
        out = rag.run(query="melanoma ABCDE dermoscopy")

        assert out.confidence > 0
        sources = [c["source"] for c in out.result["chunks"]]
        assert "Test:Melanoma" in sources, f"Expected Melanoma in top results, got {sources}"
        # Top hit should be the melanoma chunk, not psoriasis.
        assert out.result["chunks"][0]["source"] == "Test:Melanoma"
