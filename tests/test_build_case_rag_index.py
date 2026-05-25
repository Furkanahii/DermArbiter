"""Tests for ``scripts/build_case_rag_index.py``.

Designed to run anywhere — no GPU, no DermLIP weights, no internet. We inject
a small ``FakeEncoder`` that produces deterministic unit vectors so the full
pipeline (manifest loading → filtering → batched encoding → ChromaDB upsert)
can be exercised on a laptop in seconds.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import build_case_rag_index as ci  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Fake encoder
# ─────────────────────────────────────────────────────────────────────────────


class FakeEncoder:
    """Hash each image path into a deterministic 512-D unit vector."""

    embedding_dim = 8   # tiny dim — ChromaDB is fine with any positive int

    def encode_batch(self, image_paths):
        vectors = []
        for p in image_paths:
            h = hashlib.sha256(str(p).encode()).digest()
            # Take 8 bytes, map to floats in [-1, 1], then L2 normalise.
            raw = [(b / 127.5) - 1.0 for b in h[: self.embedding_dim]]
            norm = sum(x * x for x in raw) ** 0.5 or 1.0
            vectors.append([x / norm for x in raw])
        return vectors


# ─────────────────────────────────────────────────────────────────────────────
# CaseEntry
# ─────────────────────────────────────────────────────────────────────────────


class TestCaseEntry:
    def test_to_metadata_returns_scalar_strings(self, tmp_path: Path):
        e = ci.CaseEntry(
            case_id="X",
            image_path=tmp_path / "x.jpg",
            diagnosis="mel",
            location="back",
            age="55",
            sex="male",
            source="HAM10000",
        )
        meta = e.to_metadata()
        assert isinstance(meta, dict)
        assert meta["diagnosis"] == "mel"
        assert meta["sex"] == "male"
        # ChromaDB rejects nested objects; verify all values are scalars.
        for v in meta.values():
            assert isinstance(v, (str, int, float, bool))


# ─────────────────────────────────────────────────────────────────────────────
# load_ham10000_manifest
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadHam10000Manifest:
    def _write_csv(self, path: Path, rows: int = 4) -> None:
        lines = ["lesion_id,image_id,dx,dx_type,age,sex,localization,dataset"]
        classes = ["mel", "nv", "bkl", "bcc"]
        for i in range(rows):
            cls = classes[i % len(classes)]
            lines.append(f"HAM_{i:04d},ISIC_{i:07d},{cls},histo,55,male,back,vidir_modern")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_loads_entries(self, tmp_path: Path):
        self._write_csv(tmp_path / "HAM10000_metadata.csv", rows=10)
        entries = ci.load_ham10000_manifest(tmp_path)
        assert len(entries) == 10
        e = entries[0]
        assert e.case_id == "ISIC_0000000"
        assert e.image_path == tmp_path / "images" / "ISIC_0000000.jpg"
        assert e.diagnosis in {"mel", "nv", "bkl", "bcc"}
        assert e.source == "HAM10000"

    def test_lowercases_dx(self, tmp_path: Path):
        (tmp_path / "HAM10000_metadata.csv").write_text(
            "lesion_id,image_id,dx,dx_type,age,sex,localization,dataset\n"
            "HAM_0001,ISIC_001,MEL,histo,50,female,back,vidir\n",
            encoding="utf-8",
        )
        entries = ci.load_ham10000_manifest(tmp_path)
        assert entries[0].diagnosis == "mel"

    def test_raises_when_csv_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="HAM10000_metadata.csv"):
            ci.load_ham10000_manifest(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# load_local_manifest
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadLocalManifest:
    def test_loads_jsonl(self, tmp_path: Path):
        p = tmp_path / "m.jsonl"
        p.write_text(
            json.dumps({"case_id": "A1", "image_path": "imgs/a.jpg", "diagnosis": "MEL"}) + "\n"
            + json.dumps({"case_id": "B2", "image_path": "imgs/b.jpg", "diagnosis": "NV", "age": "30"}) + "\n",
            encoding="utf-8",
        )
        entries = ci.load_local_manifest(p)
        assert len(entries) == 2
        # Relative image_path is resolved against the manifest's directory.
        assert entries[0].image_path == (tmp_path / "imgs" / "a.jpg").resolve()
        assert entries[0].diagnosis == "mel"   # lower-cased
        assert entries[1].age == "30"

    def test_loads_csv(self, tmp_path: Path):
        p = tmp_path / "m.csv"
        p.write_text(
            "case_id,image_path,diagnosis,location,age,sex,source\n"
            "Z,images/z.jpg,bcc,nose,70,male,custom\n",
            encoding="utf-8",
        )
        entries = ci.load_local_manifest(p)
        assert len(entries) == 1
        assert entries[0].source == "custom"

    def test_skips_incomplete_rows(self, tmp_path: Path):
        p = tmp_path / "m.jsonl"
        p.write_text(
            json.dumps({"case_id": "A1", "image_path": "imgs/a.jpg", "diagnosis": "MEL"}) + "\n"
            + json.dumps({"case_id": "B2", "image_path": "imgs/b.jpg"}) + "\n"   # no diagnosis
            + json.dumps({"diagnosis": "NV", "image_path": "imgs/c.jpg"}) + "\n",   # no case_id
            encoding="utf-8",
        )
        entries = ci.load_local_manifest(p)
        assert len(entries) == 1

    def test_raises_when_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            ci.load_local_manifest(tmp_path / "nope.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# load_derm1m_manifest — still a stub
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadDerm1m:
    """Derm1M loader uses HF snapshot_download — we monkeypatch it so the
    tests stay network-free and never touch the user's HF token.
    """

    def _write_manifest(self, root: Path, rows: list[dict]) -> Path:
        import csv as _csv

        path = root / "Derm1M_v2_pretrain.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_raises_without_hf_token(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        with pytest.raises(SystemExit, match="HF_TOKEN"):
            ci.load_derm1m_manifest(data_dir=tmp_path)

    def test_loads_manifest_and_emits_entries(self, tmp_path, monkeypatch):
        # Pretend the snapshot is already on disk — point the fake download
        # at the test fixture directory and skip the real HF call.
        rows = [
            {"filename": "a.jpg", "disease_label": "Melanoma",
             "truncated_caption": "...", "source": "pubmed"},
            {"filename": "b.jpg", "disease_label": "Nevus",
             "truncated_caption": "...", "source": "atlas"},
            {"filename": "c.jpg", "disease_label": "Melanoma",
             "truncated_caption": "...", "source": "pubmed"},
            {"filename": "", "disease_label": "Acne",
             "truncated_caption": "x", "source": "x"},   # skipped: no filename
        ]
        self._write_manifest(tmp_path, rows)
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            (tmp_path / name).write_bytes(b"\xff\xd8\xff\xe0")  # tiny JPEG header

        monkeypatch.setenv("HF_TOKEN", "test-token-not-real")
        monkeypatch.setattr(ci, "_download_derm1m",
                            lambda data_dir, hf_token=None, allow_patterns=None: data_dir)

        entries = ci.load_derm1m_manifest(data_dir=tmp_path, max_cases=None)
        assert len(entries) == 3
        assert {e.case_id for e in entries} == {"a", "b", "c"}
        # Diagnosis is lower-cased to match the rest of the pipeline.
        assert {e.diagnosis for e in entries} == {"melanoma", "nevus"}
        # Source is namespaced so we can tell Derm1M apart from HAM10000.
        assert all(e.source.startswith("Derm1M:") for e in entries)

    def test_max_cases_uses_stratified_sample(self, tmp_path, monkeypatch):
        rows = []
        # 30 melanomas, 10 nevi → with max_cases=10 expect ≈7-8 mel + 2-3 nv.
        for i in range(30):
            rows.append({"filename": f"m{i}.jpg", "disease_label": "Melanoma",
                         "truncated_caption": "", "source": "x"})
        for i in range(10):
            rows.append({"filename": f"n{i}.jpg", "disease_label": "Nevus",
                         "truncated_caption": "", "source": "x"})
        self._write_manifest(tmp_path, rows)
        for r in rows:
            (tmp_path / r["filename"]).write_bytes(b"\xff\xd8\xff\xe0")

        monkeypatch.setenv("HF_TOKEN", "test-token-not-real")
        monkeypatch.setattr(ci, "_download_derm1m",
                            lambda data_dir, hf_token=None, allow_patterns=None: data_dir)

        entries = ci.load_derm1m_manifest(data_dir=tmp_path, max_cases=10, seed=0)
        assert len(entries) == 10
        diag_counts = {}
        for e in entries:
            diag_counts[e.diagnosis] = diag_counts.get(e.diagnosis, 0) + 1
        # Stratification: melanoma class should dominate but nevus must appear.
        assert diag_counts.get("melanoma", 0) >= 6
        assert diag_counts.get("nevus", 0) >= 1


class TestStratifiedSampler:
    def test_proportional_split(self):
        rows = (
            [{"disease_label": "a"}] * 80
            + [{"disease_label": "b"}] * 20
        )
        sub = ci._stratified_sample(rows, 10, seed=0)
        counts = {}
        for r in sub:
            counts[r["disease_label"]] = counts.get(r["disease_label"], 0) + 1
        assert sum(counts.values()) == 10
        # 80:20 → roughly 8:2
        assert counts.get("a", 0) == 8
        assert counts.get("b", 0) == 2

    def test_returns_all_when_n_exceeds_population(self):
        rows = [{"disease_label": "a"}] * 5
        sub = ci._stratified_sample(rows, 50)
        assert len(sub) == 5

    def test_handles_missing_stratify_key(self):
        rows = [{"other": "x"} for _ in range(10)]
        sub = ci._stratified_sample(rows, 4, seed=0)
        assert len(sub) == 4

    def test_deterministic_under_same_seed(self):
        rows = [{"disease_label": ("a" if i % 2 == 0 else "b"), "i": i} for i in range(20)]
        s1 = ci._stratified_sample(rows, 6, seed=42)
        s2 = ci._stratified_sample(rows, 6, seed=42)
        assert [r["i"] for r in s1] == [r["i"] for r in s2]


# ─────────────────────────────────────────────────────────────────────────────
# _filter_entries
# ─────────────────────────────────────────────────────────────────────────────


class TestFilterEntries:
    def test_drops_unsupported_extension(self, tmp_path: Path):
        good = tmp_path / "x.jpg"
        good.write_bytes(b"x")
        bad = tmp_path / "x.txt"
        bad.write_bytes(b"x")
        entries = [
            ci.CaseEntry(case_id="A", image_path=good, diagnosis="mel"),
            ci.CaseEntry(case_id="B", image_path=bad, diagnosis="nv"),
        ]
        kept, dropped = ci._filter_entries(entries, require_image=True)
        assert len(kept) == 1 and kept[0].case_id == "A"
        assert dropped == 1

    def test_drops_missing_files_when_require_image(self, tmp_path: Path):
        present = tmp_path / "ok.jpg"
        present.write_bytes(b"x")
        entries = [
            ci.CaseEntry(case_id="A", image_path=present, diagnosis="mel"),
            ci.CaseEntry(case_id="B", image_path=tmp_path / "missing.jpg", diagnosis="nv"),
        ]
        kept, dropped = ci._filter_entries(entries, require_image=True)
        assert len(kept) == 1
        assert dropped == 1

    def test_keeps_missing_files_when_not_required(self, tmp_path: Path):
        entries = [
            ci.CaseEntry(case_id="A", image_path=tmp_path / "missing.jpg", diagnosis="mel"),
        ]
        kept, dropped = ci._filter_entries(entries, require_image=False)
        assert len(kept) == 1
        assert dropped == 0


# ─────────────────────────────────────────────────────────────────────────────
# Full ingest pipeline — uses FakeEncoder + real ChromaDB
# ─────────────────────────────────────────────────────────────────────────────


_HAS_CHROMA = importlib.util.find_spec("chromadb") is not None


@pytest.mark.skipif(not _HAS_CHROMA, reason="chromadb not installed")
class TestIngestPipeline:
    def _make_entries(self, tmp_path: Path, n: int = 5) -> list[ci.CaseEntry]:
        imgs = tmp_path / "images"
        imgs.mkdir(exist_ok=True)
        out = []
        for i in range(n):
            img = imgs / f"case_{i:03d}.jpg"
            img.write_bytes(b"fake-jpg")
            out.append(ci.CaseEntry(
                case_id=f"C{i:03d}",
                image_path=img,
                diagnosis="mel" if i % 2 == 0 else "nv",
                location="back",
                age=str(40 + i),
                sex="male",
                source="test",
            ))
        return out

    def test_end_to_end_embeds_and_upserts(self, tmp_path: Path):
        entries = self._make_entries(tmp_path, n=10)
        stats = ci.ingest(
            entries=entries,
            encoder=FakeEncoder(),
            persist_dir=tmp_path / "chroma",
            collection_name="test_cases",
            batch_size=4,
            require_image=True,
            reset=True,
            dry_run=False,
        )
        assert stats.embedded == 10
        assert stats.upserted == 10
        assert stats.batches == 3   # 4 + 4 + 2

        # Now query the index and verify metadata flows through correctly.
        import chromadb
        coll = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_collection("test_cases")
        assert coll.count() == 10

        # Re-encode the first entry and look it up — should be the nearest neighbour.
        target_vec = FakeEncoder().encode_batch([entries[0].image_path])[0]
        res = coll.query(query_embeddings=[target_vec], n_results=1,
                         include=["metadatas", "distances"])
        assert res["ids"][0][0] == "C000"
        assert res["metadatas"][0][0]["diagnosis"] == "mel"

    def test_idempotent_reingest(self, tmp_path: Path):
        entries = self._make_entries(tmp_path, n=4)
        for _ in range(2):
            ci.ingest(
                entries=entries,
                encoder=FakeEncoder(),
                persist_dir=tmp_path / "chroma",
                collection_name="idem",
                batch_size=2,
                require_image=True,
                reset=False,
                dry_run=False,
            )
        import chromadb
        coll = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_collection("idem")
        assert coll.count() == 4   # not 8

    def test_dry_run_does_not_write_chromadb(self, tmp_path: Path):
        entries = self._make_entries(tmp_path, n=3)
        stats = ci.ingest(
            entries=entries,
            encoder=FakeEncoder(),
            persist_dir=tmp_path / "chroma",
            collection_name="dry",
            batch_size=2,
            require_image=True,
            reset=False,
            dry_run=True,
        )
        assert stats.embedded == 0
        assert stats.upserted == 0
        # ChromaDB directory was never touched.
        assert not (tmp_path / "chroma").exists()


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke
# ─────────────────────────────────────────────────────────────────────────────


class TestCli:
    def test_argparser_requires_source(self):
        with pytest.raises(SystemExit):
            ci._build_argparser().parse_args([])

    def test_local_requires_manifest(self, tmp_path: Path):
        args = ci._build_argparser().parse_args(["--source", "local"])
        with pytest.raises(SystemExit, match="manifest"):
            ci._load_entries(args)

    def test_source_loaders_registry(self):
        for k in ("ham10000", "local", "derm1m"):
            assert k in ci.SOURCE_LOADERS
