"""Tests for ``scripts/download_datasets.py``.

Network-free: every test uses an HTTP mock to fake server responses, so the
whole suite can run offline without ever hitting Harvard Dataverse or GitHub.
"""

from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts/ to sys.path so we can import the module by file.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import download_datasets as dd  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Small utility helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestHumanBytes:
    def test_bytes(self):
        assert dd._human_bytes(500) == "500.0 B"

    def test_kilobytes(self):
        assert dd._human_bytes(1500) == "1.5 KB"

    def test_megabytes(self):
        assert dd._human_bytes(2 * 1024 * 1024) == "2.0 MB"

    def test_gigabytes(self):
        assert dd._human_bytes(3 * 1024 ** 3) == "3.0 GB"


class TestMd5:
    def test_md5_of_known_string(self, tmp_path: Path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello world")
        assert dd._md5(p) == hashlib.md5(b"hello world").hexdigest()  # noqa: S324


class TestVerify:
    def test_missing_file(self, tmp_path: Path):
        assert dd._verify(tmp_path / "nope", None, None) is False

    def test_size_far_below_expected_fails(self, tmp_path: Path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"x" * 100)
        # expected ~1 MB, got 100 B → far below 50% → fail
        assert dd._verify(p, expected_size=1_000_000, expected_md5=None) is False

    def test_size_within_tolerance_passes(self, tmp_path: Path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"x" * 800)
        # expected ~1000, got 800 → above 50% threshold → pass
        assert dd._verify(p, expected_size=1000, expected_md5=None) is True

    def test_md5_mismatch_fails(self, tmp_path: Path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"actual content")
        assert dd._verify(p, None, expected_md5="deadbeef" * 4) is False

    def test_md5_match_passes(self, tmp_path: Path):
        p = tmp_path / "f.bin"
        content = b"actual content"
        p.write_bytes(content)
        good = hashlib.md5(content).hexdigest()  # noqa: S324
        assert dd._verify(p, None, expected_md5=good) is True

    def test_no_constraints_passes(self, tmp_path: Path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"anything")
        assert dd._verify(p, None, None) is True


# ─────────────────────────────────────────────────────────────────────────────
# _extract_zip — extraction safety + idempotency
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractZip:
    def _make_zip(self, path: Path, files: dict[str, bytes]) -> Path:
        with zipfile.ZipFile(path, "w") as zf:
            for name, data in files.items():
                zf.writestr(name, data)
        return path

    def test_extracts_all_members(self, tmp_path: Path):
        zip_path = self._make_zip(
            tmp_path / "a.zip",
            {"images/ISIC_001.jpg": b"jpg1", "images/ISIC_002.jpg": b"jpg2"},
        )
        out = tmp_path / "out"
        count = dd._extract_zip(zip_path, out)
        assert count == 2
        assert (out / "ISIC_001.jpg").read_bytes() == b"jpg1"
        assert (out / "ISIC_002.jpg").read_bytes() == b"jpg2"

    def test_skips_path_traversal_entries(self, tmp_path: Path):
        zip_path = tmp_path / "bad.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../escape.txt", b"evil")
            zf.writestr("ok.txt", b"good")
        out = tmp_path / "out"
        dd._extract_zip(zip_path, out)
        # Only ok.txt should land in out/; ../escape.txt is flattened to escape.txt
        # by Path(name).name and lands inside out/ — which is the safe behaviour.
        assert (out / "ok.txt").read_bytes() == b"good"
        assert not (tmp_path / "escape.txt").exists()

    def test_idempotent_extract(self, tmp_path: Path):
        zip_path = self._make_zip(tmp_path / "a.zip", {"x.txt": b"hello"})
        out = tmp_path / "out"
        dd._extract_zip(zip_path, out)
        # Re-extract — should still report a file and not blow up.
        count = dd._extract_zip(zip_path, out)
        assert count == 1


# ─────────────────────────────────────────────────────────────────────────────
# _download — mocked HTTP with size + MD5 verification
# ─────────────────────────────────────────────────────────────────────────────


def _make_mock_response(body: bytes, status: int = 200, headers: dict | None = None):
    """Build a context-manager mock matching requests.get(stream=True) semantics."""
    mock = MagicMock()
    mock.status_code = status
    mock.headers = headers or {"Content-Length": str(len(body))}
    mock.iter_content = MagicMock(return_value=[body])
    mock.raise_for_status = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    mock.close = MagicMock()
    return mock


class TestDownload:
    def test_skips_existing_verified_file(self, tmp_path: Path):
        dest = tmp_path / "f.bin"
        dest.write_bytes(b"already here" * 100)

        with patch("requests.get") as mock_get:
            result = dd._download("https://example.com/f.bin", dest)
        assert result == dest
        mock_get.assert_not_called()  # Nothing downloaded.

    def test_downloads_when_missing(self, tmp_path: Path):
        dest = tmp_path / "f.bin"
        body = b"fresh content"

        with patch("requests.get", return_value=_make_mock_response(body)):
            result = dd._download("https://example.com/f.bin", dest)

        assert result == dest
        assert dest.read_bytes() == body
        assert not (tmp_path / "f.bin.part").exists()

    def test_force_redownloads_even_if_present(self, tmp_path: Path):
        dest = tmp_path / "f.bin"
        dest.write_bytes(b"old content" * 100)
        new_body = b"new content from server"

        with patch("requests.get", return_value=_make_mock_response(new_body)) as mock_get:
            dd._download("https://example.com/f.bin", dest, force=True)

        mock_get.assert_called_once()
        assert dest.read_bytes() == new_body

    def test_md5_mismatch_raises_and_cleans_partial(self, tmp_path: Path):
        dest = tmp_path / "f.bin"
        body = b"wrong content"
        # Real MD5 of body is not "ff..."
        with patch("requests.get", return_value=_make_mock_response(body)):
            with pytest.raises(RuntimeError, match="verification failed"):
                dd._download(
                    "https://example.com/f.bin",
                    dest,
                    expected_md5="ff" * 16,
                )
        assert not dest.exists()
        assert not (tmp_path / "f.bin.part").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Handler smoke tests (mock _download, never hit the network)
# ─────────────────────────────────────────────────────────────────────────────


class TestHandlers:
    def test_ham10000_metadata_only(self, tmp_path: Path):
        target = tmp_path / "ham10000"

        def fake_download(url, dest, **_kw):
            dest = Path(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("lesion_id,image_id,dx,dx_type,age,sex,localization\n")
            return dest

        with patch.object(dd, "_download", side_effect=fake_download):
            report = dd.download_ham10000(target, metadata_only=True)

        assert report.dataset == "HAM10000"
        assert (target / "HAM10000_metadata.csv").exists()
        assert any("Metadata-only" in n for n in report.notes)

    def test_fitzpatrick17k_metadata_only(self, tmp_path: Path):
        target = tmp_path / "fitz17k"

        def fake_download(url, dest, **_kw):
            dest = Path(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("md5hash,label,fitzpatrick_scale,url\n")
            return dest

        with patch.object(dd, "_download", side_effect=fake_download):
            report = dd.download_fitzpatrick17k(target, with_images=False)

        assert (target / "fitzpatrick17k.csv").exists()
        assert any("Metadata-only" in n for n in report.notes)

    def test_derm7pt_prints_instructions_only(self, tmp_path: Path, capsys):
        target = tmp_path / "derm7pt"
        report = dd.download_derm7pt(target)
        captured = capsys.readouterr()
        assert "register" in captured.out.lower()
        assert any("Manual" in n for n in report.notes)
        # Nothing should have been written under target_dir besides the empty dir.
        assert target.exists() and not any(target.iterdir())

    def test_skincap_prints_instructions_only(self, tmp_path: Path, capsys):
        target = tmp_path / "skincap"
        dd.download_skincap(target)
        captured = capsys.readouterr()
        assert "huggingface" in captured.out.lower()


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────


class TestCli:
    def test_argparser_defaults(self):
        parser = dd._build_argparser()
        args = parser.parse_args(["--dataset", "ham10000", "--metadata-only"])
        assert args.dataset == "ham10000"
        assert args.metadata_only is True
        assert args.force is False

    def test_handlers_registry_contains_all_datasets(self):
        for name in ("ham10000", "fitzpatrick17k", "skincon", "derm7pt", "skincap"):
            assert name in dd.HANDLERS
