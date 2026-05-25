"""Tests for ``scripts/convert_to_jsonl.py``.

Pure-Python tests — no network, no large dataset files. Builds tiny synthetic
CSVs in tmp_path and verifies schema, lesion-aware splitting, and filter
behaviour.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import convert_to_jsonl as cj  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Schema mappers
# ─────────────────────────────────────────────────────────────────────────────


class TestHam10000RowMapper:
    def test_produces_expected_keys(self, tmp_path: Path):
        row = {
            "lesion_id": "HAM_0001",
            "image_id": "ISIC_0024306",
            "dx": "MEL",
            "dx_type": "histo",
            "age": "55",
            "sex": "female",
            "localization": "back",
        }
        rec = cj.ham10000_row_to_record(row, images_dir=tmp_path / "images")
        assert rec["case_id"] == "ISIC_0024306"
        assert rec["image_path"].endswith("images/ISIC_0024306.jpg")
        assert rec["ground_truth_label"] == "mel"   # lower-cased
        assert rec["patient_context"]["age"] == "55"
        assert rec["patient_context"]["sex"] == "female"
        assert rec["patient_context"]["localization"] == "back"
        assert "What is the diagnosis" in rec["query"]


class TestFitzpatrick17kRowMapper:
    def test_prefers_md5hash_and_fitzpatrick_scale(self, tmp_path: Path):
        row = {
            "md5hash": "abcd1234",
            "fitzpatrick_scale": "3",
            "fitzpatrick_centaur": "4",   # should be ignored — scale wins
            "label": "Psoriasis",
            "three_partition_label": "inflammatory",
            "url": "https://example.com/x.jpg",
        }
        rec = cj.fitzpatrick17k_row_to_record(row, images_dir=tmp_path / "images")
        assert rec["case_id"] == "abcd1234"
        assert rec["fitzpatrick_type"] == "III"   # numeric → Roman
        assert rec["ground_truth_label"] == "psoriasis"
        assert rec["patient_context"]["fitzpatrick_type"] == "III"

    def test_falls_back_through_skin_type_columns(self, tmp_path: Path):
        row = {
            "md5hash": "x",
            "fitzpatrick_centaur": "5",   # only centaur provided
            "label": "vitiligo",
        }
        rec = cj.fitzpatrick17k_row_to_record(row, images_dir=tmp_path)
        assert rec["fitzpatrick_type"] == "V"

    def test_unknown_skin_type_kept_as_is(self, tmp_path: Path):
        row = {"md5hash": "x", "fitzpatrick_scale": "-1", "label": "x"}
        rec = cj.fitzpatrick17k_row_to_record(row, images_dir=tmp_path)
        assert rec["fitzpatrick_type"] == "-1"


# ─────────────────────────────────────────────────────────────────────────────
# SplitFractions
# ─────────────────────────────────────────────────────────────────────────────


class TestSplitFractions:
    def test_parses_percentages(self):
        s = cj.SplitFractions.parse("70/15/15")
        assert s.train == pytest.approx(0.70)
        assert s.val == pytest.approx(0.15)
        assert s.test == pytest.approx(0.15)

    def test_parses_unit_fractions(self):
        s = cj.SplitFractions.parse("0.7/0.15/0.15")
        assert s.train == pytest.approx(0.70)

    def test_test_only_split(self):
        s = cj.SplitFractions.parse("0/0/100")
        assert s.train == 0.0 and s.val == 0.0 and s.test == 1.0

    def test_rejects_wrong_count(self):
        with pytest.raises(ValueError):
            cj.SplitFractions.parse("70/30")

    def test_rejects_non_numeric(self):
        with pytest.raises(ValueError):
            cj.SplitFractions.parse("a/b/c")

    def test_rejects_zero_total(self):
        with pytest.raises(ValueError):
            cj.SplitFractions.parse("0/0/0")


# ─────────────────────────────────────────────────────────────────────────────
# group_split — the critical lesion-aware split
# ─────────────────────────────────────────────────────────────────────────────


class TestGroupSplit:
    def test_groups_are_not_split_across_buckets(self):
        # Each lesion has 3 images; 30 lesions = 90 images.
        groups: dict[str, list[int]] = {}
        for lesion_idx in range(30):
            base = lesion_idx * 3
            groups[f"L{lesion_idx}"] = [base, base + 1, base + 2]

        fr = cj.SplitFractions.parse("70/15/15")
        train_i, val_i, test_i = cj.group_split(groups, fr, seed=42)

        # Every lesion's 3 indices must be in exactly one split.
        for lesion, idxs in groups.items():
            in_train = [i in train_i for i in idxs]
            in_val = [i in val_i for i in idxs]
            in_test = [i in test_i for i in idxs]
            count_buckets = (any(in_train), any(in_val), any(in_test))
            assert sum(count_buckets) == 1, f"Lesion {lesion} appeared in multiple buckets"
            # And every index within the lesion lives in the same bucket.
            if any(in_train):
                assert all(in_train), f"Lesion {lesion} split across train and other"
            if any(in_val):
                assert all(in_val)
            if any(in_test):
                assert all(in_test)

    def test_split_sizes_are_close_to_fractions(self):
        groups = {f"L{i}": [i] for i in range(1000)}
        fr = cj.SplitFractions.parse("70/15/15")
        train_i, val_i, test_i = cj.group_split(groups, fr, seed=0)
        # Single-row groups + 1000 of them → exact split achievable to within ±5.
        assert abs(len(train_i) - 700) <= 10
        assert abs(len(val_i) - 150) <= 10
        assert abs(len(test_i) - 150) <= 10

    def test_seed_is_deterministic(self):
        groups = {f"L{i}": [i] for i in range(100)}
        fr = cj.SplitFractions.parse("70/15/15")
        a = cj.group_split(groups, fr, seed=7)
        b = cj.group_split(groups, fr, seed=7)
        assert a == b


# ─────────────────────────────────────────────────────────────────────────────
# row_split
# ─────────────────────────────────────────────────────────────────────────────


class TestRowSplit:
    def test_split_sizes(self):
        train, val, test = cj.row_split(100, cj.SplitFractions.parse("70/15/15"), seed=0)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        assert (train | val | test) == set(range(100))
        assert (train & val) == (train & test) == (val & test) == set()


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end with synthetic CSVs
# ─────────────────────────────────────────────────────────────────────────────


def _write_ham10000_csv(path: Path, n_lesions: int = 20, imgs_per_lesion: int = 2) -> None:
    rows = ["lesion_id,image_id,dx,dx_type,age,sex,localization,dataset"]
    classes = ["mel", "nv", "bkl", "bcc"]
    for i in range(n_lesions):
        cls = classes[i % len(classes)]
        for j in range(imgs_per_lesion):
            rows.append(
                f"HAM_{i:04d},ISIC_{i*10 + j:07d},{cls},histo,50,male,back,vidir_modern"
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_fitz_csv(path: Path, n: int = 30) -> None:
    rows = ["md5hash,fitzpatrick_scale,fitzpatrick_centaur,label,three_partition_label,qc,url"]
    classes = ["psoriasis", "melanoma", "vitiligo"]
    for i in range(n):
        cls = classes[i % len(classes)]
        fitz = (i % 6) + 1   # cycle through 1..6
        qc = "3 Wrongly labelled" if i % 10 == 0 else "1 Diagnostic"
        rows.append(
            f"hash{i:04x},{fitz},{fitz},{cls},inflammatory,{qc},https://x.com/{i}.jpg"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class TestConvertHam10000:
    def test_writes_three_splits_with_correct_schema(self, tmp_path: Path):
        _write_ham10000_csv(tmp_path / "HAM10000_metadata.csv", n_lesions=30, imgs_per_lesion=2)
        stats = cj.convert_ham10000(
            input_dir=tmp_path,
            output_dir=tmp_path,
            fractions=cj.SplitFractions.parse("70/15/15"),
            seed=42,
            require_image=False,
            max_cases=None,
        )
        assert stats.total_input_rows == 60
        assert stats.total_written() == 60
        for split in ("train", "val", "test"):
            f = tmp_path / f"{split}.jsonl"
            assert f.exists(), f"{split}.jsonl was not produced"
            for line in f.read_text().splitlines():
                rec = json.loads(line)
                assert rec["case_id"].startswith("ISIC_")
                assert rec["ground_truth_label"] in {"mel", "nv", "bkl", "bcc"}
                assert "patient_context" in rec
                assert rec["query"].startswith("What is the diagnosis")

    def test_lesion_aware_split_prevents_leakage(self, tmp_path: Path):
        _write_ham10000_csv(tmp_path / "HAM10000_metadata.csv", n_lesions=40, imgs_per_lesion=3)
        cj.convert_ham10000(
            input_dir=tmp_path,
            output_dir=tmp_path,
            fractions=cj.SplitFractions.parse("70/15/15"),
            seed=42,
            require_image=False,
            max_cases=None,
        )
        # Re-derive lesion_id from case_id via the metadata CSV.
        import csv
        lesion_by_image: dict[str, str] = {}
        with (tmp_path / "HAM10000_metadata.csv").open() as fh:
            for row in csv.DictReader(fh):
                lesion_by_image[row["image_id"]] = row["lesion_id"]

        lesions_per_split: dict[str, set[str]] = {}
        for split in ("train", "val", "test"):
            lesions = set()
            for line in (tmp_path / f"{split}.jsonl").read_text().splitlines():
                rec = json.loads(line)
                lesions.add(lesion_by_image[rec["case_id"]])
            lesions_per_split[split] = lesions

        assert lesions_per_split["train"] & lesions_per_split["val"] == set()
        assert lesions_per_split["train"] & lesions_per_split["test"] == set()
        assert lesions_per_split["val"] & lesions_per_split["test"] == set()

    def test_raises_when_csv_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="HAM10000_metadata.csv"):
            cj.convert_ham10000(
                input_dir=tmp_path,
                output_dir=tmp_path,
                fractions=cj.SplitFractions.parse("100/0/0"),
                seed=0,
                require_image=False,
                max_cases=None,
            )

    def test_require_image_drops_missing_files(self, tmp_path: Path):
        _write_ham10000_csv(tmp_path / "HAM10000_metadata.csv", n_lesions=5, imgs_per_lesion=2)
        # Create only some of the image files.
        (tmp_path / "images").mkdir()
        for i in range(3):   # cover only lesions 0,1,2 → 6 images present out of 10
            for j in range(2):
                (tmp_path / "images" / f"ISIC_{i*10 + j:07d}.jpg").write_bytes(b"x")

        stats = cj.convert_ham10000(
            input_dir=tmp_path,
            output_dir=tmp_path,
            fractions=cj.SplitFractions.parse("100/0/0"),
            seed=0,
            require_image=True,
            max_cases=None,
        )
        assert stats.filtered_no_image == 4   # 10 total - 6 present


class TestConvertFitzpatrick17k:
    def test_writes_records_with_fitzpatrick_type(self, tmp_path: Path):
        _write_fitz_csv(tmp_path / "fitzpatrick17k.csv", n=30)
        stats = cj.convert_fitzpatrick17k(
            input_dir=tmp_path,
            output_dir=tmp_path,
            fractions=cj.SplitFractions.parse("70/15/15"),
            seed=0,
            require_image=False,
            max_cases=None,
            filter_qc=False,
        )
        assert stats.total_input_rows == 30
        assert stats.total_written() == 30
        # Inspect one record.
        rec = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
        assert rec["fitzpatrick_type"] in {"I", "II", "III", "IV", "V", "VI"}
        assert rec["patient_context"]["fitzpatrick_type"] == rec["fitzpatrick_type"]

    def test_filter_qc_drops_wrongly_labelled_rows(self, tmp_path: Path):
        _write_fitz_csv(tmp_path / "fitzpatrick17k.csv", n=30)
        stats = cj.convert_fitzpatrick17k(
            input_dir=tmp_path,
            output_dir=tmp_path,
            fractions=cj.SplitFractions.parse("100/0/0"),
            seed=0,
            require_image=False,
            max_cases=None,
            filter_qc=True,
        )
        # _write_fitz_csv flags every 10th row (3 rows) as QC=3.
        assert stats.filtered_qc == 3
        assert stats.total_written() == 27


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke
# ─────────────────────────────────────────────────────────────────────────────


class TestCli:
    def test_argparser_requires_dataset(self):
        with pytest.raises(SystemExit):
            cj._build_argparser().parse_args([])

    def test_argparser_accepts_known_dataset(self):
        args = cj._build_argparser().parse_args(["--dataset", "ham10000"])
        assert args.dataset == "ham10000"

    def test_handlers_registry(self):
        assert "ham10000" in cj.HANDLERS
        assert "fitzpatrick17k" in cj.HANDLERS
