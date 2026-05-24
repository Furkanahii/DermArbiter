"""Convert downloaded benchmark CSVs into the JSONL format consumed by
``dermarbiter.evaluation.benchmark_runner.DatasetLoader.load_jsonl``.

Why JSONL?
    The CSV loaders in ``benchmark_runner`` exist as a convenience, but the
    canonical input format for ``BenchmarkRunner.run_benchmark()`` is JSONL —
    one self-contained case per line, no schema-coupling to a particular
    dataset's column names. JSONL also lets you:
      * filter / subsample without re-reading the source
      * mix cases across datasets
      * version the prepared evaluation slice in git

Output schema (every line):
    {
      "case_id":              str,
      "image_path":           str,
      "query":                str,
      "ground_truth_label":   str,
      "patient_context":      dict,
      "fitzpatrick_type":     str (optional; Fitzpatrick17k only)
    }

Splits:
    HAM10000 splits are **lesion-aware** — multiple ISIC images can share a
    lesion_id, and assigning them across train/val/test would leak
    information. Splitting at the lesion level prevents that.

    Fitzpatrick17k splits are row-level because each row corresponds to a
    distinct hash.

Example
-------
HAM10000 → train/val/test JSONL with default 70/15/15 split::

    python scripts/convert_to_jsonl.py --dataset ham10000

Fitzpatrick17k, test-only split, keep only QC-passing rows::

    python scripts/convert_to_jsonl.py --dataset fitzpatrick17k \\
        --splits 0/0/100 --filter-qc
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("convert_to_jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# Schema mappers — one per dataset, each turns a CSV row into a JSONL record
# ─────────────────────────────────────────────────────────────────────────────


_FITZ_NUM_TO_ROMAN = {
    "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI",
}


def ham10000_row_to_record(row: dict[str, str], images_dir: Path) -> dict[str, Any]:
    """Translate one HAM10000_metadata.csv row → DermArbiter JSONL record."""
    image_id = (row.get("image_id") or "").strip()
    return {
        "case_id": image_id,
        "image_path": str(images_dir / f"{image_id}.jpg"),
        "query": "What is the diagnosis for this skin lesion?",
        "ground_truth_label": (row.get("dx") or "").strip().lower(),
        "patient_context": {
            "age": (row.get("age") or "").strip(),
            "sex": (row.get("sex") or "").strip(),
            "localization": (row.get("localization") or "").strip(),
        },
    }


def fitzpatrick17k_row_to_record(row: dict[str, str], images_dir: Path) -> dict[str, Any]:
    """Translate one fitzpatrick17k.csv row → DermArbiter JSONL record.

    Mirrors the column priority used by ``DatasetLoader.load_fitzpatrick17k``:
      * ``md5hash`` → ``hasher`` fallback for the unique row identifier
      * ``fitzpatrick_scale`` → ``fitzpatrick_centaur`` → raw ``fitzpatrick`` for the skin type
      * ``label`` → ``three_partition_label`` fallback for the diagnosis label
    """
    hasher = (
        (row.get("md5hash") or row.get("hasher") or "").strip()
    )
    fitz_raw = (
        (row.get("fitzpatrick_scale") or row.get("fitzpatrick_centaur") or row.get("fitzpatrick") or "").strip()
    )
    fitzpatrick_type = _FITZ_NUM_TO_ROMAN.get(fitz_raw, fitz_raw)
    label = (
        (row.get("label") or row.get("three_partition_label") or "").strip().lower()
    )

    return {
        "case_id": hasher,
        "image_path": str(images_dir / f"{hasher}.jpg"),
        "query": "What is the diagnosis for this skin condition?",
        "ground_truth_label": label,
        "fitzpatrick_type": fitzpatrick_type,
        "patient_context": {
            "fitzpatrick_type": fitzpatrick_type,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Split utilities
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SplitFractions:
    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if total <= 0:
            raise ValueError(f"Split fractions sum to {total}; expected > 0.")
        # Normalise so {70, 15, 15} and {0.7, 0.15, 0.15} both work.
        self.train /= total
        self.val /= total
        self.test /= total

    @classmethod
    def parse(cls, spec: str) -> "SplitFractions":
        """Parse forms like '70/15/15' or '0.7/0.15/0.15' or '0/0/100'."""
        parts = spec.split("/")
        if len(parts) != 3:
            raise ValueError(f"--splits must be 'train/val/test', got {spec!r}")
        try:
            t, v, te = (float(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"--splits values must be numeric: {spec!r}") from exc
        return cls(train=t, val=v, test=te)


def group_split(
    groups: dict[str, list[int]],
    fractions: SplitFractions,
    seed: int,
) -> tuple[set[int], set[int], set[int]]:
    """Assign whole groups (lesions / patients) to train/val/test.

    The number of rows per group can be unequal, but the unit of assignment is
    the whole group — this prevents data leakage when multiple images share a
    grouping key (e.g. HAM10000's ``lesion_id``).

    Returns three index sets keyed back to the source row positions.
    """
    rng = random.Random(seed)
    keys = sorted(groups.keys())
    rng.shuffle(keys)

    total_rows = sum(len(v) for v in groups.values())
    target_train = int(round(total_rows * fractions.train))
    target_val = int(round(total_rows * fractions.val))

    train_idx, val_idx, test_idx = set(), set(), set()
    cursor = 0
    for key in keys:
        rows = groups[key]
        if cursor + len(rows) <= target_train or fractions.test == 0 and fractions.val == 0:
            train_idx.update(rows)
        elif cursor + len(rows) <= target_train + target_val:
            val_idx.update(rows)
        else:
            test_idx.update(rows)
        cursor += len(rows)
    return train_idx, val_idx, test_idx


def row_split(
    n: int,
    fractions: SplitFractions,
    seed: int,
) -> tuple[set[int], set[int], set[int]]:
    """Simple per-row split for datasets with no grouping constraint."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    n_train = int(round(n * fractions.train))
    n_val = int(round(n * fractions.val))
    return (
        set(order[:n_train]),
        set(order[n_train : n_train + n_val]),
        set(order[n_train + n_val :]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConversionStats:
    total_input_rows: int = 0
    written_train: int = 0
    written_val: int = 0
    written_test: int = 0
    filtered_no_image: int = 0
    filtered_qc: int = 0
    skipped_missing_id: int = 0
    classes: dict[str, int] = None  # populated lazily

    def total_written(self) -> int:
        return self.written_train + self.written_val + self.written_test


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def convert_ham10000(
    input_dir: Path,
    output_dir: Path,
    fractions: SplitFractions,
    seed: int,
    require_image: bool,
    max_cases: Optional[int],
) -> ConversionStats:
    csv_path = input_dir / "HAM10000_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run: "
            f"python scripts/download_datasets.py --dataset ham10000 --metadata-only"
        )

    images_dir = input_dir / "images"
    stats = ConversionStats(classes=defaultdict(int))

    with csv_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    stats.total_input_rows = len(rows)

    records: list[dict[str, Any]] = []
    lesion_groups: dict[str, list[int]] = defaultdict(list)

    for row in rows:
        lesion_id = (row.get("lesion_id") or "").strip()
        image_id = (row.get("image_id") or "").strip()
        if not lesion_id or not image_id:
            stats.skipped_missing_id += 1
            continue
        rec = ham10000_row_to_record(row, images_dir)
        if require_image and not Path(rec["image_path"]).exists():
            stats.filtered_no_image += 1
            continue
        idx = len(records)
        records.append(rec)
        lesion_groups[lesion_id].append(idx)
        stats.classes[rec["ground_truth_label"]] += 1
        if max_cases and len(records) >= max_cases:
            break

    train_i, val_i, test_i = group_split(lesion_groups, fractions, seed)
    _emit_splits(records, train_i, val_i, test_i, output_dir, stats)
    return stats


def convert_fitzpatrick17k(
    input_dir: Path,
    output_dir: Path,
    fractions: SplitFractions,
    seed: int,
    require_image: bool,
    max_cases: Optional[int],
    filter_qc: bool,
) -> ConversionStats:
    csv_path = input_dir / "fitzpatrick17k.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run: "
            f"python scripts/download_datasets.py --dataset fitzpatrick17k"
        )

    images_dir = input_dir / "images"
    stats = ConversionStats(classes=defaultdict(int))

    with csv_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    stats.total_input_rows = len(rows)

    records: list[dict[str, Any]] = []
    for row in rows:
        qc_raw = (row.get("qc") or "").strip()
        # The published qc values are e.g. "1 Diagnostic", "3 Wrongly labelled".
        if filter_qc and qc_raw.startswith("3"):
            stats.filtered_qc += 1
            continue
        rec = fitzpatrick17k_row_to_record(row, images_dir)
        if not rec["case_id"]:
            stats.skipped_missing_id += 1
            continue
        if require_image and not Path(rec["image_path"]).exists():
            stats.filtered_no_image += 1
            continue
        records.append(rec)
        stats.classes[rec["ground_truth_label"]] += 1
        if max_cases and len(records) >= max_cases:
            break

    train_i, val_i, test_i = row_split(len(records), fractions, seed)
    _emit_splits(records, train_i, val_i, test_i, output_dir, stats)
    return stats


def _emit_splits(
    records: list[dict[str, Any]],
    train_i: set[int],
    val_i: set[int],
    test_i: set[int],
    output_dir: Path,
    stats: ConversionStats,
) -> None:
    train = [r for i, r in enumerate(records) if i in train_i]
    val = [r for i, r in enumerate(records) if i in val_i]
    test = [r for i, r in enumerate(records) if i in test_i]

    if train:
        stats.written_train = _write_jsonl(train, output_dir / "train.jsonl")
    if val:
        stats.written_val = _write_jsonl(val, output_dir / "val.jsonl")
    if test:
        stats.written_test = _write_jsonl(test, output_dir / "test.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


HANDLERS: dict[str, Callable[..., ConversionStats]] = {
    "ham10000": convert_ham10000,
    "fitzpatrick17k": convert_fitzpatrick17k,
}


def _print_report(dataset: str, output_dir: Path, stats: ConversionStats) -> None:
    print("\n" + "=" * 68)
    print(f" Conversion report — {dataset}")
    print("=" * 68)
    print(f"  Source rows:        {stats.total_input_rows}")
    if stats.skipped_missing_id:
        print(f"  Skipped (no ID):    {stats.skipped_missing_id}")
    if stats.filtered_no_image:
        print(f"  Filtered (no img):  {stats.filtered_no_image}")
    if stats.filtered_qc:
        print(f"  Filtered (QC=3):    {stats.filtered_qc}")
    print(f"  Written total:      {stats.total_written()}")
    print(f"    train.jsonl:      {stats.written_train}")
    print(f"    val.jsonl:        {stats.written_val}")
    print(f"    test.jsonl:       {stats.written_test}")
    if stats.classes:
        print("\n  Class distribution (over all written records):")
        # Sort by count descending.
        for cls, n in sorted(stats.classes.items(), key=lambda kv: -kv[1])[:20]:
            print(f"    {cls:<25} {n}")
        if len(stats.classes) > 20:
            print(f"    ... ({len(stats.classes) - 20} more classes)")
    print(f"\n  Output dir:         {output_dir}")
    print()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert benchmark CSVs to DermArbiter JSONL splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", choices=sorted(HANDLERS.keys()), required=True)
    p.add_argument(
        "--input-dir",
        help="Directory containing the downloaded CSV (default: data/<dataset>/).",
    )
    p.add_argument(
        "--output-dir",
        help="Where to write {train,val,test}.jsonl (default: same as --input-dir).",
    )
    p.add_argument(
        "--splits",
        default="70/15/15",
        help="Train/val/test split fractions, slash-separated. Use 0/0/100 for test-only.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--require-image",
        action="store_true",
        help="Drop rows whose image file is missing on disk.",
    )
    p.add_argument(
        "--filter-qc",
        action="store_true",
        help="Fitzpatrick17k only: drop rows flagged '3 Wrongly labelled'.",
    )
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir) if args.input_dir else Path("data") / args.dataset
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    fractions = SplitFractions.parse(args.splits)

    handler = HANDLERS[args.dataset]
    kwargs: dict = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "fractions": fractions,
        "seed": args.seed,
        "require_image": args.require_image,
        "max_cases": args.max_cases,
    }
    if args.dataset == "fitzpatrick17k":
        kwargs["filter_qc"] = args.filter_qc

    stats = handler(**kwargs)
    _print_report(args.dataset, output_dir, stats)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
