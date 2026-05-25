"""Filter our HAM10000 case base to the EXACT 642 images DermAgent (Liu et al.,
MICCAI 2026, arXiv:2605.14403) used in their HAM10000 benchmark.

Why this script exists
----------------------
DermAgent reports 81.83 % accuracy on a class-balanced 642-image subset of
HAM10000 (50 each of akiec/bcc/df/vasc, 53 bkl, 58 mel, 331 nv). The full
HAM10000 test set is dominated by nv (~67 %), so any DermArbiter accuracy
quoted on the full split is *not* directly comparable to DermAgent's number.

This script reads the upstream split list at
``data/ham10000/dermagent_subset.csv`` (pulled verbatim from
``https://github.com/YizeezLiu/DermAgent/blob/master/data/ham10000/HAM10000_benchmark_500.csv``)
and produces a DermArbiter-flavoured JSONL covering exactly those cases.

Output: ``data/ham10000/dermagent_subset.jsonl`` — one record per line, schema
identical to ``DatasetLoader.load_jsonl`` (see ``convert_to_jsonl.py``).

Example
-------
::

    python scripts/build_dermagent_subset.py
    python scripts/build_dermagent_subset.py --require-image   # drop unfetched cases
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("build_dermagent_subset")

DEFAULT_SUBSET_CSV = "data/ham10000/dermagent_subset.csv"
DEFAULT_OUTPUT = "data/ham10000/dermagent_subset.jsonl"
DEFAULT_HAM_DIR = "data/ham10000"


@dataclass
class BuildStats:
    requested: int = 0
    written: int = 0
    missing_in_ham10000: list[str] = None
    missing_image_file: list[str] = None

    def __post_init__(self):
        if self.missing_in_ham10000 is None:
            self.missing_in_ham10000 = []
        if self.missing_image_file is None:
            self.missing_image_file = []


def load_subset_ids(subset_csv: Path) -> set[str]:
    """Read the image_id column from DermAgent's subset CSV."""
    if not subset_csv.exists():
        raise FileNotFoundError(
            f"{subset_csv} not found. Pull it with:\n"
            f"  curl -L -o {subset_csv} \\\n"
            f"    https://raw.githubusercontent.com/YizeezLiu/DermAgent/master/"
            f"data/ham10000/HAM10000_benchmark_500.csv"
        )
    ids: set[str] = set()
    with subset_csv.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            iid = (row.get("image_id") or "").strip()
            if iid:
                ids.add(iid)
    return ids


def build(
    ham_dir: Path,
    subset_csv: Path,
    output_path: Path,
    require_image: bool,
) -> BuildStats:
    """Filter HAM10000_metadata.csv to subset_ids, emit JSONL records."""
    ham_meta = ham_dir / "HAM10000_metadata.csv"
    if not ham_meta.exists():
        raise FileNotFoundError(
            f"{ham_meta} not found. Run:\n"
            f"  python scripts/download_datasets.py --dataset ham10000 --metadata-only"
        )

    subset_ids = load_subset_ids(subset_csv)
    stats = BuildStats(requested=len(subset_ids))
    logger.info("DermAgent subset: %d unique image_ids requested", len(subset_ids))

    images_dir = ham_dir / "images"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    found_ids: set[str] = set()
    records: list[dict] = []
    with ham_meta.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            image_id = (row.get("image_id") or "").strip()
            if image_id not in subset_ids:
                continue
            found_ids.add(image_id)
            img_path = images_dir / f"{image_id}.jpg"
            if require_image and not img_path.exists():
                stats.missing_image_file.append(image_id)
                continue
            records.append({
                "case_id": image_id,
                "image_path": str(img_path),
                "query": "What is the diagnosis for this skin lesion?",
                "ground_truth_label": (row.get("dx") or "").strip().lower(),
                "patient_context": {
                    "age": (row.get("age") or "").strip(),
                    "sex": (row.get("sex") or "").strip(),
                    "localization": (row.get("localization") or "").strip(),
                },
                "subset_source": "DermAgent_HAM10000_benchmark_500",
            })

    stats.missing_in_ham10000 = sorted(subset_ids - found_ids)

    with output_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats.written = len(records)

    return stats


def _print_report(stats: BuildStats, output_path: Path, records_classes: Counter | None = None) -> None:
    print("\n" + "=" * 68)
    print(" DermAgent subset build report")
    print("=" * 68)
    print(f"  Requested image_ids:        {stats.requested}")
    print(f"  Written records:            {stats.written}")
    if stats.missing_in_ham10000:
        print(f"  Missing from HAM10000:      {len(stats.missing_in_ham10000)}")
        if len(stats.missing_in_ham10000) <= 5:
            for iid in stats.missing_in_ham10000:
                print(f"    - {iid}")
    if stats.missing_image_file:
        print(f"  Filtered (no image file):   {len(stats.missing_image_file)}")
    if records_classes:
        print("\n  Class distribution:")
        for cls, n in sorted(records_classes.items(), key=lambda kv: -kv[1]):
            print(f"    {cls:<10} {n}")
    print(f"\n  Output: {output_path}\n")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Filter our HAM10000 to DermAgent's 642-image benchmark subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ham-dir", default=DEFAULT_HAM_DIR,
                   help="Directory containing HAM10000_metadata.csv and images/.")
    p.add_argument("--subset-csv", default=DEFAULT_SUBSET_CSV,
                   help="DermAgent's upstream subset CSV (pulled into our repo).")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--require-image", action="store_true",
                   help="Drop records whose .jpg is not on disk (default off — JSONL still emitted).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        stats = build(
            ham_dir=Path(args.ham_dir),
            subset_csv=Path(args.subset_csv),
            output_path=Path(args.output),
            require_image=args.require_image,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    # Re-read for class distribution (cheap; subset is small).
    classes: Counter = Counter()
    with Path(args.output).open() as fh:
        for line in fh:
            if line.strip():
                classes[json.loads(line)["ground_truth_label"]] += 1
    _print_report(stats, Path(args.output), classes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
