"""Bundle a curated DermAbench set + only its referenced images into a zip
for transport to Colab (via Google Drive).

The curated gold JSONL and raw images are gitignored, so a Colab clone of
the repo won't have them. Rather than re-download/re-curate on Colab (SCIN
is public but DDI needs a short-lived SAS link), we ship the exact set we
validated locally — guaranteeing identical, deterministic data.

The zip preserves repo-relative paths so it unzips straight into the repo
root on Colab:

    data/dermabench/<set>.jsonl
    data/dermabench/raw/scin/dataset/images/<id>.png
    data/dermabench/raw/ddi/images/<id>.png

Usage:
    python scripts/make_colab_bundle.py \
        --gold data/dermabench/dermabench_v1lite.jsonl \
        --out  data/dermabench/dermabench_v1lite_bundle.zip
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path


def build_bundle(gold_path: Path, out_path: Path) -> tuple[int, int, int]:
    """Zip the gold JSONL + every image it references. Returns
    (n_cases, n_images_added, n_images_missing)."""
    cases = [json.loads(line) for line in
             gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    added = missing = 0
    seen: set[str] = set()
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # The gold itself (arcname = repo-relative path).
        zf.write(gold_path, arcname=str(gold_path))
        for c in cases:
            img = c.get("image_path", "")
            if not img or img in seen:
                continue
            seen.add(img)
            if os.path.exists(img) and os.path.getsize(img) > 0:
                zf.write(img, arcname=img)
                added += 1
            else:
                missing += 1
    return len(cases), added, missing


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", default="data/dermabench/dermabench_v1lite.jsonl")
    p.add_argument("--out", default="data/dermabench/dermabench_v1lite_bundle.zip")
    args = p.parse_args(argv)

    gold = Path(args.gold)
    if not gold.exists():
        raise SystemExit(f"gold not found: {gold}")
    out = Path(args.out)

    n, added, missing = build_bundle(gold, out)
    size_mb = out.stat().st_size / 1e6
    print(f"\n  Bundle: {out}  ({size_mb:.1f} MB)")
    print(f"  Cases: {n}  |  images added: {added}  |  images MISSING: {missing}")
    if missing:
        print("  ⚠️  Some images are missing locally — download them first "
              "(SCIN images / DDI images) so the bundle is complete.")
    else:
        print("  ✅ All referenced images bundled.\n")
        print("  Next: upload this zip to Google Drive at")
        print("        MyDrive/dermarbiter/  then run notebook 07.\n")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
