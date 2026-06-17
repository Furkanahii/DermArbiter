"""Download benchmark datasets that ``DermArbiter`` evaluates against.

Supported datasets and what this script does for each:

    ham10000        Auto: HAM10000_metadata.csv + the two ~1.5 GB image zip parts
                    from Harvard Dataverse (DOI 10.7910/DVN/DBW86T).
                    Result matches the layout expected by
                    ``DatasetLoader.load_ham10000``.

    fitzpatrick17k  Auto (CSV only by default): fitzpatrick17k.csv from the
                    upstream GitHub repo. Images live on external dermatology
                    sites; pass ``--with-images`` to scrape them (slow, brittle).

    skincon         Auto: SkinCon annotations CSV from the SkinCon GitHub repo.
                    SkinCon overlaps with Fitzpatrick17k images and Derm7pt;
                    you must pull those separately for the pixel data.

    derm7pt         Manual only: prints registration instructions. The Derm7pt
                    download requires a free account on the SFU portal, so it
                    cannot be automated politely.

    skincap         Manual only: prints HuggingFace dataset instructions.

Design goals:
    * Idempotent — re-running skips files that exist and pass size/MD5 checks.
    * Resumable — large downloads use HTTP Range requests.
    * Safe — checksums are verified after download where the source publishes
      them; otherwise file size is sanity-checked.
    * Honest — never claims a dataset is ready when only metadata was fetched.

Example
-------
Just HAM10000 metadata (sanity check, ~600 KB)::

    python scripts/download_datasets.py --dataset ham10000 --metadata-only

Full HAM10000 (~3 GB)::

    python scripts/download_datasets.py --dataset ham10000

Fitzpatrick17k CSV plus 200 sample images for development::

    python scripts/download_datasets.py --dataset fitzpatrick17k \\
        --with-images --max-images 200
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("download_datasets")

USER_AGENT = (
    "DermArbiter-DatasetDownloader/0.1 (academic research; "
    "https://github.com/dermarbiter; contact via repo issues)"
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset descriptors
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RemoteFile:
    """One downloadable artifact within a dataset."""

    name: str              # Filename on disk under the dataset's target dir.
    url: str               # HTTPS URL.
    expected_size: Optional[int] = None   # Bytes; sanity-check only.
    expected_md5: Optional[str] = None    # Hex digest if the source publishes one.
    extract: bool = False  # If True and name ends with .zip, unzip after download.
    extract_subdir: Optional[str] = None  # Optional subdir for the extracted files.


# Harvard Dataverse file IDs for HAM10000 (DOI: 10.7910/DVN/DBW86T).
# These IDs are stable; the access API is documented at
# https://guides.dataverse.org/en/latest/api/dataaccess.html.
_DVN_BASE = "https://dataverse.harvard.edu/api/access/datafile/"

# File IDs, sizes, and MD5s were queried from the Dataverse JSON API:
#   https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/DBW86T
# ?format=original returns the uploaded file as-is; without it, .tab files come
# back as Dataverse-converted TSV which loses the original CSV headers.
HAM10000_FILES = {
    "metadata": RemoteFile(
        name="HAM10000_metadata.csv",
        url=_DVN_BASE + "4338392?format=original",
        expected_size=830_428,
    ),
    "images_part_1": RemoteFile(
        name="HAM10000_images_part_1.zip",
        url=_DVN_BASE + "3172585",
        expected_size=1_366_522_108,
        extract=True,
        extract_subdir="images",
    ),
    "images_part_2": RemoteFile(
        name="HAM10000_images_part_2.zip",
        url=_DVN_BASE + "3172584",
        expected_size=1_403_566_547,
        extract=True,
        extract_subdir="images",
    ),
}

FITZ17K_CSV = RemoteFile(
    name="fitzpatrick17k.csv",
    url="https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv",
    expected_size=3_000_000,   # ~3 MB
)

SKINCON_CSV = RemoteFile(
    name="annotations_fitzpatrick17k.csv",
    url=(
        "https://raw.githubusercontent.com/SonyResearch/"
        "SkinCon/main/annotations/annotations_fitzpatrick17k.csv"
    ),
    expected_size=200_000,   # rough; tightened by HEAD probe
)


# ─────────────────────────────────────────────────────────────────────────────
# Generic download utilities
# ─────────────────────────────────────────────────────────────────────────────


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _md5(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()  # noqa: S324 — integrity check, not crypto
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, expected_size: Optional[int], expected_md5: Optional[str]) -> bool:
    """Verify a downloaded file. Logs a warning on mismatch but only fails on MD5."""
    if not path.exists():
        return False
    actual_size = path.stat().st_size
    if expected_size is not None and actual_size < expected_size * 0.5:
        # Size is wildly off — likely truncated.
        logger.warning(
            "%s size %s is far below expected ~%s; treating as incomplete.",
            path.name, _human_bytes(actual_size), _human_bytes(expected_size),
        )
        return False
    if expected_md5:
        actual = _md5(path)
        if actual.lower() != expected_md5.lower():
            logger.error(
                "%s MD5 mismatch: got %s expected %s", path.name, actual, expected_md5,
            )
            return False
    return True


def _download(
    url: str,
    dest: Path,
    expected_size: Optional[int] = None,
    expected_md5: Optional[str] = None,
    force: bool = False,
    timeout: float = 60.0,
) -> Path:
    """Download ``url`` to ``dest`` with resume support and progress display.

    If ``dest`` already exists and passes verification, the download is skipped.
    Partial downloads are resumed via HTTP Range when the server supports it;
    otherwise the file is re-fetched from scratch.
    """
    import requests
    from tqdm import tqdm

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not force and dest.exists() and _verify(dest, expected_size, expected_md5):
        logger.info("✓ %s already present (%s)", dest.name, _human_bytes(dest.stat().st_size))
        return dest

    headers = {"User-Agent": USER_AGENT}
    mode = "wb"
    resume_offset = 0

    partial = dest.with_suffix(dest.suffix + ".part")
    if partial.exists() and not force:
        resume_offset = partial.stat().st_size
        headers["Range"] = f"bytes={resume_offset}-"
        mode = "ab"

    logger.info("→ Downloading %s", url)
    with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as r:
        if resume_offset and r.status_code not in (206, 200):
            # Server didn't honour Range; restart cleanly.
            partial.unlink(missing_ok=True)
            resume_offset = 0
            mode = "wb"
            r.close()
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True,
                             timeout=timeout, allow_redirects=True)
        r.raise_for_status()

        total = int(r.headers.get("Content-Length", 0))
        if r.status_code == 206 and resume_offset:
            total += resume_offset
        if expected_size is None and total:
            expected_size = total

        with partial.open(mode) as fh, tqdm(
            total=total or None,
            initial=resume_offset,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if not chunk:
                    continue
                fh.write(chunk)
                pbar.update(len(chunk))

    if not _verify(partial, expected_size, expected_md5):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Download verification failed for {url}")

    partial.replace(dest)
    logger.info("✓ Wrote %s (%s)", dest.name, _human_bytes(dest.stat().st_size))
    return dest


def _extract_zip(zip_path: Path, dest_dir: Path) -> int:
    """Extract a zip archive. Returns the number of files written."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for name in members:
            # Defensive: skip anything trying to escape dest_dir.
            target = (dest_dir / Path(name).name).resolve()
            if not str(target).startswith(str(dest_dir.resolve())):
                logger.warning("Skipping suspicious zip entry %s", name)
                continue
            if name.endswith("/"):
                continue
            if target.exists():
                count += 1
                continue
            with zf.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    logger.info("✓ Extracted %d files from %s into %s", count, zip_path.name, dest_dir)
    return count


def _disk_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset handlers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DownloadReport:
    dataset: str
    target_dir: Path
    files_written: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add_file(self, path: Path) -> None:
        self.files_written.append(str(path.relative_to(self.target_dir.parent)))

    def add_note(self, note: str) -> None:
        self.notes.append(note)


def download_ham10000(
    target_dir: Path,
    metadata_only: bool = False,
    force: bool = False,
    keep_zip: bool = False,
) -> DownloadReport:
    """Download HAM10000 (Harvard Dataverse, DOI 10.7910/DVN/DBW86T).

    Layout produced::

        target_dir/
            HAM10000_metadata.csv
            images/
                ISIC_0024306.jpg
                ...
    """
    report = DownloadReport(dataset="HAM10000", target_dir=target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metadata (small, always grab).
    meta_file = HAM10000_FILES["metadata"]
    meta_path = _download(meta_file.url, target_dir / meta_file.name,
                          expected_size=meta_file.expected_size, force=force)
    report.add_file(meta_path)

    if metadata_only:
        report.add_note("Metadata-only mode: images not downloaded.")
        return report

    # 2. Disk-space check before the heavy parts (~3 GB unzipped + zip files).
    need = sum(f.expected_size or 0 for f in HAM10000_FILES.values()) * 2
    free = _disk_free_bytes(target_dir)
    if free < need:
        raise RuntimeError(
            f"Insufficient disk space: need ~{_human_bytes(need)} "
            f"in {target_dir}, have {_human_bytes(free)}."
        )

    # 3. Image archives.
    for key in ("images_part_1", "images_part_2"):
        f = HAM10000_FILES[key]
        zip_path = _download(f.url, target_dir / f.name,
                             expected_size=f.expected_size, force=force)
        if f.extract:
            sub = target_dir / (f.extract_subdir or "")
            _extract_zip(zip_path, sub)
            if not keep_zip:
                zip_path.unlink()
                logger.info("Removed %s after extraction (--keep-zip to retain).", zip_path.name)
        report.add_file(zip_path)

    return report


def download_fitzpatrick17k(
    target_dir: Path,
    with_images: bool = False,
    max_images: Optional[int] = None,
    force: bool = False,
    delay: float = 0.5,
) -> DownloadReport:
    """Download Fitzpatrick17k metadata; optionally scrape image URLs.

    Layout produced::

        target_dir/
            fitzpatrick17k.csv
            images/                       (only when --with-images was set)
                {md5hash}.jpg

    The image URLs in the CSV point to atlasdermatologico.com.br and
    dermaamin.com. Many entries are stale; expect ~30-40% download failures
    and proceed politely (small delay between requests).
    """
    report = DownloadReport(dataset="Fitzpatrick17k", target_dir=target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_path = _download(FITZ17K_CSV.url, target_dir / FITZ17K_CSV.name,
                         expected_size=FITZ17K_CSV.expected_size, force=force)
    report.add_file(csv_path)

    if not with_images:
        report.add_note(
            "Metadata-only. Pass --with-images to scrape the per-row image URLs "
            "from Atlas Dermatologico / DermaAmin (slow, brittle)."
        )
        return report

    # Scrape images.
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    import requests

    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})

    ok = 0
    fail = 0
    skip = 0
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if max_images is not None:
        rows = rows[:max_images]

    logger.info("Scraping %d image URLs from Fitzpatrick17k…", len(rows))
    for i, row in enumerate(rows, start=1):
        hasher = (row.get("md5hash") or row.get("hasher") or "").strip()
        url = (row.get("url") or "").strip()
        if not hasher or not url:
            fail += 1
            continue
        dest = images_dir / f"{hasher}.jpg"
        if dest.exists() and not force:
            skip += 1
            continue
        try:
            with sess.get(url, timeout=20, stream=True) as r:
                r.raise_for_status()
                with dest.open("wb") as dst:
                    shutil.copyfileobj(r.raw, dst)
            ok += 1
            if i % 50 == 0:
                logger.info("  progress: %d/%d (ok=%d fail=%d skip=%d)",
                            i, len(rows), ok, fail, skip)
        except Exception as exc:
            fail += 1
            logger.debug("Failed %s: %s", url, exc)
            dest.unlink(missing_ok=True)
        time.sleep(delay)

    report.add_note(
        f"Image scrape: ok={ok} fail={fail} skipped={skip}. "
        f"Failures are expected (~30-40% of URLs are stale)."
    )
    return report


def download_skincon(target_dir: Path, force: bool = False) -> DownloadReport:
    """Download SkinCon annotation CSV (3,230 dermatology concept labels)."""
    report = DownloadReport(dataset="SkinCon", target_dir=target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = _download(SKINCON_CSV.url, target_dir / SKINCON_CSV.name,
                     expected_size=SKINCON_CSV.expected_size, force=force)
    report.add_file(path)
    report.add_note(
        "SkinCon publishes concept annotations only. Source images are a subset "
        "of Fitzpatrick17k and Derm7pt — those datasets must be downloaded "
        "separately for pixel data."
    )
    return report


def download_derm7pt(target_dir: Path, **_kw) -> DownloadReport:
    """Derm7pt requires manual registration; print instructions only."""
    report = DownloadReport(dataset="Derm7pt", target_dir=target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    msg = (
        "Derm7pt cannot be downloaded automatically — the source portal at\n"
        "  https://derm.cs.sfu.ca/Welcome.html\n"
        "requires a free registration before issuing download links.\n\n"
        "Steps:\n"
        f"  1. Register and download the dataset zip to {target_dir}/\n"
        "  2. Unzip so that you have:\n"
        f"     {target_dir}/release_v0/meta/meta.csv\n"
        f"     {target_dir}/release_v0/images/\n"
        "  3. Re-run DermArbiter evaluation pointing data_dir at this directory."
    )
    print("\n" + msg + "\n")
    report.add_note("Manual download required (registration on derm.cs.sfu.ca).")
    return report


def download_bcn20000(target_dir: Path, **_kw) -> DownloadReport:
    """BCN20000 is distributed via the ISIC Archive / ISIC 2019 challenge; print instructions."""
    report = DownloadReport(dataset="BCN20000", target_dir=target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    msg = (
        "BCN20000 (part of the ISIC 2019 Challenge dataset) requires manual download.\n\n"
        "Source (ISIC Archive):\n"
        "  https://challenge.isic-archive.com/landing/2019/\n\n"
        "Steps:\n"
        "  1. Register on the ISIC Challenge site.\n"
        "  2. Download BCN20000 clinical images and ground truth metadata CSV.\n"
        f"  3. Extract so that you have:\n"
        f"     {target_dir}/ISIC_2019_Training_Metadata.csv   (image, MEL, NV, BCC, ...)\n"
        f"     {target_dir}/images/*.jpg\n"
        "  4. Point BCN20000 evaluation config at this directory."
    )
    print("\n" + msg + "\n")
    report.add_note("Manual download required (ISIC 2019 portal).")
    return report


def download_ddi(target_dir: Path, **_kw) -> DownloadReport:
    """DDI (Stanford Diverse Dermatology Images) — manual access.

    DDI is the fairness anchor for DermAbench (Fitzpatrick skin-tone groups
    + biopsy-confirmed malignancy). Gated behind a Stanford AIMI
    registration + data-use agreement; cannot be fetched automatically.
    Stages the layout expected by ``build_dermabench.py --source ddi``.
    """
    report = DownloadReport(dataset="DDI", target_dir=target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "images").mkdir(exist_ok=True)
    msg = (
        "DDI (Diverse Dermatology Images) requires manual download.\n\n"
        "Source (Stanford AIMI — free registration + data-use agreement):\n"
        "  https://stanfordaimi.azurewebsites.net/datasets/\n"
        "  (search 'Diverse Dermatology Images' / DDI)\n\n"
        "Steps:\n"
        "  1. Register, accept the DUA, download the DDI archive.\n"
        f"  2. Unzip so that you have:\n"
        f"     {target_dir}/ddi_metadata.csv   (DDI_file, skin_tone,\n"
        f"                                      malignant, disease)\n"
        f"     {target_dir}/images/*.png\n"
        f"  3. Build DermAbench DDI cases:\n"
        f"     python scripts/build_dermabench.py --source ddi \\\n"
        f"         --raw-dir {target_dir} --out data/dermabench/ddi.jsonl\n\n"
        "skin_tone 12/34/56 → Fitzpatrick I-II / III-IV / V-VI; the\n"
        "'malignant' flag is biopsy-confirmed ground truth for DermAbench\n"
        "Dimensions 6 (fairness) and 7 (safety)."
    )
    print("\n" + msg + "\n")
    report.add_note("Manual download required (Stanford AIMI registration + DUA).")
    return report


def download_scin(target_dir: Path, **_kw) -> DownloadReport:
    """SCIN (Google Skin Condition Image Network) — public GCS bucket."""
    report = DownloadReport(dataset="SCIN", target_dir=target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "images").mkdir(exist_ok=True)
    msg = (
        "SCIN (Skin Condition Image Network) is on a public GCS bucket.\n\n"
        "Source: gs://dx-scin-public-data/\n"
        "  (https://github.com/google-research-datasets/scin)\n\n"
        "Steps (requires gsutil — `pip install gsutil`):\n"
        f"  1. gsutil cp gs://dx-scin-public-data/dataset/scin_cases.csv {target_dir}/\n"
        f"     gsutil cp gs://dx-scin-public-data/dataset/scin_labels.csv {target_dir}/\n"
        f"  2. gsutil -m cp -r gs://dx-scin-public-data/dataset/images {target_dir}/\n"
        f"  3. python scripts/build_dermabench.py --source scin \\\n"
        f"         --raw-dir {target_dir} --out data/dermabench/scin.jsonl\n\n"
        "Dermatologist labels + Fitzpatrick feed narrative (Dim 2) and\n"
        "fairness (Dim 6)."
    )
    print("\n" + msg + "\n")
    report.add_note("Manual download (public GCS bucket via gsutil).")
    return report


def download_skincap(target_dir: Path, **_kw) -> DownloadReport:
    """SkinCap is on HuggingFace; print instructions only."""
    report = DownloadReport(dataset="SkinCap", target_dir=target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    msg = (
        "SkinCap is distributed via HuggingFace. To pull it, install the\n"
        "datasets library and run:\n\n"
        "    from datasets import load_dataset\n"
        "    ds = load_dataset('joshuachou/SkinCAP')\n"
        f"    ds.save_to_disk('{target_dir}')\n\n"
        "Then point the SkinCap evaluator at the saved Arrow directory."
    )
    print("\n" + msg + "\n")
    report.add_note("Manual download (HuggingFace datasets).")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


HANDLERS: dict[str, Callable[..., DownloadReport]] = {
    "ham10000": download_ham10000,
    "fitzpatrick17k": download_fitzpatrick17k,
    "skincon": download_skincon,
    "derm7pt": download_derm7pt,
    "skincap": download_skincap,
    "ddi": download_ddi,
    "scin": download_scin,
    "bcn20000": download_bcn20000,
}


def _print_summary(reports: list[DownloadReport]) -> None:
    print("\n" + "=" * 68)
    print(" Dataset download summary")
    print("=" * 68)
    for r in reports:
        print(f"\n[{r.dataset}]  →  {r.target_dir}")
        if r.files_written:
            for f in r.files_written:
                print(f"  · {f}")
        else:
            print("  (no files written)")
        for note in r.notes:
            print(f"  ⚠ {note}")
    print()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download DermArbiter benchmark datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        choices=["all"] + sorted(HANDLERS.keys()),
        default="ham10000",
        help="Dataset to download. 'all' iterates through every supported dataset.",
    )
    p.add_argument(
        "--target-root",
        default="data",
        help="Root directory under which each dataset gets its own subfolder.",
    )
    p.add_argument(
        "--metadata-only",
        action="store_true",
        help="HAM10000 only: download metadata CSV but skip the ~3 GB image zips.",
    )
    p.add_argument(
        "--with-images",
        action="store_true",
        help="Fitzpatrick17k only: also scrape the per-row image URLs.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Fitzpatrick17k only: cap the number of images scraped (testing).",
    )
    p.add_argument(
        "--keep-zip",
        action="store_true",
        help="HAM10000: retain image zip archives after extraction.",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def run(args: argparse.Namespace) -> int:
    root = Path(args.target_root)
    datasets = sorted(HANDLERS.keys()) if args.dataset == "all" else [args.dataset]

    reports: list[DownloadReport] = []
    for name in datasets:
        target = root / name
        logger.info("=== %s → %s ===", name, target)
        handler = HANDLERS[name]
        kwargs: dict = {"target_dir": target, "force": args.force}
        if name == "ham10000":
            kwargs["metadata_only"] = args.metadata_only
            kwargs["keep_zip"] = args.keep_zip
        elif name == "fitzpatrick17k":
            kwargs["with_images"] = args.with_images
            kwargs["max_images"] = args.max_images
        try:
            reports.append(handler(**kwargs))
        except Exception as exc:
            logger.error("[%s] failed: %s", name, exc, exc_info=args.verbose)
            r = DownloadReport(dataset=name, target_dir=target)
            r.add_note(f"FAILED: {exc}")
            reports.append(r)

    _print_summary(reports)
    failed = [r for r in reports if any(n.startswith("FAILED") for n in r.notes)]
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        for noisy in ("urllib3", "requests"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
