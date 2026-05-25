"""Build the ChromaDB index consumed by ``dermarbiter.tools.case_rag.CaseRAG``.

The ``CaseRAG`` tool only *queries* an existing ChromaDB collection of
dermatology case embeddings; this script is what *fills* that collection by
running every source image through the DermLIP image encoder and upserting
the resulting 512-D vectors along with per-case metadata.

Three ingest modes are supported:

    ham10000     Use the HAM10000 dataset (downloaded via
                 ``scripts/download_datasets.py``) as the reference case
                 base. Each ISIC image becomes one indexed case, with
                 ``diagnosis``, ``location``, ``age``, and ``sex`` metadata
                 pulled from HAM10000_metadata.csv.

    local        Generic local mode — point ``--manifest`` at a CSV or JSONL
                 file with columns/keys: ``case_id, image_path, diagnosis``
                 plus optional ``location``, ``age``, ``sex``, ``source``.

    derm1m       Placeholder for the official Derm1M (413K+ cases) flow.
                 Currently raises NotImplementedError — to enable, set
                 HUGGINGFACE_HUB_TOKEN and edit ``load_derm1m_manifest``.

Idempotency: ChromaDB upsert is keyed by ``case_id``, so re-running the
script with the same manifest does not duplicate records.

Hardware: DermLIP-ViT-B-16 is small (~600 MB) but encoding 413K images on
CPU is impractical. Designed for Colab T4 with batch_size=32.

Example
-------
HAM10000 → ChromaDB on Colab T4::

    python scripts/build_case_rag_index.py --source ham10000 --batch-size 64

Local manifest with a custom CLIP model::

    python scripts/build_case_rag_index.py --source local \\
        --manifest data/my_cases.jsonl \\
        --clip-model hf-hub:redlessone/DermLIP_ViT-B-16
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

logger = logging.getLogger("case_rag_ingest")


# ─────────────────────────────────────────────────────────────────────────────
# Constants — keep aligned with dermarbiter.tools.case_rag defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CHROMA_DIR = "data/chroma_cases"
DEFAULT_COLLECTION = "derm1m_cases"
DEFAULT_CLIP_MODEL = "hf-hub:redlessone/DermLIP_ViT-B-16"
DEFAULT_BATCH_SIZE = 32

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CaseEntry:
    """One reference case ready for embedding and upsert."""

    case_id: str          # ChromaDB primary key; e.g. "ISIC_0024306".
    image_path: Path
    diagnosis: str
    location: str = ""
    age: str = ""
    sex: str = ""
    source: str = ""      # Provenance label (e.g. "HAM10000", "Derm1M").

    def to_metadata(self) -> dict[str, str]:
        """ChromaDB only accepts scalars in metadata — flatten everything to str."""
        return {
            "diagnosis": self.diagnosis,
            "location": self.location,
            "age": self.age,
            "sex": self.sex,
            "source": self.source,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Encoder Protocol — dependency-injected so tests can use a fake
# ─────────────────────────────────────────────────────────────────────────────


class Encoder(Protocol):
    """An image batch encoder that yields one L2-normalised vector per image."""

    embedding_dim: int

    def encode_batch(self, image_paths: list[Path]) -> list[list[float]]:
        ...


class DermLipEncoder:
    """Lazy-loading wrapper around an ``open_clip`` DermLIP image tower.

    Loading is deferred until the first ``encode_batch`` call so that
    ``--dry-run`` and unit tests can skip the heavy model entirely.
    """

    def __init__(self, clip_model: str = DEFAULT_CLIP_MODEL, device: str = "auto") -> None:
        self._clip_model_name = clip_model
        self._device_str = device
        self._model = None
        self._preprocess = None
        self._device = None
        self.embedding_dim = 512   # DermLIP-ViT-B-16

    def _resolve_device(self):
        import torch

        if self._device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self._device_str)

    def _load(self) -> None:
        if self._model is not None:
            return
        import open_clip
        import torch  # noqa: F401 — used in _resolve_device

        self._device = self._resolve_device()
        logger.info("Loading DermLIP encoder '%s' on %s", self._clip_model_name, self._device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            self._clip_model_name, device=self._device,
        )
        self._model = model.eval()
        self._preprocess = preprocess

    def encode_batch(self, image_paths: list[Path]) -> list[list[float]]:
        import torch
        from PIL import Image

        self._load()
        tensors = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            tensors.append(self._preprocess(img))
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Source loaders
# ─────────────────────────────────────────────────────────────────────────────


def load_ham10000_manifest(data_dir: Path) -> list[CaseEntry]:
    """Build CaseEntry list from a downloaded HAM10000 dataset.

    Expects ``data_dir/HAM10000_metadata.csv`` and ``data_dir/images/``.
    The dx-code is kept lowercase to match the loader convention.
    """
    csv_path = data_dir / "HAM10000_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run: "
            f"python scripts/download_datasets.py --dataset ham10000"
        )

    images_dir = data_dir / "images"
    entries: list[CaseEntry] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                continue
            entries.append(CaseEntry(
                case_id=image_id,
                image_path=images_dir / f"{image_id}.jpg",
                diagnosis=(row.get("dx") or "").strip().lower(),
                location=(row.get("localization") or "").strip(),
                age=(row.get("age") or "").strip(),
                sex=(row.get("sex") or "").strip(),
                source="HAM10000",
            ))
    logger.info("Loaded %d HAM10000 entries from %s", len(entries), csv_path)
    return entries


def load_local_manifest(manifest_path: Path) -> list[CaseEntry]:
    """Read a generic manifest (CSV or JSONL).

    Required fields: ``case_id, image_path, diagnosis``.
    Optional fields: ``location, age, sex, source``.

    Image paths are resolved relative to the manifest file's parent directory
    when relative paths are given.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    base = manifest_path.parent

    rows: Iterable[dict[str, str]]
    if manifest_path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = (
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    else:
        with manifest_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

    entries: list[CaseEntry] = []
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        image_path = str(row.get("image_path") or "").strip()
        diagnosis = str(row.get("diagnosis") or "").strip().lower()
        if not (case_id and image_path and diagnosis):
            logger.warning("Skipping incomplete manifest row: %s", row)
            continue
        img = Path(image_path)
        if not img.is_absolute():
            img = (base / img).resolve()
        entries.append(CaseEntry(
            case_id=case_id,
            image_path=img,
            diagnosis=diagnosis,
            location=str(row.get("location") or "").strip(),
            age=str(row.get("age") or "").strip(),
            sex=str(row.get("sex") or "").strip(),
            source=str(row.get("source") or "").strip() or "local",
        ))
    logger.info("Loaded %d entries from manifest %s", len(entries), manifest_path)
    return entries


DERM1M_REPO_ID = "redlessone/Derm1M"
DERM1M_DEFAULT_SPLIT = "pretrain"
DERM1M_DEFAULT_DATA_DIR = "data/derm1m"


def _download_derm1m(
    data_dir: Path,
    hf_token: Optional[str] = None,
    allow_patterns: Optional[list[str]] = None,
) -> Path:
    """Snapshot-download Derm1M from HF Hub (gated). Returns the local cache root.

    The dataset ships as raw image folders + CSV manifests; ``snapshot_download``
    resumes interrupted runs and respects ``allow_patterns`` so callers can pull
    just the manifest for a dry-run before committing to the full image pull.
    """
    from huggingface_hub import snapshot_download

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGINGFACE_HUB_TOKEN"
    )
    if token is None:
        raise SystemExit(
            "Derm1M is a gated dataset. Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) "
            "in the environment, or pass --hf-token. Make sure you accepted "
            "the usage agreement at https://huggingface.co/datasets/redlessone/Derm1M"
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading Derm1M (%s) → %s%s",
        DERM1M_REPO_ID,
        data_dir,
        f" filtered to {allow_patterns}" if allow_patterns else " (full snapshot)",
    )
    local_root = snapshot_download(
        repo_id=DERM1M_REPO_ID,
        repo_type="dataset",
        local_dir=str(data_dir),
        token=token,
        allow_patterns=allow_patterns,
    )
    return Path(local_root)


def _stratified_sample(
    rows: list[dict[str, str]],
    n: int,
    *,
    stratify_by: str = "disease_label",
    seed: int = 42,
) -> list[dict[str, str]]:
    """Class-stratified sample of size n from rows.

    Falls back to a plain shuffle if no stratification key is found.
    """
    import random
    from collections import defaultdict

    rng = random.Random(seed)
    if not rows or n >= len(rows):
        return list(rows)

    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        buckets[(r.get(stratify_by) or "unknown").strip().lower()].append(r)

    # Allocate per-bucket quota proportional to bucket size, min 1.
    total = len(rows)
    quotas: dict[str, int] = {}
    remainders: list[tuple[str, float]] = []
    used = 0
    for label, bucket in buckets.items():
        exact = n * len(bucket) / total
        quotas[label] = int(exact)
        remainders.append((label, exact - int(exact)))
        used += quotas[label]
    # Distribute the remaining quota by largest remainder
    for label, _ in sorted(remainders, key=lambda x: -x[1]):
        if used >= n:
            break
        if quotas[label] < len(buckets[label]):
            quotas[label] += 1
            used += 1

    sampled: list[dict[str, str]] = []
    for label, q in quotas.items():
        bucket = buckets[label]
        rng.shuffle(bucket)
        sampled.extend(bucket[:q])
    rng.shuffle(sampled)
    return sampled


def load_derm1m_manifest(
    data_dir: Path | str = DERM1M_DEFAULT_DATA_DIR,
    split: str = DERM1M_DEFAULT_SPLIT,
    max_cases: Optional[int] = None,
    stratify_by: str = "disease_label",
    hf_token: Optional[str] = None,
    download_images: bool = True,
    seed: int = 42,
) -> list[CaseEntry]:
    """Build CaseEntry list from the gated Derm1M HF dataset.

    Args:
        data_dir: Local cache root for the HF snapshot.
        split: Either ``pretrain`` (≈1.0M pairs / ≈400K imgs) or ``validation``.
        max_cases: When set, take a stratified subset of this size (per
            ``stratify_by``) instead of the full split. Recommended for first
            ingest runs to fit Colab free-tier quotas.
        stratify_by: Manifest column used for stratified sampling.
        hf_token: Override for the HF auth token (env-based by default).
        download_images: When False, only the manifest CSV is fetched (used
            for dry-run / manifest sanity checks).
        seed: RNG seed for reproducible stratified sampling.
    """
    data_dir = Path(data_dir)
    manifest_name = (
        "Derm1M_v2_pretrain.csv"
        if split == "pretrain"
        else "Derm1M_v2_validation.csv"
    )

    # First pull just the manifest (cheap) so we can decide what image
    # patterns to allow on the full snapshot.
    _download_derm1m(
        data_dir,
        hf_token=hf_token,
        allow_patterns=[manifest_name, "ontology.json"],
    )
    manifest_path = data_dir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Derm1M manifest {manifest_path} not found after HF snapshot. "
            "Double-check the split name and your gated-access agreement."
        )

    # utf-8-sig strips the BOM that ships at the head of the published CSVs;
    # without it the first column key becomes "﻿filename" and every row
    # silently fails the `filename` lookup.
    with manifest_path.open("r", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    logger.info("Derm1M %s manifest: %d rows", split, len(rows))

    def _pick(row: dict[str, str], *keys: str) -> str:
        """First non-empty value across alternate column names (BOM-safe)."""
        for k in keys:
            if k in row and (row[k] or "").strip():
                return row[k].strip()
        # Fallback: tolerate stray BOMs we somehow missed.
        for k in keys:
            bom_key = "﻿" + k
            if bom_key in row and (row[bom_key] or "").strip():
                return row[bom_key].strip()
        return ""

    # "No <thing> information" is Derm1M's sentinel for missing values — treat
    # as empty so it doesn't pollute downstream retrieval / metadata.
    def _clean(value: str) -> str:
        v = (value or "").strip()
        if not v or v.lower().startswith("no ") and "information" in v.lower():
            return ""
        return v

    if max_cases is not None and max_cases < len(rows):
        rows = _stratified_sample(rows, max_cases, stratify_by=stratify_by, seed=seed)
        logger.info("Stratified subset → %d rows (by %s)", len(rows), stratify_by)

    if download_images:
        # Pull only the image files referenced by the chosen subset to avoid
        # downloading 80 GB+ when the user just wants 50K.
        needed = sorted({_pick(r, "filename") for r in rows if _pick(r, "filename")})
        # HF snapshot_download accepts glob patterns; passing the explicit
        # filenames keeps the transfer scoped to the subset.
        _download_derm1m(data_dir, hf_token=hf_token, allow_patterns=needed)

    entries: list[CaseEntry] = []
    missing = 0
    for row in rows:
        filename = _pick(row, "filename")
        diagnosis = _pick(row, "disease_label").lower()
        if not filename or not diagnosis:
            continue
        img = data_dir / filename
        if download_images and not img.exists():
            missing += 1
            continue
        entries.append(CaseEntry(
            case_id=Path(filename).stem,
            image_path=img,
            diagnosis=diagnosis,
            location=_clean(_pick(row, "body_location")),
            age=_clean(_pick(row, "age")),
            sex=_clean(_pick(row, "gender", "sex")),
            source=f"Derm1M:{_pick(row, 'source')}",
        ))
    if missing:
        logger.warning(
            "Derm1M: %d/%d images missing on disk (likely HF allow_patterns mismatch).",
            missing,
            len(rows),
        )
    logger.info("Loaded %d Derm1M entries (split=%s).", len(entries), split)
    return entries


SOURCE_LOADERS = {
    "ham10000": "load_ham10000",
    "local": "load_local",
    "derm1m": "load_derm1m",
}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class IngestStats:
    total_entries: int = 0
    filtered_no_image: int = 0
    embedded: int = 0
    upserted: int = 0
    batches: int = 0
    elapsed_s: float = 0.0
    diagnoses: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _filter_entries(entries: list[CaseEntry], require_image: bool) -> tuple[list[CaseEntry], int]:
    """Drop entries whose image file is missing or has an unsupported extension."""
    kept: list[CaseEntry] = []
    dropped = 0
    for e in entries:
        if e.image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            dropped += 1
            continue
        if require_image and not e.image_path.exists():
            dropped += 1
            continue
        kept.append(e)
    return kept, dropped


def _upsert_batch(
    collection: Any,
    entries: list[CaseEntry],
    embeddings: list[list[float]],
) -> None:
    collection.upsert(
        ids=[e.case_id for e in entries],
        embeddings=embeddings,
        metadatas=[e.to_metadata() for e in entries],
        # documents are optional — keep the raw_text searchable for debugging.
        documents=[f"{e.diagnosis} | {e.location} | {e.source}" for e in entries],
    )


def ingest(
    entries: list[CaseEntry],
    encoder: Encoder,
    persist_dir: Path,
    collection_name: str,
    batch_size: int,
    require_image: bool,
    reset: bool,
    dry_run: bool,
) -> IngestStats:
    """Embed every entry and upsert it into the ChromaDB collection."""
    stats = IngestStats(total_entries=len(entries))

    entries, dropped = _filter_entries(entries, require_image)
    stats.filtered_no_image = dropped
    for e in entries:
        stats.diagnoses[e.diagnosis] = stats.diagnoses.get(e.diagnosis, 0) + 1

    if dry_run:
        logger.info("--dry-run set: skipping embedding and ChromaDB upsert.")
        return stats

    if not entries:
        logger.warning("No entries to ingest.")
        return stats

    import chromadb

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    if reset:
        try:
            client.delete_collection(collection_name)
            logger.info("Dropped existing collection '%s' (--reset).", collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(collection_name)

    t0 = time.perf_counter()
    for start in range(0, len(entries), batch_size):
        batch = entries[start : start + batch_size]
        try:
            embeddings = encoder.encode_batch([e.image_path for e in batch])
        except Exception as exc:
            stats.errors.append(f"Encode failed for batch {start}-{start+len(batch)}: {exc}")
            logger.error("Encode failed (batch %d-%d): %s", start, start + len(batch), exc)
            continue
        try:
            _upsert_batch(collection, batch, embeddings)
            stats.embedded += len(batch)
            stats.upserted += len(batch)
            stats.batches += 1
        except Exception as exc:
            stats.errors.append(f"Upsert failed for batch {start}-{start+len(batch)}: {exc}")
            logger.error("Upsert failed: %s", exc, exc_info=True)
            continue
        if (start // batch_size) % 10 == 0:
            elapsed = time.perf_counter() - t0
            rate = stats.embedded / elapsed if elapsed > 0 else 0
            logger.info(
                "  progress: %d/%d (%.1f img/s)",
                stats.embedded, len(entries), rate,
            )

    stats.elapsed_s = time.perf_counter() - t0
    final = collection.count()
    logger.info(
        "Ingest complete: embedded=%d upserted=%d collection_size=%d elapsed=%.1fs",
        stats.embedded, stats.upserted, final, stats.elapsed_s,
    )
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _print_report(stats: IngestStats, output_dir: Path, collection_name: str) -> None:
    print("\n" + "=" * 68)
    print(" Case RAG ingest report")
    print("=" * 68)
    print(f"  Source entries:     {stats.total_entries}")
    print(f"  Filtered (no img):  {stats.filtered_no_image}")
    print(f"  Embedded:           {stats.embedded}")
    print(f"  Upserted:           {stats.upserted}")
    print(f"  Batches:            {stats.batches}")
    if stats.elapsed_s:
        rate = stats.embedded / stats.elapsed_s
        print(f"  Throughput:         {rate:.1f} img/s")
        print(f"  Elapsed:            {stats.elapsed_s:.1f}s")
    if stats.diagnoses:
        print("\n  Diagnosis distribution:")
        for dx, n in sorted(stats.diagnoses.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {dx:<30} {n}")
        if len(stats.diagnoses) > 10:
            print(f"    ... ({len(stats.diagnoses) - 10} more)")
    if stats.errors:
        print(f"\n  ⚠ Errors: {len(stats.errors)}")
        for err in stats.errors[:5]:
            print(f"    - {err}")
    print(f"\n  ChromaDB:           {output_dir}/{collection_name}")
    print()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the Case RAG ChromaDB index for DermArbiter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        choices=sorted(SOURCE_LOADERS.keys()),
        required=True,
        help="Which manifest source to ingest from.",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Source data directory. Defaults: data/ham10000 for ham10000, "
            "data/derm1m for derm1m."
        ),
    )
    p.add_argument(
        "--manifest",
        help="Path to a CSV/JSONL manifest (for --source local).",
    )
    p.add_argument(
        "--derm1m-split",
        choices=["pretrain", "validation"],
        default=DERM1M_DEFAULT_SPLIT,
        help="Derm1M split to ingest (default: pretrain).",
    )
    p.add_argument(
        "--derm1m-stratify-by",
        default="disease_label",
        help="Column used for stratified sampling when --max-cases is set.",
    )
    p.add_argument("--persist-dir", default=DEFAULT_CHROMA_DIR)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    p.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument(
        "--require-image",
        action="store_true",
        default=True,
        help="Drop manifest rows whose image file is missing on disk (default on).",
    )
    p.add_argument(
        "--no-require-image",
        action="store_false",
        dest="require_image",
        help="Disable the missing-image filter (encoding will then crash on missing files).",
    )
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--reset", action="store_true", help="Drop the collection before ingesting.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and filter only; do not embed or write to ChromaDB.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _load_entries(args: argparse.Namespace) -> list[CaseEntry]:
    if args.source == "ham10000":
        data_dir = Path(args.data_dir or "data/ham10000")
        return load_ham10000_manifest(data_dir)
    if args.source == "local":
        if not args.manifest:
            raise SystemExit("--manifest is required for --source local")
        return load_local_manifest(Path(args.manifest))
    if args.source == "derm1m":
        data_dir = Path(args.data_dir or DERM1M_DEFAULT_DATA_DIR)
        return load_derm1m_manifest(
            data_dir=data_dir,
            split=args.derm1m_split,
            max_cases=args.max_cases,
            stratify_by=args.derm1m_stratify_by,
            download_images=not args.dry_run,
        )
    raise SystemExit(f"Unknown source: {args.source}")


def run(args: argparse.Namespace, encoder: Optional[Encoder] = None) -> int:
    entries = _load_entries(args)
    if args.max_cases:
        entries = entries[: args.max_cases]

    if encoder is None and not args.dry_run:
        encoder = DermLipEncoder(clip_model=args.clip_model, device=args.device)

    stats = ingest(
        entries=entries,
        encoder=encoder,   # type: ignore[arg-type]   (unused when dry_run)
        persist_dir=Path(args.persist_dir),
        collection_name=args.collection,
        batch_size=args.batch_size,
        require_image=args.require_image,
        reset=args.reset,
        dry_run=args.dry_run,
    )
    _print_report(stats, Path(args.persist_dir), args.collection)
    return 1 if stats.errors else 0


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        for noisy in ("urllib3", "chromadb", "open_clip", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
