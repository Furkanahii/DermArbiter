"""Build the ChromaDB index consumed by ``dermarbiter.tools.guideline_rag.GuidelineRAG``.

The ``GuidelineRAG`` tool itself only *queries* a pre-built ChromaDB collection;
this script is what actually *fills* that collection with clinical-guideline
text chunks from DermNet NZ, Mayo Clinic, or any user-supplied source.

Three ingest modes are supported:

    local   Read documents from a local JSONL file (offline; no network required).
    urls    Fetch a curated list of URLs (YAML or JSON list) and parse the main
            article body with BeautifulSoup.
    dermnet Convenience wrapper over ``urls`` using a small built-in DermNet seed.

Each input document is split into overlapping character windows, embedded with
``sentence-transformers/all-MiniLM-L6-v2`` (CPU-friendly, ~80 MB), and upserted
into ChromaDB. Chunk IDs are deterministic SHA-1 hashes of ``source|text`` so
the script is idempotent — re-running on the same input does not duplicate
records.

Example
-------
Offline ingest of the bundled seed dataset::

    python scripts/build_guideline_rag_index.py \\
        --source local \\
        --input data/guideline_seed/guidelines.jsonl

Inspect the resulting collection::

    python -c "import chromadb; c = chromadb.PersistentClient('data/chroma_guidelines'); \\
               print(c.get_collection('clinical_guidelines').count())"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# -- Constants -----------------------------------------------------------------

DEFAULT_PERSIST_DIR = "data/chroma_guidelines"
DEFAULT_COLLECTION = "clinical_guidelines"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128
DEFAULT_BATCH_SIZE = 32

USER_AGENT = (
    "DermArbiter-GuidelineIngest/0.1 (academic research; "
    "https://github.com/dermarbiter; contact via repo issues)"
)

# A small built-in DermNet seed (used by ``--source dermnet`` when no
# --urls-file is provided). Keep it short to respect rate limits.
DERMNET_SEED_URLS = [
    "https://dermnetnz.org/topics/melanoma",
    "https://dermnetnz.org/topics/basal-cell-carcinoma",
    "https://dermnetnz.org/topics/squamous-cell-carcinoma",
    "https://dermnetnz.org/topics/actinic-keratosis",
    "https://dermnetnz.org/topics/melanocytic-naevus",
    "https://dermnetnz.org/topics/seborrhoeic-keratosis",
    "https://dermnetnz.org/topics/psoriasis",
    "https://dermnetnz.org/topics/atopic-dermatitis",
    "https://dermnetnz.org/topics/rosacea",
    "https://dermnetnz.org/topics/vitiligo",
]

logger = logging.getLogger("guideline_rag_ingest")


# -- Data models ---------------------------------------------------------------


@dataclass
class GuidelineDocument:
    """One clinical-guideline document (pre-chunking)."""

    source: str          # Stable provenance label, e.g. "DermNet:Melanoma".
    text: str            # Full article body.
    url: str = ""        # Optional canonical URL.
    title: str = ""      # Optional human-readable title.
    topic: str = ""      # Optional condition/topic key.

    def metadata(self) -> dict[str, str]:
        """Return the per-chunk metadata payload stored in ChromaDB."""
        return {
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "topic": self.topic,
        }


@dataclass
class Chunk:
    """One indexable chunk of a guideline document."""

    chunk_id: str
    text: str
    metadata: dict[str, str]


# -- Chunking ------------------------------------------------------------------


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split ``text`` into overlapping character windows aligned to sentence breaks.

    Strategy:
        1. Greedily accumulate sentences until the next sentence would push the
           window past ``chunk_size`` characters.
        2. Emit the current window, then start the next one carrying back enough
           trailing characters to satisfy ``chunk_overlap``.

    This keeps each chunk roughly self-contained and avoids cutting mid-sentence.
    """
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    sentences = _SENTENCE_SPLIT.split(text)
    chunks: list[str] = []
    buffer = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if not buffer:
            buffer = sent
            continue
        candidate = f"{buffer} {sent}"
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue

        chunks.append(buffer)
        # Carry over the tail of the previous buffer for overlap continuity.
        tail = buffer[-chunk_overlap:] if chunk_overlap else ""
        buffer = f"{tail} {sent}".strip() if tail else sent

    if buffer:
        chunks.append(buffer)

    # Defensive: if a single sentence exceeded chunk_size, fall back to a hard
    # character window for that one item so we still emit usable chunks.
    fixed: list[str] = []
    for c in chunks:
        if len(c) <= chunk_size:
            fixed.append(c)
            continue
        step = max(1, chunk_size - chunk_overlap)
        for start in range(0, len(c), step):
            fixed.append(c[start : start + chunk_size])
            if start + chunk_size >= len(c):
                break
    return fixed


def chunk_document(
    doc: GuidelineDocument,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Chunk a single document and produce stable, deterministic chunk IDs."""
    out: list[Chunk] = []
    meta_base = doc.metadata()
    for piece in chunk_text(doc.text, chunk_size, chunk_overlap):
        seed = f"{doc.source}|{piece}".encode("utf-8")
        cid = hashlib.sha1(seed).hexdigest()[:16]
        out.append(Chunk(chunk_id=cid, text=piece, metadata=meta_base))
    return out


# -- Source loaders ------------------------------------------------------------


def load_local_jsonl(path: Path) -> list[GuidelineDocument]:
    """Read documents from a JSONL file with one document per line.

    Expected schema per line::

        {"source": "DermNet:Melanoma", "text": "...", "url": "...", "title": "...", "topic": "..."}

    Only ``source`` and ``text`` are required.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    docs: list[GuidelineDocument] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d: %s", line_num, exc)
                continue
            if not obj.get("source") or not obj.get("text"):
                logger.warning(
                    "Skipping line %d: missing required 'source' or 'text' field.",
                    line_num,
                )
                continue
            docs.append(
                GuidelineDocument(
                    source=str(obj["source"]),
                    text=str(obj["text"]),
                    url=str(obj.get("url", "")),
                    title=str(obj.get("title", "")),
                    topic=str(obj.get("topic", "")),
                )
            )
    logger.info("Loaded %d documents from %s", len(docs), path)
    return docs


def _parse_html_article(html: str) -> tuple[str, str]:
    """Return ``(title, body_text)`` from an HTML page.

    Strategy: prefer the first ``<main>`` or ``<article>`` block; otherwise
    fall back to concatenated ``<p>`` text under ``<body>``. Scripts, styles,
    navigation, headers, and footers are stripped first.
    """
    from bs4 import BeautifulSoup  # lazy import; only needed for web modes

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    main = soup.find("main") or soup.find("article")
    if main is not None:
        text = main.get_text(separator=" ", strip=True)
    else:
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)

    # Collapse runs of whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return title, text


def fetch_url(url: str, timeout: float = 20.0) -> GuidelineDocument:
    """Fetch an HTML page and extract a ``GuidelineDocument``.

    Uses ``requests`` with a descriptive User-Agent. Raises on HTTP errors so
    the caller can skip and continue.
    """
    import requests  # lazy import

    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    title, text = _parse_html_article(resp.text)
    if not text:
        raise ValueError(f"Empty body extracted from {url}")

    # Derive a stable source label from the URL path tail.
    host = url.split("/")[2] if "://" in url else "url"
    slug = url.rstrip("/").split("/")[-1] or "page"
    source_label = f"{host}:{slug}"

    return GuidelineDocument(
        source=source_label,
        text=text,
        url=url,
        title=title,
        topic=slug.replace("-", " "),
    )


def load_urls(urls: Iterable[str], delay_s: float = 1.0) -> list[GuidelineDocument]:
    """Fetch each URL in sequence with a polite delay between requests."""
    docs: list[GuidelineDocument] = []
    for i, url in enumerate(urls):
        try:
            logger.info("[%d] Fetching %s", i + 1, url)
            doc = fetch_url(url)
            docs.append(doc)
        except Exception as exc:
            logger.warning("Skipped %s: %s", url, exc)
        if delay_s > 0:
            time.sleep(delay_s)
    logger.info("Fetched %d / %d URLs successfully", len(docs), sum(1 for _ in urls))
    return docs


def read_urls_file(path: Path) -> list[str]:
    """Read a URL list from a YAML or JSON file (either a flat list or {urls: [...]})."""
    if not path.exists():
        raise FileNotFoundError(f"URLs file not found: {path}")
    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml
        data = yaml.safe_load(raw_text)
    else:
        data = json.loads(raw_text)

    if isinstance(data, dict) and "urls" in data:
        data = data["urls"]
    if not isinstance(data, list):
        raise ValueError(
            f"{path} must contain a list of URLs or {{'urls': [...]}}, got {type(data).__name__}"
        )
    return [str(u).strip() for u in data if str(u).strip()]


# -- ChromaDB upsert -----------------------------------------------------------


def upsert_chunks(
    persist_dir: Path,
    collection_name: str,
    chunks: list[Chunk],
    embedding_model: str,
    batch_size: int,
    reset: bool,
) -> int:
    """Embed ``chunks`` and upsert them into the ChromaDB collection.

    Returns the post-upsert collection size.
    """
    if not chunks:
        logger.warning("No chunks to upsert; nothing to do.")
        return 0

    import chromadb
    from sentence_transformers import SentenceTransformer

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    if reset:
        try:
            client.delete_collection(collection_name)
            logger.info("Dropped existing collection '%s' (--reset).", collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(collection_name)

    logger.info("Loading embedding model '%s'", embedding_model)
    encoder = SentenceTransformer(embedding_model)

    texts = [c.text for c in chunks]
    logger.info("Embedding %d chunks (batch_size=%d)", len(texts), batch_size)
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()

    ids = [c.chunk_id for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # ChromaDB's add() errors on duplicate IDs; upsert() is the idempotent path.
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    final_count = collection.count()
    logger.info("Upserted %d chunks. Collection now holds %d records.", len(chunks), final_count)
    return final_count


# -- Pipeline ------------------------------------------------------------------


def _summarise(docs: list[GuidelineDocument], chunks: list[Chunk]) -> None:
    sources = sorted({d.source for d in docs})
    logger.info("=" * 60)
    logger.info("Ingest summary")
    logger.info("=" * 60)
    logger.info("Documents:        %d", len(docs))
    logger.info("Chunks produced:  %d", len(chunks))
    logger.info("Unique sources:   %d", len(sources))
    if chunks:
        avg_len = sum(len(c.text) for c in chunks) / len(chunks)
        logger.info("Avg chunk length: %.0f chars", avg_len)
    for s in sources[:10]:
        logger.info("  - %s", s)
    if len(sources) > 10:
        logger.info("  ... and %d more", len(sources) - 10)


def run_pipeline(args: argparse.Namespace) -> int:
    # 1. Load documents from the selected source.
    if args.source == "local":
        if not args.input:
            raise SystemExit("--input is required for --source local")
        docs = load_local_jsonl(Path(args.input))
    elif args.source == "urls":
        if not args.urls_file:
            raise SystemExit("--urls-file is required for --source urls")
        urls = read_urls_file(Path(args.urls_file))
        docs = load_urls(urls, delay_s=args.delay)
    elif args.source == "dermnet":
        urls = read_urls_file(Path(args.urls_file)) if args.urls_file else DERMNET_SEED_URLS
        docs = load_urls(urls, delay_s=args.delay)
    else:
        raise SystemExit(f"Unknown source: {args.source}")

    if not docs:
        logger.error("No documents loaded. Aborting.")
        return 1

    # 2. Chunk every document.
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, args.chunk_size, args.chunk_overlap))

    _summarise(docs, chunks)

    if args.dry_run:
        logger.info("--dry-run set: skipping embedding and ChromaDB upsert.")
        return 0

    # 3. Embed and upsert.
    upsert_chunks(
        persist_dir=Path(args.persist_dir),
        collection_name=args.collection,
        chunks=chunks,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        reset=args.reset,
    )
    return 0


# -- CLI -----------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the Guideline RAG ChromaDB index for DermArbiter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        choices=["local", "urls", "dermnet"],
        default="local",
        help="Where to load guideline documents from.",
    )
    p.add_argument("--input", help="Path to a JSONL file (required for --source local).")
    p.add_argument(
        "--urls-file",
        help="Path to YAML/JSON file with URL list (for --source urls; optional for dermnet).",
    )
    p.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between HTTP requests when fetching URLs.",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Drop the existing collection before ingesting (default: idempotent upsert).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk only; do not embed or write to ChromaDB.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down some noisy third-party loggers in verbose mode.
    if not args.verbose:
        for noisy in ("urllib3", "chromadb", "sentence_transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        return run_pipeline(args)
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
