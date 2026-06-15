"""Standard clinical code mappings for dermatological diagnoses.

Maps the canonical HAM10000 7-class labels (and common free-text synonyms)
to their ICD-10 and SNOMED-CT codes, plus a malignancy flag and the
default clinical management tier. Used by:

  * ``scripts/build_dermabench.py`` — to enrich gold-standard cases with
    reference codes during DermAbench curation.
  * ``dermarbiter/evaluation/dermabench.py`` — Dimension 3 (Coding) scores
    predicted codes against these references.

These are reference mappings for benchmark ground truth, NOT injected into
the model at inference time (that would be leakage). The model must produce
codes from its own reasoning; we score against this table afterwards.

Code sources: ICD-10-CM (2024) and SNOMED-CT International Edition. Where a
class spans multiple specific entities (e.g. bkl = seborrheic keratosis +
solar lentigo + lichen-planus-like keratosis), the most representative
code is used.
"""

from __future__ import annotations

from typing import Optional


# ── Per-class reference record ──────────────────────────────────────────────
# Each HAM10000 class → (icd10, snomed, is_malignant, default_management)
# management tiers: "biopsy" (excise/sample), "monitor" (follow-up), "reassure"
_CLASS_CODES: dict[str, dict[str, object]] = {
    "nv": {
        "name": "melanocytic nevus",
        "icd10": "D22.9",        # Melanocytic nevi, unspecified
        "snomed": "400010006",   # Melanocytic naevus
        "is_malignant": False,
        "management": "reassure",
    },
    "mel": {
        "name": "melanoma",
        "icd10": "C43.9",        # Malignant melanoma of skin, unspecified
        "snomed": "372244006",   # Malignant melanoma
        "is_malignant": True,
        "management": "biopsy",
    },
    "bkl": {
        "name": "benign keratosis-like lesion",
        "icd10": "L82.1",        # Other seborrheic keratosis
        "snomed": "65126008",    # Seborrheic keratosis
        "is_malignant": False,
        "management": "reassure",
    },
    "bcc": {
        "name": "basal cell carcinoma",
        "icd10": "C44.91",       # Basal cell carcinoma of skin, unspecified
        "snomed": "254701007",   # Basal cell carcinoma of skin
        "is_malignant": True,
        "management": "biopsy",
    },
    "akiec": {
        "name": "actinic keratosis / intraepithelial carcinoma",
        "icd10": "L57.0",        # Actinic keratosis
        "snomed": "201101007",   # Actinic keratosis
        "is_malignant": False,   # pre-malignant; flagged for biopsy regardless
        "management": "biopsy",
    },
    "df": {
        "name": "dermatofibroma",
        "icd10": "D23.9",        # Other benign neoplasm of skin, unspecified
        "snomed": "253008",      # Dermatofibroma
        "is_malignant": False,
        "management": "reassure",
    },
    "vasc": {
        "name": "vascular lesion",
        "icd10": "D18.01",       # Hemangioma of skin and subcutaneous tissue
        "snomed": "400210000",   # Haemangioma of skin
        "is_malignant": False,
        "management": "monitor",
    },
}


# ── Free-text synonym → canonical HAM10000 code ─────────────────────────────
# Mirrors the label normalisation used in run_dermagent_subset so the agent
# layer's free-text output maps to the same 7-class space the codes index.
_SYNONYM_TO_CLASS: dict[str, str] = {
    # nv
    "melanocytic_nevus": "nv", "melanocytic nevus": "nv", "nevus": "nv",
    "compound_nevus": "nv", "atypical_nevus": "nv", "atypical nevus": "nv",
    "atypical melanocytic nevus": "nv", "dysplastic nevus": "nv",
    # mel
    "melanoma": "mel", "malignant_melanoma": "mel", "malignant melanoma": "mel",
    # bkl
    "seborrheic_keratosis": "bkl", "seborrheic keratosis": "bkl",
    "irritated seborrheic keratosis": "bkl", "solar_lentigo": "bkl",
    "solar lentigo": "bkl", "lichenoid keratosis": "bkl",
    "benign_keratosis": "bkl", "benign keratosis": "bkl",
    # bcc
    "basal_cell_carcinoma": "bcc", "basal cell carcinoma": "bcc",
    "pigmented basal cell carcinoma": "bcc",
    # akiec
    "actinic_keratosis": "akiec", "actinic keratosis": "akiec",
    "squamous_cell_carcinoma": "akiec", "squamous cell carcinoma": "akiec",
    "bowen disease": "akiec", "intraepithelial carcinoma": "akiec",
    # df
    "dermatofibroma": "df",
    # vasc
    "vascular_lesion": "vasc", "vascular lesion": "vasc",
    "hemangioma": "vasc", "angioma": "vasc", "pyogenic granuloma": "vasc",
    # already-coded passthrough
    "nv": "nv", "mel": "mel", "bkl": "bkl", "bcc": "bcc",
    "akiec": "akiec", "df": "df", "vasc": "vasc",
}


def normalize_to_class(raw: str) -> str:
    """Map a free-text or coded diagnosis to a canonical HAM10000 class.

    Case-insensitive, hyphen-tolerant. Returns the lowercased raw value
    when no mapping exists (keeps unknowns visible rather than silently
    forcing a class).
    """
    if not raw:
        return ""
    key = raw.strip().lower().replace("-", "_")
    if key in _SYNONYM_TO_CLASS:
        return _SYNONYM_TO_CLASS[key]
    # try the spaced variant too
    spaced = raw.strip().lower()
    return _SYNONYM_TO_CLASS.get(spaced, key)


def icd10_for(diagnosis: str) -> Optional[str]:
    """Reference ICD-10 code for a diagnosis (free-text or class), or None."""
    cls = normalize_to_class(diagnosis)
    rec = _CLASS_CODES.get(cls)
    return rec["icd10"] if rec else None  # type: ignore[return-value]


def snomed_for(diagnosis: str) -> Optional[str]:
    """Reference SNOMED-CT code for a diagnosis, or None."""
    cls = normalize_to_class(diagnosis)
    rec = _CLASS_CODES.get(cls)
    return rec["snomed"] if rec else None  # type: ignore[return-value]


def is_malignant(diagnosis: str) -> bool:
    """Whether a diagnosis is malignant (mel, bcc). akiec is pre-malignant
    and returns False here but carries a 'biopsy' management default."""
    cls = normalize_to_class(diagnosis)
    rec = _CLASS_CODES.get(cls)
    return bool(rec["is_malignant"]) if rec else False


def default_management(diagnosis: str) -> Optional[str]:
    """Default clinical management tier: biopsy / monitor / reassure."""
    cls = normalize_to_class(diagnosis)
    rec = _CLASS_CODES.get(cls)
    return rec["management"] if rec else None  # type: ignore[return-value]


def reference_record(diagnosis: str) -> dict[str, object]:
    """Full reference enrichment for a diagnosis — used by the DermAbench
    builder to populate a gold case's ground_truth block.

    Returns a dict with class, icd10, snomed, is_malignant, management.
    Empty dict if the diagnosis can't be mapped.
    """
    cls = normalize_to_class(diagnosis)
    rec = _CLASS_CODES.get(cls)
    if not rec:
        return {}
    return {
        "diagnosis_class": cls,
        "icd10_code": rec["icd10"],
        "snomed_code": rec["snomed"],
        "is_malignant": rec["is_malignant"],
        "management": rec["management"],
    }


def all_classes() -> list[str]:
    """The 7 canonical HAM10000 class codes."""
    return list(_CLASS_CODES.keys())
