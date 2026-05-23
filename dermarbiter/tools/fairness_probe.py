"""FairnessProbe — Fitzpatrick skin type + ITA estimation (★ novel).

Estimates Individual Typology Angle (ITA) from the CIELab colour
space of a dermoscopic image and maps it to Fitzpatrick skin type.
Generates bias warnings for under-represented skin tones to enable
fairness-aware diagnostic pipelines.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np

from dermarbiter.tools.base_tool import BaseTool, ToolOutput

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ITA → Fitzpatrick mapping thresholds (published)
ITA_THRESHOLDS: list[tuple[float, str, str]] = [
    (55.0, "I", "very_light"),
    (41.0, "II", "light"),
    (28.0, "III", "intermediate"),
    (10.0, "IV", "tan"),
    (-30.0, "V", "brown"),
    (float("-inf"), "VI", "dark"),
]


def _ita_to_fitzpatrick(ita: float) -> tuple[str, str]:
    """Map ITA angle to Fitzpatrick type and category label."""
    for threshold, fitz_type, category in ITA_THRESHOLDS:
        if ita > threshold:
            return fitz_type, category
    return "VI", "dark"


class FairnessProbe(BaseTool):
    """Estimates Fitzpatrick skin type and ITA from dermoscopic images.

    This is a **novel contribution** of the DermArbiter framework,
    enabling bias-aware diagnosis by detecting the patient's skin tone
    directly from the lesion image.

    The ITA is computed as::

        ITA = arctan((L* − 50) / b*) × (180 / π)

    where *L** and *b** are CIELab colour-space values of the
    surrounding skin region (excluding the lesion itself).
    """

    @property
    def name(self) -> str:
        return "fairness_probe"

    @property
    def description(self) -> str:
        return (
            "Fairness Probe (★ novel contribution) — estimates Fitzpatrick "
            "skin type and Individual Typology Angle (ITA) from dermoscopic "
            "images for bias-aware diagnosis."
        )

    def __init__(self, border_fraction: float = 0.15) -> None:
        self._border_fraction = border_fraction
        self._loaded = True  # no model to load

    def _load_model(self) -> None:
        pass  # pure computation — no model

    def unload(self) -> None:
        pass

    def validate_input(self, image_path: str | None = None, query: str = "") -> bool:
        if image_path is None:
            return False
        path = Path(image_path)
        return path.exists() and path.suffix.lower() in SUPPORTED_EXTENSIONS

    def _extract_skin_region(self, image_path: str) -> np.ndarray:
        """Extract border pixels as a proxy for surrounding skin."""
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        h, w = arr.shape[:2]
        bh = max(1, int(h * self._border_fraction))
        bw = max(1, int(w * self._border_fraction))

        # Collect border pixels: top, bottom, left, right strips
        regions = [
            arr[:bh, :, :],      # top
            arr[-bh:, :, :],     # bottom
            arr[:, :bw, :],      # left
            arr[:, -bw:, :],     # right
        ]
        pixels = np.concatenate([r.reshape(-1, 3) for r in regions], axis=0)
        return pixels

    def _compute_ita(self, rgb_pixels: np.ndarray) -> tuple[float, list[int]]:
        """Compute ITA angle from RGB pixel array.

        Uses pure-numpy sRGB → XYZ → CIELab conversion (no skimage).
        """
        # Mean RGB of skin region
        mean_rgb = rgb_pixels.mean(axis=0).astype(np.float64)
        skin_rgb = [int(c) for c in mean_rgb.round()]

        # sRGB → linear RGB
        srgb = mean_rgb / 255.0
        linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)

        # Linear RGB → XYZ (D65 illuminant)
        x = linear[0] * 0.4124564 + linear[1] * 0.3575761 + linear[2] * 0.1804375
        y = linear[0] * 0.2126729 + linear[1] * 0.7151522 + linear[2] * 0.0721750
        z = linear[0] * 0.0193339 + linear[1] * 0.1191920 + linear[2] * 0.9503041

        # D65 reference white
        xn, yn, zn = 0.95047, 1.00000, 1.08883

        def _f(t: float) -> float:
            delta = 6.0 / 29.0
            if t > delta ** 3:
                return t ** (1.0 / 3.0)
            return t / (3.0 * delta * delta) + 4.0 / 29.0

        _fx, fy, fz = _f(x / xn), _f(y / yn), _f(z / zn)
        l_star = 116.0 * fy - 16.0
        b_star = 200.0 * (fy - fz)

        # ITA = arctan((L* - 50) / b*) × (180/π)
        if abs(b_star) < 1e-6:
            ita = 90.0 if l_star > 50 else -90.0
        else:
            ita = math.atan2(l_star - 50.0, b_star) * (180.0 / math.pi)

        return round(ita, 1), skin_rgb

    def _estimate_confidence(self, rgb_pixels: np.ndarray) -> float:
        """Estimate confidence based on colour variance in skin region.

        Low variance → high confidence (uniform skin colour).
        High variance → low confidence (artefacts, hair, etc.).
        """
        std_per_channel = rgb_pixels.std(axis=0).mean()
        # Normalise: std=0 → conf=0.95, std=80+ → conf=0.40
        conf = max(0.40, 0.95 - std_per_channel / 200.0)
        return round(conf, 2)

    def run(self, image_path: str | None = None, query: str = "") -> ToolOutput:
        t0 = time.perf_counter()

        if not self.validate_input(image_path):
            return ToolOutput(
                tool_name=self.name,
                result={"error": f"Invalid or missing image: {image_path}"},
                confidence=0.0,
                raw_text="FairnessProbe: invalid input.",
                metadata={"status": "error"},
            )

        try:
            skin_pixels = self._extract_skin_region(image_path)
            ita_angle, skin_rgb = self._compute_ita(skin_pixels)
            fitz_type, ita_category = _ita_to_fitzpatrick(ita_angle)
            confidence = self._estimate_confidence(skin_pixels)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Bias warning for under-represented types
            bias_warning = None
            if fitz_type in ("V", "VI"):
                bias_warning = (
                    f"Fitzpatrick type {fitz_type} detected. "
                    "Classifier performance may be reduced for darker "
                    "skin tones due to training data imbalance."
                )

            calibration_note = (
                "Classifier performance may vary for Fitzpatrick "
                "types V–VI. Monitor per-subgroup metrics."
            )

            result = {
                "fitzpatrick_type": fitz_type,
                "fitzpatrick_confidence": confidence,
                "ita_angle": ita_angle,
                "ita_category": ita_category,
                "skin_tone_rgb": skin_rgb,
                "bias_warning": bias_warning,
                "calibration_note": calibration_note,
            }

            raw_text = (
                f"Fitzpatrick type {fitz_type} (confidence {confidence}), "
                f"ITA = {ita_angle}° ({ita_category})."
            )
            if bias_warning:
                raw_text += f" ⚠ {bias_warning}"

            return ToolOutput(
                tool_name=self.name,
                result=result,
                confidence=confidence,
                raw_text=raw_text,
                metadata={
                    "model": "ITA-estimator-v1",
                    "contribution": "novel",
                    "latency_ms": round(elapsed_ms, 1),
                },
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error("FairnessProbe failed: %s", exc, exc_info=True)
            return ToolOutput(
                tool_name=self.name,
                result={"error": str(exc)},
                confidence=0.0,
                raw_text=f"FairnessProbe failed: {exc}",
                metadata={"status": "error", "latency_ms": round(elapsed_ms, 1)},
            )
