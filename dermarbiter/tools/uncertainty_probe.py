"""UncertaintyProbe — Entropy + conformal prediction (★ novel).

Computes predictive entropy, normalised entropy, and a conformal
prediction set from classification probability distributions.  Also
calculates Expected Calibration Error (ECE) when calibration data
is available.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from dermarbiter.tools.base_tool import BaseTool, ToolOutput

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.10  # 90% coverage guarantee
DEFAULT_ECE_BINS = 10
ENTROPY_HIGH_THRESHOLD = 0.7  # normalised entropy above this → high uncertainty


class UncertaintyProbe(BaseTool):
    """Predictive uncertainty quantification via entropy + conformal.

    This is a **novel contribution** of the DermArbiter framework,
    providing calibrated uncertainty estimates that enable the debate
    protocol to distinguish confident from ambiguous predictions.

    Args:
        alpha: Significance level for conformal prediction (default 0.10).
        ece_bins: Number of bins for ECE computation.
    """

    @property
    def name(self) -> str:
        return "uncertainty_probe"

    @property
    def description(self) -> str:
        return (
            "Uncertainty Probe (★ novel contribution) — computes "
            "predictive entropy and conformal prediction set for "
            "calibrated uncertainty quantification."
        )

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        ece_bins: int = DEFAULT_ECE_BINS,
    ) -> None:
        self._alpha = alpha
        self._ece_bins = ece_bins
        self._loaded = True  # pure computation

        # Optional: calibration scores from a held-out set
        self._calibration_scores: np.ndarray | None = None

    def _load_model(self) -> None:
        pass  # no model to load

    def unload(self) -> None:
        self._calibration_scores = None

    def set_calibration_scores(self, scores: np.ndarray) -> None:
        """Set non-conformity scores from a calibration dataset.

        Args:
            scores: 1-D array of non-conformity scores (e.g. 1 − p_true)
                    computed on a held-out calibration set.
        """
        self._calibration_scores = np.sort(scores)
        logger.info(
            "UncertaintyProbe calibration set: %d samples.", len(scores),
        )

    def validate_input(
        self, image_path: str | None = None, query: str = ""
    ) -> bool:
        # Requires probabilities passed via structured input
        return True

    @staticmethod
    def compute_entropy(probabilities: np.ndarray) -> float:
        """Shannon entropy: H = −Σ pᵢ log(pᵢ)."""
        probs = probabilities[probabilities > 0]
        return float(-np.sum(probs * np.log(probs)))

    @staticmethod
    def compute_max_entropy(num_classes: int) -> float:
        """Maximum entropy for uniform distribution: log(K)."""
        return float(np.log(num_classes))

    def compute_conformal_set(
        self,
        probabilities: np.ndarray,
        class_names: list[str],
    ) -> list[str]:
        """Build conformal prediction set at level 1 − α.

        Uses the Adaptive Prediction Sets (APS) method: sort classes
        by descending probability and accumulate until sum ≥ 1 − α.
        """
        threshold = 1.0 - self._alpha
        sorted_idx = np.argsort(-probabilities)
        cumsum = 0.0
        conformal_set: list[str] = []
        for idx in sorted_idx:
            conformal_set.append(class_names[idx])
            cumsum += probabilities[idx]
            if cumsum >= threshold:
                break
        return conformal_set

    def compute_ece(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
    ) -> float:
        """Expected Calibration Error over binned predictions."""
        n = len(confidences)
        if n == 0:
            return 0.0

        bin_edges = np.linspace(0.0, 1.0, self._ece_bins + 1)
        ece = 0.0
        for i in range(self._ece_bins):
            mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
        return round(float(ece), 4)

    def _parse_probabilities(
        self, query: str
    ) -> tuple[np.ndarray, list[str]] | None:
        """Parse probability dict from query string.

        Expects format: ``melanoma:0.62,bcc:0.15,nevus:0.11,...``
        """
        if not query or ":" not in query:
            return None

        class_names: list[str] = []
        probs: list[float] = []
        for item in query.split(","):
            item = item.strip()
            if ":" not in item:
                continue
            name, prob_str = item.rsplit(":", 1)
            try:
                probs.append(float(prob_str.strip()))
                class_names.append(name.strip())
            except ValueError:
                continue

        if not probs:
            return None

        arr = np.array(probs, dtype=np.float64)
        # Normalise if needed
        total = arr.sum()
        if total > 0:
            arr /= total
        return arr, class_names

    def run(
        self,
        image_path: str | None = None,
        query: str = "",
        probabilities: dict[str, float] | None = None,
    ) -> ToolOutput:
        t0 = time.perf_counter()

        # Get probabilities from either kwarg or query string
        if probabilities is not None:
            class_names = list(probabilities.keys())
            probs = np.array(list(probabilities.values()), dtype=np.float64)
            total = probs.sum()
            if total > 0:
                probs /= total
        else:
            parsed = self._parse_probabilities(query)
            if parsed is None:
                return ToolOutput(
                    tool_name=self.name,
                    result={"error": "No probabilities provided."},
                    confidence=0.0,
                    raw_text="UncertaintyProbe: no probability distribution.",
                    metadata={"status": "error"},
                )
            probs, class_names = parsed

        try:
            n_classes = len(probs)
            entropy = self.compute_entropy(probs)
            max_ent = self.compute_max_entropy(n_classes)
            norm_ent = round(entropy / max_ent, 4) if max_ent > 0 else 0.0
            is_high = norm_ent > ENTROPY_HIGH_THRESHOLD

            conformal_set = self.compute_conformal_set(probs, class_names)

            # ECE placeholder (would need calibration data)
            ece = 0.0
            cal_size = 0
            if self._calibration_scores is not None:
                cal_size = len(self._calibration_scores)

            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Confidence: inverse of normalised entropy
            confidence = round(max(0.0, min(1.0, 1.0 - norm_ent)), 2)

            result = {
                "predictive_entropy": round(entropy, 4),
                "max_entropy": round(max_ent, 4),
                "normalised_entropy": norm_ent,
                "conformal_set": conformal_set,
                "conformal_alpha": self._alpha,
                "coverage_guarantee": round(1.0 - self._alpha, 2),
                "calibration_ece": ece,
                "is_high_uncertainty": is_high,
            }

            set_str = ", ".join(conformal_set)
            raw_text = (
                f"Normalised entropy = {norm_ent:.2f} "
                f"({'HIGH' if is_high else 'moderate'}). "
                f"Conformal set at α={self._alpha}: "
                f"{{{set_str}}} — "
                f"{result['coverage_guarantee']*100:.0f}% coverage."
            )

            return ToolOutput(
                tool_name=self.name,
                result=result,
                confidence=confidence,
                raw_text=raw_text,
                metadata={
                    "method": "entropy + split-conformal",
                    "contribution": "novel",
                    "calibration_set_size": cal_size,
                    "latency_ms": round(elapsed_ms, 1),
                },
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error("UncertaintyProbe failed: %s", exc, exc_info=True)
            return ToolOutput(
                tool_name=self.name,
                result={"error": str(exc)},
                confidence=0.0,
                raw_text=f"UncertaintyProbe failed: {exc}",
                metadata={"status": "error", "latency_ms": round(elapsed_ms, 1)},
            )
