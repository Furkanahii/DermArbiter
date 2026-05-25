"""MedGemma-4B VQA Tool — General medical visual question answering.

MedGemma-4B is Google's medical fine-tuned variant of Gemma, designed
for multimodal medical image understanding.  It provides a non-specialist
"second opinion" perspective on dermoscopic images.

The model is gated on HuggingFace and requires ``HF_TOKEN`` for access.
It supports both text-only and image+text (multimodal) inputs.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from dermarbiter.tools.base_tool import BaseTool, ToolOutput

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_MODEL_ID = "google/medgemma-4b-it"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_QUERY = "Describe this skin lesion and suggest possible diagnoses."

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class MedGemmaVQA(BaseTool):
    """MedGemma-4B general-purpose medical VQA.

    Provides a non-specialist clinical perspective on dermoscopic
    images, offering a broad differential diagnosis that can serve
    as an independent second opinion in the multi-agent framework.

    Args:
        model_id: HuggingFace model identifier.
        max_new_tokens: Maximum number of tokens to generate.
        device: Target device (``"auto"`` / ``"cuda"`` / ``"cpu"``).
        quantize_4bit: Whether to load with 4-bit quantisation.
    """

    @property
    def name(self) -> str:
        return "general_vqa"

    @property
    def description(self) -> str:
        return (
            "MedGemma-4B — general medical VQA providing a second, "
            "non-specialist opinion on dermatoscopic images."
        )

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        device: str = "auto",
        quantize_4bit: bool = True,
    ) -> None:
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._device_str = device
        self._quantize_4bit = quantize_4bit

        # Lazily initialised
        self._model: Any = None
        self._processor: Any = None
        self._device: Any = None
        self._loaded = False

    # -- Device ------------------------------------------------------------

    def _resolve_device(self) -> Any:
        import torch

        if self._device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self._device_str)

    # -- Model lifecycle ---------------------------------------------------

    def _load_model(self) -> None:
        """Load MedGemma-4B from HuggingFace."""
        if self._loaded:
            return

        import torch
        from transformers import AutoProcessor

        self._device = self._resolve_device()
        hf_token = os.environ.get("HF_TOKEN")

        logger.info(
            "Loading MedGemma-4B from %s on %s (4-bit=%s)",
            self._model_id,
            self._device,
            self._quantize_4bit,
        )

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "token": hf_token,
        }

        if self._quantize_4bit and self._device.type == "cuda":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = {"": self._device}

        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            trust_remote_code=True,
            token=hf_token,
        )

        # MedGemma-4B-it is a multimodal model (image + text → text). With
        # AutoModelForCausalLM the image branch is silently ignored, so the
        # model runs but returns an empty answer ("0 chars" in validation).
        # Try the image-text-to-text class first, then Vision2Seq, then fall
        # back to CausalLM for any text-only checkpoint variant.
        last_err: Exception | None = None
        self._model = None
        for AutoClsName in ("AutoModelForImageTextToText",
                            "AutoModelForVision2Seq",
                            "AutoModelForCausalLM"):
            try:
                import transformers as _t
                AutoCls = getattr(_t, AutoClsName)
                self._model = AutoCls.from_pretrained(self._model_id, **load_kwargs)
                logger.info("Loaded MedGemma-4B via %s", AutoClsName)
                break
            except (AttributeError, ValueError) as e:
                last_err = e
                continue
        if self._model is None:
            raise RuntimeError(
                f"Could not load {self._model_id} with any AutoModel class. "
                f"Last error: {last_err}"
            )
        self._model.eval()
        self._loaded = True

        logger.info("MedGemma-4B loaded successfully.")

    def unload(self) -> None:
        """Release model from GPU memory."""
        if self._model is not None:
            import torch

            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._loaded = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("MedGemma-4B unloaded.")

    # -- Validation --------------------------------------------------------

    def validate_input(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> bool:
        if image_path is not None:
            path = Path(image_path)
            if not path.exists() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                return False
        return True

    # -- Inference ---------------------------------------------------------

    def _build_chat_prompt(self, query: str, has_image: bool) -> list[dict]:
        """Build a chat-style prompt for MedGemma."""
        content: list[dict[str, str]] = []

        if has_image:
            content.append({"type": "image"})

        content.append({
            "type": "text",
            "text": query,
        })

        return [{"role": "user", "content": content}]

    @staticmethod
    def _to_device(inputs: Any, device: Any) -> Any:
        """Safely move processor outputs to device."""
        if hasattr(inputs, "to"):
            return inputs.to(device)
        return inputs

    def _run_inference(
        self, image_path: str | None, query: str
    ) -> dict[str, Any]:
        """Run VQA inference with MedGemma-4B."""
        import torch
        from PIL import Image

        effective_query = query or DEFAULT_QUERY
        has_image = image_path is not None and Path(image_path).exists()

        messages = self._build_chat_prompt(effective_query, has_image)

        # Apply chat template
        prompt_text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs
        if has_image:
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(
                text=prompt_text, images=image, return_tensors="pt",
            )
        else:
            inputs = self._processor(
                text=prompt_text, return_tensors="pt",
            )

        inputs = self._to_device(inputs, self._model.device)

        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        # Decode response only (skip input tokens)
        input_len = inputs["input_ids"].shape[-1]
        answer = self._processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
        ).strip()

        return {
            "question": effective_query,
            "answer": answer,
            "model_version": "medgemma-4b-v1",
        }

    # -- Public API --------------------------------------------------------

    def run(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> ToolOutput:
        t0 = time.perf_counter()

        if not self.validate_input(image_path, query):
            return ToolOutput(
                tool_name=self.name,
                result={"error": f"Invalid input: image={image_path}"},
                confidence=0.0,
                raw_text="MedGemma: invalid input.",
                metadata={"status": "error"},
            )

        try:
            self._load_model()
            result = self._run_inference(image_path, query)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            answer = result["answer"]
            confidence = self._estimate_confidence(answer)

            summary = answer[:150].rstrip()
            if len(answer) > 150:
                summary += "..."

            return ToolOutput(
                tool_name=self.name,
                result=result,
                confidence=confidence,
                raw_text=f"General VQA: {summary}",
                metadata={
                    "model": "MedGemma-4B",
                    "source": "HuggingFace (gated)",
                    "latency_ms": round(elapsed_ms, 1),
                },
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error("MedGemma inference failed: %s", exc, exc_info=True)
            return ToolOutput(
                tool_name=self.name,
                result={"error": str(exc)},
                confidence=0.0,
                raw_text=f"MedGemma inference failed: {exc}",
                metadata={"status": "error", "latency_ms": round(elapsed_ms, 1)},
            )

    @staticmethod
    def _estimate_confidence(answer: str) -> float:
        """Heuristic confidence from answer text.

        Non-specialist baseline: slightly lower than specialist models.
        """
        if not answer:
            return 0.25

        score = 0.4

        if len(answer) > 80:
            score += 0.1
        if len(answer) > 200:
            score += 0.05

        medical_terms = [
            "lesion", "pigment", "border", "color", "melanoma",
            "carcinoma", "nevus", "differential", "diagnosis",
            "dermoscop", "biopsy", "asymmetr",
        ]
        matches = sum(1 for t in medical_terms if t.lower() in answer.lower())
        score += min(matches * 0.04, 0.2)

        return min(round(score, 2), 1.0)
