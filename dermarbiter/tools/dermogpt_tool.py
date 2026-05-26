"""DermoGPT-RL VQA Tool — Dermatology visual question answering.

DermoGPT-RL is a dermatology-specialised VQA model fine-tuned via
SFT (Supervised Fine-Tuning) and MAVIC (Multi-Agent Verification
with Iterative Correction) reinforcement learning.  It is hosted
as a gated model on HuggingFace and requires an HF_TOKEN.

The model takes a dermoscopic image and a natural-language question,
producing a free-text clinical answer with diagnostic reasoning.
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
DEFAULT_MODEL_ID = "mendicant04/DermoGPT-RL"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_QUERY = "What is the most likely diagnosis for this skin lesion?"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class DermoGPTVQA(BaseTool):
    """DermoGPT-RL dermatology VQA model.

    Built on Qwen3-VL-8B-Instruct, fine-tuned with SFT on DermoInstruct
    (211K images, 770K trajectories) and MAVIC RL for morphologically-
    anchored visual-inference-consistent reasoning.

    Loads the gated HuggingFace model with optional 4-bit quantisation
    for Colab T4 compatibility.  Requires ``HF_TOKEN`` environment
    variable for gated model access.

    Args:
        model_id: HuggingFace model identifier.
        max_new_tokens: Maximum number of tokens to generate.
        device: Target device (``"auto"`` / ``"cuda"`` / ``"cpu"``).
        quantize_4bit: Whether to load with 4-bit quantisation.
    """

    @property
    def name(self) -> str:
        return "dermogpt_vqa"

    @property
    def description(self) -> str:
        return (
            "DermoGPT-RL — dermatology-specialised VQA model fine-tuned "
            "with SFT + MAVIC reinforcement learning on Qwen3-VL-8B."
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

    @classmethod
    def from_tool_config(cls, tool_config: Any) -> "DermoGPTVQA":
        """Construct from a ``ToolConfig`` instance.

        Reads model settings from the config's ``extra`` dict, falling
        back to class defaults when keys are absent.  This is the
        preferred construction path when using the DermArbiter config
        system (``tools.yaml``).

        Expected ``extra`` keys (all optional)::

            model_id:        "mendicant04/DermoGPT-RL"
            max_new_tokens:  256
            device:          "auto"
            quantize_4bit:   true

        Args:
            tool_config: A ``ToolConfig`` (or any object with an
                ``extra`` dict attribute).

        Returns:
            A configured ``DermoGPTVQA`` instance.
        """
        extra = getattr(tool_config, "extra", {})
        return cls(
            model_id=extra.get("model_id", DEFAULT_MODEL_ID),
            max_new_tokens=int(extra.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)),
            device=extra.get("device", "auto"),
            quantize_4bit=bool(extra.get("quantize_4bit", True)),
        )

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
        """Load the DermoGPT-RL model from HuggingFace."""
        if self._loaded:
            return

        import torch
        from transformers import AutoProcessor

        self._device = self._resolve_device()
        hf_token = os.environ.get("HF_TOKEN")

        logger.info(
            "Loading DermoGPT-RL from %s on %s (4-bit=%s)",
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

        # DermoGPT-RL is a vision-language model built on Qwen3-VL — must be
        # loaded via an image-text-to-text AutoModel. AutoModelForCausalLM
        # does not register Qwen3VLConfig and crashes with
        # "Unrecognized configuration class Qwen3VLConfig".
        #
        # We try the newest API first (transformers ≥ 4.45), fall back to
        # AutoModelForVision2Seq (4.39+), and finally CausalLM for text-only
        # variants. ``trust_remote_code=True`` means the model's own python
        # wrapper takes over when these fail too.
        last_err: Exception | None = None
        self._model = None
        for AutoClsName in ("AutoModelForImageTextToText",
                            "AutoModelForVision2Seq",
                            "AutoModelForCausalLM"):
            try:
                import transformers as _t
                AutoCls = getattr(_t, AutoClsName)
                self._model = AutoCls.from_pretrained(self._model_id, **load_kwargs)
                logger.info("Loaded DermoGPT-RL via %s", AutoClsName)
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

        logger.info("DermoGPT-RL loaded successfully.")

    def unload(self) -> None:
        """Free GPU memory by unloading all model weights.

        Deletes the model, processor, and device references, then
        forces Python garbage collection and clears the CUDA cache.
        The model will be re-loaded on the next ``run()`` call.
        """
        import gc

        for attr in ("_model", "_processor"):
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
        self._model = None
        self._processor = None
        self._device = None
        self._loaded = False

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info(
            "Unloaded %s (~7 GB) to free GPU memory.", self.name,
        )

    # -- Validation --------------------------------------------------------

    def validate_input(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> bool:
        # DermoGPT can work with just text, but image is preferred
        if image_path is not None:
            path = Path(image_path)
            if not path.exists() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                return False
        return True

    # -- Inference ---------------------------------------------------------

    def _build_messages(
        self, query: str, image_path: str | None
    ) -> list[dict]:
        """Build Qwen3-VL chat-template messages."""
        content: list[dict[str, str]] = []

        if image_path and Path(image_path).exists():
            content.append({"type": "image"})

        content.append({"type": "text", "text": query})
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
        """Run VQA inference using chat template."""
        import torch
        from PIL import Image

        effective_query = query or DEFAULT_QUERY

        # Build chat messages and apply template
        messages = self._build_messages(effective_query, image_path)
        prompt_text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(
                text=prompt_text, images=image, return_tensors="pt",
            )
        else:
            inputs = self._processor(
                text=prompt_text, return_tensors="pt",
            )

        inputs = self._to_device(inputs, self._device)

        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        # Decode — skip input tokens
        input_len = inputs["input_ids"].shape[-1]
        answer = self._processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
        ).strip()

        return {
            "question": effective_query,
            "answer": answer,
            "model_version": "dermogpt-rl-v1",
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
                raw_text="DermoGPT: invalid input.",
                metadata={"status": "error"},
            )

        try:
            self._load_model()
            result = self._run_inference(image_path, query)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Heuristic confidence based on answer length and keywords
            answer = result["answer"]
            confidence = self._estimate_confidence(answer)

            # Truncate answer for raw_text summary
            summary = answer[:150].rstrip()
            if len(answer) > 150:
                summary += "..."

            return ToolOutput(
                tool_name=self.name,
                result=result,
                confidence=confidence,
                raw_text=f"DermoGPT assessment: {summary}",
                metadata={
                    "model": "DermoGPT-RL",
                    "source": "HuggingFace (gated)",
                    "latency_ms": round(elapsed_ms, 1),
                },
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error("DermoGPT inference failed: %s", exc, exc_info=True)
            return ToolOutput(
                tool_name=self.name,
                result={"error": str(exc)},
                confidence=0.0,
                raw_text=f"DermoGPT inference failed: {exc}",
                metadata={"status": "error", "latency_ms": round(elapsed_ms, 1)},
            )

    @staticmethod
    def _estimate_confidence(answer: str) -> float:
        """Heuristic confidence from answer text.

        Higher confidence if the answer contains clinical reasoning
        keywords and is sufficiently detailed.
        """
        if not answer:
            return 0.3

        score = 0.5
        # Length bonus
        if len(answer) > 100:
            score += 0.1
        if len(answer) > 200:
            score += 0.05

        # Clinical keyword bonus
        clinical_terms = [
            "melanoma", "carcinoma", "biopsy", "dermoscop",
            "diagnosis", "differential", "malignant", "benign",
            "ABCDE", "pigment", "asymmetr",
        ]
        matches = sum(1 for t in clinical_terms if t.lower() in answer.lower())
        score += min(matches * 0.05, 0.2)

        return min(round(score, 2), 1.0)
