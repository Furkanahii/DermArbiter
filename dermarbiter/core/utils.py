"""DermArbiter shared utilities.

Centralised helpers used across agents, tools, and the debate protocol.
Avoids code duplication of common patterns (JSON extraction, prompt
loading, token counting).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Approximate chars-per-token ratio for English text.
_CHARS_PER_TOKEN: float = 3.8


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from free-form LLM output.

    Strategy:
        1. Look for a fenced ``json`` code block (```json ... ```).
        2. Fall back to the first ``{ ... }`` substring.
        3. Return an empty dict if nothing parseable is found.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict, or ``{}`` on failure.
    """
    fence_match = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_DEFAULT_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(prompt_name: str, prompts_dir: Path | None = None) -> str:
    """Load a system prompt from disk.

    Args:
        prompt_name: Name of the prompt file (without extension).
        prompts_dir: Optional directory override.  Defaults to
            ``dermarbiter/core/prompts/``.

    Returns:
        The prompt text, or empty string if file not found.
    """
    if prompts_dir is None:
        prompts_dir = _DEFAULT_PROMPTS_DIR

    prompt_path = prompts_dir / f"{prompt_name}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    logger.warning("Prompt file not found: %s", prompt_path)
    return ""


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Fast approximate token count based on character length.

    Uses a ratio of ~3.8 characters per token (slightly conservative
    for English text with GPT/Gemini tokenisers).
    """
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))

