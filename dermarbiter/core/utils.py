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

def _try_parse(candidate: str) -> dict[str, Any] | None:
    """json.loads with one retry that escapes literal newlines inside strings.

    Gemini occasionally emits JSON like::

        {"reasoning": "The lesion shows...
        It is consistent with..."}

    which is invalid JSON (unescaped newline inside the string value).
    The retry escapes raw newlines/tabs that appear *between* quotes,
    leaving structural whitespace alone.
    """
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        result = json.loads(candidate)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass

    # Escape literal newlines/tabs that fall inside string values.
    in_string = False
    escape = False
    fixed_chars: list[str] = []
    for ch in candidate:
        if escape:
            fixed_chars.append(ch)
            escape = False
            continue
        if ch == "\\":
            fixed_chars.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            fixed_chars.append(ch)
            continue
        if in_string and ch == "\n":
            fixed_chars.append("\\n")
        elif in_string and ch == "\t":
            fixed_chars.append("\\t")
        elif in_string and ch == "\r":
            fixed_chars.append("\\r")
        else:
            fixed_chars.append(ch)
    try:
        result = json.loads("".join(fixed_chars))
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None


def _find_balanced_braces(text: str) -> str | None:
    """Return the substring spanning the first ``{`` and its matching ``}``.

    Walks the text once, counts brace depth (skipping braces inside string
    literals), and returns the well-balanced span. Beats the greedy
    ``{.*}`` regex when the response has multiple JSON objects or a
    trailing markdown explanation.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _salvage_truncated_json(text: str) -> dict[str, Any] | None:
    """Last-resort recovery for JSON that was cut off by max_output_tokens.

    Gemini hits its token budget mid-reasoning, the closing brace never
    arrives, _find_balanced_braces returns None, and a perfectly valid
    top3_differential array vanishes with it. This walks the text once
    looking for the ``"top3_differential": [ ... ]`` array specifically
    and reconstructs a minimal dict if found. Confidence/reasoning land
    as best-effort fallbacks when present in the prefix.
    """
    # Pull out the top3_differential array (regex tolerant of whitespace
    # and newlines between elements).
    m = re.search(
        r'"top3_differential"\s*:\s*\[\s*((?:"[^"]*"\s*,?\s*)+)\]',
        text, re.DOTALL,
    )
    if not m:
        return None
    items = re.findall(r'"([^"]+)"', m.group(1))
    if not items:
        return None

    result: dict[str, Any] = {"top3_differential": items[:5]}

    # Bonus: pull confidence if it's complete.
    c = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if c:
        try:
            result["confidence"] = float(c.group(1))
        except ValueError:
            pass

    # Bonus: prefix of reasoning, even if truncated.
    r = re.search(r'"reasoning"\s*:\s*"([^"]*)', text, re.DOTALL)
    if r:
        result["reasoning"] = r.group(1).strip()

    return result


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from free-form LLM output.

    Strategy (each step retries with newline-escape on JSONDecodeError):
        1. Fenced ```json ... ``` code block.
        2. Fenced ``` ... ``` block (no language tag) — Gemini sometimes
           omits the ``json`` after the opening fence.
        3. First balanced ``{ ... }`` span found by brace-counting.
        4. Salvage path: regex-extract ``top3_differential`` even when
           the response was truncated mid-reasoning by max_output_tokens.

    Returns the parsed dict, or ``{}`` if every strategy fails. When
    extraction fails the caller is expected to log ``text[:800]`` at
    WARNING level so we can see what the model actually returned.
    """
    # 1. Fenced ```json ... ``` block.
    fence_json = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_json:
        parsed = _try_parse(fence_json.group(1))
        if parsed is not None:
            return parsed

    # 2. Fenced ``` ... ``` block (no language tag).
    fence_any = re.search(r"```\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_any:
        parsed = _try_parse(fence_any.group(1))
        if parsed is not None:
            return parsed

    # 3. First balanced { ... } span.
    balanced = _find_balanced_braces(text)
    if balanced:
        parsed = _try_parse(balanced)
        if parsed is not None:
            return parsed

    # 4. Salvage truncated JSON (max_output_tokens cutoff).
    salvaged = _salvage_truncated_json(text)
    if salvaged is not None:
        return salvaged

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

