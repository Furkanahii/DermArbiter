"""
DermArbiter Configuration — Loader & Validation

Loads a layered configuration from YAML files:
    config_dir/
        default.yaml    — global defaults (model, temperature, budget, etc.)
        agents.yaml     — per-agent overrides (role, model, prompt path, tools)
        tools.yaml      — tool registry settings (endpoints, API keys, timeouts)

Environment variables (loaded from ``.env`` via python-dotenv) can override
any secret field.  ``GOOGLE_API_KEY`` is the primary example.

Usage:
    from dermarbiter.core.config import load_config
    cfg = load_config("config/")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """Configuration for a single agent in the panel."""

    model_config = {"extra": "ignore"}

    role: str = Field(
        ...,
        description="Agent role identifier (specialist, generalist, skeptic, moderator).",
    )
    model_backend: str = Field(
        default="google_api",
        description="Backend to use: 'google_api', 'local_hf', 'groq_api'.",
    )
    model_name: str = Field(
        default="gemini-2.0-flash",
        description="Model identifier passed to the backend.",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for this agent.",
    )
    max_output_tokens: int = Field(
        default=4096,
        ge=64,
        description="Maximum tokens the LLM may generate per call.",
    )
    system_prompt_path: str = Field(
        default="",
        description="Relative path to the agent's system prompt file.",
    )
    has_tool_access: bool = Field(
        default=True,
        description="Whether this agent may invoke diagnostic tools.",
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="Whitelist of tool names this agent may invoke. "
                    "Empty means 'all registered tools' (if has_tool_access=True).",
    )
    device: str = Field(
        default="cpu",
        description="Target device for local_hf backend ('cpu', 'cuda', 'auto').",
    )
    quantization: Optional[str] = Field(
        default=None,
        description="Quantization mode for local_hf backend ('4bit', '8bit', or None).",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Catch-all for backend-specific or experimental settings.",
    )

    @field_validator("temperature", mode="before")
    @classmethod
    def _clamp_temperature(cls, v: Any) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.3
        return max(0.0, min(2.0, v))


class DebateConfig(BaseModel):
    """Settings governing the structured debate phase."""

    max_rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of debate rounds before forced synthesis.",
    )
    early_exit_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Min avg confidence for early exit (confidence_threshold).",
    )
    max_tokens_per_turn: int = Field(
        default=100,
        ge=16,
        description="Token budget per debate turn.",
    )
    global_token_budget: int = Field(
        default=50_000,
        ge=1000,
        description="Total token budget for the full pipeline run.",
    )
    turn_order: list[str] = Field(
        default_factory=lambda: ["specialist", "generalist", "skeptic"],
        description="Ordered list of agent roles for each debate round.",
    )
    moderator_role: str = Field(
        default="moderator",
        description="Role identifier for the moderator agent.",
    )

    # --- Early Exit Gating ---
    min_agreement: int = Field(
        default=2,
        ge=1,
        description="Minimum number of agents that must agree on top-1 for early exit.",
    )
    confidence_floor: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="No individual agent's confidence may be below this for early exit.",
    )
    require_no_flags: bool = Field(
        default=True,
        description="If True, abort early exit when any agent has disagreement flags.",
    )
    require_unanimous: bool = Field(
        default=False,
        description="If True, ALL agents must agree on top-1 for early exit.",
    )

    # --- Synthesis ---
    voting_strategy: str = Field(
        default="weighted_confidence",
        description="Consensus strategy: 'weighted_confidence', 'majority', 'rank_fusion'.",
    )
    specialist_weight: float = Field(
        default=1.2,
        ge=0.0,
        description="Weight multiplier for the specialist's opinion in synthesis.",
    )
    rank_weights: list[float] = Field(
        default_factory=lambda: [1.0, 0.6, 0.3],
        description="Positional weight for rank 1, 2, 3 in differential lists.",
    )
    top_k_diagnoses: int = Field(
        default=5,
        ge=1,
        description="Maximum number of diagnoses to include in final ranking.",
    )


class ToolConfig(BaseModel):
    """Configuration for a single diagnostic tool."""

    name: str = Field(
        ...,
        description="Canonical tool name (must match tool_registry keys).",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this tool is available in the current run.",
    )
    endpoint: str = Field(
        default="",
        description="HTTP endpoint or local function path.",
    )
    api_key_env: str = Field(
        default="",
        description="Name of the env var holding the API key for this tool.",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        description="Per-call timeout in seconds.",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description="Number of automatic retries on transient failure.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific overrides.",
    )

    def resolve_api_key(self) -> Optional[str]:
        """Return the actual API key from the environment, or None."""
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class DermArbiterConfig(BaseModel):
    """Root configuration for the DermArbiter framework."""

    # --- Global ---
    project_name: str = Field(
        default="DermArbiter",
        description="Human-readable project name.",
    )
    google_api_key: str = Field(
        default="",
        description="Google Gemini API key (can be overridden by env var).",
    )
    groq_api_key: str = Field(
        default="",
        description="Groq Cloud API key (can be overridden by env var).",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (can be overridden by env var).",
    )
    default_model: str = Field(
        default="gemini-2.0-flash",
        description="Fallback model when an agent config doesn't specify one.",
    )
    default_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
    )
    log_level: str = Field(
        default="INFO",
        description="Python logging level.",
    )
    token_budget: int = Field(
        default=50_000,
        ge=1000,
        description="Global token budget for the entire pipeline run.",
    )

    # --- Nested ---
    agents: dict[str, AgentConfig] = Field(
        default_factory=dict,
        description="Mapping from agent role to its configuration.",
    )
    debate: DebateConfig = Field(
        default_factory=DebateConfig,
        description="Debate phase settings.",
    )
    tools: dict[str, ToolConfig] = Field(
        default_factory=dict,
        description="Mapping from tool name to its configuration.",
    )


# ---------------------------------------------------------------------------
# YAML deep merge
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge *override* into *base* (mutates *base*).

    - Dicts are merged recursively.
    - Lists in *override* replace lists in *base* entirely.
    - Scalars in *override* replace scalars in *base*.
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_config(config_dir: str) -> DermArbiterConfig:
    """
    Load and merge DermArbiter configuration from a directory of YAML files.

    Resolution order (later overrides earlier):
        1. Built-in defaults (Pydantic field defaults)
        2. ``default.yaml``   — global overrides
        3. ``agents.yaml``    — agent-specific overrides
        4. ``tools.yaml``     — tool-specific overrides
        5. Environment variables (via ``.env`` and ``os.environ``)

    Args:
        config_dir: Path to the directory containing YAML config files.

    Returns:
        A fully validated ``DermArbiterConfig`` instance.
    """
    config_path = Path(config_dir)

    # --- Load .env early so YAML $ENV references are resolvable ---
    dotenv_path = config_path / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
    else:
        # Try project root .env as well
        load_dotenv()

    merged: dict[str, Any] = {}

    # --- Layer YAML files ---
    yaml_files = ["default.yaml", "agents.yaml", "tools.yaml"]
    for filename in yaml_files:
        filepath = config_path / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as fh:
                content = yaml.safe_load(fh)
                if isinstance(content, dict):
                    _deep_merge(merged, content)

    # --- Flatten debate config structure if nested ---
    if "debate" in merged and isinstance(merged["debate"], dict):
        debate_raw = merged["debate"]
        flat_debate = {}
        for k, v in debate_raw.items():
            if k == "early_exit" and isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if sub_k == "confidence_threshold":
                        flat_debate["early_exit_threshold"] = sub_v
                    else:
                        flat_debate[sub_k] = sub_v
            elif k == "synthesis" and isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if sub_k == "top_k_diagnoses":
                        flat_debate["top_k_diagnoses"] = sub_v
                    else:
                        flat_debate[sub_k] = sub_v
            else:
                flat_debate[k] = v
        # Shorthand aliases for debate
        if "token_budget_per_turn" in flat_debate:
            flat_debate["max_tokens_per_turn"] = flat_debate.pop("token_budget_per_turn")
        merged["debate"] = flat_debate

    # --- Transform agents (list or dict) → dict keyed by role ---
    if "agents" in merged and isinstance(merged["agents"], list):
        agents_dict: dict[str, Any] = {}
        for agent_data in merged["agents"]:
            if isinstance(agent_data, dict) and "role" in agent_data:
                agents_dict[agent_data["role"]] = agent_data
        merged["agents"] = agents_dict

    # --- Normalize agent YAML shorthand → Pydantic field names ---
    _AGENT_FIELD_ALIASES = {
        "model": "model_name",
        "backend": "model_backend",
        "max_tokens": "max_output_tokens",
        "system_prompt_key": "system_prompt_path",
    }
    if "agents" in merged and isinstance(merged["agents"], dict):
        for role, agent_data in merged["agents"].items():
            if not isinstance(agent_data, dict):
                continue
            # Ensure role is set
            agent_data.setdefault("role", role)
            # Map shorthand keys to canonical Pydantic field names
            for short, canonical in _AGENT_FIELD_ALIASES.items():
                if short in agent_data and canonical not in agent_data:
                    agent_data[canonical] = agent_data.pop(short)

    # --- Transform tools list → dict keyed by name ---
    if "tools" in merged and isinstance(merged["tools"], list):
        tools_dict: dict[str, Any] = {}
        for tool_data in merged["tools"]:
            if isinstance(tool_data, dict) and "name" in tool_data:
                tools_dict[tool_data["name"]] = tool_data
        merged["tools"] = tools_dict

    # --- Environment variable overrides ---
    env_overrides: dict[str, tuple[str, type]] = {
        "GOOGLE_API_KEY": ("google_api_key", str),
        "GROQ_API_KEY": ("groq_api_key", str),
        "OPENAI_API_KEY": ("openai_api_key", str),
        "DERMARBITER_LOG_LEVEL": ("log_level", str),
        "DERMARBITER_TOKEN_BUDGET": ("token_budget", int),
        "DERMARBITER_DEFAULT_MODEL": ("default_model", str),
    }

    for env_var, (field_name, cast_type) in env_overrides.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            try:
                merged[field_name] = cast_type(env_value)
            except (TypeError, ValueError):
                pass  # Ignore malformed env values; Pydantic will catch later

    # Try project root or system environment standard OPENAI_API_KEY
    if not merged.get("openai_api_key") and os.environ.get("OPENAI_API_KEY"):
        merged["openai_api_key"] = os.environ.get("OPENAI_API_KEY")

    # --- Validate and return ---
    cfg = DermArbiterConfig(**merged)

    # Populate moderator's extra dict with debate early exit settings
    if "moderator" in cfg.agents:
        mod_extra = cfg.agents["moderator"].extra
        mod_extra["min_agreement"] = cfg.debate.min_agreement
        mod_extra["early_exit_threshold"] = cfg.debate.early_exit_threshold
        mod_extra["confidence_floor"] = cfg.debate.confidence_floor
        mod_extra["require_no_flags"] = cfg.debate.require_no_flags
        mod_extra["require_unanimous"] = cfg.debate.require_unanimous

    return cfg
