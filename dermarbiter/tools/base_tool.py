"""DermArbiter Base Tool — Abstract tool interface and ToolRegistry.

This module defines:
  • ``ToolOutput`` — Pydantic model for every tool's return value.
  • ``BaseTool``  — ABC that each frozen tool must implement.
  • ``ToolRegistry`` — Registry that manages available tools and supports
    batch execution.

Every tool wrapper (PanDerm, MAKE, DermoGPT, …) subclasses ``BaseTool``
and returns a ``ToolOutput``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ToolOutput — canonical return type for every tool
# ---------------------------------------------------------------------------

class ToolOutput(BaseModel):
    """Structured output returned by every ``BaseTool.run()`` call.

    Attributes:
        tool_name:  Identifier matching the tool's ``name`` property.
        result:     Tool-specific structured payload (flexible dict).
        confidence: Model/tool confidence in the result, ∈ [0, 1].
        raw_text:   Human-readable one-line summary of the result.
        metadata:   Optional auxiliary data (latency, model version, …).
        timestamp:  ISO-8601 timestamp of when the output was created.
    """

    tool_name: str = Field(
        ...,
        description="Identifier matching the tool's name property.",
    )
    result: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific structured output.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model/tool confidence score in [0, 1].",
    )
    raw_text: str = Field(
        "",
        description="Human-readable one-line summary.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional auxiliary data (latency, version, etc.).",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO-8601 creation timestamp.",
    )


# ---------------------------------------------------------------------------
# BaseTool — abstract interface for all frozen tools
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    """Abstract base class for every DermArbiter tool.

    Subclasses **must** implement ``name``, ``description``, and ``run``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique machine-readable tool identifier (snake_case)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description shown to agents during planning."""
        ...

    @abstractmethod
    def run(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> ToolOutput:
        """Execute the tool and return a structured ``ToolOutput``.

        Args:
            image_path: Local filesystem path to the dermoscopic image.
                        May be ``None`` for text-only tools.
            query:      Free-text query or instruction for the tool.

        Returns:
            A ``ToolOutput`` instance with the tool's results.
        """
        ...

    # -- Optional overrides ------------------------------------------------

    def validate_input(
        self,
        image_path: str | None = None,
        query: str = "",
    ) -> bool:
        """Pre-flight check before ``run()``.  Override for custom logic."""
        return True

    # -- Introspection helpers ---------------------------------------------

    def to_schema(self) -> dict[str, Any]:
        """Return a JSON-serialisable schema describing the tool.

        Useful for LLM function-calling manifests.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": ["string", "null"],
                        "description": "Path to dermoscopic image.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Free-text query.",
                    },
                },
                "required": [],
            },
        }

    def __repr__(self) -> str:  # noqa: D105
        return f"<{self.__class__.__name__} name={self.name!r}>"


# ---------------------------------------------------------------------------
# ToolRegistry — manages available tools
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Thread-safe registry of available ``BaseTool`` instances.

    Usage::

        registry = ToolRegistry()
        registry.register(PanDermClassifier())
        registry.register(MAKEAnnotator())

        output = registry.get("panderm_classifier").run(image_path="img.jpg")
        outputs = registry.run_batch(
            ["panderm_classifier", "make_annotator"],
            image_path="img.jpg",
        )
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # -- Mutation ----------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance.  Overwrites if name already exists."""
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"Expected a BaseTool instance, got {type(tool).__name__}"
            )
        logger.debug("Registering tool %r", tool.name)
        self._tools[tool.name] = tool

    # -- Query -------------------------------------------------------------

    def get(self, name: str) -> BaseTool:
        """Retrieve a tool by name.

        Raises:
            KeyError: If no tool with *name* is registered.
        """
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(sorted(self._tools)) or "(none)"
            raise KeyError(
                f"Tool {name!r} not found. Available: {available}"
            ) from None

    def list_tools(self) -> list[dict[str, Any]]:
        """Return a list of tool schemas for LLM planning prompts."""
        return [tool.to_schema() for tool in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    # -- Batch execution ---------------------------------------------------

    def run_batch(
        self,
        tool_names: list[str],
        image_path: str | None = None,
        query: str = "",
    ) -> list[ToolOutput]:
        """Execute multiple tools sequentially and collect outputs.

        Tools that raise exceptions are logged and skipped (their output
        is replaced with an error ``ToolOutput`` so the pipeline never
        crashes due to a single tool failure).

        Args:
            tool_names:  List of tool names to run.
            image_path:  Shared image path for all tools.
            query:       Shared query string for all tools.

        Returns:
            List of ``ToolOutput`` instances (one per tool name).
        """
        outputs: list[ToolOutput] = []
        for name in tool_names:
            try:
                tool = self.get(name)
                output = tool.run(image_path=image_path, query=query)
                outputs.append(output)
            except Exception as exc:
                logger.error("Tool %r failed: %s", name, exc, exc_info=True)
                outputs.append(
                    ToolOutput(
                        tool_name=name,
                        result={"error": str(exc)},
                        confidence=0.0,
                        raw_text=f"Tool {name} failed: {exc}",
                        metadata={"status": "error"},
                    )
                )
        return outputs

    def __repr__(self) -> str:  # noqa: D105
        names = ", ".join(sorted(self._tools))
        return f"<ToolRegistry tools=[{names}]>"
