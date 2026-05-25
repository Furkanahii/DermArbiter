"""Mock factory for CPU-only pipeline testing.

Provides factory functions for creating mock agents and tool registries
without coupling production code to the test suite.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def create_mock_agents() -> Dict[str, Any]:
    """Create mock agent instances for CPU-only testing.
    
    Delegates to tests.mocks.mock_agents.create_mock_agents().
    This indirection keeps the import localised to one file.
    """
    from tests.mocks.mock_agents import create_mock_agents as _create
    return _create()


def create_mock_registry() -> Any:
    """Create a mock tool registry for CPU-only testing.
    
    Delegates to tests.mocks.mock_tools.create_mock_registry().
    """
    from tests.mocks.mock_tools import create_mock_registry as _create
    return _create()
