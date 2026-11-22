"""DermArbiter — Multi-Agent Dermatological Diagnosis Framework."""

__version__ = "0.1.0"

__all__ = [
    "DermArbiterOrchestrator",
    "__version__",
]


def __getattr__(name):
    # Lazy: dermarbiter.research (text benchmark) must import without the
    # LangGraph / model stack, e.g. on a bare Kaggle or Colab runtime.
    if name == "DermArbiterOrchestrator":
        from dermarbiter.core.orchestrator import DermArbiterOrchestrator
        return DermArbiterOrchestrator
    raise AttributeError(f"module 'dermarbiter' has no attribute {name!r}")
