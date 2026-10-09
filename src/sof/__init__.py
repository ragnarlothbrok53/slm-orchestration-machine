"""SOF: durable, tool-using agents for small language models."""

from .contracts import AgentInput, AgentResult
from .engine import (
    PermanentError,
    RetryableError,
    RetryPolicy,
    RunContext,
    StepFailedError,
    Worker,
    Workflow,
    step_info,
)
from .store import Store
from .tools import BUILTIN_TOOLS, Tool, ToolError, ToolRegistry, idempotency_key, tool

__version__ = "0.1.0"

__all__ = [
    "BUILTIN_TOOLS",
    "AgentInput",
    "AgentResult",
    "PermanentError",
    "RetryPolicy",
    "RetryableError",
    "RunContext",
    "StepFailedError",
    "Store",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "Worker",
    "Workflow",
    "__version__",
    "idempotency_key",
    "step_info",
    "tool",
]
