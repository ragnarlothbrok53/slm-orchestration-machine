"""Data passed between the API, the workflow and activities.

Plain dataclasses: Temporal's default converter serializes them to JSON, they import cleanly in
the workflow sandbox, and they are what ends up in workflow history (so changing a field is a
compatibility decision, not a refactor).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# --- LLM step ---


@dataclass
class LlmRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


@dataclass
class LlmResponse:
    text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float | None
    latency_s: float
    request_id: str  # X-Request-ID sent to the server: grep slm-runtime's logs for it


# --- tool steps ---


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema of the arguments


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    ok: bool
    output: str


# --- workflow ---


@dataclass
class AgentInput:
    question: str
    tools: list[str] | None = None  # None = every tool registered on the worker
    max_steps: int = 6
    max_repairs: int = 2  # re-prompts allowed for unusable model output, per run
    max_tokens: int = 256
    temperature: float = 0.0
    protocol: Literal["json", "qwen"] = "json"  # see sof.prompts


StepKind = Literal["tool", "answer", "invalid"]


@dataclass
class Step:
    kind: StepKind
    model_output: str  # raw text, exactly as generated
    tool: str | None = None
    args: dict[str, Any] | None = None
    observation: str | None = None  # tool output, or the parse error fed back to the model
    llm_request_id: str = ""


@dataclass
class Usage:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_seconds: float = 0.0
    tool_calls: int = 0


RunStatus = Literal["answered", "max_steps", "max_repairs"]


@dataclass
class AgentResult:
    status: RunStatus
    answer: str | None
    steps: list[Step]
    usage: Usage
