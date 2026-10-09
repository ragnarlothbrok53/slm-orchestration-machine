"""The agent: a bounded loop of model call -> parse -> tool call, as a durable workflow.

Each model call and each tool call is a journaled step, so a crashed run resumes after its last
finished call instead of paying for every LLM call again. Everything between steps (prompt
rendering, parsing, bookkeeping) is deterministic, which is what makes replay valid.
"""

from __future__ import annotations

from .contracts import (
    AgentInput,
    AgentResult,
    LlmRequest,
    LlmResponse,
    RunStatus,
    Step,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
from .engine import PermanentError, RetryPolicy, RunContext, StepFailedError, Workflow
from .llm import LlmClient
from .metrics import Metrics
from .parsing import AnswerDecision, InvalidOutput, ToolDecision
from .prompts import PROTOCOLS, STOP
from .tools import ToolRegistry

WORKFLOW_NAME = "agent"


class AgentSteps:
    """The agent's step functions, bound to this worker's model client and tools."""

    def __init__(self, llm: LlmClient, tools: ToolRegistry, metrics: Metrics) -> None:
        self.llm = llm
        self.tools = tools
        self.metrics = metrics

    async def describe_tools(self, names: list[str] | None) -> list[ToolSpec]:
        selected = self.tools.names() if names is None else names
        unknown = [n for n in selected if n not in self.tools]
        if unknown:
            raise PermanentError(f"unknown tools {unknown}; available: {self.tools.names()}")
        return [self.tools.get(n).spec() for n in selected]

    async def complete(self, req: LlmRequest) -> LlmResponse:
        return await self.llm.complete(req)

    async def call_tool(self, call: ToolCall) -> ToolResult:
        result = await self.tools.get(call.name).invoke(call.args)
        self.metrics.tool_calls.labels(call.name, str(result.ok).lower()).inc()
        return result


def agent_workflow(
    steps: AgentSteps,
    *,
    llm_retry: RetryPolicy = RetryPolicy(max_attempts=6, initial_s=1.0, max_s=30.0),  # noqa: B008
    tool_retry: RetryPolicy = RetryPolicy(max_attempts=3, timeout_s=30.0),  # noqa: B008
) -> Workflow[AgentInput, AgentResult]:
    metrics = steps.metrics

    async def run_agent(ctx: RunContext, inp: AgentInput) -> AgentResult:
        specs = await ctx.step(
            "describe_tools", steps.describe_tools, inp.tools, retry=RetryPolicy(max_attempts=1)
        )
        names = [s.name for s in specs]
        protocol = PROTOCOLS[inp.protocol]
        history: list[Step] = []
        usage = Usage()
        repairs = 0

        def done(status: RunStatus, answer: str | None = None) -> AgentResult:
            if not ctx.replaying:
                metrics.agent_outcomes.labels(status).inc()
            return AgentResult(status, answer, history, usage)

        for _ in range(inp.max_steps):
            request = LlmRequest(
                prompt=protocol.render(specs, inp.question, history),
                max_tokens=inp.max_tokens,
                temperature=inp.temperature,
                stop=STOP,
            )
            resp = await ctx.step("llm", steps.complete, request, retry=llm_retry)
            usage.llm_calls += 1
            usage.prompt_tokens += resp.prompt_tokens
            usage.completion_tokens += resp.completion_tokens
            usage.llm_seconds += resp.latency_s

            output = protocol.prefill + resp.text
            decision = protocol.parse(output, names)
            if isinstance(decision, ToolDecision) and (seen := _earlier_result(history, decision)):
                # Small models often re-issue a call they already made instead of answering.
                # Running it again can't help, and it would loop until max_steps.
                decision = InvalidOutput(
                    f"you already called {decision.tool} with these arguments and got: {seen}. "
                    "Use that result: answer, or make a different call"
                )

            if isinstance(decision, AnswerDecision):
                history.append(Step("answer", output, llm_request_id=resp.request_id))
                return done("answered", decision.answer)

            if isinstance(decision, InvalidOutput):
                if not ctx.replaying:
                    metrics.agent_invalid_outputs.inc()
                history.append(
                    Step(
                        "invalid",
                        output,
                        observation=decision.reason,
                        llm_request_id=resp.request_id,
                    )
                )
                repairs += 1
                if repairs > inp.max_repairs:
                    return done("max_repairs")
                continue

            assert isinstance(decision, ToolDecision)
            call = ToolCall(decision.tool, decision.args)
            try:
                result = await ctx.step(
                    f"tool:{call.name}", steps.call_tool, call, retry=tool_retry
                )
            except StepFailedError as exc:
                # Retries are exhausted; let the model see it and choose another path.
                result = ToolResult(ok=False, output=f"tool failed: {exc.message}")
            usage.tool_calls += 1
            history.append(
                Step(
                    "tool",
                    output,
                    tool=call.name,
                    args=call.args,
                    observation=result.output,
                    llm_request_id=resp.request_id,
                )
            )

        return done("max_steps")

    return Workflow.define(WORKFLOW_NAME, run_agent)


def _earlier_result(history: list[Step], call: ToolDecision) -> str | None:
    for step in history:
        if step.kind == "tool" and step.tool == call.tool and step.args == call.args:
            return step.observation
    return None
