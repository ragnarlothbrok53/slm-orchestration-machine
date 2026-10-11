"""The agent loop: tool use, repair of bad output, loop guard, limits, and durable resume."""

from __future__ import annotations

import asyncio

import pytest
from helpers import ScriptedLlm, make_worker, run_to_completion, sample_value
from pydantic import TypeAdapter

from sof.agent import WORKFLOW_NAME
from sof.contracts import AgentInput, AgentResult
from sof.engine import PermanentError, RetryableError
from sof.metrics import Metrics
from sof.store import Store

pytestmark = pytest.mark.anyio

INPUT = TypeAdapter(AgentInput)
CALC = '"tool": "calculator", "args": {"expression": "17*23+4"}}'  # the prefill supplies "{"
ANSWER = '"answer": "395"}'


async def submit(store: Store, run_id: str = "r1", **kw: object) -> None:
    inp = AgentInput(question="17 pens at 23 each plus 4?", **kw)  # type: ignore[arg-type]
    await store.submit(run_id, WORKFLOW_NAME, INPUT.dump_json(inp).decode())


async def result_of(store: Store, run_id: str = "r1") -> AgentResult:
    run = await store.get(run_id)
    assert run is not None and run.status == "completed", run
    return TypeAdapter(AgentResult).validate_json(run.output or "")


async def test_tool_then_answer(store: Store) -> None:
    llm = ScriptedLlm([CALC, ANSWER])
    await submit(store)
    await run_to_completion(make_worker(store, llm), store, "r1")
    result = await result_of(store)
    assert (result.status, result.answer) == ("answered", "395")
    assert [s.kind for s in result.steps] == ["tool", "answer"]
    assert result.steps[0].observation == "395"
    assert result.usage.llm_calls == 2 and result.usage.tool_calls == 1
    # The second prompt contains the tool result, and every prompt ends in the prefill.
    assert "Result of calculator: 395" in llm.prompts[1]
    assert all(p.endswith("<|im_start|>assistant\n{") for p in llm.prompts)
    names = [s.name for s in await store.steps("r1")]
    assert names == ["describe_tools", "llm", "tool:calculator", "llm"]


async def test_invalid_output_is_repaired(store: Store) -> None:
    metrics = Metrics()
    llm = ScriptedLlm(["I think it is 395", ANSWER])
    await submit(store)
    await run_to_completion(make_worker(store, llm, metrics=metrics), store, "r1")
    result = await result_of(store)
    assert result.status == "answered" and [s.kind for s in result.steps] == ["invalid", "answer"]
    assert "Your reply was invalid: no JSON object found" in llm.prompts[1]
    assert sample_value(metrics, "sof_agent_invalid_outputs_total") == 1


async def test_repairs_are_bounded(store: Store) -> None:
    llm = ScriptedLlm(["nope", "nope", "nope"])
    await submit(store, max_repairs=2)
    await run_to_completion(make_worker(store, llm), store, "r1")
    result = await result_of(store)
    assert result.status == "max_repairs" and result.answer is None and len(llm.prompts) == 3


async def test_repeated_tool_call_is_not_executed_and_forces_an_answer(store: Store) -> None:
    llm = ScriptedLlm([CALC, CALC, '395"}'])
    await submit(store)
    metrics = Metrics()
    await run_to_completion(make_worker(store, llm, metrics=metrics), store, "r1")
    result = await result_of(store)
    assert [s.kind for s in result.steps] == ["tool", "repeat", "answer"]
    assert "you already called calculator" in (result.steps[1].observation or "")
    assert llm.prompts[2].endswith('<|im_start|>assistant\n{"answer": "')
    assert (result.status, result.answer) == ("answered", "395")
    assert sample_value(metrics, "sof_tool_calls_total", {"tool": "calculator", "ok": "true"}) == 1


async def test_last_step_must_answer(store: Store) -> None:
    llm = ScriptedLlm([CALC, "395"])  # forced answer, cut off before its closing quote
    await submit(store, max_steps=2)
    await run_to_completion(make_worker(store, llm), store, "r1")
    result = await result_of(store)
    assert llm.prompts[1].endswith('{"answer": "')
    assert (result.status, result.answer) == ("answered", "395")


async def test_step_budget(store: Store) -> None:
    calls = [f'"tool": "calculator", "args": {{"expression": "{i}+1"}}}}' for i in range(3)]
    await submit(store, max_steps=3)
    await run_to_completion(make_worker(store, ScriptedLlm(calls)), store, "r1")
    result = await result_of(store)
    assert result.status == "max_steps" and result.usage.llm_calls == 3


async def test_tool_errors_are_shown_to_the_model(store: Store) -> None:
    bad = '"tool": "calculator", "args": {"expression": "1/0"}}'
    llm = ScriptedLlm([bad, ANSWER])
    await submit(store)
    await run_to_completion(make_worker(store, llm), store, "r1")
    result = await result_of(store)
    assert result.steps[0].observation == "error: division by zero"
    assert "Result of calculator: error: division by zero" in llm.prompts[1]


async def test_transient_llm_failures_are_retried(store: Store) -> None:
    llm = ScriptedLlm([RetryableError("HTTP 503 overloaded", retry_after=0.01), ANSWER])
    await submit(store)
    await run_to_completion(make_worker(store, llm), store, "r1")
    result = await result_of(store)
    assert result.status == "answered"
    (_, llm_step) = await store.steps("r1")
    assert llm_step.attempts == 2


async def test_permanent_llm_failure_fails_the_run(store: Store) -> None:
    llm = ScriptedLlm([PermanentError("HTTP 400: max_tokens must be <= 1024")])
    await submit(store)
    run = await run_to_completion(make_worker(store, llm), store, "r1")
    assert run.status == "failed" and "max_tokens must be <= 1024" in (run.error or "")


async def test_unknown_tool_selection_fails_without_calling_the_model(store: Store) -> None:
    llm = ScriptedLlm([])
    await submit(store, tools=["web_search"])
    run = await run_to_completion(make_worker(store, llm), store, "r1")
    assert run.status == "failed" and "unknown tools ['web_search']" in (run.error or "")
    assert llm.prompts == []


async def test_qwen_protocol(store: Store) -> None:
    call = (
        '<tool_call>\n{"name": "calculator", "arguments": {"expression": "17*23+4"}}\n</tool_call>'
    )
    llm = ScriptedLlm([call, "She pays 395 rupees."])
    await submit(store, protocol="qwen")
    await run_to_completion(make_worker(store, llm), store, "r1")
    result = await result_of(store)
    assert result.answer == "She pays 395 rupees."
    assert "<tools>\n" in llm.prompts[0] and '"name": "calculator"' in llm.prompts[0]
    assert "<tool_response>\n395\n</tool_response>" in llm.prompts[1]
    assert llm.prompts[1].endswith("<|im_start|>assistant\n")  # no prefill


async def test_resumed_run_does_not_pay_for_finished_llm_calls(store: Store) -> None:
    """Worker A finished the first model call and the tool call, then stopped during the second
    model call. Worker B must make exactly one more model call and see the same transcript."""
    await submit(store)
    hang = asyncio.Event()  # never set: A's second model call never returns
    first = ScriptedLlm([CALC, hang])
    worker_a = make_worker(store, first, shutdown_grace_s=0)
    task = asyncio.create_task(worker_a.run())
    while len(first.prompts) < 2:
        await asyncio.sleep(0.005)
    worker_a.stop()
    await task

    second = ScriptedLlm([ANSWER])
    await run_to_completion(make_worker(store, second), store, "r1")
    result = await result_of(store)
    assert result.status == "answered" and [s.kind for s in result.steps] == ["tool", "answer"]
    assert len(second.prompts) == 1  # only the unfinished call was made again
    assert second.prompts[0] == first.prompts[1]  # identical prompt: replay is deterministic
