"""Tools: the calculator's safety boundary, and the registry/invoke contract."""

import asyncio

import pytest
from pydantic import BaseModel

from slom.tools import (
    ToolError,
    ToolRegistry,
    calculator,
    evaluate_arithmetic,
    format_number,
    tool,
)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("17 * 23 + 4", 395),
        ("(12.5 * 4) - 3 / 2", 48.5),
        ("2 ** 10", 1024),
        ("2^10", 1024),  # models write ^ for power
        ("1,250 * 3", 3750),  # thousands separators
        ("-(3 - 5)", 2),
        ("7 // 2 + 7 % 2", 4),
    ],
)
def test_arithmetic(expr: str, expected: float) -> None:
    assert evaluate_arithmetic(expr) == expected


@pytest.mark.parametrize(
    ("expr", "reason"),
    [
        ("__import__('os').system('echo hi')", "unsupported syntax"),
        ("abs(-1)", "unsupported syntax"),
        ("x + 1", "unsupported syntax"),
        ("(1).__class__", "unsupported syntax"),
        ("'a' * 3", "unsupported syntax"),
        ("3,5", "unsupported syntax"),  # not a thousands separator: don't guess
        ("9 ** 9 ** 9", "exponent larger"),
        ("10 ** 50", "out of range"),
        ("1 / 0", "division by zero"),
        ("(-8) ** 0.5", "out of range"),  # complex result
        ("1 +", "not an arithmetic expression"),
        ("1" * 201, "longer than"),
    ],
)
def test_calculator_rejects(expr: str, reason: str) -> None:
    with pytest.raises(ToolError, match=reason):
        evaluate_arithmetic(expr)


def test_format_number() -> None:
    assert format_number(4.0) == "4"
    assert format_number(1 / 3) == "0.3333333333"
    assert format_number(7) == "7"


@pytest.mark.anyio
async def test_invoke_turns_bad_args_and_tool_errors_into_readable_results() -> None:
    bad = await calculator.invoke({"expr": "1+1"})
    assert not bad.ok and "expression: Field required" in bad.output
    err = await calculator.invoke({"expression": "1/0"})
    assert not err.ok and err.output == "error: division by zero"
    ok = await calculator.invoke({"expression": "6*7"})
    assert ok.ok and ok.output == "42"


class EchoArgs(BaseModel):
    text: str


@pytest.mark.anyio
async def test_unexpected_exceptions_propagate_for_retry() -> None:
    @tool()
    def flaky(args: EchoArgs) -> str:
        raise ConnectionError("upstream down")

    with pytest.raises(ConnectionError):
        await flaky.invoke({"text": "x"})


@pytest.mark.anyio
async def test_sync_tools_run_off_the_event_loop_and_output_is_truncated() -> None:
    loop_thread: list[bool] = []

    @tool(description="big")
    def big(args: EchoArgs) -> str:
        try:
            asyncio.get_running_loop()
            loop_thread.append(True)
        except RuntimeError:
            loop_thread.append(False)
        return args.text * 5000

    result = await big.invoke({"text": "ab"})
    assert loop_thread == [False]
    assert len(result.output) < 2100 and result.output.endswith("[truncated 8000 chars]")


def test_decorator_and_registry_contract() -> None:
    @tool()
    async def echo(args: EchoArgs) -> str:
        """Repeat the text."""
        return args.text

    assert echo.name == "echo" and echo.description == "Repeat the text."
    assert echo.spec().parameters["required"] == ["text"]

    with pytest.raises(TypeError, match="exactly one pydantic model"):
        tool()(lambda x, y: "")  # type: ignore[arg-type,return-value]

    registry = ToolRegistry([echo])
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(echo)
    assert "echo" in registry and "nope" not in registry
