"""Tools: typed Python functions the agent can call, plus a small built-in set.

A tool is a function taking one pydantic model and returning ``str`` (sync or async)::

    class WeatherArgs(BaseModel):
        city: str = Field(description="City name")

    @tool(description="Current weather for a city.")
    async def weather(args: WeatherArgs) -> str: ...

Each tool call is a journaled step, executed **at least once**: if a worker crashes mid-call,
the call runs again on the worker that resumes the run. Read-only tools are fine; tools with
side effects should pass :func:`idempotency_key` to the system they modify.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import math
import operator
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast, get_type_hints

from pydantic import BaseModel, Field, ValidationError

from .contracts import ToolResult, ToolSpec
from .engine import step_info

MAX_OUTPUT_CHARS = 2000  # tool output goes back into a small context window


class ToolError(Exception):
    """Raise from a tool to report a failure the model should see and may correct (no retry)."""


ToolFn = Callable[[Any], str | Awaitable[str]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: ToolFn

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.args_model.model_json_schema())

    async def invoke(self, raw_args: dict[str, Any]) -> ToolResult:
        """Validate and run. Bad arguments and :class:`ToolError` become a failed result the model
        can read; any other exception propagates so the step is retried."""
        try:
            args = self.args_model.model_validate(raw_args)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or 'args'}: {e['msg']}" for e in exc.errors()
            )
            return ToolResult(ok=False, output=f"invalid arguments: {problems}")
        try:
            if inspect.iscoroutinefunction(self.fn):
                output = await self.fn(args)
            else:
                # Sync tools run on a thread so they can't stall the worker's event loop.
                output = await asyncio.to_thread(self.fn, args)
        except ToolError as exc:
            return ToolResult(ok=False, output=f"error: {exc}")
        return ToolResult(ok=True, output=_truncate(str(output)))


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"... [truncated {len(text) - MAX_OUTPUT_CHARS} chars]"


def tool(name: str | None = None, *, description: str | None = None) -> Callable[[ToolFn], Tool]:
    """Decorator: build a :class:`Tool` from a function's single pydantic-typed parameter."""

    def wrap(fn: ToolFn) -> Tool:
        params = list(inspect.signature(fn).parameters)
        hints = get_type_hints(fn)
        model = hints.get(params[0]) if len(params) == 1 else None
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise TypeError(f"{fn.__name__}: a tool takes exactly one pydantic model argument")
        desc = description or inspect.getdoc(fn) or ""
        return Tool(name or fn.__name__, desc.strip(), model, fn)

    return wrap


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"duplicate tool name: {t.name}")
        self._tools[t.name] = t

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


def idempotency_key() -> str:
    """A key that is identical across retries and crash recovery of the current tool call, and
    unique otherwise. Pass it to the system a side-effecting tool modifies."""
    return step_info().idempotency_key


# --- built-in tools -------------------------------------------------------------------------

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_MAX_EXPRESSION_CHARS = 200
_MAX_EXPONENT = 100
_MAX_MAGNITUDE = 1e30  # big ints would otherwise be computed (and printed) at any size
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")  # "1,250" -> "1250"; "3,5" stays an error


def evaluate_arithmetic(expression: str) -> int | float:
    """Evaluate ``+ - * / // % **`` over numbers. No names, calls or attributes: never ``eval``."""
    if len(expression) > _MAX_EXPRESSION_CHARS:
        raise ToolError(f"expression longer than {_MAX_EXPRESSION_CHARS} characters")
    try:
        normalized = _THOUSANDS.sub("", expression.replace("^", "**"))
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError:
        raise ToolError(f"not an arithmetic expression: {expression!r}") from None

    def ev(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = cast(int | float, node.value)
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
                raise ToolError(f"exponent larger than {_MAX_EXPONENT}")
            try:
                value = _BINARY[type(node.op)](left, right)
            except ZeroDivisionError:
                raise ToolError("division by zero") from None
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            value = _UNARY[type(node.op)](ev(node.operand))
        else:
            raise ToolError(f"unsupported syntax: {ast.unparse(node)!r}")
        if isinstance(value, complex) or abs(value) > _MAX_MAGNITUDE:
            raise ToolError("result out of range")
        return value

    return ev(tree.body)


def format_number(value: int | float) -> str:
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


class CalculatorArgs(BaseModel):
    expression: str = Field(description="e.g. (12.5 * 4) - 3 / 2")


@tool(description="Evaluate an arithmetic expression with + - * / // % ** and parentheses.")
def calculator(args: CalculatorArgs) -> str:
    return format_number(evaluate_arithmetic(args.expression))


class NoArgs(BaseModel):
    pass


@tool(description="The current date and time in UTC, ISO 8601.")
def utc_now(args: NoArgs) -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


BUILTIN_TOOLS: tuple[Tool, ...] = (calculator, utc_now)
