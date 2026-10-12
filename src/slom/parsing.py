"""Turn raw model output into a decision: call a tool, answer, or explain what was wrong.

Small models get the *intent* right far more often than the *syntax*: they wrap JSON in prose or
code fences, emit Python-style dicts, or say ``arguments`` instead of ``args``. The parser accepts
those variants and rejects everything else with a reason short enough to feed back to the model.

This module is pure and deterministic because it runs inside workflow code, which Temporal replays.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolDecision:
    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class AnswerDecision:
    answer: str


@dataclass(frozen=True)
class InvalidOutput:
    reason: str


Decision = ToolDecision | AnswerDecision | InvalidOutput

_UNTERMINATED_ANSWER = re.compile(r'\s*\{\s*"answer"\s*:\s*"([^"]+)"?\s*$')


def extract_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object (or Python dict literal) embedded anywhere in ``text``."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            obj = _literal_dict_at(text, start)
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    return None


def _literal_dict_at(text: str, start: int) -> object:
    """Parse a Python dict literal (``{'a': 1, 'b': True}``) starting at ``start``.

    ``ast.literal_eval`` only evaluates literals, so this is safe on untrusted text.
    """
    depth = 0
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return ast.literal_eval(text[start : end + 1])
                # Any failure means "not a dict literal here". literal_eval raises more than its
                # docs suggest: "{{...}}" is a set containing a dict -> TypeError (unhashable).
                except Exception:
                    return None
    return None


def parse_json_decision(text: str, tool_names: Collection[str]) -> Decision:
    """The ``json`` protocol: ``{"tool": name, "args": {...}}`` or ``{"answer": "..."}``."""
    obj = extract_object(text)
    if obj is None:
        if m := _UNTERMINATED_ANSWER.match(text):  # forced answer, cut off before its closing
            return AnswerDecision(m.group(1).strip())
        return InvalidOutput("no JSON object found")

    has_tool, has_answer = "tool" in obj, "answer" in obj
    if has_tool and has_answer:
        return InvalidOutput('use either "tool" or "answer", not both')
    if has_answer:
        answer = obj["answer"]
        if isinstance(answer, dict | list):
            answer = json.dumps(answer)
        return AnswerDecision(str(answer).strip())
    if not has_tool:
        return InvalidOutput('the object needs a "tool" or an "answer" key')
    return _tool_decision(obj["tool"], obj.get("args", obj.get("arguments", {})), tool_names)


def parse_qwen_decision(text: str, tool_names: Collection[str]) -> Decision:
    """The ``qwen`` protocol (Qwen2.5's trained format): a call is
    ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``; any other text is the answer.
    """
    start = text.find("<tool_call>")
    if start == -1:
        answer = text.strip()
        return AnswerDecision(answer) if answer else InvalidOutput("empty reply")
    body = text[start + len("<tool_call>") :].split("</tool_call>", 1)[0]
    obj = extract_object(body)
    if obj is None:
        return InvalidOutput("no JSON object inside <tool_call>")
    return _tool_decision(obj.get("name"), obj.get("arguments", obj.get("args", {})), tool_names)


def _tool_decision(name: object, args: object, tool_names: Collection[str]) -> Decision:
    if not isinstance(name, str) or name not in tool_names:
        available = ", ".join(sorted(tool_names)) or "none"
        return InvalidOutput(f"unknown tool {name!r}; available tools: {available}")
    if not isinstance(args, dict):
        return InvalidOutput("the arguments must be a JSON object")
    return ToolDecision(name, args)
