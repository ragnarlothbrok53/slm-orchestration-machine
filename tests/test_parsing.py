"""Parsing model output: the variants small models really produce, and clear rejections."""

import pytest

from sof.parsing import (
    AnswerDecision,
    InvalidOutput,
    ToolDecision,
    extract_object,
    parse_json_decision,
    parse_qwen_decision,
)

TOOLS = {"calculator", "utc_now"}


@pytest.mark.parametrize(
    "text",
    [
        '{"tool": "calculator", "args": {"expression": "2+2"}}',
        'Sure! ```json\n{"tool": "calculator", "args": {"expression": "2+2"}}\n```',
        "{'tool': 'calculator', 'args': {'expression': '2+2'}}",  # Python dict literal
        '{"tool": "calculator", "arguments": {"expression": "2+2"}}',  # OpenAI-style key
        '{"tool": "calculator", "args": {"expression": "2+2"}} and then I will answer',
        '{ "tool": "calculator", "args": { "expression": "2+2" } }',
    ],
)
def test_json_tool_call_variants(text: str) -> None:
    assert parse_json_decision(text, TOOLS) == ToolDecision("calculator", {"expression": "2+2"})


def test_json_answer_is_stringified_and_stripped() -> None:
    assert parse_json_decision('{"answer": 395}', TOOLS) == AnswerDecision("395")
    assert parse_json_decision('{"answer": "  ok "}', TOOLS) == AnswerDecision("ok")
    assert parse_json_decision('{"answer": [1, 2]}', TOOLS) == AnswerDecision("[1, 2]")


def test_tool_without_args_defaults_to_empty() -> None:
    assert parse_json_decision('{"tool": "utc_now"}', TOOLS) == ToolDecision("utc_now", {})


@pytest.mark.parametrize(
    ("text", "reason_part"),
    [
        ("I think the answer is 4", "no JSON object"),
        ('{"tool": "search", "args": {}}', "unknown tool 'search'"),
        ('{"tool": "calculator", "args": "2+2"}', "must be a JSON object"),
        ('{"tool": "calculator", "answer": "4"}', "not both"),
        ('{"result": 4}', '"tool" or an "answer"'),
        ('{"tool": 7}', "unknown tool 7"),
        ("{not json at all", "no JSON object"),
    ],
)
def test_json_rejections_explain_the_problem(text: str, reason_part: str) -> None:
    decision = parse_json_decision(text, TOOLS)
    assert isinstance(decision, InvalidOutput)
    assert reason_part in decision.reason


def test_unknown_tool_lists_available_tools() -> None:
    decision = parse_json_decision('{"tool": "x"}', TOOLS)
    assert isinstance(decision, InvalidOutput)
    assert "calculator, utc_now" in decision.reason


def test_qwen_tool_call() -> None:
    text = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "1+1"}}\n</tool_call>'
    assert parse_qwen_decision(text, TOOLS) == ToolDecision("calculator", {"expression": "1+1"})


def test_qwen_tool_call_with_template_double_braces_and_no_closing_tag() -> None:
    # Qwen2.5's template shows the call example in double braces, and the model copies them.
    text = '<tool_call>\n{{"name": "calculator", "arguments": {"expression": "1+1"}}'
    assert parse_qwen_decision(text, TOOLS) == ToolDecision("calculator", {"expression": "1+1"})


def test_qwen_plain_text_is_the_answer() -> None:
    assert parse_qwen_decision(" She pays 395.\n", TOOLS) == AnswerDecision("She pays 395.")
    assert isinstance(parse_qwen_decision("   ", TOOLS), InvalidOutput)
    assert isinstance(parse_qwen_decision("<tool_call>oops</tool_call>", TOOLS), InvalidOutput)


def test_extract_object_ignores_non_dict_json_and_unbalanced_braces() -> None:
    assert extract_object("[1, 2] then {}") == {}
    assert extract_object("{{{") is None
    assert extract_object("{'__import__': 1}") == {"__import__": 1}  # literal, never evaluated
