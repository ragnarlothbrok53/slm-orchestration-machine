"""Prompt protocols: how tools, history and replies are laid out for the model.

slim-serve serves the raw completions API, so the client owns the chat template (ChatML here,
as used by Qwen2.5 and many other small models). Two protocols are implemented, because which one
a small model follows better is an empirical question (see ``benchmarks/eval_agent.py``):

* ``json``: a generic contract (``{"tool": ..., "args": ...}`` or ``{"answer": ...}``) that works
  with any instruction-tuned ChatML model. The assistant turn is *prefilled* with ``{``, which
  forces even a 0.5B model to start a JSON object instead of chatting.
* ``qwen``: Qwen2.5's own function-calling format (``<tool_call>`` / ``<tool_response>``), which
  the model saw in training. Plain text is the final answer.

Pure and deterministic: the workflow re-renders the prompt from its history on replay.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .contracts import Step, ToolSpec
from .parsing import Decision, parse_json_decision, parse_qwen_decision

ProtocolName = Literal["json", "qwen"]
STOP = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]


def chatml(turns: Sequence[tuple[str, str]], prefill: str = "") -> str:
    body = "".join(f"<|im_start|>{role}\n{content}<|im_end|>\n" for role, content in turns)
    return f"{body}<|im_start|>assistant\n{prefill}"


@dataclass(frozen=True)
class Protocol:
    name: ProtocolName
    prefill: str
    # Prefill that leaves the model no choice but to answer, used when it is looping or out of
    # steps. None: the protocol can't force an answer this way.
    answer_prefill: str | None
    system: Callable[[Sequence[ToolSpec]], str]
    observation: Callable[[Step], str]
    parse: Callable[[str, Collection[str]], Decision]

    def prefill_for(self, force_answer: bool) -> str:
        if force_answer and self.answer_prefill is not None:
            return self.answer_prefill
        return self.prefill

    def render(
        self,
        tools: Sequence[ToolSpec],
        question: str,
        steps: Sequence[Step],
        force_answer: bool = False,
    ) -> str:
        """The full prompt for the next model call, ending in the (prefilled) assistant turn."""
        turns = [("system", self.system(tools)), ("user", question)]
        for step in steps:
            turns.append(("assistant", step.model_output))
            turns.append(("user", self.observation(step)))
        return chatml(turns, self.prefill_for(force_answer))


# --- json ------------------------------------------------------------------------------------

_JSON_FORMAT = """\
Reply with exactly ONE JSON object and nothing else, in one of these two forms:
{"tool": "<tool name>", "args": {<arguments>}}
{"answer": "<final answer>"}"""

_JSON_SYSTEM = """\
You solve the user's task step by step, using tools when they help.

Available tools:
{tools}

{format}

Always use a tool for arithmetic instead of computing in your head. After each tool call you \
receive its result. When you know the final answer, reply with the "answer" form."""


def _describe_args(parameters: dict[str, Any]) -> str:
    """Render a JSON Schema as a one-line example; small models follow examples better."""
    props: dict[str, Any] = parameters.get("properties", {})
    parts = []
    for name, schema in props.items():
        hint = schema.get("type", "value")
        if desc := schema.get("description"):
            hint = f"{hint}, {desc}"
        parts.append(f'"{name}": <{hint}>')
    return "{" + ", ".join(parts) + "}"


def _json_system(tools: Sequence[ToolSpec]) -> str:
    lines = [
        f"- {t.name}: {t.description} Arguments: {_describe_args(t.parameters)}" for t in tools
    ]
    return _JSON_SYSTEM.format(tools="\n".join(lines) or "(none)", format=_JSON_FORMAT)


def _json_observation(step: Step) -> str:
    if step.kind in ("invalid", "repeat"):
        return f"Your reply was invalid: {step.observation}.\n{_JSON_FORMAT}"
    return f"Result of {step.tool}: {step.observation}"


# --- qwen ------------------------------------------------------------------------------------

# What Qwen2.5's chat template (from the GGUF metadata) renders for requests with tools, byte for
# byte, including its literal double braces around the call example.
_QWEN_SYSTEM = """\
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within \
<tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def _qwen_system(tools: Sequence[ToolSpec]) -> str:
    schemas = [
        json.dumps(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            },
            ensure_ascii=False,
        )
        for t in tools
    ]
    return _QWEN_SYSTEM.format(tools="\n".join(schemas))


def _qwen_observation(step: Step) -> str:
    if step.kind in ("invalid", "repeat"):
        return f"<tool_response>\nerror: {step.observation}\n</tool_response>"
    return f"<tool_response>\n{step.observation}\n</tool_response>"


PROTOCOLS: dict[str, Protocol] = {
    "json": Protocol(
        "json", "{", '{"answer": "', _json_system, _json_observation, parse_json_decision
    ),
    "qwen": Protocol("qwen", "", None, _qwen_system, _qwen_observation, parse_qwen_decision),
}
