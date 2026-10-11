"""LlmClient: parsing slm-runtime's SSE stream and mapping its failures to retry decisions."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from sof.contracts import LlmRequest
from sof.engine import PermanentError, RetryableError
from sof.llm import LlmClient

pytestmark = pytest.mark.anyio


def sse(*events: object) -> bytes:
    lines = [f"data: {json.dumps(e)}\n\n" for e in events] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def chunk(text: str, finish: str | None = None, usage: dict[str, int] | None = None) -> object:
    event: dict[str, object] = {"choices": [{"index": 0, "text": text, "finish_reason": finish}]}
    if usage:
        event["usage"] = usage
    return event


def client(handler: Callable[[httpx.Request], httpx.Response], **kw: object) -> LlmClient:
    return LlmClient("http://llm", transport=httpx.MockTransport(handler), **kw)  # type: ignore[arg-type]


async def test_streams_text_usage_and_sends_request_payload() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        body = sse(
            chunk("Hel"),
            chunk("lo"),
            chunk("", "stop", {"prompt_tokens": 3, "completion_tokens": 2}),
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    llm = client(handler)
    resp = await llm.complete(LlmRequest("hi", max_tokens=8, stop=["a", "b", "c", "d", "e"]))
    assert (resp.text, resp.finish_reason) == ("Hello", "stop")
    assert (resp.prompt_tokens, resp.completion_tokens) == (3, 2)
    assert resp.ttft_s is not None and resp.latency_s >= resp.ttft_s
    payload = json.loads(seen[0].content)
    assert payload["stream"] is True and payload["max_tokens"] == 8
    assert payload["stop"] == ["a", "b", "c", "d"]  # the API allows 4
    assert seen[0].headers["x-request-id"] == resp.request_id
    await llm.aclose()


@pytest.mark.parametrize(
    ("status", "headers", "error", "retry_after"),
    [
        (503, {"retry-after": "1"}, RetryableError, 1.0),  # slm-runtime: queue full
        (503, {}, RetryableError, None),  # model still loading
        (429, {"retry-after": "7"}, RetryableError, 7.0),
        (504, {}, RetryableError, None),  # slm-runtime: deadline exceeded
        (500, {}, RetryableError, None),  # engine error
        (408, {}, RetryableError, None),
        (400, {}, PermanentError, None),  # e.g. max_tokens over the server's limit
        (404, {}, PermanentError, None),
    ],
)
async def test_status_codes_map_to_retry_decisions(
    status: int, headers: dict[str, str], error: type[Exception], retry_after: float | None
) -> None:
    body = {"error": {"message": "nope", "type": "overloaded"}}
    llm = client(lambda _: httpx.Response(status, json=body, headers=headers))
    with pytest.raises(error, match=f"HTTP {status} overloaded: nope") as info:
        await llm.complete(LlmRequest("hi"))
    if isinstance(info.value, RetryableError):
        assert info.value.retry_after == retry_after


async def test_in_band_stream_error_and_truncated_stream_are_retryable() -> None:
    err = {"error": {"message": "deadline", "type": "timeout"}}
    llm = client(lambda _: httpx.Response(200, content=sse(chunk("a"), err)))
    with pytest.raises(RetryableError, match="stream failed: timeout"):
        await llm.complete(LlmRequest("hi"))

    llm = client(lambda _: httpx.Response(200, content=sse(chunk("a"))))
    with pytest.raises(RetryableError, match="without a finish_reason"):
        await llm.complete(LlmRequest("hi"))


async def test_connection_failures_are_retryable() -> None:
    def refuse(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    with pytest.raises(RetryableError, match="ConnectError"):
        await client(refuse).complete(LlmRequest("hi"))


async def test_total_deadline_covers_slow_streams() -> None:
    class Slow(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield b'data: {"choices": [{"text": "a", "finish_reason": null}]}\n\n'
            await asyncio.sleep(5)
            yield b"data: [DONE]\n\n"

    llm = client(lambda _: httpx.Response(200, stream=Slow()), timeout_s=0.2)
    with pytest.raises(RetryableError, match=r"within 0.2s"):
        await llm.complete(LlmRequest("hi"))


async def test_concurrency_cap() -> None:
    active = 0
    peak = 0

    class Tracked(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1
            yield sse(chunk("x", "stop"))

    llm = client(lambda _: httpx.Response(200, stream=Tracked()), concurrency=2)
    await asyncio.gather(*(llm.complete(LlmRequest("hi")) for _ in range(6)))
    assert peak == 2


async def test_readiness() -> None:
    assert await client(lambda _: httpx.Response(200)).ready()
    assert not await client(lambda _: httpx.Response(503)).ready()
    assert await client(lambda _: httpx.Response(404)).ready()  # server without /readyz
