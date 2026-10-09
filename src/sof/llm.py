"""Client for an OpenAI-compatible ``/v1/completions`` server, written against slm-runtime.

It translates the server's failure contract into retry decisions for the engine:

========================================  =========================================
Server says                               Client does
========================================  =========================================
``503`` (overloaded / model loading)      retry, waiting at least ``Retry-After``
``504`` deadline, ``5xx``, ``408``        retry with backoff
``429``                                   retry, honouring ``Retry-After``
other ``4xx`` (e.g. ``max_tokens`` limit) fail the step: retrying can't fix it
connection refused / reset / timeout      retry with backoff
in-band ``data: {"error": ...}``          retry (the stream died after headers)
stream ends without ``finish_reason``     retry (truncated response)
========================================  =========================================

It always streams, for two reasons: client-side TTFT is measurable, and cancelling a run closes
the connection, which slm-runtime detects and uses to stop generating within about one token.
A non-streaming request would keep the model busy until ``max_tokens``.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

import httpx

from .contracts import LlmRequest, LlmResponse
from .engine import PermanentError, RetryableError, step_info
from .metrics import Metrics


def _retry_after(headers: httpx.Headers) -> float | None:
    try:
        return max(0.0, float(headers.get("retry-after", "")))
    except ValueError:
        return None  # HTTP-date form: rare, and the backoff covers it


def _error_message(body: bytes) -> tuple[str, str]:
    try:
        err = json.loads(body)["error"]
        return str(err.get("type", "error")), str(err.get("message", ""))
    except (ValueError, KeyError, TypeError):
        return "error", body[:200].decode("utf-8", "replace")


class LlmClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 130.0,
        concurrency: int = 1,
        metrics: Metrics | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.metrics = metrics or Metrics()
        # Caps requests in flight from this process, so excess load waits here instead of
        # overflowing the server's queue into 503s. The per-attempt deadline starts only once a
        # slot is acquired, so waiting for a slot can't time a request out.
        self._slots = asyncio.Semaphore(concurrency)
        self._http = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def ready(self) -> bool:
        """True when the server reports its model loaded (``/readyz``; servers without that
        endpoint are assumed ready once they answer)."""
        try:
            r = await self._http.get("/readyz", timeout=2.0)
        except httpx.TransportError:
            return False
        return r.status_code in (200, 404)

    async def complete(self, req: LlmRequest) -> LlmResponse:
        try:
            request_id = step_info().request_id
        except LookupError:  # called outside a step (tests, scripts)
            request_id = uuid.uuid4().hex[:16]
        payload: dict[str, Any] = {
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": True,
        }
        if req.stop:
            payload["stop"] = req.stop[:4]  # the OpenAI API allows at most 4
        if req.seed is not None:
            payload["seed"] = req.seed

        async with self._slots:
            try:
                async with asyncio.timeout(self.timeout_s):
                    return await self._stream(payload, request_id)
            except TimeoutError:
                self.metrics.llm_errors.labels("timeout").inc()
                raise RetryableError(f"no complete response within {self.timeout_s}s") from None
            except httpx.TransportError as exc:
                self.metrics.llm_errors.labels("connection").inc()
                raise RetryableError(f"{type(exc).__name__}: {exc}") from None

    async def _stream(self, payload: dict[str, Any], request_id: str) -> LlmResponse:
        t0 = time.perf_counter()
        ttft: float | None = None
        parts: list[str] = []
        finish_reason: str | None = None
        usage: dict[str, int] = {}
        headers = {"X-Request-ID": request_id}

        async with self._http.stream("POST", "/v1/completions", json=payload, headers=headers) as r:
            if r.status_code != 200:
                raise self._status_error(r.status_code, await r.aread(), r.headers)
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    err = event["error"]
                    self.metrics.llm_errors.labels(str(err.get("type", "stream"))).inc()
                    raise RetryableError(f"stream failed: {err.get('type')}: {err.get('message')}")
                choice = event["choices"][0]
                if text := choice.get("text"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(text)
                finish_reason = choice.get("finish_reason") or finish_reason
                usage = event.get("usage") or usage

        if finish_reason is None:
            self.metrics.llm_errors.labels("truncated").inc()
            raise RetryableError("stream ended without a finish_reason")
        if ttft is not None:
            self.metrics.llm_ttft.observe(ttft)
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        self.metrics.llm_tokens.labels("prompt").inc(prompt_tokens)
        self.metrics.llm_tokens.labels("completion").inc(completion_tokens)
        return LlmResponse(
            text="".join(parts),
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=ttft,
            latency_s=time.perf_counter() - t0,
            request_id=request_id,
        )

    def _status_error(
        self, status: int, body: bytes, headers: httpx.Headers
    ) -> RetryableError | PermanentError:
        type_, message = _error_message(body)
        self.metrics.llm_errors.labels(type_ if status in (503, 504) else str(status)).inc()
        text = f"HTTP {status} {type_}: {message}"
        if status in (429, 503):
            return RetryableError(text, retry_after=_retry_after(headers))
        if status == 408 or status >= 500:
            return RetryableError(text)
        return PermanentError(text)
