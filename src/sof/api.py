"""HTTP API: submit, inspect and cancel agent runs.

The API never executes runs; it writes them to the journal and reads them back. Workers may run
in the same process (``sof serve``) or separately (``sof worker``, as many as you like on the
same host). Either way the API's view of a run is exactly what is in the journal, including the
steps finished so far.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest
from pydantic import BaseModel, Field, TypeAdapter
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import __version__
from .agent import WORKFLOW_NAME
from .config import Settings
from .contracts import AgentInput
from .metrics import Metrics
from .store import TERMINAL, Run, RunConflictError, StepRecord, Store

WorkerFactory = Callable[[Store, Metrics], Coroutine[Any, Any, None]]

_SAFE_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
_INPUT = TypeAdapter(AgentInput)


# --- schemas ---------------------------------------------------------------------------------


class RunRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    tools: list[str] | None = Field(None, description="Subset of the worker's tools; null = all.")
    max_steps: int = Field(6, ge=1, le=20, description="Model calls allowed in this run.")
    max_repairs: int = Field(2, ge=0, le=5, description="Re-prompts after unparseable output.")
    max_tokens: int = Field(256, ge=1, le=1024, description="Per model call.")
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    protocol: Literal["json", "qwen"] = Field(
        "json", description="Prompt/reply format: generic JSON, or Qwen2.5's native tool calls."
    )
    run_id: str | None = Field(
        None,
        pattern=r"^[A-Za-z0-9_-]{1,40}$",
        description="Idempotency key: submitting the same run_id again returns the same run.",
    )

    def agent_input(self) -> AgentInput:
        return AgentInput(
            question=self.question,
            tools=self.tools,
            max_steps=self.max_steps,
            max_repairs=self.max_repairs,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            protocol=self.protocol,
        )


class StepView(BaseModel):
    seq: int
    name: str
    ok: bool
    attempts: int
    duration_ms: float
    error: str | None = None


class RunView(BaseModel):
    run_id: str
    status: str
    answer: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int
    created_at: float
    updated_at: float
    steps: list[StepView]


def _view(run: Run, steps: list[StepRecord]) -> RunView:
    result = json.loads(run.output) if run.output else None
    return RunView(
        run_id=run.id,
        status=run.status,
        answer=result.get("answer") if result else None,
        result=result,
        error=run.error,
        attempts=run.attempts,
        created_at=run.created_at,
        updated_at=run.updated_at,
        steps=[
            StepView(
                seq=s.seq,
                name=s.name,
                ok=s.ok,
                attempts=s.attempts,
                duration_ms=round((s.finished_at - s.started_at) * 1000, 1),
                error=s.error,
            )
            for s in steps
        ],
    )


def _error(message: str, type_: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": type_}}, status_code=status)


# --- middleware ------------------------------------------------------------------------------


class RequestContextMiddleware:
    """``X-Request-ID`` propagation and one access-log line per request (as in slm-runtime)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = supplied if _SAFE_ID.fullmatch(supplied) else uuid.uuid4().hex[:16]
        start = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        with logger.contextualize(request_id=request_id):
            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                if scope["path"] not in ("/healthz", "/readyz", "/metrics"):
                    logger.info(
                        "{method} {path} {status}",
                        method=scope["method"],
                        path=scope["path"],
                        status=status,
                        duration_ms=round((time.perf_counter() - start) * 1000, 1),
                    )


# --- app -------------------------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    *,
    store: Store | None = None,
    metrics: Metrics | None = None,
    run_worker: WorkerFactory | None = None,
) -> FastAPI:
    """Build the API. ``run_worker``, if given, runs in the background for the app's lifetime
    (``sof serve``); it receives the same store and metrics so ``/metrics`` covers both."""
    settings = settings or Settings()
    metrics = metrics or Metrics()
    owns_store = store is None
    db = store or Store(settings.db_path)
    runs_by_status = Gauge(
        "sof_runs",
        "Runs in the journal by status (sampled at scrape)",
        ["status"],
        registry=metrics.registry,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        worker_task = asyncio.create_task(run_worker(db, metrics)) if run_worker else None
        try:
            yield
        finally:
            if worker_task is not None:
                worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)
            if owns_store:
                await db.close()

    app = FastAPI(title="sof", version=__version__, lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    app.state.store = db

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
        type_ = {404: "not_found", 409: "conflict"}.get(exc.status_code, "invalid_request_error")
        return _error(str(exc.detail), type_, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        msg = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()
        )
        return _error(msg, "invalid_request_error", 400)

    async def load(run_id: str) -> RunView:
        run = await db.get(run_id)
        if run is None:
            raise HTTPException(404, f"no run {run_id!r}")
        return _view(run, await db.steps(run_id))

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        """Ready when the journal answers. Model readiness is the workers' concern: they wait
        for it before claiming, so runs queue up as ``pending`` meanwhile."""
        try:
            async with asyncio.timeout(2.0):
                counts = await db.count_by_status()
        except Exception as exc:
            return JSONResponse({"status": "unavailable", "error": repr(exc)}, status_code=503)
        return JSONResponse({"status": "ready", "runs": counts})

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        counts = await db.count_by_status()
        for status in ("pending", "running", *sorted(TERMINAL)):
            runs_by_status.labels(status).set(counts.get(status, 0))
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/runs", status_code=202)
    async def submit(
        req: RunRequest,
        response: Response,
        wait: bool = Query(False, description="Hold the request until the run finishes."),
        timeout_s: float = Query(30.0, gt=0, description="Max wait (capped by the server)."),
    ) -> RunView:
        run_id = req.run_id or uuid.uuid4().hex[:16]
        payload = _INPUT.dump_json(
            req.agent_input()
        ).decode()  # canonical: same request, same bytes
        try:
            run, created = await db.submit(run_id, WORKFLOW_NAME, payload)
        except RunConflictError as exc:
            raise HTTPException(409, str(exc)) from None
        response.headers["Location"] = f"/v1/runs/{run.id}"
        if created:
            logger.info("run submitted", run_id=run.id)
        if wait:
            deadline = time.monotonic() + min(timeout_s, settings.api_wait_timeout_s)
            while run.status not in TERMINAL and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
                run = await db.get(run_id) or run
        if run.status in TERMINAL:
            response.status_code = 200
        return _view(run, await db.steps(run_id))

    @app.get("/v1/runs/{run_id}")
    async def get_run(run_id: str) -> RunView:
        return await load(run_id)

    @app.post("/v1/runs/{run_id}/cancel", status_code=202)
    async def cancel(run_id: str) -> RunView:
        if await db.request_cancel(run_id) is None:
            raise HTTPException(404, f"no run {run_id!r}")
        return await load(run_id)

    return app
