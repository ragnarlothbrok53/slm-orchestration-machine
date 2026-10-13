"""Durable execution: journaled steps, retries, leases, recovery.

A *workflow* is an ``async def wf(ctx, input) -> output``. Everything non-deterministic (model
calls, tools, clocks, randomness) goes through ``await ctx.step(name, fn, arg)``:

* If step ``n`` of this run is already in the journal, its recorded result is returned (or its
  recorded failure re-raised) and ``fn`` is not called.
* Otherwise ``fn`` runs under its :class:`RetryPolicy` and the outcome is written to the journal
  before ``step`` returns.

After a crash, another worker re-runs the workflow from the top; journaled steps replay in
microseconds, so execution effectively resumes at the first unfinished step. Guarantees:

* a step that finished is never executed again;
* the step that was in flight during a crash runs again (at-least-once), so steps with side
  effects should be idempotent (see :attr:`StepInfo.idempotency_key`);
* workflow code between steps must be deterministic: the same journal must lead to the same
  sequence of step names. A mismatch fails the run with :class:`NonDeterminismError`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import random
import socket
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
from typing import Any, Generic, TypeVar, cast, get_type_hints

from loguru import logger
from pydantic import TypeAdapter

from .metrics import Metrics
from .store import LeaseLostError, Run, StepRecord, Store

T = TypeVar("T")
A = TypeVar("A")
In = TypeVar("In")
Out = TypeVar("Out")

# --- errors -----------------------------------------------------------------------------------


class RetryableError(Exception):
    """A transient failure. ``retry_after`` (seconds) overrides the backoff if it's longer."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PermanentError(Exception):
    """A failure that retrying cannot fix (bad request, unknown tool, ...)."""


class StepFailedError(Exception):
    """Raised into workflow code when a step failed for good. Replays identically."""

    def __init__(self, step: str, error_type: str, message: str) -> None:
        super().__init__(f"step {step!r} failed: {error_type}: {message}")
        self.step = step
        self.error_type = error_type
        self.message = message


class NonDeterminismError(Exception):
    """Workflow code asked for a different step than the journal recorded at that position."""


# --- policies and step context --------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_s: float = 0.5
    backoff: float = 2.0
    max_s: float = 30.0
    timeout_s: float | None = None  # per attempt
    jitter: float = 0.5  # fraction of each delay that is randomized; 0 = deterministic

    def delay(
        self,
        failed_attempt: int,
        retry_after: float | None,
        rand: Callable[[], float] = random.random,
    ) -> float:
        """Exponential backoff, randomized so that clients rejected together don't all retry
        together (measured: without jitter, an overloaded server sat idle between synchronized
        retry waves; see benchmarks/bench_overload.py). ``Retry-After`` is a floor, so jitter is
        only ever added on top of it."""
        backoff = min(self.initial_s * self.backoff ** (failed_attempt - 1), self.max_s)
        backoff *= 1 - self.jitter * rand()
        if retry_after:
            return max(backoff, retry_after * (1 + self.jitter * rand()))
        return backoff


@dataclass(frozen=True)
class StepInfo:
    run_id: str
    seq: int
    name: str
    attempt: int

    @property
    def request_id(self) -> str:
        """Unique per attempt; sent as ``X-Request-ID`` so server logs join to the journal."""
        return f"{self.run_id}.{self.seq}.{self.attempt}"

    @property
    def idempotency_key(self) -> str:
        """Identical across retries *and* across crash recovery of the same step."""
        return f"{self.run_id}.{self.seq}"


_current_step: ContextVar[StepInfo] = ContextVar("slom_current_step")


def step_info() -> StepInfo:
    """The step being executed. Only valid inside a step function."""
    return _current_step.get()


@cache
def _adapter(tp: Any) -> TypeAdapter[Any]:
    return TypeAdapter(tp)


def _return_type(fn: Callable[..., Any]) -> Any:
    hints = get_type_hints(fn)
    if "return" not in hints:
        raise TypeError(f"step function {fn!r} needs a return type annotation (it is journaled)")
    return hints["return"]


class RunContext:
    """Handed to workflow code. Tracks the position in the journal."""

    def __init__(self, store: Store, run: Run, journal: list[StepRecord], metrics: Metrics) -> None:
        self.run_id = run.id
        self._store = store
        self._token = run.lease_token
        self._journal = journal
        self._metrics = metrics
        self._seq = 0
        self._last_replayed = False

    @property
    def replaying(self) -> bool:
        """True if the most recent step came from the journal. Use it to keep side effects in
        workflow code (metrics, logs) from repeating during recovery."""
        return self._last_replayed

    async def step(
        self,
        name: str,
        fn: Callable[[A], Awaitable[T]],
        arg: A,
        *,
        retry: RetryPolicy | None = None,
    ) -> T:
        seq = self._seq
        self._seq += 1
        out_type = _return_type(fn)

        if seq < len(self._journal):
            rec = self._journal[seq]
            if rec.name != name:
                raise NonDeterminismError(
                    f"step {seq} is {rec.name!r} in the journal but the workflow asked for "
                    f"{name!r}: the workflow code changed under an in-flight run"
                )
            self._last_replayed = True
            self._metrics.steps_replayed.labels(name).inc()
            return cast(T, _decode(rec, out_type))

        self._last_replayed = False
        return await self._execute(seq, name, fn, arg, retry or RetryPolicy(), out_type)

    async def _execute(
        self,
        seq: int,
        name: str,
        fn: Callable[[A], Awaitable[T]],
        arg: A,
        policy: RetryPolicy,
        out_type: Any,
    ) -> T:
        started = time.time()
        t0 = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            token = _current_step.set(StepInfo(self.run_id, seq, name, attempt))
            try:
                if policy.timeout_s is None:
                    result = await fn(arg)
                else:
                    async with asyncio.timeout(policy.timeout_s):
                        result = await fn(arg)
            except PermanentError as exc:
                error: BaseException | None = exc
                retry_after = None
                retryable = False
            except RetryableError as exc:
                error, retry_after, retryable = exc, exc.retry_after, True
            except Exception as exc:  # unknown failures are assumed transient
                error, retry_after, retryable = exc, None, True
            else:
                error = None
            finally:
                _current_step.reset(token)

            if error is None:
                payload = _adapter(out_type).dump_json(result).decode()
                await self._record(seq, name, True, payload, None, attempt, started)
                self._metrics.steps.labels(name, "ok").inc()
                self._metrics.step_duration.labels(name).observe(time.perf_counter() - t0)
                return result

            error_type = type(error).__name__
            if retryable and attempt < policy.max_attempts:
                delay = policy.delay(attempt, retry_after)
                self._metrics.step_retries.labels(name, error_type).inc()
                logger.warning(
                    "step {name} attempt {attempt} failed, retrying",
                    name=name,
                    attempt=attempt,
                    run_id=self.run_id,
                    seq=seq,
                    error=f"{error_type}: {error}",
                    retry_in_s=round(delay, 2),
                )
                await asyncio.sleep(delay)
                continue

            message = str(error) or error_type
            err_json = json.dumps({"type": error_type, "message": message})
            await self._record(seq, name, False, None, err_json, attempt, started)
            self._metrics.steps.labels(name, "failed").inc()
            raise StepFailedError(name, error_type, message) from error

    async def _record(
        self,
        seq: int,
        name: str,
        ok: bool,
        output: str | None,
        error: str | None,
        attempts: int,
        started: float,
    ) -> None:
        rec = StepRecord(seq, name, ok, output, error, attempts, started, time.time())
        await self._store.record_step(self.run_id, self._token, rec)


def _decode(rec: StepRecord, out_type: Any) -> Any:
    if rec.ok:
        return _adapter(out_type).validate_json(rec.output or "null")
    err = json.loads(rec.error or "{}")
    raise StepFailedError(rec.name, err.get("type", "Error"), err.get("message", ""))


# --- workflows and the worker ---------------------------------------------------------------


@dataclass(frozen=True)
class Workflow(Generic[In, Out]):
    name: str
    fn: Callable[[RunContext, In], Awaitable[Out]]
    input_type: Any
    output_type: Any

    @classmethod
    def define(cls, name: str, fn: Callable[[RunContext, In], Awaitable[Out]]) -> Workflow[In, Out]:
        hints = get_type_hints(fn)
        params = [p for p in hints if p != "return"]
        if len(params) != 2 or "return" not in hints:
            raise TypeError(f"workflow {name!r} must be async def (ctx, input: T) -> R")
        return cls(name, fn, hints[params[1]], hints["return"])

    def encode_input(self, value: In) -> str:
        return _adapter(self.input_type).dump_json(value).decode()

    def decode_output(self, payload: str) -> Out:
        return cast(Out, _adapter(self.output_type).validate_json(payload))


class Worker:
    """Claims runs from the store and executes them, up to ``max_concurrent_runs`` at a time."""

    def __init__(
        self,
        store: Store,
        workflows: list[Workflow[Any, Any]],
        *,
        metrics: Metrics | None = None,
        max_concurrent_runs: int = 8,
        lease_s: float = 15.0,
        poll_interval_s: float = 0.2,
        max_run_attempts: int = 5,
        shutdown_grace_s: float = 10.0,
        worker_id: str | None = None,
    ) -> None:
        self.store = store
        self.workflows = {w.name: w for w in workflows}
        self.metrics = metrics or Metrics()
        self.max_concurrent_runs = max_concurrent_runs
        self.lease_s = lease_s
        self.poll_interval_s = poll_interval_s
        self.max_run_attempts = max_run_attempts
        self.shutdown_grace_s = shutdown_grace_s
        self.id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        self._active: dict[str, asyncio.Task[None]] = {}
        self._cancel_reason: dict[str, str] = {}
        self._stopping = asyncio.Event()

    def stop(self) -> None:
        """Stop claiming new runs; :meth:`run` then drains and returns."""
        self._stopping.set()

    async def run(self) -> None:
        logger.info(
            "worker started", worker_id=self.id, max_concurrent_runs=self.max_concurrent_runs
        )
        try:
            while not self._stopping.is_set():
                if len(self._active) >= self.max_concurrent_runs:
                    await asyncio.wait(
                        self._active.values(),
                        timeout=self.poll_interval_s,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    continue
                run = await self.store.claim(self.id, self.lease_s)
                if run is None:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self._stopping.wait(), self.poll_interval_s)
                    continue
                task = asyncio.create_task(self._execute(run), name=f"run:{run.id}")
                self._active[run.id] = task
                task.add_done_callback(self._forget)
        finally:
            await self._drain()
            logger.info("worker stopped", worker_id=self.id)

    def _forget(self, task: asyncio.Task[None]) -> None:
        self._active.pop(task.get_name().removeprefix("run:"), None)

    async def _drain(self) -> None:
        """Give in-flight runs the grace period, then hand them back for another worker."""
        if not self._active:
            return
        tasks = list(self._active.values())
        _, pending = await asyncio.wait(tasks, timeout=self.shutdown_grace_s)
        for task in pending:
            run_id = task.get_name().removeprefix("run:")
            self._cancel_reason[run_id] = "shutdown"
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    async def _execute(self, run: Run) -> None:
        with logger.contextualize(run_id=run.id):
            await self._execute_inner(run)

    async def _execute_inner(self, run: Run) -> None:
        wf = self.workflows.get(run.workflow)
        if wf is None:
            await self._finish(run, "failed", error=f"no workflow named {run.workflow!r}")
            return
        if run.cancel_requested:
            await self._finish(run, "cancelled")
            return
        if run.attempts > self.max_run_attempts:
            # Each earlier claim ended with a dead or stalled worker. Stop feeding it workers.
            msg = f"abandoned after {run.attempts - 1} attempts that never finished"
            await self._finish(run, "failed", error=msg)
            return

        journal = await self.store.steps(run.id)
        resumed = run.attempts > 1 or bool(journal)
        self.metrics.run_claims.labels("resumed" if resumed else "new").inc()
        if resumed:
            logger.info("resuming run", attempt=run.attempts, journaled_steps=len(journal))

        ctx = RunContext(self.store, run, journal, self.metrics)
        me = asyncio.current_task()
        assert me is not None
        finishing = asyncio.Event()  # once set, a late cancel must not interrupt the final write
        heartbeat = asyncio.create_task(self._heartbeat(run, me, finishing))
        self.metrics.runs_active.inc()
        try:
            output = await wf.fn(ctx, _adapter(wf.input_type).validate_json(run.input))
            payload = _adapter(wf.output_type).dump_json(output).decode()
            finishing.set()
            await self._finish(run, "completed", output=payload)
            self.metrics.run_duration.observe(self.store.clock() - run.created_at)
        except asyncio.CancelledError:
            reason = self._cancel_reason.pop(run.id, "shutdown")
            if reason == "cancelled":
                await self._finish(run, "cancelled")
            elif reason == "shutdown":
                await self.store.release(run.id, run.lease_token)
                logger.info("run handed back on shutdown")
            # lease_lost: someone else owns it now; write nothing.
        except LeaseLostError:
            self._lease_lost()
        except Exception as exc:
            logger.opt(exception=not isinstance(exc, StepFailedError)).error(
                "run failed", error=f"{type(exc).__name__}: {exc}"
            )
            await self._finish(run, "failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            heartbeat.cancel()
            self.metrics.runs_active.dec()

    async def _finish(
        self, run: Run, status: Any, output: str | None = None, error: str | None = None
    ) -> None:
        try:
            await self.store.finish(run.id, run.lease_token, status, output=output, error=error)
        except LeaseLostError:
            self._lease_lost()
            return
        self.metrics.runs_finished.labels(status).inc()
        logger.info("run {status}", status=status, error=error)

    def _lease_lost(self) -> None:
        self.metrics.leases_lost.inc()
        logger.warning("lease lost; another worker owns this run now")

    async def _heartbeat(self, run: Run, task: asyncio.Task[Any], finishing: asyncio.Event) -> None:
        interval = self.lease_s / 3
        while True:
            await asyncio.sleep(interval)
            if finishing.is_set():
                return  # the result is being written; the fenced write decides the outcome
            try:
                cancel = await self.store.heartbeat(run.id, run.lease_token, self.lease_s)
            except LeaseLostError:
                self._cancel_reason[run.id] = "lease_lost"
                self._lease_lost()
                task.cancel()
                return
            except Exception as exc:  # e.g. database busy: the next beat may succeed
                logger.warning("heartbeat failed", error=repr(exc))
                continue
            if cancel:
                self._cancel_reason[run.id] = "cancelled"
                task.cancel()
                return
