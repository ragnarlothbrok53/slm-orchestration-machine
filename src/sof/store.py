"""SQLite journal: runs, who holds them (leases), and the result of every finished step.

Why SQLite: the workload is small transactional writes from several processes (API, workers) on
one host. WAL mode gives concurrent readers plus one serialized writer via file locks, which is
exactly what claim-by-lease needs, and it ships with Python. ``synchronous=FULL`` means a step
reported as done survives power loss, not just a process crash. Multi-host deployments need a
server database (Postgres with ``FOR UPDATE SKIP LOCKED``); everything that touches SQL is in
this module.

All calls run on one dedicated thread per ``Store``: ``sqlite3`` is blocking, and one connection
used from one thread is the simplest correct arrangement.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

RunStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
TERMINAL: frozenset[str] = frozenset({"completed", "failed", "cancelled"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id               TEXT PRIMARY KEY,
    workflow         TEXT NOT NULL,
    input            TEXT NOT NULL,             -- JSON
    status           TEXT NOT NULL,             -- see RunStatus
    output           TEXT,                      -- JSON, once completed
    error            TEXT,                      -- once failed
    attempts         INTEGER NOT NULL DEFAULT 0,-- claims that did not end in a clean release
    lease_owner      TEXT,
    lease_until      REAL,                      -- unix time; NULL or past = claimable
    lease_token      INTEGER NOT NULL DEFAULT 0,-- +1 per claim; fences writes by stale owners
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_open ON runs (created_at) WHERE status IN ('pending', 'running');
CREATE TABLE IF NOT EXISTS steps (
    run_id      TEXT NOT NULL REFERENCES runs (id),
    seq         INTEGER NOT NULL,
    name        TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    output      TEXT,                           -- JSON result when ok
    error       TEXT,                           -- JSON {type, message} when it failed for good
    attempts    INTEGER NOT NULL,
    started_at  REAL NOT NULL,
    finished_at REAL NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"""


class RunConflictError(Exception):
    """A run with this ID already exists with a different workflow or input."""


class LeaseLostError(Exception):
    """This worker no longer owns the run (its lease expired and someone else claimed it)."""


@dataclass(frozen=True)
class Run:
    id: str
    workflow: str
    input: str
    status: RunStatus
    output: str | None
    error: str | None
    attempts: int
    lease_token: int
    cancel_requested: bool
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class StepRecord:
    seq: int
    name: str
    ok: bool
    output: str | None
    error: str | None
    attempts: int
    started_at: float
    finished_at: float


_RUN_COLS = (
    "id, workflow, input, status, output, error, attempts, lease_token, cancel_requested, "
    "created_at, updated_at"
)
_STEP_COLS = "seq, name, ok, output, error, attempts, started_at, finished_at"

T = TypeVar("T")


def _run(row: tuple[object, ...]) -> Run:
    r = list(row)
    r[8] = bool(r[8])
    return Run(*r)  # type: ignore[arg-type]


def _step(row: tuple[object, ...]) -> StepRecord:
    r = list(row)
    r[2] = bool(r[2])
    return StepRecord(*r)  # type: ignore[arg-type]


class Store:
    def __init__(self, path: str | Path, clock: Callable[[], float] = time.time) -> None:
        self.path = str(path)
        self.clock = clock  # wall clock: leases are compared across processes
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sof-db")
        self._conn: sqlite3.Connection | None = None

    # --- plumbing -------------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.path, isolation_level=None, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(_SCHEMA)
            self._conn = conn
        return self._conn

    async def _call(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(self._connect()))

    async def _tx(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``fn`` in a write transaction. IMMEDIATE takes the write lock up front, so two
        workers can't both read a run as claimable and then both claim it."""

        def tx(conn: sqlite3.Connection) -> T:
            conn.execute("BEGIN IMMEDIATE")
            try:
                result = fn(conn)
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
            return result

        return await self._call(tx)

    async def close(self) -> None:
        def close(_: sqlite3.Connection) -> None:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

        await self._call(close)
        self._executor.shutdown(wait=True)

    # --- API side -------------------------------------------------------------------------

    async def submit(self, run_id: str, workflow: str, input_json: str) -> tuple[Run, bool]:
        """Create a run, or return the existing one with this ID (idempotent submit).

        Returns ``(run, created)``. Reusing an ID with a different input is an error, not a
        silent no-op: the caller would otherwise get someone else's result.
        """
        now = self.clock()

        def tx(conn: sqlite3.Connection) -> tuple[Run, bool]:
            cur = conn.execute(
                "INSERT INTO runs (id, workflow, input, status, created_at, updated_at) "
                "VALUES (?, ?, ?, 'pending', ?, ?) ON CONFLICT (id) DO NOTHING",
                (run_id, workflow, input_json, now, now),
            )
            run = _run(
                conn.execute(f"SELECT {_RUN_COLS} FROM runs WHERE id = ?", (run_id,)).fetchone()
            )
            if cur.rowcount == 0 and (run.workflow, run.input) != (workflow, input_json):
                raise RunConflictError(f"run {run_id!r} exists with a different input")
            return run, cur.rowcount == 1

        return await self._tx(tx)

    async def get(self, run_id: str) -> Run | None:
        def q(conn: sqlite3.Connection) -> Run | None:
            row = conn.execute(f"SELECT {_RUN_COLS} FROM runs WHERE id = ?", (run_id,)).fetchone()
            return _run(row) if row else None

        return await self._call(q)

    async def steps(self, run_id: str) -> list[StepRecord]:
        def q(conn: sqlite3.Connection) -> list[StepRecord]:
            rows = conn.execute(
                f"SELECT {_STEP_COLS} FROM steps WHERE run_id = ? ORDER BY seq", (run_id,)
            ).fetchall()
            return [_step(r) for r in rows]

        return await self._call(q)

    async def request_cancel(self, run_id: str) -> Run | None:
        """Pending runs are cancelled at once; running ones are flagged for their worker."""
        now = self.clock()

        def tx(conn: sqlite3.Connection) -> Run | None:
            conn.execute(
                "UPDATE runs SET status = 'cancelled', updated_at = ? "
                "WHERE id = ? AND status = 'pending' AND lease_owner IS NULL",
                (now, run_id),
            )
            conn.execute(
                "UPDATE runs SET cancel_requested = 1, updated_at = ? "
                "WHERE id = ? AND status IN ('pending', 'running')",
                (now, run_id),
            )
            row = conn.execute(f"SELECT {_RUN_COLS} FROM runs WHERE id = ?", (run_id,)).fetchone()
            return _run(row) if row else None

        return await self._tx(tx)

    async def count_by_status(self) -> dict[str, int]:
        def q(conn: sqlite3.Connection) -> dict[str, int]:
            rows = conn.execute("SELECT status, COUNT(*) FROM runs GROUP BY status").fetchall()
            return {str(s): int(n) for s, n in rows}

        return await self._call(q)

    # --- worker side ----------------------------------------------------------------------

    async def claim(self, owner: str, lease_s: float) -> Run | None:
        """Take the oldest run that is pending, or running under an expired lease."""
        now = self.clock()

        def tx(conn: sqlite3.Connection) -> Run | None:
            row = conn.execute(
                "SELECT id FROM runs WHERE status IN ('pending', 'running') "
                "AND (lease_until IS NULL OR lease_until < ?) ORDER BY created_at LIMIT 1",
                (now,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE runs SET status = 'running', lease_owner = ?, lease_until = ?, "
                "lease_token = lease_token + 1, attempts = attempts + 1, updated_at = ? "
                "WHERE id = ?",
                (owner, now + lease_s, now, row[0]),
            )
            return _run(conn.execute(f"SELECT {_RUN_COLS} FROM runs WHERE id = ?", row).fetchone())

        return await self._tx(tx)

    async def heartbeat(self, run_id: str, token: int, lease_s: float) -> bool:
        """Extend the lease. Returns whether cancellation was requested."""
        now = self.clock()

        def tx(conn: sqlite3.Connection) -> bool:
            cur = conn.execute(
                "UPDATE runs SET lease_until = ? WHERE id = ? AND lease_token = ? "
                "AND status = 'running'",
                (now + lease_s, run_id, token),
            )
            if cur.rowcount == 0:
                raise LeaseLostError(run_id)
            row = conn.execute("SELECT cancel_requested FROM runs WHERE id = ?", (run_id,))
            return bool(row.fetchone()[0])

        return await self._tx(tx)

    async def record_step(self, run_id: str, token: int, step: StepRecord) -> None:
        """Append a finished step, only if ``token`` still holds the lease (fencing)."""

        def tx(conn: sqlite3.Connection) -> None:
            _check_owner(conn, run_id, token)
            try:
                conn.execute(
                    f"INSERT INTO steps (run_id, {_STEP_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        step.seq,
                        step.name,
                        int(step.ok),
                        step.output,
                        step.error,
                        step.attempts,
                        step.started_at,
                        step.finished_at,
                    ),
                )
            except sqlite3.IntegrityError:
                raise LeaseLostError(run_id) from None  # another owner already wrote this step

        await self._tx(tx)

    async def finish(
        self,
        run_id: str,
        token: int,
        status: Literal["completed", "failed", "cancelled"],
        output: str | None = None,
        error: str | None = None,
    ) -> None:
        now = self.clock()

        def tx(conn: sqlite3.Connection) -> None:
            _check_owner(conn, run_id, token)
            conn.execute(
                "UPDATE runs SET status = ?, output = ?, error = ?, lease_owner = NULL, "
                "lease_until = NULL, updated_at = ? WHERE id = ?",
                (status, output, error, now, run_id),
            )

        await self._tx(tx)

    async def release(self, run_id: str, token: int) -> None:
        """Hand a run back unfinished (graceful shutdown); it doesn't count as a failed attempt."""
        now = self.clock()

        def tx(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE runs SET lease_owner = NULL, lease_until = NULL, "
                "attempts = MAX(attempts - 1, 0), updated_at = ? "
                "WHERE id = ? AND lease_token = ? AND status = 'running'",
                (now, run_id, token),
            )

        await self._tx(tx)


def _check_owner(conn: sqlite3.Connection, run_id: str, token: int) -> None:
    row = conn.execute(
        "SELECT 1 FROM runs WHERE id = ? AND lease_token = ? AND status = 'running'",
        (run_id, token),
    ).fetchone()
    if row is None:
        raise LeaseLostError(run_id)
