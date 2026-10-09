"""HTTP contract: submit/wait/inspect/cancel, idempotency, validation, probes, metrics."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from helpers import ScriptedLlm, make_worker

from sof.api import create_app
from sof.config import Settings
from sof.metrics import Metrics
from sof.store import Store

pytestmark = pytest.mark.anyio

ANSWER = '"answer": "4"}'
CALC = '"tool": "calculator", "args": {"expression": "2+2"}}'


@asynccontextmanager
async def api(
    tmp_path: Path, replies: Sequence[str | asyncio.Event] | None = None
) -> AsyncIterator[httpx.AsyncClient]:
    """The app with an embedded worker fed by scripted replies (or no worker at all)."""
    settings = Settings(db_path=str(tmp_path / "api.db"), api_wait_timeout_s=5)
    llm = ScriptedLlm(list(replies or []))

    async def worker(store: Store, metrics: Metrics) -> None:
        await make_worker(store, llm, metrics=metrics, lease_s=0.3).run()

    app = create_app(settings, run_worker=worker if replies is not None else None)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def test_submit_and_wait_returns_the_answer(tmp_path: Path) -> None:
    async with api(tmp_path, [CALC, ANSWER]) as client:
        r = await client.post("/v1/runs?wait=true", json={"question": "2+2?"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "completed" and body["answer"] == "4"
        assert body["result"]["status"] == "answered"
        assert [s["name"] for s in body["steps"]] == [
            "describe_tools",
            "llm",
            "tool:calculator",
            "llm",
        ]
        assert r.headers["location"] == f"/v1/runs/{body['run_id']}"
        assert "x-request-id" in r.headers

        again = await client.get(f"/v1/runs/{body['run_id']}")
        assert again.status_code == 200 and again.json()["answer"] == "4"


async def test_without_wait_the_run_is_accepted_and_pending(tmp_path: Path) -> None:
    async with api(tmp_path) as client:  # no worker
        r = await client.post("/v1/runs", json={"question": "2+2?"})
        assert r.status_code == 202 and r.json()["status"] == "pending"
        # wait=true with nothing to run it: returns 202 after the requested timeout
        r = await client.post("/v1/runs?wait=true&timeout_s=0.1", json={"question": "x"})
        assert r.status_code == 202


async def test_submit_is_idempotent_by_run_id(tmp_path: Path) -> None:
    async with api(tmp_path) as client:
        body = {"question": "2+2?", "run_id": "order-42"}
        first = await client.post("/v1/runs", json=body)
        second = await client.post("/v1/runs", json=body)
        assert first.status_code == second.status_code == 202
        assert first.json()["run_id"] == second.json()["run_id"] == "order-42"
        assert first.json()["created_at"] == second.json()["created_at"]

        conflict = await client.post("/v1/runs", json={**body, "question": "3+3?"})
        assert conflict.status_code == 409
        assert conflict.json()["error"]["type"] == "conflict"


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({"question": ""}, "question"),
        ({"question": "x", "max_steps": 0}, "max_steps"),
        ({"question": "x", "max_tokens": 5000}, "max_tokens"),
        ({"question": "x", "run_id": "has spaces"}, "run_id"),
        ({"question": "x", "protocol": "xml"}, "protocol"),
    ],
)
async def test_validation_errors_are_400(
    tmp_path: Path, body: dict[str, object], field: str
) -> None:
    async with api(tmp_path) as client:
        r = await client.post("/v1/runs", json=body)
        assert r.status_code == 400
        err = r.json()["error"]
        assert err["type"] == "invalid_request_error" and field in err["message"]


async def test_unknown_run_is_404(tmp_path: Path) -> None:
    async with api(tmp_path) as client:
        for r in (await client.get("/v1/runs/nope"), await client.post("/v1/runs/nope/cancel")):
            assert r.status_code == 404 and r.json()["error"]["type"] == "not_found"


async def test_cancel_a_running_run(tmp_path: Path) -> None:
    hang = asyncio.Event()
    async with api(tmp_path, [hang]) as client:
        run_id = (await client.post("/v1/runs", json={"question": "x"})).json()["run_id"]
        for _ in range(200):
            if (await client.get(f"/v1/runs/{run_id}")).json()["status"] == "running":
                break
            await asyncio.sleep(0.01)
        r = await client.post(f"/v1/runs/{run_id}/cancel")
        assert r.status_code == 202
        for _ in range(300):
            status = (await client.get(f"/v1/runs/{run_id}")).json()["status"]
            if status == "cancelled":
                break
            await asyncio.sleep(0.01)
        assert status == "cancelled"


async def test_probes_and_metrics(tmp_path: Path) -> None:
    async with api(tmp_path, [ANSWER]) as client:
        assert (await client.get("/healthz")).json() == {"status": "ok"}
        await client.post("/v1/runs?wait=true", json={"question": "x"})
        ready = await client.get("/readyz")
        assert ready.status_code == 200 and ready.json()["runs"] == {"completed": 1}
        text = (await client.get("/metrics")).text
        assert 'sof_runs{status="completed"} 1.0' in text
        assert 'sof_runs_finished_total{outcome="completed"} 1.0' in text  # embedded worker
        assert 'sof_agent_runs_total{status="answered"} 1.0' in text
