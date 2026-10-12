"""Command line: ``slom serve | api | worker | run``. Flags override ``SOF_*`` variables."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Annotated, Any

import click
import typer
import uvicorn

from .agent import WORKFLOW_NAME
from .api import create_app
from .config import Settings
from .contracts import AgentInput
from .logs import configure_logging
from .metrics import Metrics
from .store import TERMINAL, Store
from .worker import agent_worker, serve_worker

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

Db = Annotated[str | None, typer.Option("--db", help="SQLite journal path [SOF_DB_PATH].")]
LlmUrl = Annotated[str | None, typer.Option("--llm-url", help="LLM server [SOF_LLM_BASE_URL].")]


def _settings(**overrides: Any) -> Settings:
    names = {"db": "db_path", "llm_url": "llm_base_url", "host": "api_host", "port": "api_port"}
    values = {names.get(k, k): v for k, v in overrides.items() if v is not None}
    return Settings(**values)


@app.command()
def serve(
    db: Db = None,
    llm_url: LlmUrl = None,
    host: Annotated[str | None, typer.Option(help="[SOF_API_HOST]")] = None,
    port: Annotated[int | None, typer.Option(help="[SOF_API_PORT]")] = None,
) -> None:
    """API and one worker in a single process: the simplest way to run SLOM."""
    settings = _settings(db=db, llm_url=llm_url, host=host, port=port)
    configure_logging(settings.log_level, settings.log_json)

    async def embedded_worker(store: Store, metrics: Metrics) -> None:
        async with agent_worker(settings, store, metrics) as worker:
            await worker.run()

    api = create_app(settings, run_worker=embedded_worker)
    uvicorn.run(api, host=settings.api_host, port=settings.api_port, log_config=None)


@app.command("api")
def api_only(
    db: Db = None,
    host: Annotated[str | None, typer.Option(help="[SOF_API_HOST]")] = None,
    port: Annotated[int | None, typer.Option(help="[SOF_API_PORT]")] = None,
) -> None:
    """The API alone; run workers separately with `slom worker`."""
    settings = _settings(db=db, host=host, port=port)
    configure_logging(settings.log_level, settings.log_json)
    uvicorn.run(
        create_app(settings), host=settings.api_host, port=settings.api_port, log_config=None
    )


@app.command()
def worker(db: Db = None, llm_url: LlmUrl = None) -> None:
    """A worker process. Start several to share the load and take over each other's runs."""
    asyncio.run(serve_worker(_settings(db=db, llm_url=llm_url)))


@app.command()
def run(
    question: str,
    tool: Annotated[list[str] | None, typer.Option(help="Allow only these tools.")] = None,
    max_steps: int = 6,
    protocol: Annotated[
        str, typer.Option(click_type=click.Choice(["json", "qwen"]), help="Prompt protocol.")
    ] = "json",
    as_json: Annotated[bool, typer.Option("--json", help="Print the full result.")] = False,
    db: Db = None,
    llm_url: LlmUrl = None,
) -> None:
    """Run one agent task to completion in this process and print the trace."""
    settings = _settings(db=db, llm_url=llm_url)
    configure_logging("WARNING", settings.log_json)
    status, result, error = asyncio.run(
        _run_once(settings, AgentInput(question, tool, max_steps, protocol=protocol))  # type: ignore[arg-type]  # validated by click.Choice
    )
    if as_json:
        typer.echo(json.dumps({"status": status, "result": result, "error": error}, indent=2))
        return
    if result is None:
        typer.echo(f"run {status}: {error}", err=True)
        raise typer.Exit(1)
    for step in result["steps"]:
        typer.echo(f"model> {step['model_output']}")
        if step["observation"] is not None:
            label = "tool" if step["kind"] == "tool" else "parser"
            typer.echo(f"{label}> {step['observation']}")
    typer.echo(f"\n{result['status']}: {result['answer']}")
    usage = result["usage"]
    typer.echo(
        f"[{usage['llm_calls']} LLM calls, {usage['tool_calls']} tool calls, "
        f"{usage['prompt_tokens']}+{usage['completion_tokens']} tokens, "
        f"{usage['llm_seconds']:.1f}s in the model]"
    )


async def _run_once(
    settings: Settings, inp: AgentInput
) -> tuple[str, dict[str, Any] | None, str | None]:
    from pydantic import TypeAdapter

    store = Store(settings.db_path, synchronous=settings.db_synchronous)
    try:
        run_id = uuid.uuid4().hex[:16]
        await store.submit(run_id, WORKFLOW_NAME, TypeAdapter(AgentInput).dump_json(inp).decode())
        async with agent_worker(settings, store, Metrics()) as w:
            task = asyncio.create_task(w.run())
            # Poll the journal: it is the source of truth, whichever process writes it.
            while (current := await store.get(run_id)) and current.status not in TERMINAL:  # noqa: ASYNC110
                await asyncio.sleep(0.1)
            w.stop()
            await task
        assert current is not None
        result = json.loads(current.output) if current.output else None
        return current.status, result, current.error
    finally:
        await store.close()
