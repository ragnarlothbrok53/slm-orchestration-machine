"""Task success with a real small model: does the orchestration make it reliably right?

Modes, all against the same slm-runtime server and model, greedy decoding (temperature 0):

* ``cot``: one model call, no tools; the prompt asks it to reason step by step and end with
  ``Answer: <number>``. The baseline: what the model does on its own.
* ``agent-json``: SOF's agent with the calculator, generic JSON protocol (prefill, repair, guard).
* ``agent-qwen``: the same, with Qwen2.5's native tool-call format.

Agents run through the full stack (journal, worker, retries), not a shortcut. Grading: the last
number in the final answer must equal the expected integer.

Requires a running slm-runtime with a real model, e.g. in the slm-inference-engine repo:

    uv run slm-runtime serve            # Qwen2.5-0.5B-Instruct Q4_K_M by default
    uv run python benchmarks/eval_agent.py --llm-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from common import summarize, write_results
from pydantic import TypeAdapter

from sof.agent import WORKFLOW_NAME
from sof.config import Settings
from sof.contracts import AgentInput, AgentResult, LlmRequest
from sof.llm import LlmClient
from sof.logs import configure_logging
from sof.metrics import Metrics
from sof.prompts import STOP, chatml
from sof.store import TERMINAL, Store
from sof.worker import agent_worker

DATA = Path(__file__).parent / "data" / "arith.jsonl"
# Problems looked at while developing the agent loop (repeat guard, forced answer). Accuracy is
# also reported without them, since the design could be overfit to these.
DEV_IDS = {"arith-00", "arith-01", "arith-02"}
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
COT_SYSTEM = (
    "You are a careful assistant. Solve the problem step by step. "
    "End with a final line of the form 'Answer: <number>'."
)


def final_number(text: str | None) -> float | None:
    if not text:
        return None
    tail = text.rsplit("Answer:", 1)[-1]
    found = NUMBER.findall(tail) or NUMBER.findall(text)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


async def run_cot(llm: LlmClient, q: dict[str, Any]) -> dict[str, Any]:
    prompt = chatml([("system", COT_SYSTEM), ("user", q["question"])])
    t0 = time.perf_counter()
    resp = await llm.complete(LlmRequest(prompt=prompt, max_tokens=384, temperature=0.0, stop=STOP))
    got = final_number(resp.text)
    return {
        "answer_text": resp.text.strip()[-200:],
        "got": got,
        "llm_calls": 1,
        "tool_calls": 0,
        "invalid": 0,
        "status": resp.finish_reason,
        "seconds": round(time.perf_counter() - t0, 2),
        "completion_tokens": resp.completion_tokens,
    }


async def run_agent(store: Store, q: dict[str, Any], protocol: str) -> dict[str, Any]:
    inp = AgentInput(question=q["question"], protocol=protocol, max_steps=6)  # type: ignore[arg-type]
    run_id = f"{protocol}-{q['id']}"
    t0 = time.perf_counter()
    await store.submit(run_id, WORKFLOW_NAME, TypeAdapter(AgentInput).dump_json(inp).decode())
    while (run := await store.get(run_id)) and run.status not in TERMINAL:
        await asyncio.sleep(0.05)
    assert run is not None
    if run.status != "completed":
        return {
            "answer_text": run.error,
            "got": None,
            "llm_calls": None,
            "tool_calls": None,
            "invalid": None,
            "status": run.status,
            "seconds": None,
            "completion_tokens": None,
        }
    result = TypeAdapter(AgentResult).validate_json(run.output or "")
    return {
        "answer_text": (result.answer or "")[-200:],
        "got": final_number(result.answer),
        "llm_calls": result.usage.llm_calls,
        "tool_calls": result.usage.tool_calls,
        "invalid": sum(s.kind == "invalid" for s in result.steps),
        "status": result.status,
        "seconds": round(time.perf_counter() - t0, 2),
        "completion_tokens": result.usage.completion_tokens,
        "transcript": json.dumps([s.model_output for s in result.steps])[:2000],
    }


def model_name(url: str) -> str:
    try:
        return str(httpx.get(f"{url}/v1/models", timeout=5).json()["data"][0]["id"])
    except Exception:
        return "unknown"


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--llm-url", default="http://127.0.0.1:8000")
    ap.add_argument("--modes", nargs="+", default=["cot", "agent-json", "agent-qwen"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--name", default="eval-agent")
    args = ap.parse_args()
    configure_logging("WARNING")

    questions = [json.loads(line) for line in DATA.read_text().splitlines()][: args.limit]
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as d:
        settings = Settings(
            db_path=str(Path(d) / "eval.db"),
            llm_base_url=args.llm_url,
            poll_interval_s=0.02,
            max_concurrent_runs=1,
        )
        store = Store(settings.db_path)
        llm = LlmClient(args.llm_url, timeout_s=settings.llm_timeout_s)
        async with agent_worker(settings, store, Metrics()) as worker:
            task = asyncio.create_task(worker.run())
            # Interleave modes per question, so slow drift on the host affects all modes alike.
            for q in questions:
                for mode in args.modes:
                    if mode == "cot":
                        out = await run_cot(llm, q)
                    else:
                        out = await run_agent(store, q, mode.removeprefix("agent-"))
                    out["correct"] = int(
                        out["got"] is not None and abs(out["got"] - q["answer"]) < 1e-6
                    )
                    rows.append({"id": q["id"], "mode": mode, "expected": q["answer"], **out})
                    print(
                        f"{q['id']} {mode:10s} expected={q['answer']:<6} got={out['got']} "
                        f"{'OK ' if out['correct'] else 'BAD'} calls={out['llm_calls']} "
                        f"{out['seconds']}s"
                    )
            worker.stop()
            await task
        await llm.aclose()
        await store.close()

    summary: dict[str, Any] = {
        "model": model_name(args.llm_url),
        "questions": len(questions),
        "dataset": str(DATA.name),
    }
    for mode in args.modes:
        mine = [r for r in rows if r["mode"] == mode]
        ok = [r for r in mine if r["seconds"] is not None]
        held_out = [r for r in mine if r["id"] not in DEV_IDS]
        summary[mode] = {
            "accuracy": round(sum(r["correct"] for r in mine) / len(mine), 3),
            "correct": f"{sum(r['correct'] for r in mine)}/{len(mine)}",
            "held_out_correct": f"{sum(r['correct'] for r in held_out)}/{len(held_out)}",
            "status": {s: sum(r["status"] == s for r in mine) for s in {r["status"] for r in mine}},
            "llm_calls_mean": round(sum(r["llm_calls"] for r in ok) / max(len(ok), 1), 2),
            "used_tool": sum(1 for r in ok if r["tool_calls"]),
            "runs_with_invalid_output": sum(1 for r in ok if r["invalid"]),
            "seconds": summarize([r["seconds"] for r in ok]),
        }
        print(mode, summary[mode])
    print("wrote", write_results(args.name, summary, rows))


if __name__ == "__main__":
    asyncio.run(main())
