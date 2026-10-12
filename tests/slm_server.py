"""slim-serve (the real server, v0.1.0) running in-process with a scripted engine."""

from __future__ import annotations

import re
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import uvicorn
from slim_serve.app import create_app as create_slm_app
from slim_serve.config import Settings as SlmSettings
from slim_serve.engines import Chunk, Engine, GenerationParams

CALL = '"tool": "calculator", "args": {"expression": "17*23+4"}}'
ANSWER = '"answer": "395"}'


class ScriptedEngine(Engine):
    """Answers like a model that uses the calculator once, then answers. Thread-safe counters."""

    name = "scripted"

    def __init__(self, token_latency_s: float = 0.0, max_text: str | None = None) -> None:
        self.token_latency_s = token_latency_s
        self.max_text = max_text  # if set, always generate this until max_tokens
        self.calls = 0
        self.prompts: list[str] = []
        self.tokens = 0
        self._lock = threading.Lock()

    def load(self) -> None:
        pass

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str, params: GenerationParams) -> Iterator[Chunk]:
        with self._lock:
            self.calls += 1
            self.prompts.append(prompt)
        if self.max_text is not None:
            pieces = [self.max_text] * params.max_tokens
        else:
            reply = ANSWER if "Result of calculator" in prompt else CALL
            pieces = re.findall(r"\S+\s*", reply)
        for piece in pieces:
            if self.token_latency_s:
                time.sleep(self.token_latency_s)
            with self._lock:
                self.tokens += 1
            yield Chunk(text=piece)
        yield Chunk(text="", finish_reason="stop" if self.max_text is None else "length")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def slim_serve(engine: Engine, **settings: object) -> Iterator[str]:
    port = _free_port()
    app = create_slm_app(SlmSettings(engine="fake", **settings), engines=[engine])  # type: ignore[arg-type]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{base}/readyz").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.05)
        yield base
    finally:
        server.should_exit = True
        thread.join(timeout=10)
