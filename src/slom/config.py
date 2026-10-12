"""Configuration, loaded from environment variables (prefix ``SOF_``) or a ``.env`` file."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SOF_", env_file=".env", extra="ignore")

    # --- durable state ---
    db_path: str = Field("slom.db", description="SQLite journal, shared by the API and workers.")
    db_synchronous: Literal["FULL", "NORMAL"] = Field(
        "FULL",
        description="FULL: finished steps survive power loss. NORMAL: they survive process "
        "crashes only, but commits are cheaper.",
    )

    # --- worker ---
    max_concurrent_runs: int = Field(8, ge=1, description="Runs one worker executes at once.")
    lease_s: float = Field(
        15.0,
        gt=0,
        description="How long a run stays claimed without a heartbeat. After a worker crash, "
        "another worker resumes the run once this expires.",
    )
    poll_interval_s: float = Field(0.2, gt=0, description="Idle delay between claim attempts.")
    max_run_attempts: int = Field(
        5,
        ge=1,
        description="Claims allowed per run; protects workers from a run that crashes them.",
    )
    shutdown_grace_s: float = Field(10.0, ge=0, description="Time in-flight runs get on SIGTERM.")

    # --- LLM backend (any OpenAI-compatible /v1/completions server; built for slim-serve) ---
    llm_base_url: str = "http://127.0.0.1:8000"
    llm_timeout_s: float = Field(
        130.0,
        gt=0,
        description="Per-attempt limit. Keep it above slim-serve's SLM_REQUEST_TIMEOUT_S (120) "
        "so the server's own 504 arrives first.",
    )
    llm_max_attempts: int = Field(
        8,
        ge=1,
        description="Attempts per LLM call, including the first. With the default backoff "
        "(1s doubling to 30s, jittered) the 7 waits add up to 45-90s of server downtime, "
        "e.g. a model reload.",
    )
    llm_concurrency: int = Field(
        1,
        ge=1,
        description="LLM calls one worker sends at once. Match slim-serve's SLM_REPLICAS: more "
        "requests only wait in its queue, and past replicas + queue size it answers 503.",
    )

    # --- API ---
    api_host: str = "127.0.0.1"
    api_port: int = 8080
    api_wait_timeout_s: float = Field(
        60.0, gt=0, description="Upper bound for POST /v1/runs?wait=true before answering 202."
    )

    # --- observability ---
    metrics_port: int | None = Field(
        9464, description="Worker Prometheus port (the API serves /metrics itself); 0 disables it."
    )
    log_level: str = "INFO"
    log_json: bool = False
