"""Prometheus metrics. Each ``Metrics`` owns its registry, so tests and embedded workers stay apart.

Everything here counts *live* work only. Steps served from the journal during recovery are counted
separately (``sof_steps_replayed_total``) and never inflate latency, token or error metrics.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

_LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry

        # --- runs (engine) ---
        self.runs_active = Gauge("sof_runs_active", "Runs this worker is executing", registry=r)
        self.run_claims = Counter(
            "sof_run_claims_total",
            "Runs claimed; kind=resumed means another worker had started it (crash/shutdown)",
            ["kind"],
            registry=r,
        )
        self.runs_finished = Counter(
            "sof_runs_finished_total", "Runs finished, by outcome", ["outcome"], registry=r
        )
        self.run_duration = Histogram(
            "sof_run_duration_seconds",
            "Submit to completion, for completed runs (includes queueing and recovery)",
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.leases_lost = Counter(
            "sof_leases_lost_total",
            "Runs abandoned because another worker took the lease",
            registry=r,
        )

        # --- steps (engine) ---
        self.steps = Counter(
            "sof_steps_total",
            "Steps executed live, by final outcome",
            ["step", "outcome"],
            registry=r,
        )
        self.step_duration = Histogram(
            "sof_step_duration_seconds",
            "Wall time of a live step, all attempts and backoff included",
            ["step"],
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.step_retries = Counter(
            "sof_step_retries_total",
            "Failed attempts that were retried",
            ["step", "error"],
            registry=r,
        )
        self.steps_replayed = Counter(
            "sof_steps_replayed_total",
            "Steps answered from the journal instead of re-executed (recovery at work)",
            ["step"],
            registry=r,
        )

        # --- LLM (client) ---
        self.llm_ttft = Histogram(
            "sof_llm_ttft_seconds",
            "Client-side time to first token (queueing at the server + prefill)",
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.llm_tokens = Counter(
            "sof_llm_tokens_total", "Tokens reported by the server", ["kind"], registry=r
        )
        self.llm_errors = Counter(
            "sof_llm_errors_total", "Failed LLM attempts by cause", ["cause"], registry=r
        )

        # --- agent ---
        self.agent_outcomes = Counter(
            "sof_agent_runs_total", "Agent runs by how they ended", ["status"], registry=r
        )
        self.agent_invalid_outputs = Counter(
            "sof_agent_invalid_outputs_total",
            "Model replies that could not be parsed",
            registry=r,
        )
        self.tool_calls = Counter(
            "sof_tool_calls_total", "Tool calls by tool and result", ["tool", "ok"], registry=r
        )
