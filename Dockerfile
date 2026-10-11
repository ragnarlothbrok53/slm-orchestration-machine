# syntax=docker/dockerfile:1
# One image for every role: `sof serve` (default), `sof api`, `sof worker`.
# The SQLite journal lives in /data; share that volume between the API and the workers.
#
#   docker build -t sof .
#   docker run --rm -p 8080:8080 -v sof-data:/data -e SOF_LLM_BASE_URL=http://host:8000 sof

FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, so source edits don't invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.13-slim
RUN useradd --create-home --uid 10001 app && mkdir /data && chown app /data
COPY --from=builder /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    SOF_DB_PATH=/data/sof.db \
    SOF_API_HOST=0.0.0.0 \
    SOF_LOG_JSON=true
USER app
WORKDIR /home/app
VOLUME /data
# 8080: API (also serves /metrics). 9464: Prometheus endpoint of `sof worker`.
EXPOSE 8080 9464
CMD ["sof", "serve"]
