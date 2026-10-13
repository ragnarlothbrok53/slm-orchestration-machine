"""Logging setup: human-readable by default, one JSON object per line with ``SLOM_LOG_JSON=true``.

Same format and conventions as slim-serve, so the two services' logs can be read side by side.
"""

from __future__ import annotations

import inspect
import logging
import sys
from types import FrameType

from loguru import logger


class _InterceptHandler(logging.Handler):
    """Route stdlib logging (uvicorn, httpx) through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame: FrameType | None = inspect.currentframe()
        depth = 0
        while frame is not None and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame, depth = frame.f_back, depth + 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    logger.remove()
    if json:
        logger.add(sys.stderr, level=level, serialize=True)
    else:
        logger.add(
            sys.stderr,
            level=level,
            format="<green>{time:HH:mm:ss.SSS}</green> <level>{level: <7}</level> "
            "{message} <dim>{extra}</dim>",
        )
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers = [_InterceptHandler()]
        logging.getLogger(name).propagate = False
    logging.getLogger("uvicorn.access").disabled = True  # the API writes its own access log
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per LLM call is enough
