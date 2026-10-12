from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from slom.store import Store


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeClock:
    """Wall clock for leases that tests advance by hand instead of sleeping."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
async def store(tmp_path: Path, clock: FakeClock) -> AsyncIterator[Store]:
    s = Store(tmp_path / "slom.db", clock=clock)
    yield s
    await s.close()
