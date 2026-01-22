from __future__ import annotations

import os


def pytest_configure() -> None:
    # Ensure matplotlib uses a non-interactive backend for all tests.
    os.environ.setdefault("MPLBACKEND", "Agg")
