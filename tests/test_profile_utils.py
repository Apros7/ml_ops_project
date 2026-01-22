from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from ml_ops import profile as profile_mod


def test_torch_profiler_disabled_by_flag(tmp_path: Path) -> None:
    with profile_mod.torch_profiler(enabled=False, output_dir=tmp_path, trace_name="x") as prof:
        assert prof is None

    assert not (tmp_path / "x.json").exists()


def test_torch_profiler_disabled_by_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(profile_mod, "PROFILING_ENABLED", False)

    with profile_mod.torch_profiler(enabled=True, output_dir=tmp_path, trace_name="x") as prof:
        assert prof is None

    assert not (tmp_path / "x.json").exists()


def test_torch_profiler_enabled_exports_trace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(profile_mod, "PROFILING_ENABLED", True)
    monkeypatch.setattr(profile_mod.torch.cuda, "is_available", lambda: False)

    calls: dict[str, Any] = {}

    class _FakeKeyAverages:
        def __init__(self) -> None:
            self.sort_by: str | None = None

        def table(self, *, sort_by: str) -> str:
            self.sort_by = sort_by
            return "table"

    class _FakeProfiler:
        def __init__(self) -> None:
            self.exported_path: str | None = None
            self._ka = _FakeKeyAverages()

        def export_chrome_trace(self, path: str) -> None:
            self.exported_path = path

        def key_averages(self) -> _FakeKeyAverages:
            return self._ka

    @contextmanager
    def _fake_profile(**kwargs: Any):
        calls.update(kwargs)
        prof = _FakeProfiler()
        yield prof

    monkeypatch.setattr(profile_mod.torch.profiler, "profile", _fake_profile)

    with profile_mod.torch_profiler(enabled=True, output_dir=tmp_path, trace_name="trace") as prof:
        assert prof is not None

    assert "activities" in calls
    assert Path(prof.exported_path) == tmp_path / "trace.json"
    assert prof.key_averages().sort_by == "self_cpu_time_total"


def test_cprofile_context_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(profile_mod, "PROFILING_ENABLED", False)
    with profile_mod.cprofile_context(enabled=True, output_dir=tmp_path, profile_name="p") as prof:
        assert prof is None
    assert not (tmp_path / "p.stats").exists()


def test_cprofile_context_enabled_writes_stats(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(profile_mod, "PROFILING_ENABLED", True)

    # Keep log output quiet in test runs.
    monkeypatch.setattr(profile_mod.logger, "info", lambda *_args, **_kwargs: None)

    with profile_mod.cprofile_context(enabled=True, output_dir=tmp_path, profile_name="p", print_stats=1) as prof:
        assert prof is not None
        sum(i * i for i in range(100))

    assert (tmp_path / "p.stats").exists()


def test_profile_function_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_mod, "PROFILING_ENABLED", False)

    @profile_mod.profile_function
    def f(x: int) -> int:
        return x + 1

    assert f(1) == 2


def test_profile_function_uses_cprofile_context_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_mod, "PROFILING_ENABLED", True)

    called: list[str] = []

    @contextmanager
    def _fake_cprofile_context(*, profile_name: str, **_kwargs: Any):
        called.append(profile_name)
        yield object()

    monkeypatch.setattr(profile_mod, "cprofile_context", _fake_cprofile_context)

    expected_profile_name = f"{__name__}.f"

    @profile_mod.profile_function
    def f(x: int) -> int:
        return x + 1

    assert f(1) == 2
    assert called == [expected_profile_name]


def test_should_profile_disabled_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_mod, "PROFILING_ENABLED", False)
    assert profile_mod.should_profile(0, profile_every_n_steps=1) is False


def test_should_profile_enabled_step_mod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_mod, "PROFILING_ENABLED", True)
    assert profile_mod.should_profile(0, profile_every_n_steps=3) is True
    assert profile_mod.should_profile(1, profile_every_n_steps=3) is False
    assert profile_mod.should_profile(3, profile_every_n_steps=3) is True
