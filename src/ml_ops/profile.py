"""Profiling utilities for performance optimization."""

import cProfile
import os
import pstats
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from loguru import logger

# Check if profiling is enabled via environment variable
PROFILING_ENABLED = os.getenv("ENABLE_PROFILING", "false").lower() == "true"
PROFILE_OUTPUT_DIR = Path(os.getenv("PROFILE_OUTPUT_DIR", "runs/profiling"))


@contextmanager
def torch_profiler(
    enabled: bool = True,
    activities: list | None = None,
    record_shapes: bool = True,
    profile_memory: bool = True,
    output_dir: Path | None = None,
    trace_name: str = "trace",
):
    """Context manager for PyTorch profiling.

    Args:
        enabled: Whether profiling is enabled.
        activities: List of activities to profile (CPU, CUDA, etc.).
        record_shapes: Whether to record tensor shapes.
        profile_memory: Whether to profile memory usage.
        output_dir: Directory to save profiling results.
        trace_name: Name for the trace file.

    Yields:
        Profiler instance if enabled, None otherwise.
    """
    if not enabled or not PROFILING_ENABLED:
        yield None
        return

    if activities is None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

    output_dir = output_dir or PROFILE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=True,
    ) as prof:
        yield prof

    # Export results
    trace_path = output_dir / f"{trace_name}.json"
    prof.export_chrome_trace(str(trace_path))
    logger.info(f"PyTorch profiler trace saved to: {trace_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PyTorch Profiler Summary")
    logger.info("=" * 60)
    logger.info(prof.key_averages().table(sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"))


@contextmanager
def cprofile_context(
    enabled: bool = True,
    output_dir: Path | None = None,
    profile_name: str = "profile",
    sort_by: str = "cumulative",
    print_stats: int = 20,
):
    """Context manager for cProfile profiling.

    Args:
        enabled: Whether profiling is enabled.
        output_dir: Directory to save profiling results.
        profile_name: Name for the profile file.
        sort_by: How to sort stats ('cumulative', 'time', 'calls', etc.).
        print_stats: Number of top functions to print.

    Yields:
        Profiler instance if enabled, None otherwise.
    """
    if not enabled or not PROFILING_ENABLED:
        yield None
        return

    profiler = cProfile.Profile()
    profiler.enable()
    yield profiler
    profiler.disable()

    output_dir = output_dir or PROFILE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save stats to file
    stats_path = output_dir / f"{profile_name}.stats"
    with open(stats_path, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats(sort_by)
        stats.print_stats()

    logger.info(f"cProfile stats saved to: {stats_path}")

    # Print summary to console
    logger.info("\n" + "=" * 60)
    logger.info(f"cProfile Summary ({profile_name})")
    logger.info("=" * 60)
    stats = pstats.Stats(profiler)
    stats.sort_stats(sort_by)
    stats.print_stats(print_stats)


def profile_function(func: Any) -> Any:
    """Decorator to profile a function using cProfile.

    Args:
        func: Function to profile.

    Returns:
        Wrapped function with profiling.
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not PROFILING_ENABLED:
            return func(*args, **kwargs)

        profile_name = f"{func.__module__}.{func.__name__}"
        with cprofile_context(profile_name=profile_name):
            return func(*args, **kwargs)

    return wrapper


def should_profile(step: int, profile_every_n_steps: int = 1) -> bool:
    """Check if profiling should be enabled for this step.

    Args:
        step: Current step/batch index.
        profile_every_n_steps: Profile every N steps (1 = every step, 10 = every 10th step).

    Returns:
        True if profiling should be enabled.
    """
    if not PROFILING_ENABLED:
        return False
    return step % profile_every_n_steps == 0
