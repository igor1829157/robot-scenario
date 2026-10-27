"""Единый вывод результатов robomap-эксперимента."""

from __future__ import annotations

from experiments.robomap.runner import ExperimentResult


def print_experiment_result(result: ExperimentResult) -> None:
    """Печатает стандартную сводку после run_experiment."""
    print("session:", result.session_id)
    print("run_dir:", result.run_dir)
    print("snapshot:", result.metrics["snapshot"])
    print("metrics:", result.metrics_path)
    if result.plot_path is not None:
        print("plot:", result.plot_path)
    if result.videos is not None:
        print("videos:", result.videos)
