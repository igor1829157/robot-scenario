from typing import Any

from robomap_client.metrics.exploration import calculate_exploration_metrics


def build_exploration_timeseries(
    frames: list[dict[str, Any]],
    *,
    robot_id: str = "robot_1",
) -> list[dict[str, Any]]:
    """Строит timeseries exploration по кадрам симуляции."""
    series: list[dict[str, Any]] = []
    for frame in frames:
        metrics = calculate_exploration_metrics(frame, robot_id=robot_id)
        series.append(
            {
                "tick": frame.get("tick", 0),
                "discovered_objects": metrics["discovered_objects"],
                "fully_explored_objects": metrics["fully_explored_objects"],
                "discovery_percentage": metrics["discovery_percentage"],
                "fully_explored_percentage": metrics["fully_explored_percentage"],
            }
        )
    return series


def summarize_timeseries(
    series: list[dict[str, Any]],
    total_objects: int,
) -> dict[str, Any]:
    """Сводка timeseries; AUC — средний % exploration за прогон (0–100)."""
    if not series:
        return {"points": 0, "total_top_level_objects": total_objects}

    def _normalized_auc(key: str) -> float:
        """Time-weighted average percentage: integral / tick span."""
        if len(series) == 1:
            return round(float(series[0][key]), 2)

        tick_span = series[-1]["tick"] - series[0]["tick"]
        if tick_span <= 0:
            return round(float(series[-1][key]), 2)

        area = 0.0
        for a, b in zip(series, series[1:]):
            dt = b["tick"] - a["tick"]
            area += 0.5 * (a[key] + b[key]) * dt
        return round(area / tick_span, 2)

    return {
        "points": len(series),
        "tick_start": series[0]["tick"],
        "tick_end": series[-1]["tick"],
        "final_discovered": series[-1]["discovered_objects"],
        "final_fully_explored": series[-1]["fully_explored_objects"],
        "first_discovery_tick": next(
            (p["tick"] for p in series if p["discovered_objects"] > 0), None
        ),
        "first_full_explore_tick": next(
            (p["tick"] for p in series if p["fully_explored_objects"] > 0), None
        ),
        "total_top_level_objects": total_objects,
        "discovery_auc": _normalized_auc("discovery_percentage"),
        "fully_explored_auc": _normalized_auc("fully_explored_percentage"),
    }
