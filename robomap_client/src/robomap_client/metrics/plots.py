from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_exploration_timeseries(
    series: list[dict[str, Any]],
    *,
    title: str,
    out_path: str | Path,
) -> Path:
    """Строит график относительных exploration-метрик (%)."""
    out_path = Path(out_path)
    ticks = [p["tick"] for p in series]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title)

    ax.plot(ticks, [p["discovery_percentage"] for p in series], label="discovery %")
    ax.plot(ticks, [p["fully_explored_percentage"] for p in series], label="fully %")
    ax.set_xlabel("tick")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
