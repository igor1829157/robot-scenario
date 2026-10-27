"""Проверка неизменности frozen-полей AgentConfig."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from meta_agent.config.schema import AgentConfig


FROZEN_FIELDS: tuple[str, ...] = (
    "model",
    "provider_routing",
    "reasoning",
    "model_kwargs",
    "initial_user_message",
)


def _frozen_snapshot(config: AgentConfig) -> dict[str, Any]:
    data = asdict(config)
    return {name: data[name] for name in FROZEN_FIELDS}


def assert_frozen_unchanged(
    candidate: AgentConfig,
    baseline: AgentConfig,
) -> None:
    """Падает, если candidate изменил инфраструктурные/бюджетные поля.

    Args:
        candidate: Конфиг мета-агента.
        baseline: Parent/baseline для сравнения.

    Raises:
        ValueError: При любом отличии в ``FROZEN_FIELDS``.
    """
    left = _frozen_snapshot(candidate)
    right = _frozen_snapshot(baseline)
    changed = [
        name for name in FROZEN_FIELDS if left[name] != right[name]
    ]
    if changed:
        raise ValueError(
            "frozen fields changed vs baseline: "
            + ", ".join(changed)
        )
