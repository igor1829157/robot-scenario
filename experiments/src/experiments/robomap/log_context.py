"""Контекст scenario/session для логов параллельных прогонов."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from collections.abc import Iterator

_scenario: ContextVar[str | None] = ContextVar(
    "robomap_log_scenario",
    default=None,
)
_session_id: ContextVar[str | None] = ContextVar(
    "robomap_log_session_id",
    default=None,
)


class RunContextFilter(logging.Filter):
    """Добавляет в LogRecord поле ``run_ctx`` с scenario/session."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Заполняет ``record.run_ctx`` из contextvars.

        Args:
            record: Запись лога.

        Returns:
            Всегда ``True`` (запись не отфильтровывается).
        """
        parts: list[str] = []
        scenario = _scenario.get()
        session_id = _session_id.get()
        if scenario:
            parts.append(f"scenario={scenario}")
        if session_id:
            parts.append(f"session={session_id}")
        record.run_ctx = f" {' '.join(parts)}" if parts else ""
        return True


@contextmanager
def run_log_context(
    *,
    scenario: str | None = None,
    session_id: str | None = None,
) -> Iterator[None]:
    """Временно привязывает scenario/session к текущему потоку.

    Args:
        scenario: Имя сценария (мира).
        session_id: ID сессии Robomap.
    """
    tokens: list[tuple[ContextVar[str | None], Token]] = []
    if scenario is not None:
        tokens.append((_scenario, _scenario.set(scenario)))
    if session_id is not None:
        tokens.append((_session_id, _session_id.set(session_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


@contextmanager
def bind_session(session_id: str) -> Iterator[None]:
    """Привязывает ``session_id`` к логам текущего потока.

    Args:
        session_id: ID сессии Robomap.
    """
    with run_log_context(session_id=session_id):
        yield


def configure_logging(*, level: int = logging.INFO) -> None:
    """Настраивает root logger с префиксом scenario/session.

    Args:
        level: Уровень логирования root logger.
    """
    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        logging.basicConfig(level=level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s]%(run_ctx)s %(message)s",
        datefmt="%H:%M:%S",
    )
    context_filter = RunContextFilter()

    for handler in root.handlers:
        handler.setFormatter(formatter)
        if not any(
            isinstance(existing, RunContextFilter)
            for existing in handler.filters
        ):
            handler.addFilter(context_filter)
