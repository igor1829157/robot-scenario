"""Локальное исполнение generated-кода с ограничением по времени."""

from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from typing import Any, Callable

from meta_agent.runtime.source_validation import (
    SAFE_BUILTINS,
    validate_source,
)


DEFAULT_COMPONENT_TIMEOUT_SEC = 1.0


class LocalRuntime:
    """Исполняет проверенный generated-код в процессе с timeout."""

    def __init__(self) -> None:
        self._functions: dict[str, Callable[..., Any]] = {}
        self._timeouts: dict[str, float] = {}

    def register(
        self,
        component_id: str,
        source_code: str,
        entrypoint: str,
        timeout_sec: float | None = None,
    ) -> None:
        """Компилирует и регистрирует функцию компонента.

        Args:
            component_id: Идентификатор компонента.
            source_code: Исходный код.
            entrypoint: Имя функции-точки входа.
            timeout_sec: Лимит вызова; по умолчанию 1.0 с.
        """
        validate_source(source_code, entrypoint)

        namespace: dict[str, Any] = {}

        exec(
            compile(source_code, component_id, "exec"),
            {"__builtins__": SAFE_BUILTINS},
            namespace,
        )

        function = namespace.get(entrypoint)

        if not callable(function):
            raise ValueError(
                f"Entrypoint {entrypoint} is not a callable function"
            )

        self._functions[component_id] = function
        self._timeouts[component_id] = (
            DEFAULT_COMPONENT_TIMEOUT_SEC
            if timeout_sec is None
            else float(timeout_sec)
        )

    def invoke(
        self,
        component_id: str,
        arguments: dict[str, Any],
        timeout_sec: float | None = None,
    ) -> Any:
        """Вызывает зарегистрированный компонент с timeout.

        Args:
            component_id: Идентификатор компонента.
            arguments: Именованные аргументы entrypoint.
            timeout_sec: Переопределение timeout регистрации.

        Returns:
            Результат вызова функции.

        Raises:
            KeyError: Компонент не зарегистрирован.
            TimeoutError: Превышен лимит времени.
        """
        function = self._functions[component_id]
        timeout = (
            float(timeout_sec)
            if timeout_sec is not None
            else self._timeouts[component_id]
        )

        if timeout <= 0:
            return function(**arguments)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(function, **arguments)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError as error:
                raise TimeoutError(
                    f"component {component_id} timed out "
                    f"after {timeout} seconds"
                ) from error

    def close(self) -> None:
        """Сбрасывает зарегистрированные компоненты."""
        self._functions.clear()
        self._timeouts.clear()
