from typing import Any, Protocol


class CodeRuntime(Protocol):

    def register(
        self,
        component_id: str,
        source_code: str,
        entrypoint: str,
        timeout_sec: float | None = None,
    ) -> None: ...

    def invoke(
        self,
        component_id: str,
        arguments: dict[str, Any],
        timeout_sec: float | None = None,
    ) -> Any: ...

    def close(self) -> None: ...
