from typing import Any

from meta_agent.generated_tools.schema import GeneratedToolSpec
from meta_agent.runtime import CodeRuntime


class GeneratedToolRegistry:
    def __init__(
        self,
        runtime: CodeRuntime,
    ):
        self.runtime = runtime

        self._tools: dict[
            str,
            GeneratedToolSpec,
        ] = {}

        self._tool_names: dict[str, str] = {}

    @staticmethod
    def _component_id(tool_id: str) -> str:
        return f"tool:{tool_id}"

    def register(
        self,
        spec: GeneratedToolSpec,
    ) -> None:
        existing = self._tools.get(
            spec.tool_id
        )

        if existing is not None:
            if existing.source_hash == spec.source_hash:
                return

            raise ValueError(
                "generated tool ID is already "
                "registered with different code: "
                f"{spec.tool_id}"
            )

        existing_tool_id = self._tool_names.get(
            spec.name
        )

        if existing_tool_id is not None:
            raise ValueError(
                "generated tool name is already "
                f"registered: {spec.name}"
            )

        self.runtime.register(
            component_id=self._component_id(
                spec.tool_id
            ),
            source_code=spec.source_code,
            entrypoint=spec.entrypoint,
            timeout_sec=spec.timeout_sec,
        )

        self._tools[spec.tool_id] = spec
        self._tool_names[spec.name] = spec.tool_id

    def invoke(
        self,
        tool_id: str,
        arguments: dict[str, Any],
    ) -> Any:
        spec = self.get(tool_id)

        return self.runtime.invoke(
            component_id=self._component_id(tool_id),
            arguments=arguments,
            timeout_sec=spec.timeout_sec,
        )

    def get(
        self,
        tool_id: str,
    ) -> GeneratedToolSpec:
        try:
            return self._tools[tool_id]

        except KeyError as error:
            raise ValueError(
                f"unknown generated tool: {tool_id}"
            ) from error

    def get_by_name(
        self,
        name: str,
    ) -> GeneratedToolSpec:
        try:
            tool_id = self._tool_names[name]

        except KeyError as error:
            raise ValueError(
                f"unknown generated tool name: {name}"
            ) from error

        return self._tools[tool_id]

    def snapshot(
        self,
    ) -> list[dict[str, Any]]:
        return [
            {
                "tool_id": spec.tool_id,
                "name": spec.name,
                "description": spec.description,
                "entrypoint": spec.entrypoint,
                "input_schema": dict(spec.input_schema),
                "output_schema": dict(spec.output_schema),
                "created_by": spec.created_by,
                "created_at": spec.created_at,
                "parent_id": spec.parent_id,
                "rationale": spec.rationale,
                "timeout_sec": spec.timeout_sec,
                "source_hash": spec.source_hash,
                "source_code": spec.source_code,
            }
            for spec in self._tools.values()
        ]
