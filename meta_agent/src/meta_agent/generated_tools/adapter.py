import re
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, create_model

from .registry import GeneratedToolRegistry
from .schema import GeneratedToolSpec


SCHEMA_TYPES: dict[str, type[Any]] = {
    "number": float,
    "integer": int,
    "string": str,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def build_args_schema(
    spec: GeneratedToolSpec,
) -> type[BaseModel]:

    fields: dict[
        str,
        tuple[type[Any], Any],
    ] = {}

    for argument_name, type_name in (
        spec.input_schema.items()
    ):
        if not argument_name.isidentifier():
            raise ValueError(
                "invalid generated tool argument "
                f"name: {argument_name}"
            )

        try:
            python_type = SCHEMA_TYPES[
                type_name
            ]

        except KeyError as error:
            raise ValueError(
                "unsupported generated tool "
                f"argument type: {type_name}"
            ) from error

        # Ellipsis означает обязательное поле Pydantic.
        fields[argument_name] = (
            python_type,
            ...,
        )

    model_name = re.sub(
        r"[^a-zA-Z0-9_]",
        "_",
        spec.name,
    )

    return create_model(
        f"{model_name}_arguments",
        **fields,
    )


def build_langchain_tool(
    spec: GeneratedToolSpec,
    registry: GeneratedToolRegistry,
) -> StructuredTool:

    if not spec.name.isidentifier():
        raise ValueError(
            "generated tool name must be "
            f"a valid identifier: {spec.name}"
        )

    args_schema = build_args_schema(spec)

    def invoke_generated_tool(
        **arguments: Any,
    ) -> Any:
        return registry.invoke(
            tool_id=spec.tool_id,
            arguments=arguments,
        )

    invoke_generated_tool.__name__ = spec.name
    invoke_generated_tool.__doc__ = spec.description

    return StructuredTool.from_function(
        func=invoke_generated_tool,
        name=spec.name,
        description=spec.description,
        args_schema=args_schema,
        infer_schema=False,
    )
