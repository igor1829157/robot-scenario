"""Smoke-входы и gate плоской схемы для verify без test_inputs."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Any

from meta_agent.generated_tools.adapter import SCHEMA_TYPES
from meta_agent.generated_tools.schema import GeneratedToolSpec


SMOKE_SEED = 0

_VALID_PROTOTYPES: dict[str, Any] = {
    "string": "sample",
    "integer": 1,
    "number": 1.0,
    "boolean": True,
    "array": [1],
    "object": {"k": 1},
}

_OBSERVATION_STUBS: dict[str, Any] = {
    "action": "move",
    "status": "SUCCESS",
    "final_message": "ok",
    "elapsed_ticks": 1,
    "events": [{"tick": 0, "message": "sample"}],
}


def entrypoint_param_names(
    source_code: str,
    entrypoint: str,
) -> list[str]:
    """Имена позиционных/keyword-параметров entrypoint без *args/**kwargs."""
    tree = ast.parse(source_code, mode="exec")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entrypoint:
            args = node.args
            names = [arg.arg for arg in args.posonlyargs]
            names.extend(arg.arg for arg in args.args)
            if args.vararg is not None:
                raise ValueError(
                    f"entrypoint {entrypoint!r} must not use *args"
                )
            if args.kwarg is not None:
                raise ValueError(
                    f"entrypoint {entrypoint!r} must not use **kwargs"
                )
            names.extend(arg.arg for arg in args.kwonlyargs)
            return names
    raise ValueError(f"entrypoint {entrypoint!r} not found")


def assert_tool_schema(spec: GeneratedToolSpec) -> None:
    """Проверяет плоский input_schema и соответствие параметрам entrypoint.

    Args:
        spec: Спецификация generated tool.

    Raises:
        ValueError: Схема пуста, типы неизвестны или не сходятся с сигнатурой.
    """
    if not spec.input_schema:
        raise ValueError("input_schema must not be empty")

    schema_keys: list[str] = []
    for name, type_name in spec.input_schema.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(
                f"invalid input_schema key: {name!r}"
            )
        if type_name not in SCHEMA_TYPES:
            raise ValueError(
                f"unsupported input_schema type for {name!r}: "
                f"{type_name!r}"
            )
        schema_keys.append(name)

    params = entrypoint_param_names(spec.source_code, spec.entrypoint)
    schema_set = set(schema_keys)
    param_set = set(params)

    if schema_set != param_set:
        raise ValueError(
            "input_schema keys must match entrypoint parameters: "
            f"schema={sorted(schema_set)}, params={sorted(param_set)}"
        )


def valid_tool_args(schema: dict[str, Any]) -> dict[str, Any]:
    """Строит один валидный набор аргументов из плоской схемы."""
    args: dict[str, Any] = {}
    for name, type_name in schema.items():
        args[name] = _VALID_PROTOTYPES[str(type_name)]
    return args


def invalid_tool_args(schema: dict[str, Any]) -> dict[str, Any]:
    """Строит один заведомо невалидный набор (дыра в required).

    При одном поле подменяет тип; при нескольких — удаляет первый ключ.
    """
    if not schema:
        raise ValueError("input_schema must not be empty")

    valid = valid_tool_args(schema)
    keys = list(schema.keys())

    if len(keys) == 1:
        only = keys[0]
        type_name = str(schema[only])
        if type_name == "string":
            return {only: 1}
        return {only: "sample"}

    broken = dict(valid)
    del broken[keys[0]]
    return broken


def valid_observation_transform_data(
    included_fields: Sequence[str],
) -> dict[str, Any]:
    """Smoke-вход observation transform по included_fields."""
    data: dict[str, Any] = {}
    for field_name in included_fields:
        if field_name in _OBSERVATION_STUBS:
            data[field_name] = _OBSERVATION_STUBS[field_name]
        else:
            data[field_name] = "sample"
    return data


def valid_context_transform_data(sections: str) -> dict[str, Any]:
    """Smoke-вход context transform: секция контракта с непустым stub."""
    section = sections or "memory"
    return {
        section: {
            "objects": [{"id": "obj_1", "status": "partial"}],
        }
    }


def invalid_transform_data() -> Any:
    """Невалидный data для transform (не dict)."""
    return "not-a-dict"
