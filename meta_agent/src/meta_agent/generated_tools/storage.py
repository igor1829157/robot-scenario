import json
from pathlib import Path
from typing import Any

from .schema import GeneratedToolSpec


SCHEMA_VERSION = 1


def generated_tool_to_dict(
    spec: GeneratedToolSpec,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool_id": spec.tool_id,
        "name": spec.name,
        "description": spec.description,
        "source_code": spec.source_code,
        "source_hash": spec.source_hash,
        "entrypoint": spec.entrypoint,
        "input_schema": spec.input_schema,
        "output_schema": spec.output_schema,
        "created_by": spec.created_by,
        "created_at": spec.created_at,
        "parent_id": spec.parent_id,
        "rationale": spec.rationale,
        "timeout_sec": spec.timeout_sec,
    }


def save_generated_tool(
    spec: GeneratedToolSpec,
    path: str | Path,
) -> Path:
    target = Path(path)

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = target.with_suffix(
        target.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            generated_tool_to_dict(spec),
            file,
            ensure_ascii=False,
            indent=2,
        )

        file.write("\n")

    temporary.replace(target)

    return target


def load_generated_tool(
    path: str | Path,
) -> GeneratedToolSpec:
    source = Path(path)

    with source.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    schema_version = data.get(
        "schema_version"
    )

    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            "unsupported generated tool "
            f"schema version: {schema_version}"
        )

    spec = GeneratedToolSpec(
        tool_id=data["tool_id"],
        name=data["name"],
        description=data["description"],
        source_code=data["source_code"],
        entrypoint=data.get(
            "entrypoint",
            "run",
        ),
        input_schema=dict(
            data.get("input_schema", {})
        ),
        output_schema=dict(
            data.get("output_schema", {})
        ),
        created_by=data.get(
            "created_by",
            "meta_agent",
        ),
        created_at=data["created_at"],
        parent_id=data.get("parent_id"),
        rationale=data.get(
            "rationale",
            "",
        ),
        timeout_sec=float(
            data.get("timeout_sec", 1.0)
        ),
    )

    expected_hash = data.get("source_hash")

    if (
        expected_hash is not None
        and expected_hash != spec.source_hash
    ):
        raise ValueError(
            "generated tool source hash mismatch: "
            f"{spec.tool_id}"
        )

    return spec
