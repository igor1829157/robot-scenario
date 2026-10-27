import json
from pathlib import Path
from stat import filemode
from typing import Any

from meta_agent.transforms.registry import TransformSpec


SCHEMA_VERSION = 1


def transform_to_dict(
    spec: TransformSpec,
) -> dict[str, Any]:

    return {
        "schema_version": SCHEMA_VERSION,
        "transform_id": spec.transform_id,
        "source_code": spec.source_code,
        "source_hash": spec.source_hash,
        "entrypoint": spec.entrypoint,
        "created_by": spec.created_by,
        "created_at": spec.created_at,
        "parent_id": spec.parent_id,
        "rationale": spec.rationale
    }


def save_transform(
    spec: TransformSpec,
    path: str | Path
) -> Path:

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    temporary = target.with_suffix(target.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(
            transform_to_dict(spec),
            file,
            ensure_ascii=False,
            indent=2,
        )

        file.write("\n")

    temporary.replace(target)

    return target


def load_transform(
    path: str | Path
) -> TransformSpec:

    source = Path(path)

    with source.open("r", encoding="utf-8") as file:
        data = json.load(file)

    schema_version = data.get("schema_version")

    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")

    spec = TransformSpec(
        transform_id=data["transform_id"],
        source_code=data["source_code"],
        entrypoint=data.get("entrypoint", "transform"),
        created_by=data.get("created_by", "meta-agent"),
        created_at=data["created_at"],
        parent_id=data.get("parent_id"),
        rationale=data.get("rationale", "")
    )

    expected_hash = data.get("source_hash")

    if (
        expected_hash is not None
        and expected_hash != spec.source_hash
    ):
        raise ValueError(f"source hash mismatch for transform {spec.transform_id}")

    return spec
