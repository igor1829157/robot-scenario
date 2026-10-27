from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from meta_agent.runtime import CodeRuntime
from meta_agent.util.time import utc_now_iso


@dataclass(frozen=True)
class TransformSpec:
    transform_id: str
    source_code: str
    entrypoint: str = "transform"
    created_by: str = "meta-agent"
    created_at: str = field(default_factory=utc_now_iso)
    parent_id: str | None = None
    rationale: str = ""

    @property
    def source_hash(self) -> str:
        return sha256(
            self.source_code.encode("utf-8")
        ).hexdigest()


class TransformRegistry:
    def __init__(
        self,
        runtime: CodeRuntime,
    ):
        self.runtime = runtime
        self._transforms: dict[str, TransformSpec] = {}

    @staticmethod
    def _component_id(transform_id: str) -> str:
        return f"transform:{transform_id}"

    def register(
        self,
        spec: TransformSpec,
    ) -> None:
        existing = self._transforms.get(spec.transform_id)

        if existing is not None:
            if existing.source_hash == spec.source_hash:
                return
            
            raise ValueError(f"transform ID is already registered with a different source code {spec.transform_id}")

        self.runtime.register(
            component_id=self._component_id(spec.transform_id),
            source_code=spec.source_code,
            entrypoint=spec.entrypoint,
            timeout_sec=1.0,
        )

        self._transforms[spec.transform_id] = spec

    def execute(
        self,
        transform_id: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        if transform_id not in self._transforms:
            raise ValueError(f"unknown transform: {transform_id}")
        
        result = self.runtime.invoke(
            component_id=self._component_id(transform_id),
            arguments={"data": data},
            timeout_sec=1.0,
        )

        if not isinstance(result, dict):
            raise TypeError("transform result must be a dictionary")

        return result

    def get(
        self,
        transform_id: str,
    ) -> TransformSpec:

        try:
            return self._transforms[transform_id]

        except KeyError as error:
            raise ValueError(f"unknown transform: {transform_id}") from error

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "transform_id": spec.transform_id,
                "entrypoint": spec.entrypoint,
                "created_by": spec.created_by,
                "parent_id": spec.parent_id,
                "rationale": spec.rationale,
                "source_hash": spec.source_hash,
                "source_code": spec.source_code,
                "created_at": spec.created_at,
            }
            for spec in self._transforms.values()
        ]
