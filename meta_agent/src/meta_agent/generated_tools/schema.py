from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from meta_agent.util.time import utc_now_iso


@dataclass(frozen=True)
class GeneratedToolSpec:
    tool_id: str
    name: str
    description: str
    source_code: str

    entrypoint: str = "run"

    input_schema: dict[str, Any] = field(default_factory=dict)

    output_schema: dict[str, Any] = field(default_factory=dict)

    created_by: str = "meta_agent"

    created_at: str = field(default_factory=utc_now_iso)

    parent_id: str | None = None
    rationale: str = ""
    timeout_sec: float = 1.0

    @property
    def source_hash(self) -> str:
        return sha256(
            self.source_code.encode("utf-8")
        ).hexdigest()
