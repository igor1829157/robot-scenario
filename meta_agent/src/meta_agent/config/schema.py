from dataclasses import dataclass, field
from typing import Any

from meta_agent.util.time import utc_now_iso


@dataclass(frozen=True)
class ContextContract:
    """Определяет информацию, доступную агенту перед действием"""

    sections: str = "memory"
    format_style: str = "json"
    include_geometry: bool = False

    transform: str | None = None


@dataclass(frozen=True)
class ObservationContract:
    """Определяет информацию, которую агент может получить в результате действия"""
    included_fields: tuple[str, ...] = (
        "action",
        "status",
        "final_message",
        "elapsed_ticks",
        "events"
    )
    max_events: int | None = None
    format_style: str = "json"

    transform: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    """Описывает один разрешенный инструмент агента"""

    name: str
    enabled: bool = True
    max_wait_ticks: int | None = None
    kwargs_defaults: dict[str, Any] = field(default_factory=dict)
    description: str | None = None


@dataclass(frozen=True)
class OpenRouterRouting:
    order: tuple[str, ...]
    allow_fallbacks: bool = False
    require_parameters: bool = True

    def __post_init__(self) -> None:
        if not self.order:
            raise ValueError("provider_routing.order must not be empty")


@dataclass(frozen=True)
class ReasoningConfig:

    enabled: bool
    effort: str | None = None
    max_tokens: int | None = None
    exclude: bool = False


def default_tool_specs() -> tuple[ToolSpec, ...]:

    return (
        ToolSpec(name="move", max_wait_ticks=200),
        ToolSpec(name="rotate", max_wait_ticks=200),
        ToolSpec(name="finish_exploration")
    )


@dataclass(frozen=True)
class AgentConfig:
    """Конфигурация агента"""

    config_id: str
    created_by: str
    model: str
    provider_routing: OpenRouterRouting
    reasoning: ReasoningConfig
    system_prompt: str

    model_kwargs: dict[str, Any] = field(default_factory=dict)
    
    parent_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    initial_user_message: str = "Start exploring the map"

    context_contract: ContextContract = field(default_factory=ContextContract)
    observation_contract: ObservationContract = field(default_factory=ObservationContract)

    tool_specs: tuple[ToolSpec, ...] = field(default_factory=default_tool_specs)

    generated_tool_ids: tuple[str, ...] = field(default_factory=tuple)

    recursion_limit: int = 300
    rationale: str = ""
