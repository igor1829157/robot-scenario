import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from meta_agent.config.schema import (
    AgentConfig,
    ContextContract,
    ObservationContract,
    OpenRouterRouting,
    ReasoningConfig,
    ToolSpec,
)


def config_to_dict(config: AgentConfig) -> dict[str, Any]:
    return asdict(config)


def save_config(config: AgentConfig, path: str | Path) -> Path:

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            config_to_dict(config),
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


def load_config(path: str | Path) -> AgentConfig:

    input_path = Path(path)

    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("config file must contain a JSON object")

    provider_routing_payload = payload.pop("provider_routing")
    reasoning_payload = payload.pop("reasoning")
    context_payload = payload.pop("context_contract", {})
    observation_payload = payload.pop("observation_contract", {})
    tools_payload = payload.pop("tool_specs", [])
    generated_tool_ids_payload = payload.pop(
        "generated_tool_ids",
        [],
    )

    if "included_fields" in observation_payload:
        observation_payload["included_fields"] = tuple(observation_payload["included_fields"])

    context_contract = ContextContract(**context_payload)
    observation_contract = ObservationContract(**observation_payload)
    tool_specs = tuple(
        ToolSpec(**tool_payload)
        for tool_payload in tools_payload
    )

    if not isinstance(
        generated_tool_ids_payload,
        list,
    ):
        raise ValueError(
            "generated_tool_ids must be a list"
        )

    if not all(
        isinstance(tool_id, str)
        and tool_id
        for tool_id in generated_tool_ids_payload
    ):
        raise ValueError(
            "generated_tool_ids must contain "
            "non-empty strings"
        )

    generated_tool_ids = tuple(
        generated_tool_ids_payload
    )

    if not isinstance(provider_routing_payload, dict):
        raise ValueError("provider_routing must be an object")

    order_payload = provider_routing_payload.get("order")

    if not isinstance(order_payload, list):
        raise ValueError("provider_routing.order must be a list")

    provider_routing_payload["order"] = tuple(order_payload)
    provider_routing = OpenRouterRouting(**provider_routing_payload)

    if not isinstance(reasoning_payload, dict):
        raise ValueError("reasoning must be an object")

    reasoning = ReasoningConfig(**reasoning_payload)

    return AgentConfig(
        **payload,
        provider_routing=provider_routing,
        reasoning=reasoning,
        context_contract=context_contract,
        observation_contract=observation_contract,
        tool_specs=tool_specs,
        generated_tool_ids=generated_tool_ids,
    )
