import json
from typing import Any

from robomap_client.core.client import RobomapClient

from meta_agent.config.schema import (
    ContextContract,
    ObservationContract,
)

from meta_agent.transforms import TransformRegistry
from meta_agent.runtime import CodeRuntime


class ConfigurableAgentPolicy:

    def __init__(
        self,
        *,
        system_prompt: str,
        context_contract: ContextContract,
        observation_contract: ObservationContract,
        transform_registry: TransformRegistry | None = None,
        runtime: CodeRuntime | None = None
    ) -> None:
        if "{context}" not in system_prompt:
            raise ValueError("system_prompt must contain {context}")
        
        if (
            observation_contract.max_events is not None
            and observation_contract.max_events <= 0
        ):
            raise ValueError("max_events must be greater than 0")

        self.system_prompt = system_prompt
        self.context_contract = context_contract
        self.observation_contract = observation_contract
        self.transform_registry = transform_registry
        self.runtime = runtime

    def format_context(
        self,
        client: RobomapClient,
    ) -> str:

        contract = self.context_contract

        payload = client.get_context(
            format_style=contract.format_style,
            sections=contract.sections,
            include_geometry=contract.include_geometry,
        )

        if contract.transform is not None:
            if self.transform_registry is None:
                raise ValueError(
                    "context transform is configured, "
                    "but transform_registry is not provided"
                )

            if contract.format_style != "json":
                raise ValueError(
                    "context transforms currently support "
                    "only JSON format"
                )

            if isinstance(payload, str):
                try:
                    transform_input = json.loads(payload)

                except json.JSONDecodeError as error:
                    raise ValueError(
                        "context payload is not valid JSON"
                    ) from error

            else:
                transform_input = payload

            if not isinstance(transform_input, dict):
                raise ValueError(
                    "context transform input must be "
                    "a dictionary"
                )

            payload = self.transform_registry.execute(
                transform_id=contract.transform,
                data=transform_input,
            )

        if contract.format_style == "json":
            if isinstance(payload, str):
                return payload

            return json.dumps(
                payload,
                ensure_ascii=False,
            )

        if isinstance(payload, str):
            return payload

        raise ValueError(
            f"unexpected context format: "
            f"{contract.format_style}"
        )

    def format_observation(
        self,
        result: dict[str, Any],
    ) -> str:

        contract = self.observation_contract

        filtered = {
            field_name: result[field_name]
            for field_name in contract.included_fields
            if field_name in result
        }

        if (
            "events" in filtered
            and contract.max_events is not None
        ):
            events = filtered["events"]

            if isinstance(events, list):
                filtered["events"] = events[
                    -contract.max_events:
                ]

        if contract.transform is not None:
            if self.transform_registry is None:
                raise ValueError(
                    "observation transform is configured, "
                    "but transform_registry is not provided"
                )

            filtered = self.transform_registry.execute(
                transform_id=contract.transform,
                data=filtered,
            )

        if contract.format_style == "json":
            return json.dumps(
                filtered,
                ensure_ascii=False,
            )

        raise ValueError(
            "unexpected observation format: "
            f"{contract.format_style}"
        )

    def close(self) -> None:
        if self.runtime is not None:
            self.runtime.close()
