import json
from typing import Any, Protocol

from robomap_client.core.client import RobomapClient

from agents.react.prompts import SYSTEM_PROMPT


class AgentPolicy(Protocol):
    system_prompt: str

    def format_context(self, client: RobomapClient) -> str: ...

    def format_observation(self, result: dict[str, Any]) -> str: ...


class BaseAgentPolicy:

    system_prompt = SYSTEM_PROMPT

    def format_context(self, client: RobomapClient) -> str:
        payload = client.get_context(
            format_style="json",
            sections="memory",
            include_geometry=False
        )

        return json.dumps(payload, ensure_ascii=False)
    
    def format_observation(self, result: dict[str, Any]) -> str:
        return json.dumps(result, ensure_ascii=False)
