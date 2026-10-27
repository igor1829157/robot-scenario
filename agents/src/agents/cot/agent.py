import logging
from pathlib import Path
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from robomap_client.core.client import RobomapClient

from agents.react.agent import ReActAgent
from agents.react.policy import AgentPolicy

logger = logging.getLogger(__name__)


class CoTAgent(ReActAgent):
    """Chain-of-Thought агент."""
    name = "cot"

    def __init__(
            self,
            model: BaseChatModel,
            policy: AgentPolicy | None = None,
            *,
            recursion_limit: int = 300,
            initial_user_message: str = "Start exploring the map",
            tool_options: dict[str, dict[str, Any]] | None = None,
            additional_tools: Sequence[BaseTool] | None = None,
            config_id: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            policy=policy,
            recursion_limit=recursion_limit,
            initial_user_message=initial_user_message,
            tool_options=tool_options,
            additional_tools=additional_tools,
            config_id=config_id,
            metadata=metadata,
        )

        # Добавляем инструкцию для CoT
        if self.policy.system_prompt:
            cot_instruction = (
                "\n\n## Chain-of-Thought Instruction\n"
                "Before calling any tool, you MUST write down your reasoning step-by-step "
                "inside <thought>...</thought> tags. Explain what you see, what your goal is, "
                "and why you are choosing the next action. Only after closing the tag, call the tool."
            )
            self.policy.system_prompt += cot_instruction

    def run(self, client: RobomapClient, *, log_dir: Path | None = None) -> None:
        """Запускает агента, сохраняя логи в подпапку 'cot'."""
        actual_log_dir = None
        if log_dir is not None:
            actual_log_dir = log_dir / "cot"
            actual_log_dir.mkdir(parents=True, exist_ok=True)

        super().run(client, log_dir=actual_log_dir)