import logging
from pathlib import Path
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from robomap_client.core.client import RobomapClient

from agents.cot.agent import CoTAgent
from agents.react.policy import AgentPolicy

logger = logging.getLogger(__name__)


class CoTSCAgent(CoTAgent):
    """Self-Consistency Chain-of-Thought агент."""
    name = "cot_sc"

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
        # Для Self-Consistency повышаем температуру, чтобы модель генерировала разные варианты
        if hasattr(model, "temperature"):
            model.temperature = 0.7

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

        # Добавляем инструкцию для Self-Consistency
        if self.policy.system_prompt:
            sc_instruction = (
                "\n\n## Self-Consistency Instruction\n"
                "Generate multiple distinct reasoning paths for the current situation. "
                "Evaluate them internally and choose the most reliable one before acting."
            )
            self.policy.system_prompt += sc_instruction

    def run(self, client: RobomapClient, *, log_dir: Path | None = None) -> None:
        """Запускает агента, сохраняя логи в подпапку 'cot_sc'."""
        actual_log_dir = None
        if log_dir is not None:
            actual_log_dir = log_dir / "cot_sc"
            actual_log_dir.mkdir(parents=True, exist_ok=True)

        super().run(client, log_dir=actual_log_dir)
        super().run(client, log_dir=actual_log_dir)