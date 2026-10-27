import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from robomap_client.core.client import RobomapClient

from agents.react.chat_log import ChatLogWriter
from agents.react.policy import AgentPolicy, BaseAgentPolicy
from agents.react.tools import create_robot_tools

logger = logging.getLogger(__name__)

_THOUGHT_LOG_MAX_LEN = 150


def _content_to_str(content: Any) -> str:
    """Нормализует content сообщения (str или блоки Gemini) в строку."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return " ".join(parts)
    return str(content)


def _format_observation_log(content: str) -> str:
    """Форматирует observation для терминала: status + final_message."""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        first_line = content.split("\n", maxsplit=1)[0].strip()
        return first_line or "Observation received."

    if not isinstance(payload, dict):
        return content.split("\n", maxsplit=1)[0].strip() or "Observation received."

    status = payload.get("status", "")
    final_message = payload.get("final_message", "")
    if status and final_message:
        return f"status: {status} | {final_message}"
    if status:
        return f"status: {status}"
    return content.split("\n", maxsplit=1)[0].strip() or "Observation received."


def _log_stream_message(client: RobomapClient, message: BaseMessage) -> None:
    """Пишет в терминал USER / THOUGHT / ACTION / OBSERVATION."""
    tick = client.get_current_tick()
    content = _content_to_str(message.content)

    if isinstance(message, HumanMessage):
        logger.info("[%04d] [USER] %s", tick, content)
        return

    if isinstance(message, AIMessage):
        thought = content.strip().replace("\n", " ")
        if thought:
            if len(thought) > _THOUGHT_LOG_MAX_LEN:
                thought = thought[:_THOUGHT_LOG_MAX_LEN] + "..."
            logger.info("[%04d] [THOUGHT] %s", tick, thought)

        for tool_call in message.tool_calls or []:
            name = tool_call.get("name", "unknown")
            args = tool_call.get("args", {})
            logger.info("[%04d] [ACTION] Calling %s(%s)", tick, name, args)
        return

    if isinstance(message, ToolMessage):
        logger.info(
            "[%04d] [OBSERVATION] %s",
            tick,
            _format_observation_log(content),
        )


class ReActAgent:
    """ReAct-агент: действие → ожидание → observation → следующее решение."""

    name = "react"

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
        """
        Args:
            model: Готовая LangChain chat-модель.
            policy: Формат prompt/context/observation; по умолчанию JSON memory.
            recursion_limit: Лимит шагов графа.
        """
        self.model = model
        self.policy = policy or BaseAgentPolicy()
        self.recursion_limit = recursion_limit
        self.initial_user_message = initial_user_message
        self.tool_options = tool_options or {}
        self.additional_tools = list(
            additional_tools or ()
        )
        self.config_id = config_id
        self.metadata = metadata or {}
    
    def run(self, client: RobomapClient, *, log_dir: Path | None = None) -> None:
        """Запускает агента в уже созданной runner'ом сессии.

        Args:
            client: Клиент активной сессии Robomap.
            log_dir: Каталог прогона; при указании пишет ``chat.jsonl``.
        """
        tools = create_robot_tools(
            client, 
            self.policy,
            tool_options=self.tool_options,
        )
        tools.extend(self.additional_tools)

        tool_names = [
            tool.name
            for tool in tools
        ]

        duplicate_names = {
            name
            for name in tool_names
            if tool_names.count(name) > 1
        }

        if duplicate_names:
            raise ValueError(
                "duplicate agent tool names: "
                + ", ".join(
                    sorted(duplicate_names)
                )
            )

        chat_log: ChatLogWriter | None = None
        if log_dir is not None:
            chat_log_path = log_dir / "chat.jsonl"
            chat_log = ChatLogWriter(chat_log_path)
            logger.info("chat log → %s", chat_log_path)

        @dynamic_prompt
        def inject_context(request: ModelRequest) -> str:
            context = self.policy.format_context(client)
            prompt = self.policy.system_prompt.format(context=context)
            if chat_log is not None:
                chat_log.begin_llm_step()
                chat_log.write_system_prompt(client.get_current_tick(), prompt)
            return prompt

        graph = create_agent(
            model=self.model,
            tools=tools,
            middleware=[inject_context],
        )
        inputs = {
            "messages": [HumanMessage(content=self.initial_user_message)]
        }
        config = {"recursion_limit": self.recursion_limit}

        logger.info("%s: starting exploration loop", self.name)

        try:
            for event in graph.stream(inputs, stream_mode="values", config=config):
                messages = event["messages"]
                tick = client.get_current_tick()
                if chat_log is not None:
                    chat_log.write_messages(tick, messages)

                last_message = messages[-1]
                _log_stream_message(client, last_message)

                if (
                    isinstance(last_message, ToolMessage)
                    and last_message.name == "finish_exploration"
                ):
                    logger.info(
                        "[%04d] [SYSTEM] Agent declared exploration finished.",
                        tick,
                    )
                    if chat_log is not None:
                        chat_log.write_event(tick, {"status": "mission_completed"})
                    break
        finally:
            if chat_log is not None:
                chat_log.close()

            close_policy = getattr(
                self.policy,
                "close",
                None
            )

            if callable(close_policy):
                try:
                    close_policy()
                
                except Exception:
                    logger.exception("Failed to close agent policy resources")
