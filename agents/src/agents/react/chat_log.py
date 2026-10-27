"""Запись полного LLM-чата и system prompt в JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TextIO

from langchain_core.messages import BaseMessage

LogKind = Literal["system_prompt", "message", "event"]


def serialize_message(message: BaseMessage) -> dict[str, Any]:
    """Сериализует LangChain-сообщение в JSON-совместимый dict."""
    if hasattr(message, "model_dump"):
        return message.model_dump()
    if hasattr(message, "dict"):
        return message.dict()
    return {
        "type": type(message).__name__,
        "content": message.content,
    }


class ChatLogWriter:
    """Пишет system prompt, сообщения и события агента в JSONL."""

    def __init__(self, path: Path) -> None:
        """Открывает JSONL-файл для дозаписи.

        Args:
            path: Путь к ``chat.jsonl`` в каталоге прогона.
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = open(path, "a", encoding="utf-8")
        self._llm_step = 0
        self._logged_messages = 0

    def begin_llm_step(self) -> None:
        """Увеличивает номер шага перед очередным вызовом LLM."""
        self._llm_step += 1

    def write_system_prompt(self, tick: int, prompt: str) -> None:
        """Записывает system prompt, отправленный модели."""
        self._write(tick, "system_prompt", prompt)

    def write_messages(self, tick: int, messages: list[BaseMessage]) -> None:
        """Записывает новые сообщения из истории графа."""
        for message in messages[self._logged_messages :]:
            self._write(tick, "message", serialize_message(message))
        self._logged_messages = len(messages)

    def write_event(self, tick: int, data: dict[str, Any]) -> None:
        """Записывает служебное событие (например, завершение миссии)."""
        self._write(tick, "event", data)

    def _write(self, tick: int, kind: LogKind, data: Any) -> None:
        record = {
            "tick": tick,
            "step": self._llm_step,
            "kind": kind,
            "data": data,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Закрывает файл лога."""
        self._file.close()
