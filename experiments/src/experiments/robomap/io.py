"""Общие I/O-утилиты для Robomap-экспериментов."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Атомарно записывает JSON-объект.

    Args:
        path: Целевой путь файла.
        payload: Сериализуемый словарь.
    """
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp_path = path.with_name(path.name + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")

    os.replace(tmp_path, path)


def load_json_object(path: Path) -> dict[str, Any]:
    """Загружает JSON-объект из файла.

    Args:
        path: Путь к JSON-файлу.

    Returns:
        Словарь верхнего уровня.

    Raises:
        ValueError: Если файл не является JSON-объектом.
        OSError: Если файл нельзя прочитать.
        json.JSONDecodeError: Если JSON некорректен.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"JSON must be an object: {path}")

    return data


def canonical_json_hash(payload: Any) -> str:
    """Возвращает SHA-256 от канонического JSON.

    Args:
        payload: Произвольный JSON-совместимый объект.

    Returns:
        Hex-строка хеша.
    """
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_relative(base: Path, value: str | Path) -> Path:
    """Резолвит путь относительно базового файла.

    Args:
        base: Базовый файл (обычно experiment/suite JSON).
        value: Абсолютный или относительный путь.

    Returns:
        Абсолютный путь.
    """
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base.parent / path).resolve()


def relpath_from(base: Path, target: str | Path) -> str:
    """Возвращает переносимый относительный путь от base к target.

    Args:
        base: Базовая директория или файл.
        target: Целевой путь.

    Returns:
        Относительный путь в POSIX-формате.
    """
    base_dir = base if base.is_dir() else base.parent
    return os.path.relpath(
        Path(target).resolve(),
        base_dir.resolve(),
    )
