"""Кодирование и декодирование файлов экспорта сессии (.msgpack.gz)."""

from __future__ import annotations

import copy
import gzip
from pathlib import Path
from typing import Any

import msgpack

# v1 — полные кадры в поле frames;
# v2 — base_frame + deltas с материализацией frames при загрузке.
EXPORT_FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = {1, 2}
EXPORT_FILE_SUFFIX = ".msgpack.gz"


def _entity_id(entity: dict[str, Any]) -> str:
    """Возвращает строковый id сущности для индексного сравнения."""
    return str(entity["id"])


def _index_entities(entities: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Строит индекс сущностей по id."""
    return {_entity_id(entity): entity for entity in entities if "id" in entity}


def _compute_list_delta(
    prev: list[dict[str, Any]],
    curr: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Считает дельту списка сущностей: upsert изменённых/новых и delete удалённых."""
    prev_idx = _index_entities(prev)
    curr_idx = _index_entities(curr)

    upsert: list[dict[str, Any]] = []
    for entity_id, entity in curr_idx.items():
        previous = prev_idx.get(entity_id)
        if previous is None or previous != entity:
            upsert.append(entity)

    delete = [entity_id for entity_id in prev_idx if entity_id not in curr_idx]
    if not upsert and not delete:
        return None
    return {"upsert": upsert, "delete": delete}


def _apply_list_delta(
    prev: list[dict[str, Any]],
    delta: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Применяет дельту списка сущностей, сохраняя порядок где возможно."""
    if not delta:
        return list(prev)

    delete_set = {str(entity_id) for entity_id in delta.get("delete", [])}
    upsert_map = {
        _entity_id(entity): entity
        for entity in delta.get("upsert", [])
        if "id" in entity
    }

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entity in prev:
        if "id" not in entity:
            continue
        entity_id = _entity_id(entity)
        if entity_id in delete_set:
            continue
        if entity_id in upsert_map:
            result.append(upsert_map[entity_id])
        else:
            result.append(entity)
        seen.add(entity_id)

    for entity_id, entity in upsert_map.items():
        if entity_id not in seen:
            result.append(entity)

    return result


def _compute_memory_delta(
    prev: dict[str, list[dict[str, Any]]],
    curr: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Считает дельту памяти роботов по robot_id."""
    robots: dict[str, dict[str, Any]] = {}
    delete_robots = [robot_id for robot_id in prev if robot_id not in curr]

    for robot_id, curr_list in curr.items():
        prev_list = prev.get(robot_id, [])
        if robot_id not in prev:
            robots[robot_id] = {"upsert": list(curr_list), "delete": []}
            continue
        list_delta = _compute_list_delta(prev_list, curr_list)
        if list_delta is not None:
            robots[robot_id] = list_delta

    if not robots and not delete_robots:
        return None

    result: dict[str, Any] = {}
    if robots:
        result["robots"] = robots
    if delete_robots:
        result["delete_robots"] = delete_robots
    return result


def _apply_memory_delta(
    prev: dict[str, list[dict[str, Any]]],
    delta: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Применяет дельту памяти роботов."""
    if not delta:
        return {robot_id: list(entities) for robot_id, entities in prev.items()}

    result = {robot_id: list(entities) for robot_id, entities in prev.items()}
    for robot_id in delta.get("delete_robots", []):
        result.pop(robot_id, None)

    for robot_id, list_delta in delta.get("robots", {}).items():
        result[robot_id] = _apply_list_delta(result.get(robot_id, []), list_delta)
    return result


def compute_frame_delta(
    prev: dict[str, Any],
    curr: dict[str, Any],
) -> dict[str, Any]:
    """Строит дельту кадра относительно предыдущего полного состояния.

    Args:
        prev: Предыдущий полный кадр.
        curr: Текущий полный кадр.

    Returns:
        Дельта, достаточная для восстановления текущего кадра.
    """
    delta: dict[str, Any] = {"tick": curr.get("tick")}

    if curr.get("events"):
        delta["events"] = curr["events"]

    if curr.get("scenario_stage") != prev.get("scenario_stage"):
        delta["scenario_stage"] = curr.get("scenario_stage")

    if curr.get("robots_state") != prev.get("robots_state"):
        delta["robots_state"] = curr.get("robots_state", {})

    real_map_delta = _compute_list_delta(
        prev.get("real_map", []),
        curr.get("real_map", []),
    )
    if real_map_delta is not None:
        delta["real_map_delta"] = real_map_delta

    memory_delta = _compute_memory_delta(
        prev.get("robots_memory", {}),
        curr.get("robots_memory", {}),
    )
    if memory_delta is not None:
        delta["robots_memory_delta"] = memory_delta

    return delta


def _apply_frame_delta(
    prev: dict[str, Any],
    delta: dict[str, Any],
) -> dict[str, Any]:
    """Разворачивает дельту в полный кадр на базе предыдущего."""
    return {
        "tick": delta.get("tick", prev.get("tick")),
        "real_map": _apply_list_delta(
            prev.get("real_map", []),
            delta.get("real_map_delta"),
        ),
        "robots_memory": _apply_memory_delta(
            prev.get("robots_memory", {}),
            delta.get("robots_memory_delta"),
        ),
        "robots_state": (
            delta["robots_state"]
            if "robots_state" in delta
            else copy.deepcopy(prev.get("robots_state", {}))
        ),
        "events": delta.get("events", []),
        "scenario_stage": (
            delta["scenario_stage"]
            if "scenario_stage" in delta
            else prev.get("scenario_stage")
        ),
    }


def encode_frames_as_deltas(
    frames: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Кодирует список полных кадров в base_frame + deltas."""
    if not frames:
        raise ValueError("Нельзя кодировать пустой список кадров.")

    base_frame = copy.deepcopy(frames[0])
    deltas: list[dict[str, Any]] = []
    prev = base_frame
    for frame in frames[1:]:
        delta = compute_frame_delta(prev, frame)
        deltas.append(delta)
        prev = frame
    return base_frame, deltas


def materialize_frames_from_deltas(
    base_frame: dict[str, Any],
    deltas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Разворачивает base_frame + deltas в список полных кадров."""
    frames = [copy.deepcopy(base_frame)]
    prev = frames[0]
    for delta in deltas:
        full_frame = _apply_frame_delta(prev, delta)
        frames.append(full_frame)
        prev = full_frame
    return frames


def validate_map_boundaries(bundle: dict[str, Any]) -> None:
    """Проверяет корректность map_boundaries в bundle."""
    map_boundaries = bundle.get("map_boundaries")
    if not isinstance(map_boundaries, list) or len(map_boundaries) != 2:
        raise ValueError("Файл экспорта не содержит корректного поля map_boundaries.")


def materialize_bundle_frames(bundle: dict[str, Any]) -> dict[str, Any]:
    """Проверяет версию формата и материализует полные frames при необходимости."""
    format_version = bundle.get("format_version")
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"Неподдерживаемая версия формата экспорта: {format_version}"
        )

    validate_map_boundaries(bundle)

    if format_version == 1:
        frames = bundle.get("frames")
        if not isinstance(frames, list):
            raise ValueError("Файл экспорта не содержит корректного поля frames.")
        return bundle

    base_frame = bundle.get("base_frame")
    deltas = bundle.get("deltas")
    if not isinstance(base_frame, dict):
        raise ValueError("Файл экспорта не содержит корректного поля base_frame.")
    if not isinstance(deltas, list):
        raise ValueError("Файл экспорта не содержит корректного поля deltas.")

    bundle["frames"] = materialize_frames_from_deltas(base_frame, deltas)
    return bundle


def read_bundle(path: str | Path) -> dict[str, Any]:
    """Читает сырой bundle с диска (.msgpack.gz)."""
    export_path = Path(path)
    if not export_path.name.endswith(EXPORT_FILE_SUFFIX):
        raise ValueError(f"Неподдерживаемый суффикс файла экспорта: {export_path}")

    with gzip.open(export_path, "rb") as export_file:
        bundle = msgpack.unpackb(
            export_file.read(),
            raw=False,
            strict_map_key=False,
        )

    if not isinstance(bundle, dict):
        raise ValueError("Файл экспорта имеет некорректную структуру верхнего уровня.")
    return bundle


def write_bundle(path: str | Path, bundle: dict[str, Any]) -> None:
    """Пишет bundle на диск в формате .msgpack.gz."""
    output_path = Path(path)
    if not str(output_path).endswith(EXPORT_FILE_SUFFIX):
        raise ValueError(f"Неподдерживаемый суффикс файла экспорта: {output_path}")

    payload = msgpack.packb(bundle, use_bin_type=True)
    with gzip.open(output_path, "wb") as export_file:
        export_file.write(payload)


def load_export_bundle(path: str | Path) -> dict[str, Any]:
    """Загружает export и возвращает bundle с полными frames."""
    export_path = Path(path)
    if not export_path.is_file():
        raise FileNotFoundError(f"Файл экспорта не найден: {export_path}")
    return materialize_bundle_frames(read_bundle(export_path))
