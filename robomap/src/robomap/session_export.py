"""Экспорт и загрузка истории сессии для повторного рендера."""

import datetime
import os
from typing import Any, Dict, List, Tuple

from robomap.export_codec import (
    EXPORT_FILE_SUFFIX,
    EXPORT_FORMAT_VERSION,
    materialize_bundle_frames,
    read_bundle,
    write_bundle,
)
from robomap.session_manager import SimulationSession

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

EXPORTED_SESSIONS_DIR = os.path.join(project_path, "exported_sessions")


def build_export_bundle(session: SimulationSession) -> Dict[str, Any]:
    """Собирает bundle для экспорта из живой сессии (format v2: base + deltas)."""
    try:
        base_frame, deltas = session.recorder.get_encoded_history()
    except ValueError as exc:
        raise ValueError(
            "History is empty. Nothing to export."
        ) from exc

    return {
        "format_version": EXPORT_FORMAT_VERSION,
        "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "session_id": session.session_id,
        "scenario_name": session.scenario_name,
        "tick_count": session.world.tick_count,
        "map_boundaries": list(session.world.map_boundaries),
        "vis_config": session.config.get("visualization", {}),
        "frames_count": session.recorder.frame_count,
        "base_frame": base_frame,
        "deltas": deltas,
    }


def normalize_export_filename(filename: str) -> str:
    """Нормализует имя файла экспорта: basename + суффикс .msgpack.gz."""
    safe_name = os.path.basename(filename)
    if safe_name.endswith(EXPORT_FILE_SUFFIX):
        return safe_name
    return f"{safe_name}{EXPORT_FILE_SUFFIX}"


def _assert_inside_export_dir(resolved_path: str, export_dir: str, filename: str) -> None:
    """Защита от path traversal за пределы каталога экспорта."""
    if not resolved_path.startswith(export_dir + os.sep):
        raise ValueError(f"Недопустимое имя файла экспорта: {filename}")


def resolve_export_path(filename: str, *, for_write: bool = False) -> str:
    """Возвращает абсолютный путь к файлу экспорта с защитой от path traversal."""
    safe_name = os.path.basename(filename)
    export_dir = os.path.realpath(EXPORTED_SESSIONS_DIR)

    if safe_name.endswith(EXPORT_FILE_SUFFIX):
        resolved_path = os.path.realpath(os.path.join(export_dir, safe_name))
        _assert_inside_export_dir(resolved_path, export_dir, filename)
        return resolved_path

    suffix = EXPORT_FILE_SUFFIX
    resolved_path = os.path.realpath(os.path.join(export_dir, f"{safe_name}{suffix}"))
    _assert_inside_export_dir(resolved_path, export_dir, filename)

    if for_write or os.path.isfile(resolved_path):
        return resolved_path

    return resolved_path


def save_session_export(session: SimulationSession) -> str:
    """Сохраняет историю сессии в exported_sessions/{session_id}.msgpack.gz."""
    os.makedirs(EXPORTED_SESSIONS_DIR, exist_ok=True)
    output_path = resolve_export_path(session.session_id, for_write=True)
    bundle = build_export_bundle(session)
    write_bundle(output_path, bundle)
    return output_path


def load_session_export(filename: str) -> Dict[str, Any]:
    """Загружает bundle из файла экспорта и материализует полные frames."""
    export_path = resolve_export_path(filename, for_write=False)

    if not os.path.isfile(export_path):
        raise FileNotFoundError(
            f"Файл экспорта не найден: {normalize_export_filename(filename)}"
        )

    bundle = read_bundle(export_path)
    return materialize_bundle_frames(bundle)


def get_export_basename(filename: str) -> str:
    """Возвращает basename файла экспорта без суффикса .msgpack.gz."""
    normalized = normalize_export_filename(filename)
    if normalized.endswith(EXPORT_FILE_SUFFIX):
        return normalized[: -len(EXPORT_FILE_SUFFIX)]
    return os.path.splitext(normalized)[0]


def bundle_map_boundaries(bundle: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """Преобразует map_boundaries из bundle в tuple для рендера."""
    boundaries = bundle["map_boundaries"]
    return (boundaries[0], boundaries[1])
