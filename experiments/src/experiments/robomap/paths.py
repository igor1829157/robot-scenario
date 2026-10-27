"""Пути и идентификаторы experiment run."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


RUN_MANIFEST_FILENAME = "run.json"
LEGACY_RUN_MANIFEST_FILENAME = "config_run.json"
ALLOWED_SPLITS = frozenset({"train", "validation"})


def generate_readable_run_id(
    now: datetime | None = None,
) -> str:
    """Генерирует читаемый ID запуска.

    Args:
        now: Момент времени; по умолчанию текущий UTC.

    Returns:
        Строка вида ``YYYY-MM-DD_HH-MM-SS``.
    """
    moment = now or datetime.now(timezone.utc)
    return moment.strftime("%Y-%m-%d_%H-%M-%S")


def legacy_run_id_to_readable(legacy_run_id: str) -> str:
    """Преобразует старый config_run_id в читаемый run_id.

    Args:
        legacy_run_id: Старый ID вида ``YYYYMMDD_HHMMSS_...``.

    Returns:
        Читаемый run_id.
    """
    parts = legacy_run_id.split("_")
    if len(parts) < 2 or len(parts[0]) != 8 or len(parts[1]) != 6:
        return legacy_run_id

    date_part = parts[0]
    time_part = parts[1]
    return (
        f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}_"
        f"{time_part[:2]}-{time_part[2:4]}-{time_part[4:6]}"
    )


def allocate_run_dir(
    *,
    experiment_id: str,
    config_id: str,
    split: str,
    runs_root: Path,
    now: datetime | None = None,
) -> tuple[Path, str]:
    """Выделяет директорию нового запуска без коллизий.

    Args:
        experiment_id: ID эксперимента.
        config_id: ID конфига агента.
        split: Имя split (`train` / `validation`).
        runs_root: Корень ``runs/``.
        now: Момент времени для ID.

    Returns:
        Пара (директория запуска, run_id).

    Raises:
        ValueError: Если split не поддерживается.
    """
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported split: {split}")

    config_dir = runs_root / experiment_id / config_id / split
    base_id = generate_readable_run_id(now)
    run_id = base_id
    suffix = 2

    while (config_dir / run_id).exists():
        run_id = f"{base_id}_{suffix:02d}"
        suffix += 1

    return config_dir / run_id, run_id


def experiment_run_root(
    *,
    experiment_id: str,
    config_id: str,
    split: str,
    run_id: str,
    runs_root: Path,
) -> Path:
    """Строит директорию одного запуска.

    Args:
        experiment_id: ID эксперимента.
        config_id: ID конфига.
        split: Имя split (`train` / `validation`).
        run_id: Читаемый ID запуска.
        runs_root: Корень ``runs/``.

    Returns:
        Путь ``runs/<experiment_id>/<config_id>/<split>/<run_id>/``.

    Raises:
        ValueError: Если split не поддерживается.
    """
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported split: {split}")

    return runs_root / experiment_id / config_id / split / run_id


def resolve_run_manifest_path(path: Path) -> Path:
    """Резолвит путь к run manifest из каталога или файла.

    Args:
        path: Каталог запуска или путь к manifest JSON.

    Returns:
        Абсолютный путь к ``run.json`` или legacy ``config_run.json``.

    Raises:
        FileNotFoundError: Если manifest не найден.
    """
    resolved = path.resolve()
    if resolved.is_dir():
        for name in (RUN_MANIFEST_FILENAME, LEGACY_RUN_MANIFEST_FILENAME):
            candidate = resolved / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"run manifest not found in directory: {resolved}"
        )
    return resolved
