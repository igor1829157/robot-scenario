"""Разрешение путей к generated tools/transforms и verified.json."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_GENERATED_ROOT = Path(__file__).resolve().parents[2] / "generated"


@dataclass(frozen=True)
class GeneratedPaths:
    """Каталоги generated-компонентов для одного корня."""

    root: Path
    tools_dir: Path
    transforms_dir: Path
    verified_path: Path


def resolve_generated_paths(
    generated_root: str | Path | None = None,
    *,
    tools_dir: str | Path | None = None,
    transforms_dir: str | Path | None = None,
    verified_path: str | Path | None = None,
) -> GeneratedPaths:
    """Собирает пути к tools/transforms/verified.json.

    Args:
        generated_root: Корень ``generated/``; по умолчанию
            ``meta_agent/generated``.
        tools_dir: Явный каталог tools (override).
        transforms_dir: Явный каталог transforms (override).
        verified_path: Явный путь к ``verified.json`` (override).

    Returns:
        Нормализованные абсолютные пути.
    """
    root = (
        Path(generated_root).resolve()
        if generated_root is not None
        else DEFAULT_GENERATED_ROOT.resolve()
    )

    return GeneratedPaths(
        root=root,
        tools_dir=(
            Path(tools_dir).resolve()
            if tools_dir is not None
            else (root / "tools")
        ),
        transforms_dir=(
            Path(transforms_dir).resolve()
            if transforms_dir is not None
            else (root / "transforms")
        ),
        verified_path=(
            Path(verified_path).resolve()
            if verified_path is not None
            else (root / "verified.json")
        ),
    )
