"""Точка входа CLI редактора сценариев."""

from __future__ import annotations

import argparse
import sys

from scenario_editor.editor import run_editor
from scenario_editor.io import load_scenario


def build_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Редактор карты сценариев Robomap")
    parser.add_argument(
        "scenario",
        nargs="?",
        default="exploring",
        help="Имя исходного сценария (папка в robomap/data/scenarios/)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Имя нового сценария при сохранении (без запроса)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Запуск редактора."""
    args = build_parser().parse_args(argv)

    try:
        bundle = load_scenario(args.scenario)
    except FileNotFoundError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    run_editor(bundle, default_out_name=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
