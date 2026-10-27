"""Общие CLI-аргументы для одиночных run_* скриптов."""

from __future__ import annotations

import argparse


def parse_run_args(
    description: str,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Парсит --scenario и --fps для запуска эксперимента.

    Args:
        description: Описание скрипта для argparse.
        argv: Аргументы командной строки; None означает sys.argv.

    Returns:
        Распарсенные аргументы с полями scenario, fps,
        random_robot_pose и robot_pose_seed.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--scenario",
        default="exploring",
        help="Имя сценария (папка в robomap/data/scenarios/)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="FPS симуляции",
    )
    parser.add_argument(
        "--random-robot-pose",
        action="store_true",
        help=(
            "Случайная стартовая позиция робота "
            "внутри bbox карты"
        ),
    )
    parser.add_argument(
        "--robot-pose-seed",
        type=int,
        default=None,
        help="Seed для случайного спавна робота (опционально)",
    )
    return parser.parse_args(argv)
