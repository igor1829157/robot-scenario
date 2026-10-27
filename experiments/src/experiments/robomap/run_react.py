import logging
import os
from pathlib import Path

from experiments.robomap.log_context import configure_logging
from experiments.robomap.run_args import parse_run_args
from experiments.robomap.report import print_experiment_result
from experiments.robomap.runner import run_experiment
from meta_agent.config import (
    build_agent_from_config,
    load_config,
)

configure_logging()

DEFAULT_CONFIG_PATH = Path(
    "meta_agent/configs/react_v1.json"
)


def main() -> None:
    args = parse_run_args(
        "Запуск ReAct-агента из файла конфигурации"
    )

    config_path = Path(
        os.getenv("REACT_CONFIG", str(DEFAULT_CONFIG_PATH))
    )

    if not config_path.is_file():
        raise SystemExit(f"файл конфигурации не найден: {config_path}")

    config = load_config(config_path)
    agent = build_agent_from_config(config)

    logging.info("config -> %s", config_path)
    logging.info("config_id -> %s", config.config_id)
    logging.info("model -> %s/%s", config.provider, config.model)

    result = run_experiment(
        agent,
        scenario=args.scenario,
        token="react",
        fps=args.fps,
        random_robot_pose=args.random_robot_pose,
        robot_pose_seed=args.robot_pose_seed,
        runs_root="runs",
    )
    print_experiment_result(result)


if __name__ == "__main__":
    main()
