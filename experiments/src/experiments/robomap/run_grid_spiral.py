import logging

from agents.baselines.grid_spiral import GridSpiralAgent
from experiments.robomap.log_context import configure_logging
from experiments.robomap.run_args import parse_run_args
from experiments.robomap.report import print_experiment_result
from experiments.robomap.runner import run_experiment

configure_logging()


def main() -> None:
    args = parse_run_args("Запуск baseline-агента GridSpiralAgent")
    result = run_experiment(
        GridSpiralAgent(),
        scenario=args.scenario,
        token="bs2",
        fps=args.fps,
        random_robot_pose=args.random_robot_pose,
        robot_pose_seed=args.robot_pose_seed,
        runs_root="runs",
    )
    print_experiment_result(result)


if __name__ == "__main__":
    main()
