"""CLI экспериментального пайплайна Robomap."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from experiments.robomap.log_context import configure_logging
from experiments.robomap.paths import ALLOWED_SPLITS
from experiments.robomap.pipeline import run_config_experiment
from experiments.robomap.pool import RobomapPool
from experiments.robomap.suite import DEFAULT_BASE_URL, normalize_base_urls
from meta_agent.config import build_agent_from_config, load_config
from meta_agent.config.freeze import assert_frozen_unchanged
from meta_agent.generated_paths import resolve_generated_paths
from meta_agent.smoke_inputs import SMOKE_SEED
from meta_agent.verification import verify_config_components


def _parse_base_urls_csv(value: str) -> list[str]:
    """Разбирает ``--base-urls`` как CSV-список URL."""
    parts = [item.strip() for item in value.split(",")]
    return [item for item in parts if item]


def _add_generated_path_args(
    parser: argparse.ArgumentParser,
) -> None:
    """Добавляет флаги путей к generated-компонентам."""
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=None,
        help=(
            "Корень generated/ соседнего проекта "
            "(tools/, transforms/, verified.json)"
        ),
    )
    parser.add_argument(
        "--tools-dir",
        type=Path,
        default=None,
        help="Override каталога generated tools",
    )
    parser.add_argument(
        "--transforms-dir",
        type=Path,
        default=None,
        help="Override каталога transforms",
    )
    parser.add_argument(
        "--verified-path",
        type=Path,
        default=None,
        help="Override пути к verified.json",
    )


def _add_output_arg(parser: argparse.ArgumentParser) -> None:
    """Добавляет ``--output text|json``."""
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Формат stdout: text (по умолчанию) или json",
    )


def _add_baseline_arg(parser: argparse.ArgumentParser) -> None:
    """Добавляет ``--baseline`` для freeze-check."""
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=(
            "Parent/baseline AgentConfig JSON; "
            "сверяет frozen-поля (model, routing, reasoning, "
            "model_kwargs, initial_user_message). "
            "Обязателен, если у candidate задан parent_id"
        ),
    )


def _assert_frozen_if_needed(args: argparse.Namespace) -> None:
    """Сверяет frozen-поля candidate с baseline при необходимости."""
    candidate = load_config(args.config)
    if args.baseline is None:
        if candidate.parent_id:
            raise ValueError(
                "parent_id is set; pass --baseline <parent.json> "
                "to verify frozen fields"
            )
        return

    baseline = load_config(args.baseline)
    assert_frozen_unchanged(candidate, baseline)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robomap-experiment",
        description=(
            "Проверка конфигов и прогон train/validation "
            "экспериментов Robomap"
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    verify_parser = subparsers.add_parser(
        "verify",
        help=(
            "Верифицировать generated-компоненты, "
            "указанные в config, в Docker sandbox"
        ),
    )
    verify_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Путь к agent config JSON",
    )
    _add_generated_path_args(verify_parser)
    _add_output_arg(verify_parser)

    build_parser = subparsers.add_parser(
        "build-check",
        help=(
            "Собрать агента из config без прогона suite "
            "(проверка verified hashes + сборка)"
        ),
    )
    build_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Путь к agent config JSON",
    )
    _add_generated_path_args(build_parser)
    _add_baseline_arg(build_parser)
    _add_output_arg(build_parser)

    run_parser = subparsers.add_parser(
        "run",
        help=(
            "Прогнать один config на выбранном split "
            "experiment suites с несколькими trials"
        ),
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Путь к agent config JSON",
    )
    run_parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Путь к experiment JSON",
    )
    run_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help=(
            "Корень robot-scenario для монтирования в контейнеры "
            "Robomap (по умолчанию текущая директория)"
        ),
    )
    run_parser.add_argument(
        "--split",
        choices=sorted(ALLOWED_SPLITS),
        required=True,
        help="Какой split прогнать: train или validation",
    )
    run_parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Корень директории артефактов",
    )
    run_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Сколько кейсов suite гонять параллельно "
            "внутри одного trial (по умолчанию 1)"
        ),
    )
    run_parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Один URL Robomap API (если не задан --base-urls)",
    )
    run_parser.add_argument(
        "--base-urls",
        type=_parse_base_urls_csv,
        default=None,
        help=(
            "Список URL Robomap через запятую "
            "(round-robin по кейсам)"
        ),
    )
    run_parser.add_argument(
        "--spawn-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Поднять batch_size контейнеров Robomap на время "
            "прогона (порты 18000+). По умолчанию включено; "
            "отключить: --no-spawn-pool"
        ),
    )
    _add_generated_path_args(run_parser)
    _add_baseline_arg(run_parser)
    _add_output_arg(run_parser)

    return parser


def _print_payload(
    payload: dict[str, Any],
    *,
    output: str,
) -> None:
    """Печатает ответ в text или json."""
    if output == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if payload.get("ok"):
        for key, value in payload.items():
            if key == "ok":
                continue
            print(f"{key}={value}")
    else:
        print(f"error={payload.get('error', 'unknown')}")


def _run_result_payload(
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    """Собирает машиночитаемый ответ команды ``run``."""
    trials = []
    for trial in manifest.get("trials", []):
        summary = trial.get("summary") or {}
        trials.append(
            {
                "trial_id": trial.get("trial_id"),
                "completed": summary.get("completed"),
                "failed": summary.get("failed"),
            }
        )

    return {
        "ok": True,
        "run_dir": str(manifest_path.parent.resolve()),
        "run_id": manifest.get("run_id"),
        "config_id": manifest.get("config_id"),
        "split": manifest.get("split"),
        "experiment_id": manifest.get("experiment_id"),
        "manifest_path": str(manifest_path.resolve()),
        "trials": trials,
    }


def _cmd_verify(args: argparse.Namespace) -> int:
    """Команда verify: schema-smoke в sandbox для компонентов config.

    Стенд сам строит 1 valid + 1 invalid вход из схемы/контракта;
    ``test_inputs`` не используются.
    """
    try:
        paths = resolve_generated_paths(
            args.generated_root,
            tools_dir=args.tools_dir,
            transforms_dir=args.transforms_dir,
            verified_path=args.verified_path,
        )
        config = load_config(args.config)
        verified_ids, results = verify_config_components(
            config,
            paths,
        )
        payload = {
            "ok": True,
            "seed": SMOKE_SEED,
            "config_id": config.config_id,
            "verified": verified_ids,
            "results": [
                {
                    "id": item.component_id,
                    "valid": item.valid,
                    "invalid": item.invalid,
                }
                for item in results
            ],
            "verified_path": str(paths.verified_path),
            "generated_root": str(paths.root),
        }
        _print_payload(payload, output=args.output)
        return 0
    except Exception as error:
        payload = {
            "ok": False,
            "seed": SMOKE_SEED,
            "error": str(error),
        }
        _print_payload(payload, output=args.output)
        return 1


def _cmd_run(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    raw_argv: list[str] | None,
) -> int:
    """Команда run: прогон train/validation."""
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    check_argv = raw_argv if raw_argv is not None else sys.argv
    explicit_spawn = "--spawn-pool" in check_argv

    if args.base_urls is not None and explicit_spawn:
        parser.error(
            "нельзя совмещать --spawn-pool и --base-urls"
        )

    generated_kwargs = {
        "generated_root": args.generated_root,
        "transform_directory": args.transforms_dir,
        "generated_tool_directory": args.tools_dir,
        "verified_path": args.verified_path,
    }
    project_root = args.project_root.resolve()

    try:
        _assert_frozen_if_needed(args)
    except Exception as error:
        payload = {"ok": False, "error": str(error)}
        _print_payload(payload, output=args.output)
        return 1

    def _run(base_urls: list[str], *, pool_spawned: bool) -> int:
        try:
            manifest, manifest_path = run_config_experiment(
                args.config,
                args.experiment,
                split=args.split,
                runs_root=args.runs_root,
                batch_size=args.batch_size,
                base_urls=base_urls,
                pool_spawned=pool_spawned,
                project_root=project_root,
                **generated_kwargs,
            )
            payload = _run_result_payload(manifest, manifest_path)
            if args.output == "json":
                _print_payload(payload, output="json")
            else:
                print(
                    f"split={manifest['split']} "
                    f"config_id={manifest['config_id']} "
                    f"config_hash={manifest['config_hash'][:12]} "
                    f"run_id={manifest['run_id']} "
                    f"trials={len(manifest['trials'])}/"
                    f"{manifest['trial_count']}"
                )
                for trial in manifest["trials"]:
                    summary = trial["summary"]
                    print(
                        f"  {trial['trial_id']}: "
                        f"completed={summary['completed']} "
                        f"failed={summary['failed']}"
                    )
                print(f"run={manifest_path.parent}")
            return 0
        except Exception as error:
            payload = {"ok": False, "error": str(error)}
            _print_payload(payload, output=args.output)
            return 1

    if args.base_urls is not None:
        urls = normalize_base_urls(args.base_urls)
        return _run(urls, pool_spawned=False)

    if args.spawn_pool:
        with RobomapPool(project_root=project_root) as pool:
            urls = pool.start(args.batch_size)
            return _run(urls, pool_spawned=True)

    urls = normalize_base_urls([args.base_url])
    return _run(urls, pool_spawned=False)


def _stub_model_factory(_config: Any) -> Any:
    """Заглушка LLM для ``build-check`` без сетевых вызовов."""
    from unittest.mock import MagicMock

    from langchain_core.language_models import BaseChatModel

    return MagicMock(spec=BaseChatModel)


def _cmd_build_check(args: argparse.Namespace) -> int:
    """Команда build-check: freeze-check + сборка агента без suite."""
    agent = None
    try:
        _assert_frozen_if_needed(args)
        config = load_config(args.config)
        agent = build_agent_from_config(
            config,
            model_factory=_stub_model_factory,
            generated_root=args.generated_root,
            transform_directory=args.transforms_dir,
            generated_tool_directory=args.tools_dir,
            verified_path=args.verified_path,
        )
        payload = {
            "ok": True,
            "config_id": config.config_id,
            "agent_type": type(agent).__name__,
        }
        _print_payload(payload, output=args.output)
        return 0
    except Exception as error:
        payload = {"ok": False, "error": str(error)}
        _print_payload(payload, output=args.output)
        return 1
    finally:
        if agent is not None:
            close = getattr(
                getattr(agent, "policy", None),
                "close",
                None,
            )
            if close is not None:
                try:
                    close()
                except Exception:
                    logging.getLogger(__name__).exception(
                        "policy.close() failed"
                    )


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI.

    Args:
        argv: Аргументы командной строки.

    Returns:
        Код завершения процесса.
    """
    configure_logging(level=logging.INFO)

    parser = _build_parser()
    args = parser.parse_args(argv)
    raw_argv = list(argv) if argv is not None else None

    if args.command == "verify":
        return _cmd_verify(args)
    if args.command == "build-check":
        return _cmd_build_check(args)
    if args.command == "run":
        return _cmd_run(args, parser=parser, raw_argv=raw_argv)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
