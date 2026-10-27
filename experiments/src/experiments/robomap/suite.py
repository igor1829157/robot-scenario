"""Загрузка suite/experiment и прогон одного suite."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.robomap.io import (
    canonical_json_hash,
    load_json_object,
    relpath_from,
    resolve_relative,
    write_json_atomic,
)
from experiments.robomap.log_context import run_log_context
from experiments.robomap.runner import ArtifactMode, run_experiment
from meta_agent.config import (
    AgentConfig,
    build_agent_from_config,
    config_to_dict,
)

DEFAULT_BASE_URL = "http://localhost:8000"


logger = logging.getLogger(__name__)

SUITE_SCHEMA_VERSION = 1
EXPERIMENT_SCHEMA_VERSION = 2


class SuiteError(ValueError):
    """Ошибка валидации suite или experiment specification."""


@dataclass(frozen=True)
class SuiteCase:
    """Один кейс suite."""

    scenario: str
    fps: float
    random_robot_pose: bool
    robot_pose_seed: int


@dataclass(frozen=True)
class SuiteSpec:
    """Валидированный suite."""

    suite_id: str
    cases: tuple[SuiteCase, ...]
    path: Path
    suite_hash: str


@dataclass(frozen=True)
class ExperimentSpec:
    """Валидированная experiment specification."""

    experiment_id: str
    path: Path
    experiment_hash: str
    train_suite: SuiteSpec
    validation_suite: SuiteSpec
    trial_count: int


def _require_non_empty_str(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SuiteError(f"{field_name} must be a non-empty string")
    return value


def _require_positive_float(
    value: Any,
    field_name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SuiteError(f"{field_name} must be a number")
    result = float(value)
    if result <= 0:
        raise SuiteError(f"{field_name} must be positive")
    return result


def _require_int(
    value: Any,
    field_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SuiteError(f"{field_name} must be an integer")
    return value


def _require_bool(
    value: Any,
    field_name: str,
) -> bool:
    if not isinstance(value, bool):
        raise SuiteError(f"{field_name} must be a boolean")
    return value


def _require_positive_int(
    value: Any,
    field_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SuiteError(f"{field_name} must be an integer")
    if value <= 0:
        raise SuiteError(f"{field_name} must be positive")
    return value


def normalize_base_urls(
    base_urls: Sequence[str] | str | None = None,
    *,
    base_url: str | None = None,
) -> list[str]:
    """Нормализует список URL Robomap API.

    Args:
        base_urls: Список URL или один URL-строка.
        base_url: Устаревший одиночный URL (если ``base_urls`` не задан).

    Returns:
        Непустой список непустых URL без хвостовых ``/``.

    Raises:
        SuiteError: Если список пуст или содержит пустые строки.
    """
    if base_urls is None:
        raw: Sequence[str] = (
            [base_url] if base_url is not None else [DEFAULT_BASE_URL]
        )
    elif isinstance(base_urls, str):
        raw = [base_urls]
    else:
        raw = base_urls

    urls = [item.strip().rstrip("/") for item in raw if item is not None]
    if not urls:
        raise SuiteError("base_urls must be a non-empty list")
    for index, url in enumerate(urls):
        if not url:
            raise SuiteError(
                f"base_urls[{index}] must be a non-empty string"
            )
    return urls


def load_suite(path: Path) -> SuiteSpec:
    """Загружает и строго валидирует suite JSON.

    Args:
        path: Путь к suite-файлу.

    Returns:
        Валидированный SuiteSpec.

    Raises:
        SuiteError: При нарушении контракта suite.
    """
    suite_path = path.resolve()
    data = load_json_object(suite_path)

    schema_version = data.get("schema_version", SUITE_SCHEMA_VERSION)
    if schema_version != SUITE_SCHEMA_VERSION:
        raise SuiteError(
            f"unsupported suite schema_version: {schema_version}"
        )

    suite_id = _require_non_empty_str(
        data.get("suite_id"),
        "suite_id",
    )
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SuiteError("suite.cases must be a non-empty list")

    cases: list[SuiteCase] = []
    seen_keys: set[tuple[str, int]] = set()

    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise SuiteError(f"suite.cases[{index}] must be an object")

        scenario = _require_non_empty_str(
            item.get("scenario"),
            f"suite.cases[{index}].scenario",
        )
        fps = _require_positive_float(
            item.get("fps", 2.0),
            f"suite.cases[{index}].fps",
        )
        random_robot_pose = _require_bool(
            item.get(
                "random_robot_pose",
                item.get("robot_random_pose", False),
            ),
            f"suite.cases[{index}].random_robot_pose",
        )
        robot_pose_seed = _require_int(
            item.get("robot_pose_seed"),
            f"suite.cases[{index}].robot_pose_seed",
        )

        key = scenario, robot_pose_seed
        if key in seen_keys:
            raise SuiteError(
                "duplicate suite case: "
                f"{scenario}:seed={robot_pose_seed}"
            )
        seen_keys.add(key)

        cases.append(
            SuiteCase(
                scenario=scenario,
                fps=fps,
                random_robot_pose=random_robot_pose,
                robot_pose_seed=robot_pose_seed,
            )
        )

    canonical = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "suite_id": suite_id,
        "cases": [
            {
                "scenario": case.scenario,
                "fps": case.fps,
                "random_robot_pose": case.random_robot_pose,
                "robot_pose_seed": case.robot_pose_seed,
            }
            for case in cases
        ],
    }

    return SuiteSpec(
        suite_id=suite_id,
        cases=tuple(cases),
        path=suite_path,
        suite_hash=canonical_json_hash(canonical),
    )


def load_experiment(path: Path) -> ExperimentSpec:
    """Загружает experiment specification и связанные suites.

    Args:
        path: Путь к experiment JSON.

    Returns:
        Валидированный ExperimentSpec.

    Raises:
        SuiteError: При нарушении контракта.
    """
    experiment_path = path.resolve()
    data = load_json_object(experiment_path)

    schema_version = data.get(
        "schema_version",
        EXPERIMENT_SCHEMA_VERSION,
    )
    if schema_version != EXPERIMENT_SCHEMA_VERSION:
        raise SuiteError(
            "unsupported experiment schema_version: "
            f"{schema_version}"
        )

    experiment_id = _require_non_empty_str(
        data.get("experiment_id"),
        "experiment_id",
    )
    train_suite_value = _require_non_empty_str(
        data.get("train_suite"),
        "train_suite",
    )
    validation_suite_value = _require_non_empty_str(
        data.get("validation_suite"),
        "validation_suite",
    )

    trial_count = _require_positive_int(
        data.get("trial_count", 1),
        "trial_count",
    )

    train_suite = load_suite(
        resolve_relative(experiment_path, train_suite_value)
    )
    validation_suite = load_suite(
        resolve_relative(
            experiment_path,
            validation_suite_value,
        )
    )

    canonical = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "train_suite": train_suite.suite_hash,
        "validation_suite": validation_suite.suite_hash,
        "trial_count": trial_count,
    }

    return ExperimentSpec(
        experiment_id=experiment_id,
        path=experiment_path,
        experiment_hash=canonical_json_hash(canonical),
        train_suite=train_suite,
        validation_suite=validation_suite,
        trial_count=trial_count,
    )


def config_hash(config: AgentConfig) -> str:
    """Хеширует полный AgentConfig.

    Args:
        config: Конфигурация агента.

    Returns:
        SHA-256 hex-строка.
    """
    return canonical_json_hash(config_to_dict(config))


def suite_for_split(
    experiment: ExperimentSpec,
    split: str,
) -> SuiteSpec:
    """Возвращает suite для указанного split.

    Args:
        experiment: Спецификация эксперимента.
        split: Имя split (`train` / `validation`).

    Returns:
        Соответствующий SuiteSpec.

    Raises:
        ValueError: Если split не поддерживается.
    """
    if split == "train":
        return experiment.train_suite
    if split == "validation":
        return experiment.validation_suite
    raise ValueError(f"unsupported split: {split}")


def artifact_mode_for_split(split: str) -> ArtifactMode:
    """Определяет политику артефактов по split.

    Args:
        split: Имя split (`train` / `validation`).

    Returns:
        Режим сохранения артефактов.

    Raises:
        ValueError: Если split не поддерживается.
    """
    if split == "train":
        return ArtifactMode.FULL
    if split == "validation":
        return ArtifactMode.METRICS_ONLY
    raise ValueError(f"unsupported split: {split}")


def world_dir_name(case: SuiteCase) -> str:
    """Строит имя каталога мира для кейса suite.

    Args:
        case: Кейс suite.

    Returns:
        Имя каталога вида ``<scenario>__seed_<seed>``.
    """
    return f"{case.scenario}__seed_{case.robot_pose_seed}"


def _normalize_run_entry(
    manifest_dir: Path,
    run_entry: dict[str, Any],
) -> dict[str, Any]:
    """Переводит абсолютные пути артефактов в относительные."""
    entry = dict(run_entry)
    run_dir = entry.get("run_dir")
    if isinstance(run_dir, str) and run_dir:
        entry["run_dir"] = relpath_from(manifest_dir, run_dir)
    metrics_path = entry.get("metrics_path")
    if isinstance(metrics_path, str) and metrics_path:
        entry["metrics_path"] = relpath_from(
            manifest_dir,
            metrics_path,
        )
    return entry


def _run_case(
    config: AgentConfig,
    case: SuiteCase,
    *,
    world_dir: Path,
    artifact_mode: ArtifactMode,
    base_url: str = DEFAULT_BASE_URL,
    project_root: Path | None = None,
    generated_root: Path | None = None,
    transform_directory: Path | None = None,
    generated_tool_directory: Path | None = None,
    verified_path: Path | None = None,
) -> dict[str, Any]:
    """Прогоняет один кейс suite в каталоге мира."""
    agent = build_agent_from_config(
        config,
        generated_root=generated_root,
        transform_directory=transform_directory,
        generated_tool_directory=generated_tool_directory,
        verified_path=verified_path,
    )

    with run_log_context(scenario=case.scenario):
        try:
            result = run_experiment(
                agent,
                base_url=base_url,
                scenario=case.scenario,
                token=config.config_id,
                fps=case.fps,
                random_robot_pose=case.random_robot_pose,
                robot_pose_seed=case.robot_pose_seed,
                run_dir=world_dir,
                artifact_mode=artifact_mode,
                project_root=project_root,
            )
            return {
                "scenario": case.scenario,
                "robot_pose_seed": case.robot_pose_seed,
                "status": "completed",
                "run_dir": result.run_dir,
                "metrics_path": result.metrics_path,
                "error": None,
            }
        except Exception as error:
            logger.exception(
                "case failed: robot_pose_seed=%s",
                case.robot_pose_seed,
            )
            return {
                "scenario": case.scenario,
                "robot_pose_seed": case.robot_pose_seed,
                "status": "failed",
                "run_dir": None,
                "metrics_path": None,
                "error": str(error),
            }
        finally:
            close = getattr(
                getattr(agent, "policy", None),
                "close",
                None,
            )
            if close is not None:
                try:
                    close()
                except Exception:
                    logger.exception("policy.close() failed")


def run_suite(
    config: AgentConfig,
    suite: SuiteSpec,
    *,
    config_path: Path,
    config_hash_value: str,
    split: str,
    manifest_path: Path,
    run_id: str,
    trial_id: str,
    trial_index: int,
    artifact_mode: ArtifactMode,
    batch_size: int = 1,
    base_urls: Sequence[str] | str | None = None,
    base_url: str | None = None,
    project_root: Path | None = None,
    generated_root: Path | None = None,
    transform_directory: Path | None = None,
    generated_tool_directory: Path | None = None,
    verified_path: Path | None = None,
) -> dict[str, Any]:
    """Прогоняет suite (опционально параллельно) и пишет manifest.

    Args:
        config: Загруженная конфигурация агента.
        suite: Валидированный suite.
        config_path: Путь к config JSON.
        config_hash_value: Хеш конфига.
        split: Имя split (`train` / `validation`).
        manifest_path: Куда сохранить suite manifest.
        run_id: Читаемый ID запуска.
        trial_id: ID текущего trial.
        trial_index: Порядковый номер trial (1-based).
        artifact_mode: Политика сохранения артефактов миров.
        batch_size: Сколько кейсов гонять одновременно (>= 1).
        base_urls: Список URL Robomap API (round-robin по кейсам).
        base_url: Устаревший одиночный URL, если ``base_urls`` не задан.
        project_root: Корень robot-scenario с экспортами Robomap.
        generated_root: Корень generated соседнего проекта.
        transform_directory: Override каталога transforms.
        generated_tool_directory: Override каталога tools.
        verified_path: Override пути ``verified.json``.

    Returns:
        Записанный suite manifest.
    """
    workers = _require_positive_int(batch_size, "batch_size")
    urls = normalize_base_urls(base_urls, base_url=base_url)

    started_at = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    manifest_dir = manifest_path.parent
    worlds_dir = manifest_dir / "worlds"
    worlds_dir.mkdir(parents=True, exist_ok=True)

    case_world_dirs = [
        (case, worlds_dir / world_dir_name(case))
        for case in suite.cases
    ]
    for _, world_dir in case_world_dirs:
        world_dir.mkdir(parents=True, exist_ok=True)

    results_by_index: dict[int, dict[str, Any]] = {}

    logger.info(
        "suite %s: %s cases, batch_size=%s, base_urls=%s",
        suite.suite_id,
        len(suite.cases),
        workers,
        urls,
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_index = {
            pool.submit(
                _run_case,
                config,
                case,
                world_dir=world_dir,
                artifact_mode=artifact_mode,
                base_url=urls[index % len(urls)],
                project_root=project_root,
                generated_root=generated_root,
                transform_directory=transform_directory,
                generated_tool_directory=generated_tool_directory,
                verified_path=verified_path,
            ): index
            for index, (case, world_dir) in enumerate(
                case_world_dirs
            )
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results_by_index[index] = _normalize_run_entry(
                manifest_dir,
                future.result(),
            )

    runs = [
        results_by_index[index]
        for index in range(len(suite.cases))
    ]

    finished_at = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )

    manifest = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "suite_id": suite.suite_id,
        "suite_hash": suite.suite_hash,
        "suite_path": relpath_from(manifest_dir, suite.path),
        "config_id": config.config_id,
        "config_hash": config_hash_value,
        "config_path": relpath_from(manifest_dir, config_path),
        "run_id": run_id,
        "trial_id": trial_id,
        "trial_index": trial_index,
        "split": split,
        "artifact_mode": artifact_mode.value,
        "batch_size": workers,
        "base_urls": urls,
        "started_at": started_at,
        "finished_at": finished_at,
        "runs": runs,
    }

    write_json_atomic(manifest_path, manifest)
    logger.info("suite manifest -> %s", manifest_path)
    return manifest
