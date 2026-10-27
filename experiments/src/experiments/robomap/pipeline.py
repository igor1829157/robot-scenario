"""Оркестрация прогона конфига на одном split."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.robomap.io import (
    relpath_from,
    write_json_atomic,
)
from experiments.robomap.paths import (
    RUN_MANIFEST_FILENAME,
    allocate_run_dir,
)
from experiments.robomap.suite import (
    ExperimentSpec,
    artifact_mode_for_split,
    config_hash,
    load_experiment,
    normalize_base_urls,
    run_suite,
    suite_for_split,
)
from meta_agent.config import load_config


logger = logging.getLogger(__name__)

RUN_SCHEMA_VERSION = 4
TRIAL_SCHEMA_VERSION = 2


def _suite_summary(manifest: dict[str, Any]) -> dict[str, int]:
    runs = manifest.get("runs", [])
    completed = sum(
        1 for run in runs if run.get("status") == "completed"
    )
    failed = len(runs) - completed
    return {
        "completed": completed,
        "failed": failed,
    }


def _build_run_manifest(
    *,
    output_dir: Path,
    experiment: ExperimentSpec,
    config_id: str,
    config_hash_value: str,
    config_path: Path,
    run_id: str,
    split: str,
    suite_id: str,
    suite_hash: str,
    started_at: str,
    finished_at: str | None,
    trials: list[dict[str, Any]],
    robomap_pool: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Собирает run manifest schema v4."""
    payload: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "split": split,
        "experiment_id": experiment.experiment_id,
        "experiment_hash": experiment.experiment_hash,
        "experiment_path": relpath_from(
            output_dir,
            experiment.path,
        ),
        "config_id": config_id,
        "config_hash": config_hash_value,
        "config_path": relpath_from(output_dir, config_path),
        "trial_count": experiment.trial_count,
        "suite_id": suite_id,
        "suite_hash": suite_hash,
        "started_at": started_at,
        "finished_at": finished_at,
        "trials": trials,
    }
    if robomap_pool is not None:
        payload["robomap_pool"] = robomap_pool
    return payload


def run_config_experiment(
    config_path: Path,
    experiment_path: Path,
    *,
    split: str,
    runs_root: Path | None = None,
    batch_size: int = 1,
    base_urls: Sequence[str] | str | None = None,
    base_url: str | None = None,
    pool_spawned: bool = False,
    project_root: Path | None = None,
    generated_root: Path | None = None,
    transform_directory: Path | None = None,
    generated_tool_directory: Path | None = None,
    verified_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Прогоняет config на одном split с несколькими trials.

    Args:
        config_path: Путь к config JSON.
        experiment_path: Путь к experiment JSON.
        split: Имя split (`train` / `validation`).
        runs_root: Корень артефактов; по умолчанию `runs`.
        batch_size: Параллельность кейсов внутри trial.
        base_urls: Список URL Robomap API.
        base_url: Устаревший одиночный URL, если ``base_urls`` не задан.
        pool_spawned: ``True``, если URL поднял оркестратор пула.
        project_root: Корень robot-scenario с экспортами Robomap.
        generated_root: Корень generated соседнего проекта.
        transform_directory: Override каталога transforms.
        generated_tool_directory: Override каталога tools.
        verified_path: Override пути ``verified.json``.

    Returns:
        Пара (run manifest dict, путь к run.json).
    """
    urls = normalize_base_urls(base_urls, base_url=base_url)
    robomap_pool = {
        "spawned": pool_spawned,
        "base_urls": urls,
    }

    root = Path(runs_root or "runs").resolve()
    experiment = load_experiment(experiment_path)
    config = load_config(config_path)
    hash_value = config_hash(config)
    suite = suite_for_split(experiment, split)
    artifact_mode = artifact_mode_for_split(split)

    output_dir, run_id = allocate_run_dir(
        experiment_id=experiment.experiment_id,
        config_id=config.config_id,
        split=split,
        runs_root=root,
    )
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    trials: list[dict[str, Any]] = []
    manifest_path = output_dir / RUN_MANIFEST_FILENAME

    write_json_atomic(
        manifest_path,
        _build_run_manifest(
            output_dir=output_dir,
            experiment=experiment,
            config_id=config.config_id,
            config_hash_value=hash_value,
            config_path=config_path,
            run_id=run_id,
            split=split,
            suite_id=suite.suite_id,
            suite_hash=suite.suite_hash,
            started_at=started_at,
            finished_at=None,
            trials=trials,
            robomap_pool=robomap_pool,
        ),
    )

    for trial_index in range(1, experiment.trial_count + 1):
        trial_id = f"trial_{trial_index:03d}"
        trial_dir = trials_dir / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "starting trial %s/%s (%s, split=%s)",
            trial_index,
            experiment.trial_count,
            trial_id,
            split,
        )

        trial_started = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )

        suite_manifest_path = trial_dir / "manifest.json"
        suite_manifest = run_suite(
            config,
            suite,
            config_path=config_path,
            config_hash_value=hash_value,
            split=split,
            manifest_path=suite_manifest_path,
            run_id=run_id,
            trial_id=trial_id,
            trial_index=trial_index,
            artifact_mode=artifact_mode,
            batch_size=batch_size,
            base_urls=urls,
            project_root=project_root,
            generated_root=generated_root,
            transform_directory=transform_directory,
            generated_tool_directory=generated_tool_directory,
            verified_path=verified_path,
        )

        trial_finished = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        suite_summary = _suite_summary(suite_manifest)
        suite_manifest_rel = relpath_from(
            output_dir,
            suite_manifest_path,
        )
        trial_rel = relpath_from(output_dir, trial_dir / "trial.json")

        trial_payload = {
            "schema_version": TRIAL_SCHEMA_VERSION,
            "trial_id": trial_id,
            "trial_index": trial_index,
            "run_id": run_id,
            "split": split,
            "config_id": config.config_id,
            "config_hash": hash_value,
            "experiment_id": experiment.experiment_id,
            "experiment_hash": experiment.experiment_hash,
            "suite_id": suite.suite_id,
            "suite_hash": suite.suite_hash,
            "manifest_path": suite_manifest_rel,
            "summary": suite_summary,
            "started_at": trial_started,
            "finished_at": trial_finished,
        }

        trial_path = trial_dir / "trial.json"
        write_json_atomic(trial_path, trial_payload)

        trials.append(
            {
                "trial_id": trial_id,
                "trial_index": trial_index,
                "trial_path": trial_rel,
                "summary": suite_summary,
            }
        )

        write_json_atomic(
            manifest_path,
            _build_run_manifest(
                output_dir=output_dir,
                experiment=experiment,
                config_id=config.config_id,
                config_hash_value=hash_value,
                config_path=config_path,
                run_id=run_id,
                split=split,
                suite_id=suite.suite_id,
                suite_hash=suite.suite_hash,
                started_at=started_at,
                finished_at=None,
                trials=trials,
                robomap_pool=robomap_pool,
            ),
        )
        logger.info(
            "completed trial %s/%s",
            trial_index,
            experiment.trial_count,
        )

    finished_at = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    manifest = _build_run_manifest(
        output_dir=output_dir,
        experiment=experiment,
        config_id=config.config_id,
        config_hash_value=hash_value,
        config_path=config_path,
        run_id=run_id,
        split=split,
        suite_id=suite.suite_id,
        suite_hash=suite.suite_hash,
        started_at=started_at,
        finished_at=finished_at,
        trials=trials,
        robomap_pool=robomap_pool,
    )
    write_json_atomic(manifest_path, manifest)
    logger.info("run manifest -> %s", manifest_path)
    return manifest, manifest_path
