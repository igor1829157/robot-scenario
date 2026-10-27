import inspect
import json
import logging
import os
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from git import Repo

from robomap.export_codec import EXPORT_FORMAT_VERSION, load_export_bundle
from robomap_client.core.client import RobomapClient
from robomap_client.metrics.exploration import calculate_exploration_metrics
from robomap_client.metrics.plots import plot_exploration_timeseries
from robomap_client.metrics.timeseries import (
    build_exploration_timeseries,
    summarize_timeseries,
)

from experiments.robomap.log_context import bind_session, run_log_context
from experiments.telemetry.openrouter import (
    collect_openrouter_metadata,
)


logger = logging.getLogger(__name__)

_WORKSPACE_PREFIX = "/workspace/robot-scenario"


class ArtifactMode(str, Enum):
    """Политика сохранения артефактов прогона."""

    FULL = "full"
    METRICS_ONLY = "metrics_only"


class Agent(Protocol):
    name: str

    def run(self, client: RobomapClient) -> None: ...


@dataclass
class ExperimentResult:
    """Результат одного прогона мира."""

    session_id: str
    run_dir: str
    metrics: dict[str, Any]
    metrics_path: str
    timeseries: list[dict[str, Any]] | None = None
    videos: dict[str, str] | None = None
    timeseries_path: str | None = None
    plot_path: str | None = None


def _project_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    return Path.cwd().resolve()


def _to_local_path(path: str | Path, project_root: Path) -> Path:
    """Преобразует путь из контейнера (/workspace/...) в путь на хосте."""
    raw = Path(path)
    if raw.exists():
        return raw

    text = str(path)
    if text.startswith(_WORKSPACE_PREFIX):
        rel = text[len(_WORKSPACE_PREFIX) :].lstrip("/")
        return project_root / rel

    return project_root / raw


def _make_run_dir(runs_root: Path, agent_name: str, session_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / "robomap" / f"{agent_name}_{session_id}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _prepare_full_run_dir(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_meta_atomic(path: Path, meta: dict[str, Any]) -> None:
    """Атомарно записывает meta.json через временный файл."""
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=4, ensure_ascii=False)
    os.replace(tmp_path, path)


def _relative_artifact(run_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _extract_robot_pose(state: dict[str, Any], robot_id: str) -> tuple[float | None, float | None]:
    """Извлекает координаты робота из ответа GET /state."""
    robot_state = (state.get("robots_state") or {}).get(robot_id) or {}
    x = robot_state.get("x")
    y = robot_state.get("y")
    return x, y


def _run_agent(agent: Agent, client: RobomapClient, run_dir: Path) -> None:
    """Запускает агента; передаёт log_dir, если run() это поддерживает."""
    run_fn = agent.run
    if "log_dir" in inspect.signature(run_fn).parameters:
        run_fn(client, log_dir=run_dir)
    else:
        run_fn(client)


def _build_llm_meta(
    report: dict[str, Any],
    config_snapshot: dict[str, Any] | None
) -> dict[str, Any]:

    config_snapshot = config_snapshot or {}

    inline = report.get("inline") or {}
    inline_usage = inline.get("usage") or {}
    summary = report.get("summary") or {}

    model_permaslugs = summary.get("model_permaslugs") or []
    upstream_providers = summary.get("upstream_providers") or []
    endpoint_ids = summary.get("endpoint_ids") or []

    requested_model = config_snapshot.get("model")

    return {
        "status": report.get("status"),
        "requested": {
            "integration": "langchain-openrouter",
            "model": requested_model,
            "provider_routing": config_snapshot.get(
                "provider_routing"
            ),
            "reasoning": config_snapshot.get(
                "reasoning"
            ),
            "model_kwargs": config_snapshot.get(
                "model_kwargs"
            ),
        },
        "actual": {
            "calls": inline.get("calls", 0),
            "resolved_generations": summary.get(
                "resolved_generations",
                0,
            ),
            "failed_generations": summary.get(
                "failed_generations",
                0,
            ),
            "models": summary.get("models", []),
            "model_permaslugs": model_permaslugs,
            "upstream_providers": upstream_providers,
            "endpoint_ids": endpoint_ids,
            "single_model_version": (
                len(model_permaslugs) == 1
            ),
            "single_upstream_provider": (
                len(upstream_providers) == 1
            ),
            "single_endpoint": (
                len(endpoint_ids) == 1
            ),
            "requested_model_observed": (
                isinstance(requested_model, str)
                and requested_model in model_permaslugs
            ),
        },
        "usage": {
            "input_tokens": summary.get(
                "native_prompt_tokens",
                inline_usage.get("input_tokens", 0),
            ),
            "completion_tokens": summary.get(
                "native_completion_tokens",
                inline_usage.get("output_tokens", 0),
            ),
            "reasoning_tokens": summary.get(
                "native_reasoning_tokens",
                inline_usage.get("reasoning_tokens", 0),
            ),
            "cached_tokens": summary.get(
                "native_cached_tokens",
                inline_usage.get("cache_read_tokens", 0),
            ),
            "max_input_tokens_per_call": (
                inline_usage.get(
                    "max_input_tokens_per_call",
                    0,
                )
            ),
            "max_output_tokens_per_call": (
                inline_usage.get(
                    "max_output_tokens_per_call",
                    0,
                )
            ),
            "max_reasoning_tokens_per_call": (
                inline_usage.get(
                    "max_reasoning_tokens_per_call",
                    0,
                )
            ),
            "reasoning_share_of_output": (
                inline_usage.get(
                    "reasoning_share_of_output",
                    "0",
                )
            ),
            "inline_cost_usd": inline_usage.get(
                "inline_cost_usd"
            ),
            "verified_cost_usd": summary.get(
                "verified_cost_usd"
            ),
            "currency": "USD",
        },
        "reasoning_output": inline.get(
            "reasoning_output",
            {},
        ),
    }


def _git_meta(root: Path) -> dict[str, str | None]:

    try:
        repo = Repo(root, search_parent_directories=True)
        return {
            "branch": None if repo.head.is_detached else repo.active_branch.name,
            "commit_sha": repo.head.commit.hexsha,
        }
    except Exception:
        logger.warning("Git metadata is unavailable", exc_info=True)
        return {"branch": None, "commit_sha": None}


def _build_metrics_payload(
    *,
    agent_name: str,
    session_id: str,
    run_dir: Path,
    snapshot: dict[str, Any],
    ts_summary: dict[str, Any],
    frames_count: int,
) -> dict[str, Any]:
    return {
        "agent": agent_name,
        "session_id": session_id,
        "run_dir": str(run_dir),
        "snapshot": snapshot,
        "timeseries_summary": ts_summary,
        "frames_count": frames_count,
    }


def _compute_metrics_from_export(
    export_path: Path,
    *,
    robot_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], int]:
    bundle = load_export_bundle(export_path)
    frames = bundle.get("frames") or []
    if not frames:
        raise ValueError(f"В экспорте нет кадров: {export_path}")

    frames_count = len(frames)
    series = build_exploration_timeseries(frames, robot_id=robot_id)
    snapshot = calculate_exploration_metrics(frames[-1], robot_id=robot_id)
    ts_summary = summarize_timeseries(
        series,
        snapshot["total_top_level_objects"],
    )
    return series, snapshot, ts_summary, frames_count


def _run_experiment_metrics_only(
    agent: Agent,
    *,
    base_url: str,
    scenario: str,
    robot_id: str,
    token: str | None,
    fps: float,
    random_robot_pose: bool,
    robot_pose_seed: int | None,
    run_dir: Path,
    project_root: Path,
) -> ExperimentResult:
    """Прогон с сохранением только metrics.json в каталоге мира."""
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"

    client = RobomapClient(base_url=base_url)
    session_id = client.init_session(
        scenario,
        robot_id=robot_id,
        token=token or agent.name,
        fps=fps,
        random_robot_pose=random_robot_pose,
        robot_pose_seed=robot_pose_seed,
    )

    temp_log_dir = Path(
        tempfile.mkdtemp(prefix="robomap_agent_logs_")
    )

    with bind_session(session_id):
        try:
            try:
                _run_agent(agent, client, temp_log_dir)
            except Exception:
                logger.exception("agent failed in metrics_only mode")
                raise

            client.pause_session()
            export_payload = client.export_session()
            export_src = _to_local_path(
                export_payload["export_path"],
                project_root,
            )
            if not export_src.is_file():
                raise FileNotFoundError(f"export not found: {export_src}")

            _, snapshot, ts_summary, frames_count = (
                _compute_metrics_from_export(
                    export_src,
                    robot_id=robot_id,
                )
            )
            metrics = _build_metrics_payload(
                agent_name=agent.name,
                session_id=session_id,
                run_dir=run_dir,
                snapshot=snapshot,
                ts_summary=ts_summary,
                frames_count=frames_count,
            )
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=4, ensure_ascii=False)

            logger.info("metrics_only run_dir → %s", run_dir)
            logger.info("metrics → %s", metrics_path)

            return ExperimentResult(
                session_id=session_id,
                run_dir=str(run_dir),
                metrics=metrics,
                metrics_path=str(metrics_path),
            )
        finally:
            try:
                client.delete_session()
            except Exception:
                logger.debug(
                    "session delete skipped or failed",
                    exc_info=True,
                )
            shutil.rmtree(temp_log_dir, ignore_errors=True)


def _run_experiment_full(
    agent: Agent,
    *,
    base_url: str,
    scenario: str,
    robot_id: str,
    token: str | None,
    fps: float,
    random_robot_pose: bool,
    robot_pose_seed: int | None,
    runs_root: Path,
    run_dir: Path | None,
    project_root: Path,
    video_wait_timeout_sec: float,
    render_workers: int | None,
) -> ExperimentResult:
    """Полный прогон: session → agent → metrics (без export/video в артефактах)."""
    del video_wait_timeout_sec, render_workers

    client = RobomapClient(base_url=base_url)
    session_id = client.init_session(
        scenario,
        robot_id=robot_id,
        token=token or agent.name,
        fps=fps,
        random_robot_pose=random_robot_pose,
        robot_pose_seed=robot_pose_seed,
    )
    with bind_session(session_id):
        if run_dir is None:
            run_dir = _make_run_dir(runs_root, agent.name, session_id)
        else:
            run_dir = _prepare_full_run_dir(run_dir)
        initial_state = client.get_full_state()
        robot_x, robot_y = _extract_robot_pose(initial_state, robot_id)
        meta_path = run_dir / "meta.json"
        started_at = datetime.now().isoformat(timespec="seconds")
        started_mono = datetime.now().timestamp()

        agent_metadata = getattr(agent, "metadata", {})

        if not isinstance(agent_metadata, dict):
            agent_metadata = {}

        meta: dict[str, Any] = {
            "schema_version": 3,
            "status": "running",
            "git": _git_meta(project_root),
            "agent": agent.name,
            "agent_config_id": agent_metadata.get("agent_config_id"),
            "prompt_hash": agent_metadata.get("prompt_hash"),
            "config_snapshot": agent_metadata.get("config_snapshot"),
            "transform_snapshots": agent_metadata.get(
                "transform_snapshots",
                [],
            ),
            "generated_tool_snapshots": agent_metadata.get(
                "generated_tool_snapshots",
                [],
            ),
            "session_id": session_id,
            "scenario": scenario,
            "robot_id": robot_id,
            "fps": fps,
            "random_robot_pose": random_robot_pose,
            "robot_pose_seed": robot_pose_seed,
            "robot_x": robot_x,
            "robot_y": robot_y,
            "base_url": base_url,
            "started_at": started_at,
            "finished_at": None,
            "duration_sec": None,
            "error": None,
            "frames_count": None,
            "final_tick": None,
            "export_format_version": None,
            "export_bytes": None,
            "llm": {
                "status": "pending"
            },
            "artifacts": {}
        }
        _write_meta_atomic(meta_path, meta)
        logger.info("agent=%s run_dir=%s", agent.name, run_dir)

        timeseries_path = run_dir / "timeseries.json"
        plot_path = run_dir / "exploration.png"
        metrics_path = run_dir / "metrics.json"
        chat_log_path = run_dir / "chat.jsonl"
        llm_generations_path = run_dir / "llm_generations.json"
        series: list[dict[str, Any]] = []
        snapshot: dict[str, Any] = {}
        ts_summary: dict[str, Any] = {}
        frames_count = 0

        try:
            try:
                _run_agent(agent, client, run_dir)
            except Exception:
                logger.exception(
                    "agent failed; export/video artifacts are not saved"
                )
                raise

            client.pause_session()
            export_payload = client.export_session()
            export_src = _to_local_path(
                export_payload["export_path"],
                project_root,
            )
            if not export_src.is_file():
                raise FileNotFoundError(f"export not found: {export_src}")

            series, snapshot, ts_summary, frames_count = (
                _compute_metrics_from_export(
                    export_src,
                    robot_id=robot_id,
                )
            )

            with open(timeseries_path, "w", encoding="utf-8") as handle:
                json.dump(series, handle, indent=2)

            plot_exploration_timeseries(
                series,
                title=f"{agent.name} | {session_id}",
                out_path=plot_path,
            )

            finished_at = datetime.now().isoformat(timespec="seconds")
            duration_sec = round(datetime.now().timestamp() - started_mono, 3)
            meta.update(
                {
                    "status": "completed",
                    "finished_at": finished_at,
                    "duration_sec": duration_sec,
                    "frames_count": frames_count,
                    "final_tick": snapshot.get("final_tick"),
                    "export_format_version": EXPORT_FORMAT_VERSION,
                    "export_bytes": None,
                    "artifacts": {
                        "timeseries_json": _relative_artifact(
                            run_dir,
                            timeseries_path,
                        ),
                        "plot_png": _relative_artifact(run_dir, plot_path),
                        "chat_log": _relative_artifact(
                            run_dir,
                            chat_log_path,
                        ),
                        "metrics_json": _relative_artifact(
                            run_dir,
                            metrics_path,
                        ),
                    },
                }
            )

            metrics = _build_metrics_payload(
                agent_name=agent.name,
                session_id=session_id,
                run_dir=run_dir,
                snapshot=snapshot,
                ts_summary=ts_summary,
                frames_count=frames_count,
            )
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=4, ensure_ascii=False)

            logger.info("run_dir → %s", run_dir)
            logger.info("metrics → %s", metrics_path)
            logger.info("plot → %s", plot_path)

            return ExperimentResult(
                session_id=session_id,
                run_dir=str(run_dir),
                metrics=metrics,
                metrics_path=str(metrics_path),
                timeseries=series,
                videos=None,
                timeseries_path=str(timeseries_path),
                plot_path=str(plot_path),
            )
        except Exception as exc:
            meta.update(
                {
                    "status": "failed",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "duration_sec": round(
                        datetime.now().timestamp() - started_mono,
                        3,
                    ),
                    "error": str(exc),
                    "artifacts": {
                        "timeseries_json": _relative_artifact(
                            run_dir,
                            timeseries_path,
                        )
                        if timeseries_path.is_file()
                        else None,
                        "plot_png": _relative_artifact(run_dir, plot_path)
                        if plot_path.is_file()
                        else None,
                        "chat_log": _relative_artifact(
                            run_dir,
                            chat_log_path,
                        )
                        if chat_log_path.is_file()
                        else None,
                        "metrics_json": _relative_artifact(
                            run_dir,
                            metrics_path,
                        )
                        if metrics_path.is_file()
                        else None,
                    },
                }
            )
            logger.error("experiment failed: %s", exc)
            logger.debug(traceback.format_exc())
            raise
        finally:
            try:
                client.delete_session()
            except Exception:
                logger.debug("session delete skipped or failed", exc_info=True)

            if chat_log_path.is_file():
                try:
                    llm_report = (
                        collect_openrouter_metadata(
                            chat_log_path
                        )
                    )

                    _write_meta_atomic(
                        llm_generations_path,
                        llm_report,
                    )

                    meta["llm"] = _build_llm_meta(
                        llm_report,
                        meta.get("config_snapshot"),
                    )

                    artifacts = meta.setdefault(
                        "artifacts",
                        {},
                    )
                    artifacts["llm_generations"] = (
                        _relative_artifact(
                            run_dir,
                            llm_generations_path,
                        )
                    )

                except Exception as metadata_error:
                    logger.exception(
                        "failed to collect LLM metadata"
                    )

                    meta["llm"] = {
                        "status": "collection_failed",
                        "error": {
                            "type": type(
                                metadata_error
                            ).__name__,
                            "message": str(
                                metadata_error
                            ),
                        },
                    }
            else:
                meta["llm"] = {
                    "status": "unavailable",
                    "reason": (
                        "chat log was not created"
                    ),
                }

            _write_meta_atomic(meta_path, meta)


def run_experiment(
    agent: Agent,
    *,
    base_url: str = "http://localhost:8000",
    scenario: str = "exploring",
    robot_id: str = "robot_1",
    token: str | None = None,
    fps: float = 2.0,
    random_robot_pose: bool = False,
    robot_pose_seed: int | None = None,
    runs_root: str = "runs",
    project_root: str | Path | None = None,
    video_wait_timeout_sec: float = 3600.0,
    render_workers: int | None = None,
    run_dir: str | Path | None = None,
    artifact_mode: ArtifactMode = ArtifactMode.FULL,
) -> ExperimentResult:
    """
    Прогон: session → agent → metrics из export (без сохранения export/video).

    В режиме ``metrics_only`` сохраняется только ``metrics.json`` в ``run_dir``.
    """
    root = _project_root(project_root)
    runs_base = Path(runs_root)
    if not runs_base.is_absolute():
        runs_base = root / runs_base

    resolved_run_dir = (
        Path(run_dir).resolve() if run_dir is not None else None
    )

    with run_log_context(scenario=scenario):
        if artifact_mode == ArtifactMode.METRICS_ONLY:
            if resolved_run_dir is None:
                raise ValueError(
                    "run_dir is required for metrics_only artifact mode"
                )
            return _run_experiment_metrics_only(
                agent,
                base_url=base_url,
                scenario=scenario,
                robot_id=robot_id,
                token=token,
                fps=fps,
                random_robot_pose=random_robot_pose,
                robot_pose_seed=robot_pose_seed,
                run_dir=resolved_run_dir,
                project_root=root,
            )

        return _run_experiment_full(
            agent,
            base_url=base_url,
            scenario=scenario,
            robot_id=robot_id,
            token=token,
            fps=fps,
            random_robot_pose=random_robot_pose,
            robot_pose_seed=robot_pose_seed,
            runs_root=runs_base,
            run_dir=resolved_run_dir,
            project_root=root,
            video_wait_timeout_sec=video_wait_timeout_sec,
            render_workers=render_workers,
        )
