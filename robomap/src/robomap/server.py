import os
import uuid
import datetime
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field, model_validator

from robomap.session_manager import SimulationSession
from robomap.visualization import create_map_figure
from robomap.headless_utils import render_history_to_video
from robomap.fast_headless_renderer import render_history_to_fast_video
from robomap.session_export import (
    bundle_map_boundaries,
    get_export_basename,
    load_session_export,
    normalize_export_filename,
    save_session_export,
)
from robomap.state_loader import load_and_merge_configs

logging.basicConfig(level=logging.INFO,format="%(message)s")
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SESSIONS: Dict[str, SimulationSession] = {}
RENDER_JOBS: Dict[str, Dict[str, Any]] = {}
RENDER_JOB_TTL_SEC = 3600


def _cleanup_render_jobs(now: float | None = None) -> None:
    """Удаляет terminal render jobs старше TTL."""
    current = now if now is not None else time.time()
    expired = []
    for job_id, job in RENDER_JOBS.items():
        if job.get("status") not in {"done", "failed"}:
            continue
        finished_at = job.get("finished_at")
        if finished_at is None:
            continue
        if current - float(finished_at) > RENDER_JOB_TTL_SEC:
            expired.append(job_id)
    for job_id in expired:
        RENDER_JOBS.pop(job_id, None)


def _render_job_to_api(job: Dict[str, Any]) -> Dict[str, Any]:
    """Преобразует job с Unix timestamps в API-ответ с ISO-строками."""
    payload = dict(job)
    for key in ("created_at", "finished_at"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            payload[key] = datetime.datetime.fromtimestamp(
                value,
                tz=datetime.timezone.utc,
            ).isoformat(timespec="seconds")
    return payload


def _register_render_job(output_path: str, frames_to_render: int) -> str:
    """Регистрирует новую задачу рендера и чистит устаревшие terminal jobs."""
    _cleanup_render_jobs()
    job_id = f"render_{uuid.uuid4().hex[:10]}"
    RENDER_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "output_path": output_path,
        "frames_to_render": frames_to_render,
        "error": None,
        "created_at": time.time(),
        "finished_at": None,
    }
    return job_id

tags_metadata = [
    {"name": "Sessions", "description": "Создание, удаление и управление состояниями миров."},
    {"name": "Simulation", "description": "Управление физикой (Play/Pause, Тики) и передача команд сущностям."},
    {"name": "Players", "description": "Регистрация агентов, управление правами доступа."},
    {"name": "Media & UI", "description": "Генерация MP4 видео, получение графиков Plotly и Live Viewer."},
    {"name": "System", "description": "Системные запросы (список доступных плагинов-сценариев)."}
]

app = FastAPI(
    title="RoboMap API", 
    version="1.0.0",
    openapi_tags=tags_metadata
)

class CreateSessionRequest(BaseModel):
    scenario: str = "bear_attack"
    auto_robot: bool = False
    random_robot_pose: bool = False
    robot_pose_seed: Optional[int] = None
    robot_id: str = "robot_1"
    weather_patterns: Optional[List[int]] = None
    weather_seed: Optional[int] = None
    
class RegisterPlayerRequest(BaseModel):
    token: str
    entity_ids: List[str]

class CommandRequest(BaseModel):
    token: str
    entity_id: str
    action: str
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}

class CommandResponse(BaseModel):
    success: bool
    message: str

class TickResponse(BaseModel):
    tick: int
    stage: str
    events: List[str]

class RenderVideoRequest(BaseModel):
    filename: str = "output.mp4"
    layout_type: str = "pip_real_main" 
    fps: int = 10
    width: int = 1920 
    height: int = 1080
    target_robot_id: str = "robot_1"

class FastRenderRequest(BaseModel):
    """Запрос на быстрый рендер из живой сессии или файла экспорта."""
    session_id: Optional[str] = None
    export_file: Optional[str] = None
    filename: str = "output.mp4"
    layout_type: str = "pip_real_main"
    fps: int = 10
    width: int = 1920
    height: int = 1080
    target_robot_id: str = "robot_1"
    workers: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_source(self) -> "FastRenderRequest":
        """Проверяет, что указан ровно один источник данных."""
        has_session = bool(self.session_id)
        has_export = bool(self.export_file)
        if has_session == has_export:
            raise ValueError(
                "Укажите ровно один источник: session_id или export_file."
            )
        return self

class ExportSessionRequest(BaseModel):
    """Пустое тело запроса для экспорта сессии."""
    pass

class AgentStatusResponse(BaseModel):
    success: bool
    token: str
    data: Dict[str, Any]

@app.post("/api/v1/sessions", tags=["Sessions"], response_model=Dict[str, str], summary="Создать новую сессию")
def create_session(req: CreateSessionRequest):
    """Инициализирует новый изолированный мир и загружает указанный сценарий."""
    session_id = f"sess_{uuid.uuid4().hex[:6]}"
    try:
        session = SimulationSession(
            session_id=session_id,
            auto_robot=req.auto_robot,
            scenario_name=req.scenario,
            random_robot_pose=req.random_robot_pose,
            robot_pose_seed=req.robot_pose_seed,
            robot_id=req.robot_id,
            weather_patterns=req.weather_patterns,
            weather_seed=req.weather_seed,
        )
        SESSIONS[session_id] = session
        logger.info("API | Создана сессия: %s (Сценарий: %s)", session_id, req.scenario)
        return {"session_id": session_id, "message": "Session created successfully."}
    except Exception as e:
        logger.error("API | Ошибка создания сессии: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/sessions", tags=["Sessions"], summary="Список активных сессий")
def list_sessions():
    """Возвращает информацию обо всех запущенных мирах на сервере."""
    sessions_info = [
        {
            "session_id": sid,
            "scenario": session.scenario_name,
            "tick": session.world.tick_count,
            "is_running": session.is_running,
            "players_count": len(session.permissions)
        }
        for sid, session in SESSIONS.items()
    ]
    return {"total_sessions": len(SESSIONS), "sessions": sessions_info}

@app.delete("/api/v1/sessions/{session_id}", tags=["Sessions"], summary="Удалить сессию")
def delete_session(session_id: str):
    """Останавливает фоновые процессы и полностью удаляет мир из оперативной памяти."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = SESSIONS[session_id]
    if session.is_running:
        session.stop_realtime()
        
    del SESSIONS[session_id]
    logger.info("API | Сессия %s удалена.", session_id)
    return {"message": f"Session '{session_id}' terminated and deleted."}

@app.post("/api/v1/sessions/{session_id}/reset", tags=["Sessions"], summary="Сбросить мир")
def reset_session(session_id: str):
    """Очищает историю и возвращает все объекты на стартовые позиции."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    SESSIONS[session_id].reset()
    return {"message": "Session has been reset to tick 0."}

@app.get("/api/v1/sessions/{session_id}/state", tags=["Sessions"], summary="Получить текущее состояние")
def get_state(session_id: str):
    """Возвращает полный JSON-слепок мира (карта, память, робот) для текущего тика."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = SESSIONS[session_id]
    return session.recorder.get_latest_frame() or {}

@app.post("/api/v1/sessions/{session_id}/play", tags=["Simulation"], summary="▶ Запустить Real-time")
def play_session(session_id: str, fps: float = 2.0):
    """Запускает фоновый таймер. Движок начинает симуляцию в реальном времени."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    SESSIONS[session_id].start_realtime(fps)
    return {"message": f"Real-time simulation started at {fps} FPS."}

@app.post("/api/v1/sessions/{session_id}/pause", tags=["Simulation"], summary="⏸ Остановить Real-time")
def pause_session(session_id: str):
    """Ставит мир на паузу (переход в ручной пошаговый режим)."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    SESSIONS[session_id].stop_realtime()
    return {"message": "Simulation paused."}

@app.post("/api/v1/sessions/{session_id}/tick", tags=["Simulation"], response_model=TickResponse, summary="Сделать 1 шаг (Turn-based)")
def advance_tick(session_id: str):
    """Сдвигает мир на 1 логический такт вперед (используется для пошагового управления)."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return SESSIONS[session_id].tick()

@app.post("/api/v1/sessions/{session_id}/run_auto", tags=["Simulation"], summary="Запустить перемотку (Movie Mode)")
def run_auto(session_id: str, background_tasks: BackgroundTasks, max_ticks: int = 200):
    """Запускает мир в фоне на X тиков без задержек. Удобно для быстрой генерации датасетов."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    def _run_auto_task(sid: str, ticks: int):
        if sid in SESSIONS:
            SESSIONS[sid].run_auto(ticks)
            
    background_tasks.add_task(_run_auto_task, session_id, max_ticks)
    return {"message": f"Auto-run started in background for up to {max_ticks} ticks."}

@app.post("/api/v1/sessions/{session_id}/command", tags=["Simulation"], response_model=CommandResponse, summary="Отправить команду сущности")
def send_command(session_id: str, req: CommandRequest):
    """Добавляет действие в очередь объекта. Возвращает текстовый статус операции."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = SESSIONS[session_id]
    
    feedback_message = "Command queued successfully."

    # Один захват реентрантного локa вместо двух: иначе запрос дважды
    # встаёт в очередь за фоновым тиком и латентность удваивается.
    with session.lock:
        success = session.execute_command(req.token, req.entity_id, req.action, *req.args, **req.kwargs)
        if success:
            entity = session.world.get_object_by_id(req.entity_id)
            if entity and "last_action_result" in entity.state:
                feedback_message = entity.state.pop("last_action_result")

    if not success:
        raise HTTPException(status_code=403, detail="Command rejected. Check permissions or entity_id.")

    return CommandResponse(success=True, message=feedback_message)

@app.get("/api/v1/sessions/{session_id}/feedback", tags=["Simulation"], summary="Получить результаты команд")
def get_feedback(session_id: str, token: str):
    """
    1. Извлекает и удаляет (прочитывает) все накопившиеся логи ошибок и успехов 
    для сущностей, которыми управляет данный токен.
    2. Текущую очередь выполняемых действий (action_queue) для этих сущностей.
    """
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = SESSIONS[session_id]
    
    allowed_entities = session.permissions.get(token, [])
        
    report = {}
    
    with session.lock:
        for entity_id in allowed_entities:
            entity = session.world.get_object_by_id(entity_id)
            if entity:
                feedback_list = []
                if "feedback" in entity.state and entity.state["feedback"]:
                    feedback_list = entity.state["feedback"].copy()
                    entity.state["feedback"].clear()
                
                ent_data = entity.to_dict()
                
                report[entity_id] = {
                    "feedback": feedback_list,
                    "action_queue": ent_data.get("action_queue", [])
                }
                
    return AgentStatusResponse(
        success=True,
        token=token,
        data=report
    )

@app.get("/api/v1/sessions/{session_id}/logs", tags=["Simulation"], summary="Получить отфильтрованные логи робота")
def get_robot_logs(
    session_id: str,
    token: str,
    robot_id: str = "robot_1",
    log_type: str = "all",
    format_style: str = "json",
    tick_start: int = 0,
    include_env_progress: bool = True
):
    """
    Возвращает логи робота с поддержкой инкрементального скачивания и кумулятивной фильтрации.

    - **token**: Токен агента.
    - **robot_id**: Идентификатор робота.
    - **log_type**: Категории логов ('all' - все, 'self' - только свои действия, 'others' - наблюдения и слух, 'env' - события погоды и окружения).
    - **format_style**: Формат вывода ('json' - json, 'text' - только сообщения, 'brief' - краткий лог [tick] [cat] [msg]).
    - **tick_start**: Начать вывод логов с указанного тика симуляции (дефолт: 0).
    - **include_env_progress**: Включать ли промежуточные логи (in progress) для событий погоды/окружения (по умолчанию: true). При false выводятся только старт и завершение.
    """
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = SESSIONS[session_id]
    
    if token != "admin":
        allowed_entities = session.permissions.get(token, [])
        if robot_id not in allowed_entities:
            raise HTTPException(status_code=403, detail="Access denied to this robot's logs.")

    with session.lock:
        robot = session.world.robots.get(robot_id)
        if not robot:
            raise HTTPException(status_code=404, detail=f"Robot '{robot_id}' not found in this session.")
            
        formatted_logs = robot.get_formatted_logs(
            log_type=log_type,
            format_style=format_style,
            tick_start=tick_start,
            include_env_progress=include_env_progress
        )
        
        return {
            "success": True,
            "robot_id": robot_id,
            "log_type": log_type,
            "format_style": format_style,
            "tick_start": tick_start,
            "include_env_progress": include_env_progress,
            "tick": session.world.tick_count,
            "total_logs": len(formatted_logs),
            "logs": formatted_logs
        }

@app.get("/api/v1/sessions/{session_id}/local-context", tags=["Simulation"], summary="Получить контекст локального мира агента")
def get_llm_context(
    session_id: str,
    token: str,
    robot_id: str = "robot_1",
    format_style: str = "json",
    sections: str = "all",
    include_geometry: bool = False
):
    """
    Генерирует контекст памяти для агента.
    
    - **token**: Токен агента.
    - **robot_id**: ID робота (по умолчанию robot_1).
    - **format_style**: Формат вывода ('json', 'text-md', 'text-yaml', 'text-narrative').
    - **sections**: Список секций через запятую ('all', 'current_state', 'visible_now', 'inventory'). По умолчанию 'all'.
    """
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = SESSIONS[session_id]
    
    if token != "admin":
        allowed_entities = session.permissions.get(token, [])
        if robot_id not in allowed_entities:
            raise HTTPException(status_code=403, detail="Access denied to this robot's context.")

    with session.lock:
        robot = session.world.robots.get(robot_id)
        if not robot:
            raise HTTPException(status_code=404, detail=f"Robot '{robot_id}' not found in this session.")
            
        sections_list = [s.strip() for s in sections.split(",")] if sections else ["all"]
        
        context = robot.get_llm_context(
            current_tick=session.world.tick_count,
            sections=sections_list,
            format_style=format_style,
            include_geometry=include_geometry
        )
        
    if format_style.startswith("text"):
        return PlainTextResponse(content=context)
    
    return context

@app.post("/api/v1/sessions/{session_id}/players", tags=["Players"], summary="Зарегистрировать агента (Токен)")
def register_player(session_id: str, req: RegisterPlayerRequest):
    """Выдает права токену на управление указанными ID сущностей."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    SESSIONS[session_id].register_player(req.token, req.entity_ids)
    return {"message": f"Player '{req.token}' registered for {req.entity_ids}"}

@app.get("/api/v1/sessions/{session_id}/players", tags=["Players"], summary="Список агентов в сессии")
def list_players(session_id: str):
    """Возвращает список токенов и сущностей, которыми они имеют право управлять."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return {"session_id": session_id, "permissions": SESSIONS[session_id].permissions}

@app.delete("/api/v1/sessions/{session_id}/players/{token}", tags=["Players"], summary="Удалить агента")
def remove_player(session_id: str, token: str):
    """Отзывает права доступа у токена, запрещая ему отдавать команды."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    success = SESSIONS[session_id].remove_player(token)
    if not success:
        raise HTTPException(status_code=404, detail=f"Token '{token}' not found in this session.")
        
    return {"message": f"Player '{token}' has been removed from session '{session_id}'."}

@app.get("/api/v1/scenarios", tags=["System"], summary="Список доступных плагинов-сценариев")
def list_scenarios():
    """
    Сканирует папку сценариев (путь берется из config.json).
    Возвращает список миссий и информацию о том, переопределены ли конфиг, 
    карта и объекты в локальном файле конфигурации сценария.
    """
    config_path = os.path.join(project_path, "data", "config.json")
    scenarios_dir_rel = "data/scenarios"
    
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            scenarios_dir_rel = cfg.get("paths", {}).get("scenarios_dir", "data/scenarios")
            
    scenarios_dir = os.path.join(project_path, scenarios_dir_rel)
    
    available = [{
        "name": "empty", 
        "has_custom_config": False, 
        "has_custom_state": False, 
        "has_custom_objects": False
    }] 
    
    if os.path.exists(scenarios_dir):
        for folder in os.listdir(scenarios_dir):
            folder_path = os.path.join(scenarios_dir, folder)
            
            if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, "script.py")):
                
                local_config_path = os.path.join(folder_path, "config.json")
                has_config = os.path.exists(local_config_path)
                
                has_state = False
                has_objects = False
                
                if has_config:
                    try:
                        with open(local_config_path, "r", encoding="utf-8") as f:
                            local_cfg = json.load(f)
                        
                        local_paths = local_cfg.get("paths", {})
                        
                        if "default_state_path" in local_paths:
                            has_state = True
                        if "objects_meta_path" in local_paths:
                            has_objects = True
                            
                    except Exception as e:
                        logger.error("API | Ошибка чтения локального конфига сценария '%s': %s", folder, str(e))
                
                available.append({
                    "name": folder,
                    "has_custom_config": has_config,
                    "has_custom_state": has_state,
                    "has_custom_objects": has_objects
                })
                
    return {"total": len(available), "scenarios": available}

def _get_renders_dir() -> str:
    """Возвращает абсолютный путь к директории рендеров из глобального конфига."""
    config = load_and_merge_configs("data/config.json")
    renders_dir_rel = config.get("paths", {}).get("renders_dir", "renders")
    return os.path.join(project_path, renders_dir_rel)


def _prepare_fast_render_output_path(
    renders_dir: str,
    name_prefix: str,
    filename: str,
) -> str:
    """Формирует путь к выходному MP4 для fast-render."""
    os.makedirs(renders_dir, exist_ok=True)
    time_str = datetime.datetime.now().strftime("%H%M%S")
    safe_filename = os.path.basename(filename)
    if safe_filename.endswith(".mp4"):
        safe_filename = safe_filename[:-4]
    final_filename = f"{name_prefix}_{time_str}_fast_{safe_filename}.mp4"
    return os.path.join(renders_dir, final_filename)


def _resolve_fast_render_source(
    req: FastRenderRequest,
) -> Tuple[list, tuple, dict, str]:
    """Возвращает данные для fast-render из живой сессии или файла экспорта."""
    if req.session_id:
        if req.session_id not in SESSIONS:
            raise HTTPException(status_code=404, detail="Session not found")

        session = SESSIONS[req.session_id]
        history_snapshot = list(session.recorder.get_history())
        if not history_snapshot:
            raise HTTPException(
                status_code=400,
                detail="History is empty. Nothing to render.",
            )

        return (
            history_snapshot,
            session.world.map_boundaries,
            session.config.get("visualization", {}),
            req.session_id,
        )

    try:
        bundle = load_session_export(req.export_file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    history_snapshot = bundle.get("frames", [])
    if not history_snapshot:
        raise HTTPException(
            status_code=400,
            detail="Export file has empty history.",
        )

    name_prefix = bundle.get("session_id") or get_export_basename(req.export_file)
    return (
        history_snapshot,
        bundle_map_boundaries(bundle),
        bundle.get("vis_config") or {},
        name_prefix,
    )


def _enqueue_fast_render_task(
    req: FastRenderRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Запускает фоновый fast-render и возвращает ответ API с job_id."""
    history_snapshot, bounds, vis_config, name_prefix = _resolve_fast_render_source(req)
    renders_dir = _get_renders_dir()
    output_path = _prepare_fast_render_output_path(
        renders_dir,
        name_prefix,
        req.filename,
    )

    job_id = _register_render_job(output_path, len(history_snapshot))

    background_tasks.add_task(
        _fast_render_video_task,
        job_id,
        history_snapshot,
        output_path,
        req.layout_type,
        bounds,
        req.fps,
        req.width,
        req.height,
        vis_config,
        req.target_robot_id,
        req.workers,
    )

    return {
        "message": "Быстрый рендер видео запущен в фоновом режиме.",
        "job_id": job_id,
        "status": "queued",
        "frames_to_render": len(history_snapshot),
        "expected_output_path": output_path,
    }


def _render_video_task(history_snapshot: list, output_path: str, layout_type: str, bounds: tuple, fps: int, w: int, h: int, vis_config: dict):
    """Фоновый воркер для генерации видеофайла."""
    try:
        render_history_to_video(
            history_frames=history_snapshot,
            output_path=output_path,
            vis_config=vis_config,
            layout_type=layout_type,
            map_boundaries=bounds,
            fps=fps,
            width=w,
            height=h,
            target_robot_id="robot_1"
        )
    except Exception as e:
        logger.error("API | Ошибка фонового рендера видео: %s", str(e))

def _fast_render_video_task(
    job_id: str,
    history_snapshot: list,
    output_path: str,
    layout_type: str,
    bounds: tuple,
    fps: int,
    w: int,
    h: int,
    vis_config: dict,
    target_robot_id: str,
    workers: Optional[int] = None,
):
    """Фоновый воркер для быстрой генерации видеофайла без Plotly."""
    job = RENDER_JOBS.get(job_id)
    if job is not None:
        job["status"] = "running"

    try:
        render_history_to_fast_video(
            history_frames=history_snapshot,
            output_path=output_path,
            vis_config=vis_config,
            layout_type=layout_type,
            map_boundaries=bounds,
            fps=fps,
            width=w,
            height=h,
            target_robot_id=target_robot_id,
            workers=workers,
        )
        if job is not None:
            job["status"] = "done"
            job["finished_at"] = time.time()
    except Exception as e:
        logger.error("API | Ошибка фонового быстрого рендера видео: %s", str(e))
        if job is not None:
            job["status"] = "failed"
            job["error"] = str(e)
            job["finished_at"] = time.time()


@app.get("/api/v1/renders/{job_id}", tags=["Media & UI"], summary="Статус задачи рендера")
def get_render_job_status(job_id: str):
    """Возвращает статус фоновой задачи fast-render."""
    _cleanup_render_jobs()
    job = RENDER_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Render job not found")
    return _render_job_to_api(job)


@app.post("/api/v1/sessions/{session_id}/render", tags=["Media & UI"], summary="Сгенерировать видео (MP4)")
def trigger_video_render(session_id: str, req: RenderVideoRequest, background_tasks: BackgroundTasks):
    """Формирует задачу на рендер MP4 из истории сессии. Выполняется в фоновом режиме."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = SESSIONS[session_id]
    history_snapshot = list(session.recorder.get_history())
    
    if not history_snapshot:
        raise HTTPException(status_code=400, detail="History is empty. Nothing to render.")

    renders_dir_rel = session.config.get("paths", {}).get("renders_dir", "renders")
    renders_dir = os.path.join(project_path, renders_dir_rel)
    os.makedirs(renders_dir, exist_ok=True)
    
    time_str = datetime.datetime.now().strftime("%H%M%S")
    safe_filename = os.path.basename(req.filename)
    if safe_filename.endswith(".mp4"):
        safe_filename = safe_filename[:-4]
        
    final_filename = f"{session_id}_{time_str}_{safe_filename}.mp4"
    output_path = os.path.join(renders_dir, final_filename)

    bounds = session.world.map_boundaries
    vis_config = session.config.get("visualization", {})

    background_tasks.add_task(
        _render_video_task, history_snapshot, output_path, 
        req.layout_type, bounds, req.fps, req.width, req.height, vis_config
    )

    return {
        "message": "Рендер видео запущен в фоновом режиме.",
        "frames_to_render": len(history_snapshot),
        "expected_output_path": output_path
    }

@app.post("/api/v1/sessions/{session_id}/export", tags=["Sessions"], summary="Экспортировать сессию в msgpack.gz")
def export_session(session_id: str, req: ExportSessionRequest = ExportSessionRequest()):
    """Сохраняет историю сессии в exported_sessions/{session_id}.msgpack.gz."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    session = SESSIONS[session_id]
    frames_count = session.recorder.frame_count
    if frames_count == 0:
        raise HTTPException(status_code=400, detail="History is empty. Nothing to export.")

    try:
        export_path = save_session_export(session)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    export_filename = normalize_export_filename(session_id)
    file_size_bytes = os.path.getsize(export_path)

    return {
        "message": "Сессия успешно экспортирована.",
        "export_path": export_path,
        "filename": export_filename,
        "frames_count": frames_count,
        "file_size_bytes": file_size_bytes,
    }


@app.post("/api/v1/fast-render", tags=["Media & UI"], summary="Быстрый рендер из сессии или экспорта")
def trigger_unified_fast_video_render(
    req: FastRenderRequest,
    background_tasks: BackgroundTasks,
):
    """Формирует задачу на быстрый рендер MP4 из живой сессии или файла экспорта."""
    return _enqueue_fast_render_task(req, background_tasks)


@app.post("/api/v1/sessions/{session_id}/fast-render", tags=["Media & UI"], summary="Сгенерировать видео быстро (MP4)")
def trigger_fast_video_render(session_id: str, req: RenderVideoRequest, background_tasks: BackgroundTasks):
    """Alias для быстрого рендера из живой сессии (обратная совместимость)."""
    fast_req = FastRenderRequest(
        session_id=session_id,
        filename=req.filename,
        layout_type=req.layout_type,
        fps=req.fps,
        width=req.width,
        height=req.height,
        target_robot_id=req.target_robot_id,
    )
    return _enqueue_fast_render_task(fast_req, background_tasks)


def _build_plotly_figure(map_data: list, robots_states: dict, title: str, boundaries: tuple, is_minimap: bool, vis_config: dict, clusters_data: Optional[list] = None):
    """Хелпер для создания объекта go.Figure для UI Viewer'а."""
    x_range, y_range = boundaries
    all_types = sorted(list(set(obj.get('object', obj.get('name', 'unknown')) for obj in map_data)))
    
    def _asset_getter(icon_path: str):
        if not icon_path: return None
        base_dir = vis_config.get("asset_base_path", "data/assets")
        local_path = os.path.join(project_path, base_dir, icon_path)
        try:
            import base64
            with open(local_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode()
            return f"data:image/svg+xml;base64,{encoded}"
        except Exception:
            return None

    fig = create_map_figure(
        map_data=map_data,
        robots_states=robots_states,
        title=title,
        asset_url_getter_func=_asset_getter,
        current_visible_types=None, 
        all_legend_types_sorted=all_types, 
        overall_map_x_range=x_range,
        overall_map_y_range=y_range,
        vis_config=vis_config,
        show_legend=(not is_minimap),
        clusters_data=clusters_data,
        is_minimap=is_minimap
    )
    
    if not is_minimap:
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    else:
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        
    return fig

def _build_live_data(session: Any, frame: Dict[str, Any], robot_id: Optional[str], target_robot_state: Dict[str, Any]) -> Dict[str, Any]:
    """Формирует данные для живой телеметрии робота, глобального состояния погоды и активных модификаторов."""
    env_manager = getattr(session, "env_manager", None)
    scenario = getattr(session, "scenario", None)
    if not scenario and hasattr(session, "world"):
        scenario = getattr(session.world, "scenario_controller", None)

    weather = None
    if scenario:
        if hasattr(scenario, "weather") and scenario.weather:
            weather = scenario.weather
        elif hasattr(scenario, "weather_scenario") and hasattr(scenario.weather_scenario, "weather"):
            weather = scenario.weather_scenario.weather

        if not env_manager:
            if hasattr(scenario, "env") and scenario.env:
                env_manager = scenario.env
            elif hasattr(scenario, "weather_scenario") and hasattr(scenario.weather_scenario, "env"):
                env_manager = scenario.weather_scenario.env

    if weather and hasattr(weather, "state_machine"):
        st = getattr(weather.state_machine.current_state, "name", str(weather.state_machine.current_state))
        patts = []
        for p in getattr(weather, "patterns", []):
            p_name = getattr(p, "name", None) or f"Pattern_{getattr(p, 'pattern_id', p)}"
            patts.append(p_name)
        active_z = len(getattr(weather.zlm, "active_zones", {})) if hasattr(weather, "zlm") else 0
        cur_tick = getattr(weather.state_machine, "total_ticks", session.world.tick_count if hasattr(session, "world") else 0)
        global_weather = {
            "enabled": True,
            "state": st,
            "patterns": patts,
            "active_zones": active_z,
            "tick": cur_tick
        }
    else:
        global_weather = {
            "enabled": False,
            "state": "NORMAL",
            "patterns": [],
            "active_zones": 0,
            "tick": session.world.tick_count if hasattr(session, "world") else 0
        }

    robot = None
    if hasattr(session, "world") and hasattr(session.world, "robots") and robot_id:
        robot = session.world.robots.get(robot_id)

    rx = target_robot_state.get("x")
    ry = target_robot_state.get("y")
    if robot and hasattr(robot, "x") and robot.x is not None:
        rx = round(robot.x, 2)
    elif rx is not None:
        rx = round(rx, 2)

    if robot and hasattr(robot, "y") and robot.y is not None:
        ry = round(robot.y, 2)
    elif ry is not None:
        ry = round(ry, 2)

    angle = target_robot_state.get("angle", 0.0)
    if robot and hasattr(robot, "angle") and robot.angle is not None:
        angle = round(robot.angle, 1)

    speed = target_robot_state.get("speed", 1.0)
    if robot and hasattr(robot, "speed") and robot.speed is not None:
        speed = round(robot.speed, 2)

    vision_radius = target_robot_state.get("vision_radius", target_robot_state.get("radius", 5.0))
    if robot and hasattr(robot, "vision_radius") and robot.vision_radius is not None:
        vision_radius = round(robot.vision_radius, 2)

    robot_telemetry = {
        "id": robot_id,
        "x": rx,
        "y": ry,
        "angle": angle,
        "speed": speed,
        "vision_radius": vision_radius
    }

    active_modifiers = []
    if env_manager and robot_id and hasattr(env_manager, "get_active_modifiers"):
        active_modifiers = env_manager.get_active_modifiers(robot_id)

    return {
        "global_weather": global_weather,
        "robot": robot_telemetry,
        "active_modifiers": active_modifiers
    }

@app.get("/api/v1/sessions/{session_id}/plots", tags=["Media & UI"], summary="Получить Plotly графики (JSON)")
def get_plots(session_id: str, robot_id: Optional[str] = None):
    """Генерирует графики реальности и памяти и возвращает их в JSON-формате для фронтенда."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = SESSIONS[session_id]
    frame = session.recorder.get_latest_frame()
    if frame is None:
        return {"real": None, "mem": None, "active_robots": [], "live_data": None}
    
    vis_config = session.config.get("visualization", {})
    
    bounds = session.world.map_boundaries
    
    robots_state = frame.get('robots_state', {})
    robots_memory = frame.get('robots_memory', {})
    
    active_robots = list(robots_state.keys())
    
    if not robot_id and active_robots:
        robot_id = active_robots[0]
        
    target_robot_state = robots_state.get(robot_id, {})
    target_robot_memory = robots_memory.get(robot_id, [])
    clusters_data = target_robot_state.get("clusters", None)

    real_map_data = list(frame['real_map'])
    virtual_weather = []
    source_mem = target_robot_memory or (list(robots_memory.values())[0] if robots_memory else [])
    for m in source_mem:
        if m.get("is_virtual") and (m.get("render_only") or str(m.get("notes", "")).startswith("[ENV]")):
            virtual_weather.append(m)
    if virtual_weather:
        real_map_data.extend(virtual_weather)

    real_fig = _build_plotly_figure(real_map_data, robots_state, "Real World Map", bounds, False, vis_config)
    
    rx = target_robot_state.get('x', 0)
    ry = target_robot_state.get('y', 0)
    mem_bounds = ([rx - 20, rx + 20], [ry - 20, ry + 20]) if (rx and ry) else bounds
    
    title_mem = f"Robot Memory: {robot_id}" if robot_id else "Robot Memory"
    mem_fig = _build_plotly_figure(target_robot_memory, {robot_id: target_robot_state} if target_robot_state else {}, title_mem, mem_bounds, False, vis_config, clusters_data=clusters_data)
    
    live_data = _build_live_data(session, frame, robot_id, target_robot_state)

    return {
        "real": real_fig.to_json(),
        "mem": mem_fig.to_json(),
        "active_robots": active_robots,
        "live_data": live_data
    }

@app.get("/api/v1/sessions/{session_id}/viewer", tags=["Media & UI"], response_class=HTMLResponse, summary="Открыть Live-Viewer")
def open_viewer(session_id: str):
    """Возвращает HTML-страницу с монитором, который обновляет карту в реальном времени."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
        
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Live Viewer - Session: {session_id}</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 8px 12px; background: #1a1a24; color: #e0e0e0; }}
            .header-panel {{ display: flex; justify-content: space-between; align-items: center; gap: 15px; margin-bottom: 6px; padding: 6px 12px; background: #252533; border-radius: 6px; }}
            .header-title {{ font-size: 16px; font-weight: bold; color: #ffffff; }}
            .dashboard-bar {{ display: grid; grid-template-columns: 1.1fr 1.3fr 1.6fr 1fr; gap: 8px; margin-bottom: 6px; }}
            .card {{ background: #252533; border-radius: 6px; padding: 6px 10px; border-left: 4px solid #3498db; }}
            .card.weather-card {{ border-left-color: #3498db; }}
            .card.robot-card {{ border-left-color: #2ecc71; }}
            .card.mod-card {{ border-left-color: #e67e22; }}
            .card.legend-card {{ border-left-color: #9b59b6; }}
            .card-title {{ font-size: 11px; text-transform: uppercase; color: #8888aa; letter-spacing: 0.5px; margin-bottom: 3px; font-weight: bold; }}
            .card-value {{ font-size: 12px; line-height: 1.35; }}
            .badge {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: bold; color: #fff; margin-right: 4px; margin-bottom: 2px; }}
            .weather-badge {{ background: #2980b9; font-size: 11px; padding: 2px 7px; }}
            .container {{ display: flex; width: 100%; height: calc(100vh - 145px); gap: 8px; }}
            .plot-box {{ flex: 1; background: white; border: 1px solid #3d3d52; border-radius: 6px; padding: 4px; }}
            #real-map, #mem-map {{ width: 100%; height: 100%; }}
            select {{ padding: 4px 8px; font-size: 13px; background: #353547; color: #ffffff; border: 1px solid #555570; border-radius: 4px; outline: none; }}
        </style>
    </head>
    <body>
        <div class="header-panel">
            <div class="header-title">Live Viewer | Session: {session_id}</div>
            <div>
                <select id="robot-select">
                    <option value="">Loading robots...</option>
                </select>
            </div>
        </div>

        <div class="dashboard-bar">
            <div class="card weather-card">
                <div class="card-title">Global Weather</div>
                <div id="weather-info" class="card-value">Loading weather...</div>
            </div>
            <div class="card robot-card">
                <div class="card-title">Robot Telemetry</div>
                <div id="robot-info" class="card-value">Loading robot...</div>
            </div>
            <div class="card mod-card">
                <div class="card-title">Active Modifiers on Robot</div>
                <div id="modifier-info" class="card-value">Checking modifiers...</div>
            </div>
            <div class="card legend-card">
                <div class="card-title">Modifier Color Legend</div>
                <div class="card-value" style="font-size:10px; line-height: 1.5;">
                    <span class="badge" style="background:#E67E22;">Speed</span>
                    <span class="badge" style="background:#9B59B6;">Vision</span>
                    <span class="badge" style="background:#1ABC9C;">Drift</span>
                    <span class="badge" style="background:#E74C3C;">Spd+Vis</span>
                    <span class="badge" style="background:#2980B9;">Spd+Drift</span>
                    <span class="badge" style="background:#8E44AD;">Multi</span>
                </div>
            </div>
        </div>

        <div class="container">
            <div class="plot-box"><div id="real-map"></div></div>
            <div class="plot-box"><div id="mem-map"></div></div>
        </div>

        <script>
            function preserveAndReact(divId, newPlot) {{
                const graphDiv = document.getElementById(divId);
                
                if (graphDiv && graphDiv.layout) {{
                    if (graphDiv.layout.xaxis && graphDiv.layout.xaxis.range) {{
                        newPlot.layout.xaxis.range = graphDiv.layout.xaxis.range;
                        newPlot.layout.xaxis.autorange = false;
                    }}
                    if (graphDiv.layout.yaxis && graphDiv.layout.yaxis.range) {{
                        newPlot.layout.yaxis.range = graphDiv.layout.yaxis.range;
                        newPlot.layout.yaxis.autorange = false;
                    }}
                    
                    if (graphDiv.data) {{
                        const visibilityMap = {{}};
                        const hiddenCoords = new Set(); 

                        graphDiv.data.forEach(trace => {{
                            if (trace.name) visibilityMap[trace.name] = trace.visible;
                        }});

                        newPlot.data.forEach(trace => {{
                            if (trace.name && visibilityMap[trace.name] !== undefined) {{
                                trace.visible = visibilityMap[trace.name];
                            }}
                            if (trace.visible === 'legendonly' || trace.visible === false) {{
                                if (trace.x && trace.y && trace.mode && trace.mode.includes('markers')) {{
                                    for(let i = 0; i < trace.x.length; i++) {{
                                        hiddenCoords.add(trace.x[i] + '_' + trace.y[i]);
                                    }}
                                }}
                            }}
                        }});

                        if (newPlot.layout && newPlot.layout.images) {{
                            newPlot.layout.images = newPlot.layout.images.filter(img => {{
                                const coordKey = img.x + '_' + img.y;
                                return !hiddenCoords.has(coordKey);
                            }});
                        }}
                    }}
                }}
                
                newPlot.layout.uirevision = 'true';
                Plotly.react(divId, newPlot.data, newPlot.layout);
            }}

            function updateSelectOptions(activeRobots) {{
                const select = document.getElementById("robot-select");
                const currentVal = select.value;
                
                if (activeRobots.length === 0) {{
                    select.innerHTML = '<option value="">No robots</option>';
                    return;
                }}

                if (select.options.length <= 1 && select.options[0].value === "") {{
                    select.innerHTML = '';
                    activeRobots.forEach(r => {{
                        const opt = document.createElement('option');
                        opt.value = r;
                        opt.textContent = "Robot: " + r;
                        select.appendChild(opt);
                    }});
                    if(currentVal && activeRobots.includes(currentVal)) {{
                        select.value = currentVal;
                    }}
                }} else {{
                    const existingVals = Array.from(select.options).map(o => o.value);
                    activeRobots.forEach(r => {{
                        if (!existingVals.includes(r)) {{
                            const opt = document.createElement('option');
                            opt.value = r;
                            opt.textContent = "Robot: " + r;
                            select.appendChild(opt);
                        }}
                    }});
                }}
            }}

            function updateLiveDashboard(live) {{
                if (!live) return;

                const gw = live.global_weather || {{}};
                const weatherEl = document.getElementById("weather-info");
                if (gw.enabled) {{
                    const st = gw.state || 'NORMAL';
                    let bg = '#27ae60';
                    if (st.indexOf('SNOW') !== -1) bg = '#2980b9';
                    else if (st.indexOf('BLIZZARD') !== -1) bg = '#8e44ad';
                    else if (st.indexOf('WHITEOUT') !== -1) bg = '#c0392b';
                    const patternsStr = (gw.patterns && gw.patterns.length > 0) ? gw.patterns.join(', ') : 'None';
                    weatherEl.innerHTML = '<span class="badge weather-badge" style="background:' + bg + ';">' + st + '</span> ' +
                        '<span>Patterns: <b>' + patternsStr + '</b></span><br>' +
                        '<span style="color:#aaa;">Active Zones: <b>' + (gw.active_zones || 0) + '</b> | Tick: <b>' + (gw.tick || 0) + '</b></span>';
                }} else {{
                    weatherEl.innerHTML = '<span class="badge" style="background:#555;">WEATHER OFF</span> <span style="color:#aaa;">Standard conditions</span>';
                }}

                const rob = live.robot || {{}};
                const robotEl = document.getElementById("robot-info");
                if (rob.id) {{
                    robotEl.innerHTML = '<b>' + rob.id + '</b> &bull; Pos: <b>(' + (rob.x !== undefined ? rob.x : '-') + ', ' + (rob.y !== undefined ? rob.y : '-') + ')</b> &bull; Dir: <b>' + (rob.angle !== undefined ? rob.angle : 0) + '&deg;</b><br>' +
                        'Speed: <b style="color:#f39c12;">' + (rob.speed !== undefined ? rob.speed : 1.0) + '</b> &bull; Vision: <b style="color:#bb86fc;">' + (rob.vision_radius !== undefined ? rob.vision_radius : 5.0) + 'm</b>';
                }} else {{
                    robotEl.innerHTML = '<span style="color:#888;">No robot detected</span>';
                }}

                const mods = live.active_modifiers || [];
                const modEl = document.getElementById("modifier-info");
                if (mods.length === 0) {{
                    modEl.innerHTML = '<span style="color:#888;">No active zone modifiers (Normal)</span>';
                }} else {{
                    let html = '';
                    mods.forEach(m => {{
                        const color = m.color || '#e67e22';
                        const attr = (m.attribute || m.target || m.type || 'EFFECT').toUpperCase();
                        let valStr = '';
                        if (m.base_value !== undefined && m.final_value !== undefined && m.base_value > 0) {{
                            const pct = Math.round((m.final_value / m.base_value - 1.0) * 100);
                            valStr = (pct >= 0 ? '+' : '') + pct + '% (' + m.final_value + ')';
                        }} else if (m.value !== undefined) {{
                            valStr = '' + m.value;
                        }} else {{
                            valStr = 'Active';
                        }}
                        const src = m.rule ? ' [' + m.rule + ']' : '';
                        html += '<span class="badge" style="background:' + color + ';" title="' + (m.rule || '') + '">' + attr + ': ' + valStr + src + '</span> ';
                    }});
                    modEl.innerHTML = html;
                }}
            }}

            async function fetchAndRender() {{
                try {{
                    const select = document.getElementById("robot-select");
                    const selectedRobot = select.value;
                    const queryParam = selectedRobot ? '?robot_id=' + selectedRobot : '';

                    const response = await fetch('/api/v1/sessions/{session_id}/plots' + queryParam);
                    if (!response.ok) return;
                    
                    const data = await response.json();
                    
                    if (data.active_robots) updateSelectOptions(data.active_robots);
                    if (data.live_data) updateLiveDashboard(data.live_data);

                    if (data.real) preserveAndReact('real-map', JSON.parse(data.real));
                    if (data.mem) preserveAndReact('mem-map', JSON.parse(data.mem));
                    
                }} catch (error) {{
                    console.error("Failed to fetch plots:", error);
                }}
            }}

            setInterval(fetchAndRender, 500);
            fetchAndRender(); 
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)