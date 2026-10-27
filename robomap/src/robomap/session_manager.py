import os
import random
import threading
import time
import logging
from typing import Dict, Any, List, Optional

from robomap.world import World
from robomap.api import API
from robomap.export_codec import (
    compute_frame_delta,
    materialize_frames_from_deltas,
)
from robomap.scenario import load_scenario
from robomap.state_loader import load_state_to_world, load_and_merge_configs

logger = logging.getLogger(__name__)

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class StateRecorder:

    def __init__(self):
        """Инициализирует компактное хранилище истории кадров."""
        self.base_frame: Optional[Dict[str, Any]] = None
        self.deltas: List[Dict[str, Any]] = []
        self.latest_frame: Optional[Dict[str, Any]] = None

    def record_frame(self, tick: int, world: World, scenario: Any = None):
        """Записывает кадр как базовое состояние или дельту."""
        events = []
        
        entities = world.real_map_data.copy()
        entities.extend(world.robots.values())
            
        for ent in entities:
            speaking = ent.state.get("speaking")
            if speaking:
                events.append(f"[{ent.type_name}] {speaking}")

        if scenario:
            scenario_stage = getattr(scenario, 'stage', 'unknown')
            previous_stage_frame = self.latest_frame
            if (
                previous_stage_frame is None
                or previous_stage_frame.get('scenario_stage') != scenario_stage
            ):
                events.append(f"СЦЕНАРИЙ: {scenario_stage.upper()}")

        frame_data = {
            "tick": tick,
            "real_map": world.get_real_map_data_dict(),
            "robots_memory": world.get_robots_memory_dict(),
            "robots_state": {
                robot_id: robot.to_dict()
                for robot_id, robot in world.robots.items()
            },
            "events": [],
            "scenario_stage": scenario.stage if scenario else None
        }
        if self.base_frame is None:
            self.base_frame = frame_data
        else:
            previous_frame = self.latest_frame
            if previous_frame is None:
                raise RuntimeError("Latest frame is missing")
            self.deltas.append(
                compute_frame_delta(previous_frame, frame_data)
            )
        self.latest_frame = frame_data

    def get_history(self) -> List[Dict[str, Any]]:
        """Материализует полную историю кадров по сохранённым дельтам."""
        if self.base_frame is None:
            return []
        return materialize_frames_from_deltas(
            self.base_frame,
            list(self.deltas),
        )

    def get_encoded_history(
        self,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Возвращает снимок истории в формате base frame + deltas."""
        if self.base_frame is None:
            raise ValueError("History is empty")
        return self.base_frame, list(self.deltas)

    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """Возвращает последний полный кадр без материализации истории."""
        return self.latest_frame

    @property
    def frame_count(self) -> int:
        """Возвращает количество записанных кадров."""
        if self.base_frame is None:
            return 0
        return len(self.deltas) + 1

    @property
    def frames(self) -> List[Dict[str, Any]]:
        """Материализует кадры для обратной совместимости."""
        return self.get_history()


class SimulationSession:
    def __init__(
        self,
        session_id: str,
        config_path: str = "data/config.json",
        auto_robot: bool = True,
        scenario_name: str = "bear_attack",
        *,
        random_robot_pose: bool = False,
        robot_pose_seed: Optional[int] = None,
        robot_id: str = "robot_1",
        weather_patterns: Optional[List[int]] = None,
        weather_seed: Optional[int] = None,
    ):
        self.session_id = session_id
        self.scenario_name = scenario_name
        self.global_config_path = config_path
        self.lock = threading.RLock()

        self.is_running = False
        self._bg_thread = None
        
        self.config = load_and_merge_configs(self.global_config_path, self.scenario_name)
        
        self.world = World(config_dict=self.config)
        self.api = API(self.world, self.lock)
        self.recorder = StateRecorder()
        
        self.permissions: Dict[str, List[str]] = {}
        
        state_path = self._get_state_path()
        objects_rel_path = self.world.config.get("paths", {}).get("objects_meta_path", "data/objects/objects.json")
        objects_path = os.path.join(project_path, objects_rel_path)
        
        load_state_to_world(self.world, state_path, objects_path)

        if random_robot_pose:
            rng = random.Random(robot_pose_seed) if robot_pose_seed is not None else random.Random()
            if self.world.place_robot_randomly(robot_id, rng):
                robot = self.world.robots.get(robot_id)
                if robot is not None:
                    self.world._update_robot_memory(robot)

        self.scenario = load_scenario(self.scenario_name, self.api, auto_robot)
        
        if weather_patterns:
            from robomap.arctic_weather import ArcticWeatherScenario, WeatherAttachedScenario
            weather_sc = ArcticWeatherScenario(
                self.api,
                auto_robot=auto_robot,
                weather_config={
                    "enabled": True,
                    "patterns": weather_patterns,
                    "seed": 42 if weather_seed is None else weather_seed
                }
            )
            if self.scenario is None:
                self.scenario = weather_sc
            elif isinstance(self.scenario, ArcticWeatherScenario) and getattr(self.scenario, "weather", None) is None:
                self.scenario.weather_config = {
                    "enabled": True,
                    "patterns": weather_patterns,
                    "seed": 42 if weather_seed is None else weather_seed
                }
            else:
                self.scenario = WeatherAttachedScenario(self.scenario, weather_sc)

        if self.scenario:
            self.scenario.setup()
            self.world.scenario_controller = self.scenario
        else:
            self.world.scenario_controller = None

        self.world._calculate_boundaries()
        self.recorder.record_frame(self.world.tick_count, self.world, self.scenario)

    def _get_state_path(self) -> str:
        scenarios_dir = self.world.config.get("paths", {}).get("scenarios_dir", "data/scenarios")
        
        if self.scenario_name not in ["empty", "sandbox", "none", ""]:
            custom_map_rel = os.path.join(scenarios_dir, self.scenario_name, "initial_state.json")
            custom_map_abs = os.path.join(project_path, custom_map_rel)
            if os.path.exists(custom_map_abs):
                logger.info("SESSION [%s] | Найдена кастомная карта сценария: %s", self.session_id, custom_map_rel)
                return custom_map_abs
            
        default_map_rel = self.world.config.get("paths", {}).get("default_state_path", "data/init_state/initial_state.json")
        default_map_abs = os.path.join(project_path, default_map_rel)
        logger.info("SESSION [%s] | Загружается базовая карта: %s", self.session_id, default_map_rel)
        return default_map_abs

    def start_realtime(self, fps: float = 2.0):
        if self.is_running:
            return
        self.is_running = True
        self._bg_thread = threading.Thread(target=self._realtime_loop, args=(fps,), daemon=True)
        self._bg_thread.start()
        logger.info("SESSION [%s] | Real-time mode STARTED (%.1f fps).", self.session_id, fps)

    def stop_realtime(self):
        if not self.is_running:
            return
        self.is_running = False
        if self._bg_thread:
            self._bg_thread.join(timeout=1.0)
        logger.info("SESSION [%s] | Real-time mode STOPPED.", self.session_id)

    def _realtime_loop(self, fps: float):
        sleep_time = 1.0 / fps
        while self.is_running:
            try:
                self.tick()
            except Exception as e:
                logger.error("Error in realtime loop: %s", str(e), exc_info=True)
            time.sleep(sleep_time)

    def register_player(self, token: str, entity_ids: List[str]):
        with self.lock:
            self.permissions[token] = entity_ids
            logger.info("SESSION [%s] | Player '%s' registered to control: %s", self.session_id, token, entity_ids)

    def remove_player(self, token: str) -> bool:
        with self.lock:
            if token in self.permissions:
                del self.permissions[token]
                logger.info("SESSION [%s] | Player '%s' access revoked.", self.session_id, token)
                return True
            return False

    def tick(self) -> Dict[str, Any]:
        with self.lock:
            try:
                self.world.tick()
                self.recorder.record_frame(self.world.tick_count, self.world, self.scenario)
            except Exception as e:
                logger.error("CRITICAL ERROR DURING WORLD TICK: %s", str(e), exc_info=True)
                
                for r_id, robot in self.world.robots.items():
                    if hasattr(robot, "add_event_log"):
                        robot.add_event_log(
                            tick=self.world.tick_count,
                            category="SYSTEM",
                            action_type="error",
                            message=f"Critical system failure during physical simulation: {str(e)}. Action queue cleared.",
                            status="ERROR"
                        )
                    if hasattr(robot, "clear_actions"):
                        robot.clear_actions()

            latest_frame = self.recorder.get_latest_frame()
            events_list = (
                latest_frame["events"]
                if latest_frame is not None
                else []
            )
            return {
                "tick": self.world.tick_count,
                "stage": getattr(self.scenario, "stage", "unknown"),
                "events": events_list
            }

    def execute_command(self, token: str, entity_id: str, action: str, *args, **kwargs) -> bool:
        with self.lock:
            if token != "admin":
                allowed_entities = self.permissions.get(token, [])
                if entity_id not in allowed_entities:
                    logger.warning("SECURITY | Session [%s] Token '%s' denied access to command '%s'.", self.session_id, token, entity_id)
                    return False

            if action == "move":
                return self.api.command_move(entity_id, *args, **kwargs)
            else:
                return self.api.command_action(entity_id, action, *args, **kwargs)

    def run_auto(self, max_ticks: int = 150) -> str:
        logger.info("SESSION [%s] | Запуск Авто-прогона...", self.session_id)
        with self.lock:
            for _ in range(max_ticks):
                self.world.tick()
                self.recorder.record_frame(self.world.tick_count, self.world, self.scenario)
                terminal_stages = getattr(self.scenario, "terminal_stages", ["completed", "failed"])
                if getattr(self.scenario, "stage", "") in terminal_stages:
                    for _ in range(5):
                        self.world.tick()
                        self.recorder.record_frame(self.world.tick_count, self.world, self.scenario)
                    break
        stage = getattr(self.scenario, "stage", "unknown")
        logger.info("SESSION [%s] | Авто-прогон завершен. Тиков: %d. Статус: %s", self.session_id, self.world.tick_count, stage)
        return stage

    def reset(self):
        was_running = self.is_running
        self.stop_realtime()
        
        with self.lock:
            logger.info("SESSION [%s] | Сброс симуляции...", self.session_id)
            auto_robot = getattr(self.scenario, "auto_robot", True) if self.scenario else False
            
            self.config = load_and_merge_configs(self.global_config_path, self.scenario_name)
            
            self.world = World(config_dict=self.config)
            self.api = API(self.world, self.lock)
            self.recorder = StateRecorder() 
            
            state_path = self._get_state_path()
            objects_rel_path = self.world.config.get("paths", {}).get("objects_meta_path", "data/objects/objects.json")
            objects_path = os.path.join(project_path, objects_rel_path)
            
            load_state_to_world(self.world, state_path, objects_path)

            self.scenario = load_scenario(self.scenario_name, self.api, auto_robot)
            
            if self.scenario:
                self.scenario.setup()
                self.world.scenario_controller = self.scenario
            else:
                self.world.scenario_controller = None

            self.world._calculate_boundaries()
            self.recorder.record_frame(self.world.tick_count, self.world, self.scenario)
            logger.info("SESSION [%s] | Сброс завершен.", self.session_id)
            
        if was_running:
            self.start_realtime()