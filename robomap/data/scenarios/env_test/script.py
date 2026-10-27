import os
import json

from robomap.api import API
from robomap.environment_manager import EnvironmentManager



class Scenario:
    def __init__(self, api: API, auto_robot: bool = True):
        self.api = api
        self.world = api.world
        self.auto_robot = auto_robot
        self.stage = "init"
        self.env = None
        self.robot = None
        self.route_step = 0
        self.log_file_path = os.path.join(os.path.dirname(__file__), "scenario_run.log")

    def setup(self):
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.write("--- SCENARIO START ---\n")

        with self.api.lock:
            self.robot = self.world.robots.get("robot_1")
            
            self.robot.speed = 2.0
            self.robot.vision_radius = 20.0
            
            if self.robot.clusterer is not None:
                self.robot.clusterer = None

            self.robot.state["charge"] = 100.0

            self.world.add_object("building", 0, 5, shape=[[-2, -2], [2, -2], [2, 2], [-2, 2]], obj_id="house_1")

            config_path = os.path.join(os.path.dirname(__file__), "env_config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                env_config = json.load(f)

            self.env = EnvironmentManager(self.world, env_config)
            self.env.register_robot(self.robot.id)

    def tick(self, world):
        if not self.robot or not self.env:
            return

        self.env.process_environment()
        self.env.sync_virtual_zones(self.api, self.robot.id)

        charge = getattr(self.robot, "state", {}).get("charge", 0.0)
        
        log_str = (
            f"[Tick {world.tick_count:03d}] "
            f"Pos:({self.robot.x:5.1f}, {self.robot.y:5.1f}) | "
            f"Speed:{self.robot.speed:.2f} | "
            f"Vis:{self.robot.vision_radius:4.1f} | "
            f"Charge:{charge:5.1f}"
        )
        
        print(log_str)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(log_str + "\n")

        if not self.auto_robot:
            return

        tick = world.tick_count

        if tick == 70:
            self._log_command("Moving to Moving_Fog_Zone (20, 20)")
            self.api.command_move(self.robot.id, 20, 20)
            self.route_step = 1

        elif self.route_step == 1 and len(self.robot.action_queue) == 0:
            self._log_command("Moving to Expanding_Mud_Zone (20, -30)")
            self.api.command_move(self.robot.id, 20, -30)
            self.route_step = 2

        elif self.route_step == 2 and len(self.robot.action_queue) == 0:
            self._log_command("Moving to Ice_Zone (-20, -20)")
            self.api.command_move(self.robot.id, -20, -20)
            self.route_step = 3

        elif self.route_step == 3 and len(self.robot.action_queue) == 0:
            self._log_command("Moving to Fast_Discharge_Zone (-20, 20)")
            self.api.command_move(self.robot.id, -20, 20)
            self.route_step = 4

        elif self.route_step == 4 and len(self.robot.action_queue) == 0:
            self._log_command("Returning to base (0, 0)")
            self.api.command_move(self.robot.id, 0, 0)
            self.route_step = 5
            self.stage = "completed"

    def _log_command(self, msg: str):
        print(f"[COMMAND] {msg}")
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"[COMMAND] {msg}\n")