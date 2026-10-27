import logging

import numpy as np

from agents.baselines.common import wait_until_queue_empty
from robomap_client.core.client import RobomapClient

logger = logging.getLogger(__name__)


def get_spiral_path(side: float, n: int, robot_vision_range: float) -> np.ndarray:

    turns = side // (robot_vision_range / np.sqrt(2))
    t = np.linspace(0, 2 * np.pi * turns, n)
    r = side * t / t.max()
    path = np.array([r * np.cos(t), r * np.sin(t)]).T

    return path[1:]


class SpiralAgent:
    name = "spiral"

    def __init__(
        self,
        map_size: float = 100,
        spiral_points: int = 440,
        robot_vision_range: float = 15,
    ):
        self.map_size = map_size
        self.spiral_points = spiral_points
        self.robot_vision_range = robot_vision_range

    def run(self, client: RobomapClient) -> None:
        path = get_spiral_path(self.map_size, self.spiral_points, self.robot_vision_range)
        logger.info("%s: submitting %d waypoints", self.name, len(path))

        for point in path:
            client.command("move", target_x=float(point[0]), target_y=float(point[1]))

        wait_until_queue_empty(client)
