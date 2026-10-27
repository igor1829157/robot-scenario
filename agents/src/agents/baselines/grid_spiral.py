import logging

import numpy as np

from agents.baselines.common import wait_until_queue_empty
from robomap_client.core.client import RobomapClient

logger = logging.getLogger(__name__)


def build_grid(size: float, robot_vision: float) -> np.ndarray:
    step = robot_vision / np.sqrt(2)
    max_k = int(np.ceil(size / step))
    coords_1d = np.arange(-max_k * step, max_k * step + step * 0.5, step)
    coords_1d = np.clip(coords_1d, -size, size)
    x_grid, y_grid = np.meshgrid(coords_1d, coords_1d, indexing="ij")
    return np.stack([x_grid, y_grid], axis=-1)


def build_spiral_grid_path(coords: np.ndarray) -> np.ndarray:
    x_size, y_size = coords.shape[0], coords.shape[1]
    x, y = x_size // 2, y_size // 2
    directions = np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])
    dir_idx = 0
    steps = 1
    path = []
    max_len = x_size * y_size - 1

    while len(path) < max_len:
        for _ in range(2):
            dx, dy = directions[dir_idx]
            for _ in range(steps):
                x += dx
                y += dy
                if 0 <= x < x_size and 0 <= y < y_size:
                    path.append(np.round(coords[x, y], 2))
                if len(path) >= max_len:
                    break
            dir_idx = (dir_idx + 1) % 4
            if len(path) >= max_len:
                break
        steps += 1

    return np.array(path)


class GridSpiralAgent:
    name = "grid_spiral"

    def __init__(
        self,
        map_size: float = 100,
        robot_vision_range: float = 15,
    ) -> None:
        self.map_size = map_size
        self.robot_vision_range = robot_vision_range

    def run(self, client: RobomapClient) -> None:
        grid = build_grid(self.map_size, self.robot_vision_range)
        path = build_spiral_grid_path(grid)
        logger.info("%s: submitting %d waypoints", self.name, len(path))

        for point in path:
            client.command("move", target_x=float(point[0]), target_y=float(point[1]))

        wait_until_queue_empty(client)
