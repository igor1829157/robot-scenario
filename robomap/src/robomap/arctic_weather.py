import math
import random
from typing import Dict, Any, Tuple, List, Optional
from shapely.geometry import Polygon
from robomap.environment_manager import EnvironmentManager


class WeatherStateMachine:
    def __init__(self, seed: Optional[int] = None, initial_state: str = "CLEAR_FROST"):
        self.seed = 42 if seed is None else int(seed)
        self.rng = random.Random(self.seed)
        self.states = {
            "CLEAR_FROST": {"vision_mult": 1.0, "speed_mult": 1.0, "duration_range": (80, 160)},
            "WIND_RISE": {"vision_mult": 0.8, "speed_mult": 0.9, "duration_range": (40, 80)},
            "BLIZZARD": {"vision_mult": 0.25, "speed_mult": 0.7, "duration_range": (60, 120)},
            "POST_STORM_SNOW": {"vision_mult": 0.6, "speed_mult": 0.8, "duration_range": (50, 100)},
            "FREEZING_FOG": {"vision_mult": 0.3, "speed_mult": 1.0, "duration_range": (60, 140)},
            "THAW_GLAZE": {"vision_mult": 0.7, "speed_mult": 0.85, "duration_range": (50, 90)}
        }
        self.transitions = {
            "CLEAR_FROST": [("WIND_RISE", 0.6), ("FREEZING_FOG", 0.4)],
            "WIND_RISE": [("BLIZZARD", 0.7), ("CLEAR_FROST", 0.3)],
            "BLIZZARD": [("POST_STORM_SNOW", 0.9), ("WIND_RISE", 0.1)],
            "POST_STORM_SNOW": [("CLEAR_FROST", 0.5), ("FREEZING_FOG", 0.3), ("THAW_GLAZE", 0.2)],
            "FREEZING_FOG": [("CLEAR_FROST", 0.7), ("WIND_RISE", 0.3)],
            "THAW_GLAZE": [("CLEAR_FROST", 0.8), ("WIND_RISE", 0.2)]
        }
        self.current_state = initial_state if initial_state in self.states else "CLEAR_FROST"
        self.previous_state = None
        self.ticks_in_state = 0
        self.total_ticks = 0
        self.target_duration = self._sample_duration(self.current_state)
        self.state_history = [(0, self.current_state)]

    def _sample_duration(self, state: str) -> int:
        min_d, max_d = self.states[state]["duration_range"]
        return self.rng.randint(min_d, max_d)

    def step(self) -> Dict[str, Any]:
        self.ticks_in_state += 1
        self.total_ticks += 1
        transition_occurred = False
        if self.ticks_in_state >= self.target_duration:
            self.previous_state = self.current_state
            options = self.transitions[self.current_state]
            states, weights = zip(*options)
            self.current_state = self.rng.choices(states, weights=weights)[0]
            self.ticks_in_state = 0
            self.target_duration = self._sample_duration(self.current_state)
            self.state_history.append((self.total_ticks, self.current_state))
            transition_occurred = True
        return {
            "state": self.current_state,
            "previous_state": self.previous_state,
            "transition": transition_occurred,
            "ticks_in_state": self.ticks_in_state,
            "target_duration": self.target_duration,
            "vision_mult": self.states[self.current_state]["vision_mult"],
            "speed_mult": self.states[self.current_state]["speed_mult"],
            "total_ticks": self.total_ticks
        }

    def reset(self, seed: Optional[int] = None, initial_state: str = "CLEAR_FROST"):
        if seed is not None:
            self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.current_state = initial_state if initial_state in self.states else "CLEAR_FROST"
        self.previous_state = None
        self.ticks_in_state = 0
        self.total_ticks = 0
        self.target_duration = self._sample_duration(self.current_state)
        self.state_history = [(0, self.current_state)]

    def get_state_info(self) -> Dict[str, Any]:
        return {
            "state": self.current_state,
            "previous_state": self.previous_state,
            "ticks_in_state": self.ticks_in_state,
            "target_duration": self.target_duration,
            "vision_mult": self.states[self.current_state]["vision_mult"],
            "speed_mult": self.states[self.current_state]["speed_mult"],
            "total_ticks": self.total_ticks
        }

    def get_current_rule(self) -> Dict[str, Any]:
        st = self.states[self.current_state]
        return {
            "name": f"Atmosphere_{self.current_state}",
            "target_type": "robot",
            "effects": [
                {"attribute": "vision_radius", "type": "multiply", "value": st["vision_mult"]},
                {"attribute": "speed", "type": "multiply", "value": st["speed_mult"]}
            ],
            "silent": True,
            "render_only": True,
            "is_obstacle": False
        }


AtmosphericStateMachine = WeatherStateMachine


class SpatialZoneAdapter:
    def __init__(self, map_boundaries: Optional[Tuple[List[float], List[float]]] = None, padding: float = 10.0):
        if map_boundaries and len(map_boundaries) >= 2:
            self.x_min = float(map_boundaries[0][0])
            self.x_max = float(map_boundaries[0][1])
            self.y_min = float(map_boundaries[1][0])
            self.y_max = float(map_boundaries[1][1])
        else:
            self.x_min, self.x_max = -100.0, 100.0
            self.y_min, self.y_max = -100.0, 100.0
        self.padding = padding
        self.inner_x_min = min(self.x_min + self.padding, self.x_max - self.padding)
        self.inner_x_max = max(self.x_min + self.padding, self.x_max - self.padding)
        self.inner_y_min = min(self.y_min + self.padding, self.y_max - self.padding)
        self.inner_y_max = max(self.y_min + self.padding, self.y_max - self.padding)
        self.width = max(1.0, self.inner_x_max - self.inner_x_min)
        self.height = max(1.0, self.inner_y_max - self.inner_y_min)

    def normalize_point(self, u: float, v: float) -> Tuple[float, float]:
        x = self.inner_x_min + max(0.0, min(1.0, float(u))) * self.width
        y = self.inner_y_min + max(0.0, min(1.0, float(v))) * self.height
        return round(x, 3), round(y, 3)

    def generate_box(self, center_uv: Tuple[float, float], size_uv: Tuple[float, float]) -> List[List[float]]:
        cx, cy = self.normalize_point(center_uv[0], center_uv[1])
        hw = 0.5 * max(0.01, min(1.0, float(size_uv[0]))) * self.width
        hh = 0.5 * max(0.01, min(1.0, float(size_uv[1]))) * self.height
        p1 = [round(cx - hw, 3), round(cy - hh, 3)]
        p2 = [round(cx + hw, 3), round(cy - hh, 3)]
        p3 = [round(cx + hw, 3), round(cy + hh, 3)]
        p4 = [round(cx - hw, 3), round(cy + hh, 3)]
        return [p1, p2, p3, p4]

    def _validate_coords(self, coords: List[List[float]]) -> List[List[float]]:
        if len(coords) < 3:
            return coords
        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or not poly.is_valid:
                poly = Polygon(coords).convex_hull
            if poly.geom_type == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda g: g.area)
            res = [[round(p[0], 3), round(p[1], 3)] for p in poly.exterior.coords[:-1]]
            return res if len(res) >= 3 else coords
        except Exception:
            return coords

    def generate_irregular_zone(self, center_uv: Tuple[float, float], radius_uv: float, num_vertices: int = 12, rng: Optional[random.Random] = None) -> List[List[float]]:
        return self.generate_organic_zone(center_uv, radius_uv, num_vertices=num_vertices, rng=rng)

    def generate_organic_zone(
        self,
        center_uv: Tuple[float, float],
        radius_uv: float,
        num_vertices: int = 14,
        rng: Optional[random.Random] = None
    ) -> List[List[float]]:
        cx, cy = self.normalize_point(center_uv[0], center_uv[1])
        base_r = max(0.01, min(0.5, float(radius_uv))) * min(self.width, self.height)
        p1 = rng.uniform(0.0, 6.28) if rng is not None else 0.0
        p2 = rng.uniform(0.0, 6.28) if rng is not None else 1.2
        p3 = rng.uniform(0.0, 6.28) if rng is not None else 2.5
        n = max(8, int(num_vertices))
        coords = []
        for i in range(n):
            angle = 2.0 * math.pi * i / n
            h1 = 0.20 * math.sin(2.0 * angle + p1)
            h2 = 0.12 * math.cos(3.0 * angle + p2)
            h3 = 0.08 * math.sin(5.0 * angle + p3)
            jit = rng.uniform(-0.06, 0.06) if rng is not None else 0.0
            r = base_r * max(0.4, 1.0 + h1 + h2 + h3 + jit)
            px = max(self.inner_x_min, min(self.inner_x_max, cx + r * math.cos(angle)))
            py = max(self.inner_y_min, min(self.inner_y_max, cy + r * math.sin(angle)))
            coords.append([round(px, 3), round(py, 3)])
        return self._validate_coords(coords)

    def generate_corridor(self, start_uv: Tuple[float, float], end_uv: Tuple[float, float], width_uv: float) -> List[List[float]]:
        return self.generate_meandering_corridor(start_uv, end_uv, width_uv=width_uv, num_segments=10, meander_amplitude=0.04)

    def generate_meandering_corridor(
        self,
        start_uv: Tuple[float, float],
        end_uv: Tuple[float, float],
        width_uv: float,
        num_segments: int = 12,
        meander_amplitude: float = 0.05,
        rng: Optional[random.Random] = None
    ) -> List[List[float]]:
        x1, y1 = self.normalize_point(start_uv[0], start_uv[1])
        x2, y2 = self.normalize_point(end_uv[0], end_uv[1])
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 0.001:
            return self.generate_box(start_uv, (width_uv, width_uv))
            return self.generate_organic_zone(start_uv, radius_uv=0.5 * width_uv, num_vertices=8, rng=rng)
        nx = -dy / length
        ny = dx / length
        base_w = max(0.01, min(0.5, float(width_uv))) * min(self.width, self.height)
        amp = max(0.0, float(meander_amplitude)) * min(self.width, self.height)
        phase_offset = rng.uniform(0.0, 2.0 * math.pi) if rng is not None else 0.0
        n = max(6, int(num_segments))
        left_bank = []
        right_bank = []
        for i in range(n + 1):
            t = i / float(n)
            sine_wave = math.sin(math.pi * t) * math.sin(2.5 * math.pi * t + phase_offset)
            jitter = rng.uniform(-0.15, 0.15) if rng is not None else 0.0
            lateral = (sine_wave + jitter) * amp
            width_mod = base_w * (1.0 + 0.25 * math.sin(3.0 * math.pi * t + phase_offset))
            px = x1 + t * dx + lateral * nx
            py = y1 + t * dy + lateral * ny
            px = max(self.inner_x_min, min(self.inner_x_max, px))
            py = max(self.inner_y_min, min(self.inner_y_max, py))
            lx = max(self.inner_x_min, min(self.inner_x_max, px + 0.5 * width_mod * nx))
            ly = max(self.inner_y_min, min(self.inner_y_max, py + 0.5 * width_mod * ny))
            rx = max(self.inner_x_min, min(self.inner_x_max, px - 0.5 * width_mod * nx))
            ry = max(self.inner_y_min, min(self.inner_y_max, py - 0.5 * width_mod * ny))
            left_bank.append([round(lx, 3), round(ly, 3)])
            right_bank.append([round(rx, 3), round(ry, 3)])
        return self._validate_coords(left_bank + list(reversed(right_bank)))

    def generate_curved_band(
        self,
        center_uv: Tuple[float, float],
        length_uv: float,
        width_uv: float,
        curvature: float = 0.06,
        num_points: int = 10
    ) -> List[List[float]]:
        cx, cy = self.normalize_point(center_uv[0], center_uv[1])
        len_m = max(0.05, min(1.0, float(length_uv))) * self.width
        w_m = max(0.01, min(0.5, float(width_uv))) * self.height
        curve_m = float(curvature) * self.height
        n = max(6, int(num_points))
        top_pts = []
        bot_pts = []
        for i in range(n + 1):
            s = -1.0 + 2.0 * (i / float(n))
            dx = 0.5 * len_m * s
            dy = curve_m * (1.0 - s * s)
            px = max(self.inner_x_min, min(self.inner_x_max, cx + dx))
            py = cy + dy
            ty = max(self.inner_y_min, min(self.inner_y_max, py + 0.5 * w_m))
            by = max(self.inner_y_min, min(self.inner_y_max, py - 0.5 * w_m))
            top_pts.append([round(px, 3), round(ty, 3)])
            bot_pts.append([round(px, 3), round(by, 3)])
        return self._validate_coords(top_pts + list(reversed(bot_pts)))

    def generate_jagged_ridge(
        self,
        start_uv: Tuple[float, float],
        end_uv: Tuple[float, float],
        width_uv: float,
        num_teeth: int = 10,
        rng: Optional[random.Random] = None
    ) -> List[List[float]]:
        x1, y1 = self.normalize_point(start_uv[0], start_uv[1])
        x2, y2 = self.normalize_point(end_uv[0], end_uv[1])
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 0.001:
            return self.generate_box(start_uv, (width_uv, width_uv))
            return self.generate_organic_zone(start_uv, radius_uv=0.5 * width_uv, num_vertices=8, rng=rng)
        nx = -dy / length
        ny = dx / length
        base_w = max(0.01, min(0.5, float(width_uv))) * min(self.width, self.height)
        n = max(6, int(num_teeth))
        top_teeth = []
        bot_teeth = []
        for i in range(n + 1):
            t = i / float(n)
            px = x1 + t * dx
            py = y1 + t * dy
            sign = 1.0 if (i % 2 == 1) else -1.0
            jitter_w = (rng.uniform(0.7, 1.3) if rng is not None else 1.0) * base_w * 0.5
            w_top = jitter_w + sign * 0.2 * base_w
            w_bot = jitter_w - sign * 0.2 * base_w
            tx = max(self.inner_x_min, min(self.inner_x_max, px + w_top * nx))
            ty = max(self.inner_y_min, min(self.inner_y_max, py + w_top * ny))
            bx = max(self.inner_x_min, min(self.inner_x_max, px - w_bot * nx))
            by = max(self.inner_y_min, min(self.inner_y_max, py - w_bot * ny))
            top_teeth.append([round(tx, 3), round(ty, 3)])
            bot_teeth.append([round(bx, 3), round(by, 3)])
        return self._validate_coords(top_teeth + list(reversed(bot_teeth)))


class ZoneLifecycleManager:
    def __init__(self, spatial_adapter: SpatialZoneAdapter):
        self.adapter = spatial_adapter
        self.active_zones: Dict[str, Dict[str, Any]] = {}

    def add_zone(
        self,
        zone_id: str,
        shape: List[List[float]],
        effects: List[Dict[str, Any]],
        lifetime_ticks: Optional[int] = None,
        velocity: Optional[List[float]] = None,
        expansion_rate: float = 0.0,
        log_message: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        zone_data = {
            "id": zone_id,
            "shape": [list(p) for p in shape],
            "effects": list(effects),
            "velocity": list(velocity) if velocity else [0.0, 0.0],
            "expansion_rate": float(expansion_rate),
            "lifetime_ticks": lifetime_ticks,
            "remaining_ticks": lifetime_ticks,
            "log_message": log_message,
            "is_obstacle": False,
            "render_only": True
        }
        self.active_zones[zone_id] = zone_data
        return zone_data

    def remove_zone(self, zone_id: str) -> bool:
        if zone_id in self.active_zones:
            del self.active_zones[zone_id]
            return True
        return False

    def clear_dynamic_zones(self):
        to_del = [z_id for z_id, z in self.active_zones.items() if z["lifetime_ticks"] is not None]
        for z_id in to_del:
            del self.active_zones[z_id]

    def step(self) -> List[str]:
        expired = []
        for z_id, zone in list(self.active_zones.items()):
            if zone["remaining_ticks"] is not None:
                zone["remaining_ticks"] -= 1
                if zone["remaining_ticks"] <= 0:
                    expired.append(z_id)
                    del self.active_zones[z_id]
                    continue
            vel = zone.get("velocity", [0.0, 0.0])
            if vel != [0.0, 0.0]:
                poly_pts = zone["shape"]
                cx = sum(p[0] for p in poly_pts) / len(poly_pts)
                cy = sum(p[1] for p in poly_pts) / len(poly_pts)
                if cx + vel[0] < self.adapter.inner_x_min or cx + vel[0] > self.adapter.inner_x_max:
                    vel[0] = -vel[0]
                if cy + vel[1] < self.adapter.inner_y_min or cy + vel[1] > self.adapter.inner_y_max:
                    vel[1] = -vel[1]
                zone["velocity"] = vel
                zone["shape"] = [[round(p[0] + vel[0], 3), round(p[1] + vel[1], 3)] for p in poly_pts]
        return expired

    def get_environment_rules(self) -> List[Dict[str, Any]]:
        rules = []
        for z_id, z in self.active_zones.items():
            rule = {
                "name": z_id,
                "zone": {
                    "shape": [list(p) for p in z["shape"]],
                    "velocity": [0.0, 0.0],
                    "expansion_rate": z["expansion_rate"],
                    "is_obstacle": False,
                    "render_only": True
                },
                "target_type": "robot",
                "effects": [dict(e) for e in z["effects"]],
                "silent": True
            }
            if z.get("log_message"):
                rule["log_message"] = z["log_message"]
                rule["silent"] = False
            rules.append(rule)
        return rules


class WeatherPattern:
    def __init__(self, pattern_id: int, name: str):
        self.pattern_id = pattern_id
        self.name = name

    def setup(self, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        pass

    def on_weather_transition(self, prev_state: Optional[str], new_state: str, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        pass

    def on_tick(self, tick: int, weather_info: Dict[str, Any], adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        pass


class BlizzardWakeDriftsPattern(WeatherPattern):
    def __init__(self, pattern_id: int = 1, name: str = "BlizzardWakeDrifts"):
        super().__init__(pattern_id, name)
        self.stream_zone_id = f"p1_frozen_stream_{self.pattern_id}"
        self.drift_counter = 0

    def setup(self, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        corridor_pts = adapter.generate_meandering_corridor((0.05, 0.25), (0.95, 0.75), width_uv=0.08, meander_amplitude=0.06, rng=rng)
        zlm.add_zone(
            self.stream_zone_id,
            corridor_pts,
            [{"attribute": "position", "type": "physics", "operation": "drift", "value": 0.88}],
            log_message={"message": "Traction warning: Frozen meandering stream.", "level": "WARNING"}
        )

    def on_weather_transition(self, prev_state: Optional[str], new_state: str, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        if prev_state == "BLIZZARD" and new_state == "POST_STORM_SNOW":
            num_drifts = rng.randint(2, 4)
            for i in range(num_drifts):
                self.drift_counter += 1
                cu = rng.uniform(0.15, 0.85)
                cv = rng.uniform(0.15, 0.85)
                rad = rng.uniform(0.08, 0.13)
                shape = adapter.generate_organic_zone((cu, cv), radius_uv=rad, num_vertices=12, rng=rng)
                lifetime = rng.randint(120, 180)
                zlm.add_zone(
                    f"p1_snowdrift_{self.drift_counter}",
                    shape,
                    [{"attribute": "speed", "type": "multiply", "value": 0.30}],
                    lifetime_ticks=lifetime,
                    log_message={"message": "Terrain alert: Deep snowdrift.", "level": "INFO"}
                )


class WindfrontSastrugiPattern(WeatherPattern):
    def __init__(self, pattern_id: int = 2, name: str = "WindfrontSastrugi"):
        super().__init__(pattern_id, name)
        self.lake_zone_id = f"p2_frozen_lake_{self.pattern_id}"
        self.front_counter = 0
        self.sastrugi_counter = 0

    def setup(self, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        lake_pts = adapter.generate_organic_zone((0.7, 0.4), radius_uv=0.14, num_vertices=14, rng=rng)
        zlm.add_zone(
            self.lake_zone_id,
            lake_pts,
            [{"attribute": "position", "type": "physics", "operation": "drift", "value": 0.92}],
            log_message={"message": "Traction warning: Glaze ice lake.", "level": "WARNING"}
        )

    def on_weather_transition(self, prev_state: Optional[str], new_state: str, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        if new_state == "WIND_RISE":
            self.front_counter += 1
            front_shape = adapter.generate_curved_band((0.5, 0.9), length_uv=0.90, width_uv=0.18, curvature=0.08, num_points=12)
            vy = -0.3 * (adapter.height / 200.0)
            zlm.add_zone(
                f"p2_windfront_{self.front_counter}",
                front_shape,
                [
                    {"attribute": "vision_radius", "type": "multiply", "value": 0.35},
                    {"attribute": "speed", "type": "multiply", "value": 0.65}
                ],
                lifetime_ticks=100,
                velocity=[0.0, vy],
                log_message={"message": "Weather alert: Moving snow squall.", "level": "WARNING"}
            )
        if prev_state in ["WIND_RISE", "BLIZZARD"] and new_state not in ["WIND_RISE", "BLIZZARD"]:
            self.sastrugi_counter += 1
            r1 = adapter.generate_meandering_corridor((0.15, 0.35), (0.75, 0.35), width_uv=0.05, meander_amplitude=0.03, rng=rng)
            r2 = adapter.generate_meandering_corridor((0.25, 0.65), (0.85, 0.65), width_uv=0.05, meander_amplitude=0.03, rng=rng)
            zlm.add_zone(
                f"p2_sastrugi_{self.sastrugi_counter}_a",
                r1,
                [{"attribute": "speed", "type": "multiply", "value": 0.45}],
                lifetime_ticks=140
            )
            zlm.add_zone(
                f"p2_sastrugi_{self.sastrugi_counter}_b",
                r2,
                [{"attribute": "speed", "type": "multiply", "value": 0.45}],
                lifetime_ticks=140
            )


class CyclonicSurgePattern(WeatherPattern):
    def __init__(self, pattern_id: int = 3, name: str = "CyclonicSurge"):
        super().__init__(pattern_id, name)
        self.mirror_zone_id = f"p3_ice_mirror_{self.pattern_id}"
        self.surge_counter = 0

    def setup(self, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        mirror_pts = adapter.generate_organic_zone((0.5, 0.5), radius_uv=0.16, num_vertices=16, rng=rng)
        zlm.add_zone(
            self.mirror_zone_id,
            mirror_pts,
            [{"attribute": "position", "type": "physics", "operation": "drift", "value": 0.90}],
            log_message={"message": "Traction warning: Central ice shelf.", "level": "WARNING"}
        )

    def on_weather_transition(self, prev_state: Optional[str], new_state: str, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        if new_state == "BLIZZARD":
            self.surge_counter += 1
            arm1 = adapter.generate_organic_zone((0.2, 0.8), radius_uv=0.14, num_vertices=12, rng=rng)
            arm2 = adapter.generate_organic_zone((0.8, 0.2), radius_uv=0.14, num_vertices=12, rng=rng)
            zlm.add_zone(
                f"p3_snow_arm_nw_{self.surge_counter}",
                arm1,
                [{"attribute": "speed", "type": "multiply", "value": 0.40}],
                lifetime_ticks=110,
                expansion_rate=0.005
            )
            zlm.add_zone(
                f"p3_snow_arm_se_{self.surge_counter}",
                arm2,
                [{"attribute": "speed", "type": "multiply", "value": 0.40}],
                lifetime_ticks=110,
                expansion_rate=0.005
            )


class GlazeIceWaterwaysPattern(WeatherPattern):
    def __init__(self, pattern_id: int = 4, name: str = "GlazeIceWaterways"):
        super().__init__(pattern_id, name)
        self.main_channel_id = f"p4_main_channel_{self.pattern_id}"
        self.branch_channel_id = f"p4_branch_channel_{self.pattern_id}"
        self.patch_counter = 0

    def setup(self, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        ch1 = adapter.generate_meandering_corridor((0.05, 0.5), (0.95, 0.55), width_uv=0.07, meander_amplitude=0.06, rng=rng)
        ch2 = adapter.generate_meandering_corridor((0.48, 0.05), (0.52, 0.95), width_uv=0.07, meander_amplitude=0.05, rng=rng)
        zlm.add_zone(
            self.main_channel_id,
            ch1,
            [{"attribute": "position", "type": "physics", "operation": "drift", "value": 0.94}, {"attribute": "speed", "type": "multiply", "value": 1.05}],
            log_message={"message": "Traction warning: Glaze ice waterway.", "level": "WARNING"}
        )
        zlm.add_zone(
            self.branch_channel_id,
            ch2,
            [{"attribute": "position", "type": "physics", "operation": "drift", "value": 0.94}, {"attribute": "speed", "type": "multiply", "value": 1.05}],
            log_message={"message": "Traction warning: Glaze ice branch.", "level": "WARNING"}
        )

    def on_tick(self, tick: int, weather_info: Dict[str, Any], adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        if tick > 0 and tick % 120 == 0:
            self.patch_counter += 1
            cu = rng.uniform(0.2, 0.8)
            cv = rng.uniform(0.2, 0.8)
            patch_shape = adapter.generate_organic_zone((cu, cv), radius_uv=0.10, num_vertices=12, rng=rng)
            zlm.add_zone(
                f"p4_ice_patch_{self.patch_counter}",
                patch_shape,
                [{"attribute": "position", "type": "physics", "operation": "drift", "value": 0.93}],
                lifetime_ticks=100,
                expansion_rate=0.002
            )


class FlashFreezePattern(WeatherPattern):
    def __init__(self, pattern_id: int = 5, name: str = "FlashFreeze"):
        super().__init__(pattern_id, name)
        self.zone_ids = [f"p5_slush_zone_1_{self.pattern_id}", f"p5_slush_zone_2_{self.pattern_id}"]
        self.shapes: List[List[List[float]]] = []

    def setup(self, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        z1 = adapter.generate_organic_zone((0.35, 0.35), radius_uv=0.15, num_vertices=14, rng=rng)
        z2 = adapter.generate_organic_zone((0.65, 0.65), radius_uv=0.15, num_vertices=14, rng=rng)
        self.shapes = [z1, z2]
        zlm.add_zone(
            self.zone_ids[0],
            z1,
            [{"attribute": "speed", "type": "multiply", "value": 0.60}],
            log_message={"message": "Terrain alert: Snow crust.", "level": "INFO"}
        )
        zlm.add_zone(
            self.zone_ids[1],
            z2,
            [{"attribute": "speed", "type": "multiply", "value": 0.60}],
            log_message={"message": "Terrain alert: Snow crust.", "level": "INFO"}
        )

    def on_weather_transition(self, prev_state: Optional[str], new_state: str, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        if new_state == "THAW_GLAZE":
            for i, zid in enumerate(self.zone_ids):
                zlm.add_zone(
                    zid,
                    self.shapes[i],
                    [{"attribute": "speed", "type": "multiply", "value": 0.35}],
                    log_message={"message": "Terrain warning: Wet slush.", "level": "WARNING"}
                )
        elif prev_state == "THAW_GLAZE" and new_state == "CLEAR_FROST":
            for i, zid in enumerate(self.zone_ids):
                zlm.add_zone(
                    zid,
                    self.shapes[i],
                    [{"attribute": "position", "type": "physics", "operation": "drift", "value": 0.92}, {"attribute": "speed", "type": "multiply", "value": 1.0}],
                    log_message={"message": "Traction alert: Flash frozen ice crust.", "level": "WARNING"}
                )
        elif new_state in ["POST_STORM_SNOW", "BLIZZARD"]:
            for i, zid in enumerate(self.zone_ids):
                zlm.add_zone(
                    zid,
                    self.shapes[i],
                    [{"attribute": "speed", "type": "multiply", "value": 0.50}]
                )


class PressureRidgeChaosPattern(WeatherPattern):
    def __init__(self, pattern_id: int = 6, name: str = "PressureRidgeChaos"):
        super().__init__(pattern_id, name)
        self.ridge_zone_id = f"p6_hummock_ridge_{self.pattern_id}"
        self.shelf_w_id = f"p6_shelf_w_{self.pattern_id}"
        self.shelf_e_id = f"p6_shelf_e_{self.pattern_id}"
        self.lead_zone_id = f"p6_moving_lead_{self.pattern_id}"

    def setup(self, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        ridge_pts = adapter.generate_jagged_ridge((0.2, 0.45), (0.8, 0.55), width_uv=0.10, num_teeth=14, rng=rng)
        zlm.add_zone(
            self.ridge_zone_id,
            ridge_pts,
            [{"attribute": "speed", "type": "multiply", "value": 0.30}],
            log_message={"message": "Terrain warning: Rough ice hummocks.", "level": "WARNING"}
        )
        shelf_w = adapter.generate_organic_zone((0.15, 0.5), radius_uv=0.18, num_vertices=12, rng=rng)
        shelf_e = adapter.generate_organic_zone((0.85, 0.5), radius_uv=0.18, num_vertices=12, rng=rng)
        zlm.add_zone(
            self.shelf_w_id,
            shelf_w,
            [{"attribute": "position", "type": "physics", "operation": "drift", "value": 0.90}]
        )
        zlm.add_zone(
            self.shelf_e_id,
            shelf_e,
            [{"attribute": "position", "type": "physics", "operation": "drift", "value": 0.90}]
        )
        crack_pts = adapter.generate_meandering_corridor((0.38, 0.15), (0.62, 0.35), width_uv=0.04, meander_amplitude=0.03, rng=rng)
        zlm.add_zone(
            self.lead_zone_id,
            crack_pts,
            [{"attribute": "position", "type": "physics", "operation": "drift", "value": 0.94}, {"attribute": "speed", "type": "multiply", "value": 0.50}],
            velocity=[0.02, 0.01],
            log_message={"message": "Traction warning: Active ice lead crack.", "level": "WARNING"}
        )


PATTERN_REGISTRY = {
    1: BlizzardWakeDriftsPattern,
    2: WindfrontSastrugiPattern,
    3: CyclonicSurgePattern,
    4: GlazeIceWaterwaysPattern,
    5: FlashFreezePattern,
    6: PressureRidgeChaosPattern
}


class DriftingDiamondDustPattern(WeatherPattern):
    def __init__(self, pattern_id: int = 7, name: str = "DriftingDiamondDust"):
        super().__init__(pattern_id, name)
        self.fog_bank_ids = [f"p7_fog_bank_1_{self.pattern_id}", f"p7_fog_bank_2_{self.pattern_id}"]
        self.surge_counter = 0

    def setup(self, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        shape1 = adapter.generate_organic_zone((0.3, 0.4), radius_uv=0.12, num_vertices=12, rng=rng)
        shape2 = adapter.generate_organic_zone((0.7, 0.6), radius_uv=0.14, num_vertices=14, rng=rng)
        zlm.add_zone(
            self.fog_bank_ids[0],
            shape1,
            [{"attribute": "vision_radius", "type": "multiply", "value": 0.20}],
            velocity=[0.08, 0.05],
            log_message={"message": "Opto-sensory warning: Drifting diamond dust fog.", "level": "WARNING"}
        )
        zlm.add_zone(
            self.fog_bank_ids[1],
            shape2,
            [{"attribute": "vision_radius", "type": "multiply", "value": 0.20}],
            velocity=[-0.07, 0.06],
            log_message={"message": "Opto-sensory warning: Drifting diamond dust fog.", "level": "WARNING"}
        )

    def on_weather_transition(self, prev_state: Optional[str], new_state: str, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        if new_state == "FREEZING_FOG":
            self.surge_counter += 1
            cu = rng.uniform(0.3, 0.7)
            cv = rng.uniform(0.3, 0.7)
            extra_shape = adapter.generate_organic_zone((cu, cv), radius_uv=0.16, num_vertices=14, rng=rng)
            vx = rng.uniform(-0.1, 0.1)
            vy = rng.uniform(-0.1, 0.1)
            zlm.add_zone(
                f"p7_dense_fog_cloud_{self.surge_counter}",
                extra_shape,
                [{"attribute": "vision_radius", "type": "multiply", "value": 0.15}],
                lifetime_ticks=130,
                velocity=[round(vx, 3), round(vy, 3)],
                expansion_rate=0.002,
                log_message={"message": "Opto-sensory warning: Dense freezing fog surge.", "level": "WARNING"}
            )


class PolarTwilightDriftPattern(WeatherPattern):
    def __init__(self, pattern_id: int = 8, name: str = "PolarTwilightDrift"):
        super().__init__(pattern_id, name)
        self.drift_lake_id = f"p8_twilight_lake_{self.pattern_id}"
        self.jet_counter = 0

    def setup(self, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        lake_pts = adapter.generate_organic_zone((0.45, 0.55), radius_uv=0.15, num_vertices=14, rng=rng)
        zlm.add_zone(
            self.drift_lake_id,
            lake_pts,
            [{"attribute": "position", "type": "physics", "operation": "drift", "value": 0.88}],
            log_message={"message": "Traction alert: Twilight glaze ice.", "level": "WARNING"}
        )

    def on_weather_transition(self, prev_state: Optional[str], new_state: str, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        if new_state in ["WIND_RISE", "BLIZZARD"]:
            self.jet_counter += 1
            cu = rng.uniform(0.2, 0.8)
            jet_shape = adapter.generate_meandering_corridor((0.05, cu), (0.95, cu), width_uv=0.06, meander_amplitude=0.04, rng=rng)
            zlm.add_zone(
                f"p8_ground_drift_jet_{self.jet_counter}",
                jet_shape,
                [
                    {"attribute": "speed", "type": "multiply", "value": 0.80},
                    {"attribute": "vision_radius", "type": "multiply", "value": 0.60},
                    {"attribute": "position", "type": "physics", "operation": "drift", "value": 0.88}
                ],
                lifetime_ticks=90,
                velocity=[0.25, 0.0],
                log_message={"message": "Visibility alert: Ground blizzard jet.", "level": "INFO"}
            )

    def on_tick(self, tick: int, weather_info: Dict[str, Any], adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        if tick > 0 and tick % 150 == 0:
            self.jet_counter += 1
            cv = rng.uniform(0.15, 0.85)
            band = adapter.generate_curved_band((0.5, cv), length_uv=0.7, width_uv=0.08, curvature=0.04, num_points=10)
            zlm.add_zone(
                f"p8_twilight_band_{self.jet_counter}",
                band,
                [
                    {"attribute": "vision_radius", "type": "multiply", "value": 0.50},
                    {"attribute": "speed", "type": "multiply", "value": 0.85}
                ],
                lifetime_ticks=80,
                velocity=[-0.15, 0.0]
            )


class FlatLightWhiteoutPattern(WeatherPattern):
    def __init__(self, pattern_id: int = 9, name: str = "FlatLightWhiteout"):
        super().__init__(pattern_id, name)
        self.pit_zone_ids = [f"p9_snow_pit_{i}_{self.pattern_id}" for i in range(1, 4)]
        self.glaze_id = f"p9_flat_glaze_{self.pattern_id}"
        self.surge_counter = 0

    def setup(self, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        pit_centers = [(0.25, 0.3), (0.75, 0.35), (0.45, 0.75)]
        for i, center in enumerate(pit_centers):
            shape = adapter.generate_organic_zone(center, radius_uv=0.08, num_vertices=10, rng=rng)
            zlm.add_zone(
                self.pit_zone_ids[i],
                shape,
                [{"attribute": "speed", "type": "multiply", "value": 0.35}],
                log_message={"message": "Terrain hazard: Blind snow pit.", "level": "WARNING"}
            )
        glaze_shape = adapter.generate_organic_zone((0.8, 0.8), radius_uv=0.12, num_vertices=12, rng=rng)
        zlm.add_zone(
            self.glaze_id,
            glaze_shape,
            [{"attribute": "position", "type": "physics", "operation": "drift", "value": 0.91}]
        )

    def on_weather_transition(self, prev_state: Optional[str], new_state: str, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        if new_state in ["BLIZZARD", "FREEZING_FOG"]:
            self.surge_counter += 1
            cu = rng.uniform(0.2, 0.8)
            cv = rng.uniform(0.2, 0.8)
            whiteout_veil = adapter.generate_organic_zone((cu, cv), radius_uv=0.22, num_vertices=14, rng=rng)
            zlm.add_zone(
                f"p9_whiteout_veil_{self.surge_counter}",
                whiteout_veil,
                [{"attribute": "vision_radius", "type": "multiply", "value": 0.40}],
                lifetime_ticks=120,
                expansion_rate=0.003,
                velocity=[0.03, -0.02],
                log_message={"message": "Opto-sensory alert: Flat-light whiteout veil.", "level": "WARNING"}
            )
            extra_pit = adapter.generate_organic_zone((rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)), radius_uv=0.07, num_vertices=10, rng=rng)
            zlm.add_zone(
                f"p9_masked_pit_{self.surge_counter}",
                extra_pit,
                [{"attribute": "speed", "type": "multiply", "value": 0.30}],
                lifetime_ticks=150,
                log_message={"message": "Terrain hazard: Freshly covered snow pit.", "level": "WARNING"}
            )


class WindTunnelGorgePattern(WeatherPattern):
    def __init__(self, pattern_id: int = 10, name: str = "WindTunnelGorge"):
        super().__init__(pattern_id, name)
        self.gorge_corridor_id = f"p10_gorge_corridor_{self.pattern_id}"
        self.outlet_fan_id = f"p10_outlet_fan_{self.pattern_id}"
        self.ice_scour_id = f"p10_ice_scour_{self.pattern_id}"
        self.gust_counter = 0

    def setup(self, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        corridor_pts = adapter.generate_meandering_corridor((0.08, 0.45), (0.75, 0.55), width_uv=0.07, meander_amplitude=0.07, rng=rng)
        zlm.add_zone(
            self.gorge_corridor_id,
            corridor_pts,
            [
                {"attribute": "position", "type": "physics", "operation": "drift", "value": 0.95},
                {"attribute": "speed", "type": "multiply", "value": 1.15}
            ],
            log_message={"message": "Wind warning: Gorge wind tunnel jet.", "level": "WARNING"}
        )
        fan_pts = adapter.generate_organic_zone((0.85, 0.55), radius_uv=0.18, num_vertices=14, rng=rng)
        zlm.add_zone(
            self.outlet_fan_id,
            fan_pts,
            [{"attribute": "speed", "type": "multiply", "value": 0.40}],
            log_message={"message": "Terrain alert: Wind tunnel deposition fan.", "level": "INFO"}
        )
        scour_pts = adapter.generate_organic_zone((0.35, 0.48), radius_uv=0.09, num_vertices=12, rng=rng)
        zlm.add_zone(
            self.ice_scour_id,
            scour_pts,
            [{"attribute": "position", "type": "physics", "operation": "drift", "value": 0.94}]
        )

    def on_weather_transition(self, prev_state: Optional[str], new_state: str, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        if new_state in ["WIND_RISE", "BLIZZARD"]:
            self.gust_counter += 1
            cloud = adapter.generate_curved_band((0.45, 0.52), length_uv=0.40, width_uv=0.08, curvature=0.05, num_points=10)
            zlm.add_zone(
                f"p10_gorge_gust_{self.gust_counter}",
                cloud,
                [{"attribute": "vision_radius", "type": "multiply", "value": 0.35}],
                lifetime_ticks=100,
                velocity=[0.18, 0.02],
                log_message={"message": "Opto-sensory alert: Blinding snow gust in gorge.", "level": "WARNING"}
            )


class PermafrostMuskegPattern(WeatherPattern):
    def __init__(self, pattern_id: int = 11, name: str = "PermafrostMuskeg"):
        super().__init__(pattern_id, name)
        self.hummock_ids = [f"p11_hummock_{i}_{self.pattern_id}" for i in range(1, 4)]
        self.squall_counter = 0
        self.powder_counter = 0

    def setup(self, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        centers = [(0.25, 0.30), (0.75, 0.70), (0.35, 0.75)]
        radii = [0.20, 0.22, 0.18]
        for i in range(3):
            pts = adapter.generate_organic_zone(centers[i], radius_uv=radii[i], num_vertices=14, rng=rng)
            zlm.add_zone(
                self.hummock_ids[i],
                pts,
                [{"attribute": "speed", "type": "multiply", "value": 0.45}],
                log_message={"message": "Terrain alert: Rugged permafrost hummocks.", "level": "INFO"}
            )

    def on_weather_transition(self, prev_state: Optional[str], new_state: str, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        if new_state in ["WIND_RISE", "BLIZZARD"]:
            self.squall_counter += 1
            squall_shape = adapter.generate_curved_band((0.5, 0.8), length_uv=0.75, width_uv=0.16, curvature=0.06, num_points=12)
            zlm.add_zone(
                f"p11_muskeg_squall_{self.squall_counter}",
                squall_shape,
                [
                    {"attribute": "speed", "type": "multiply", "value": 0.70},
                    {"attribute": "vision_radius", "type": "multiply", "value": 0.35}
                ],
                lifetime_ticks=100,
                velocity=[-0.15, -0.15],
                log_message={"message": "Weather alert: Muskeg snow squall.", "level": "WARNING"}
            )
        elif prev_state in ["WIND_RISE", "BLIZZARD"] and new_state == "POST_STORM_SNOW":
            for _ in range(2):
                self.powder_counter += 1
                cu = rng.uniform(0.2, 0.8)
                cv = rng.uniform(0.2, 0.8)
                pocket = adapter.generate_organic_zone((cu, cv), radius_uv=0.08, num_vertices=10, rng=rng)
                zlm.add_zone(
                    f"p11_powder_pocket_{self.powder_counter}",
                    pocket,
                    [{"attribute": "speed", "type": "multiply", "value": 0.25}],
                    lifetime_ticks=140,
                    log_message={"message": "Terrain alert: Fresh fluffy snow trap.", "level": "INFO"}
                )


class FastIceLeadEdgePattern(WeatherPattern):
    def __init__(self, pattern_id: int = 12, name: str = "FastIceLeadEdge"):
        super().__init__(pattern_id, name)
        self.fast_ice_id = f"p12_fast_ice_shelf_{self.pattern_id}"
        self.thin_edge_id = f"p12_thin_lead_edge_{self.pattern_id}"
        self.sea_smoke_id = f"p12_sea_smoke_{self.pattern_id}"

    def setup(self, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        shelf_pts = adapter.generate_organic_zone((0.2, 0.5), radius_uv=0.28, num_vertices=16, rng=rng)
        zlm.add_zone(
            self.fast_ice_id,
            shelf_pts,
            [{"attribute": "position", "type": "physics", "operation": "drift", "value": 0.85}]
        )
        edge_pts = adapter.generate_meandering_corridor((0.45, 0.05), (0.55, 0.95), width_uv=0.08, meander_amplitude=0.07, rng=rng)
        zlm.add_zone(
            self.thin_edge_id,
            edge_pts,
            [
                {"attribute": "position", "type": "physics", "operation": "drift", "value": 0.95},
                {"attribute": "speed", "type": "multiply", "value": 0.50}
            ],
            log_message={"message": "Traction alert: Thin ice along lead edge.", "level": "WARNING"}
        )
        smoke_pts = adapter.generate_organic_zone((0.6, 0.4), radius_uv=0.15, num_vertices=14, rng=rng)
        zlm.add_zone(
            self.sea_smoke_id,
            smoke_pts,
            [{"attribute": "vision_radius", "type": "multiply", "value": 0.25}],
            velocity=[-0.06, 0.02],
            expansion_rate=0.001,
            log_message={"message": "Opto-sensory warning: Arctic sea smoke.", "level": "WARNING"}
        )


class SnowbankAvalancheCreepPattern(WeatherPattern):
    def __init__(self, pattern_id: int = 13, name: str = "SnowbankAvalancheCreep"):
        super().__init__(pattern_id, name)
        self.avalanche_toe_id = f"p13_avalanche_toe_{self.pattern_id}"
        self.scour_id = f"p13_scoured_bedrock_{self.pattern_id}"
        self.sluff_counter = 0

    def setup(self, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        toe_pts = adapter.generate_organic_zone((0.80, 0.85), radius_uv=0.18, num_vertices=14, rng=rng)
        zlm.add_zone(
            self.avalanche_toe_id,
            toe_pts,
            [{"attribute": "speed", "type": "multiply", "value": 0.30}],
            velocity=[-0.04, -0.04],
            expansion_rate=0.003,
            log_message={"message": "Terrain warning: Creeping avalanche debris cone.", "level": "WARNING"}
        )
        scour_pts = adapter.generate_organic_zone((0.88, 0.90), radius_uv=0.14, num_vertices=12, rng=rng)
        zlm.add_zone(
            self.scour_id,
            scour_pts,
            [{"attribute": "position", "type": "physics", "operation": "drift", "value": 0.93}],
            log_message={"message": "Traction warning: Scoured bedrock ice slope.", "level": "WARNING"}
        )

    def on_weather_transition(self, prev_state: Optional[str], new_state: str, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        if new_state == "BLIZZARD":
            self.sluff_counter += 1
            sluff = adapter.generate_curved_band((0.65, 0.70), length_uv=0.35, width_uv=0.10, curvature=0.04, num_points=10)
            zlm.add_zone(
                f"p13_sluff_slide_{self.sluff_counter}",
                sluff,
                [{"attribute": "speed", "type": "multiply", "value": 0.25}],
                lifetime_ticks=80,
                velocity=[-0.12, -0.10],
                log_message={"message": "Hazard alert: Fresh secondary sluff slide.", "level": "WARNING"}
            )


class FullStochasticArcticPattern(WeatherPattern):
    def __init__(self, pattern_id: int = 14, name: str = "FullStochasticArctic"):
        super().__init__(pattern_id, name)
        self.river_id = f"p14_glacial_river_{self.pattern_id}"
        self.floe1_id = f"p14_drift_floe_1_{self.pattern_id}"
        self.floe2_id = f"p14_drift_floe_2_{self.pattern_id}"
        self.squall_counter = 0
        self.drift_counter = 0

    def setup(self, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        river_pts = adapter.generate_meandering_corridor((0.1, 0.1), (0.9, 0.9), width_uv=0.06, meander_amplitude=0.08, rng=rng)
        zlm.add_zone(
            self.river_id,
            river_pts,
            [{"attribute": "position", "type": "physics", "operation": "drift", "value": 0.90}],
            log_message={"message": "Traction warning: Glacial braided melt river.", "level": "WARNING"}
        )
        floe1 = adapter.generate_organic_zone((0.3, 0.7), radius_uv=0.13, num_vertices=14, rng=rng)
        floe2 = adapter.generate_organic_zone((0.7, 0.3), radius_uv=0.13, num_vertices=14, rng=rng)
        zlm.add_zone(
            self.floe1_id,
            floe1,
            [
                {"attribute": "speed", "type": "multiply", "value": 0.50},
                {"attribute": "position", "type": "physics", "operation": "drift", "value": 0.91}
            ]
        )
        zlm.add_zone(
            self.floe2_id,
            floe2,
            [
                {"attribute": "speed", "type": "multiply", "value": 0.50},
                {"attribute": "position", "type": "physics", "operation": "drift", "value": 0.91}
            ]
        )

    def on_weather_transition(self, prev_state: Optional[str], new_state: str, adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        if prev_state == "BLIZZARD" and new_state == "POST_STORM_SNOW":
            for _ in range(2):
                self.drift_counter += 1
                cu = rng.uniform(0.15, 0.85)
                cv = rng.uniform(0.15, 0.85)
                shape = adapter.generate_organic_zone((cu, cv), radius_uv=0.10, num_vertices=12, rng=rng)
                zlm.add_zone(
                    f"p14_leeward_drift_{self.drift_counter}",
                    shape,
                    [{"attribute": "speed", "type": "multiply", "value": 0.30}],
                    lifetime_ticks=160,
                    log_message={"message": "Terrain alert: Leeward snow accumulation.", "level": "INFO"}
                )

    def on_tick(self, tick: int, weather_info: Dict[str, Any], adapter: SpatialZoneAdapter, zlm: ZoneLifecycleManager, rng: random.Random):
        if tick > 0 and tick % 140 == 0:
            self.squall_counter += 1
            theta = 0.005 * tick
            vx = 0.10 * math.cos(theta)
            vy = 0.10 * math.sin(theta)
            cu = rng.uniform(0.3, 0.7)
            cv = rng.uniform(0.3, 0.7)
            squall_shape = adapter.generate_curved_band((cu, cv), length_uv=0.60, width_uv=0.12, curvature=0.05, num_points=10)
            zlm.add_zone(
                f"p14_dynamic_squall_{self.squall_counter}",
                squall_shape,
                [
                    {"attribute": "vision_radius", "type": "multiply", "value": 0.30},
                    {"attribute": "speed", "type": "multiply", "value": 0.70}
                ],
                lifetime_ticks=100,
                velocity=[round(vx, 3), round(vy, 3)],
                log_message={"message": "Weather alert: Dynamic arctic squall front.", "level": "WARNING"}
            )


PATTERN_REGISTRY = {
    1: BlizzardWakeDriftsPattern,
    2: WindfrontSastrugiPattern,
    3: CyclonicSurgePattern,
    4: GlazeIceWaterwaysPattern,
    5: FlashFreezePattern,
    6: PressureRidgeChaosPattern,
    7: DriftingDiamondDustPattern,
    8: PolarTwilightDriftPattern,
    9: FlatLightWhiteoutPattern,
    10: WindTunnelGorgePattern,
    11: PermafrostMuskegPattern,
    12: FastIceLeadEdgePattern,
    13: SnowbankAvalancheCreepPattern,
    14: FullStochasticArcticPattern
}


def create_pattern(pattern_id: int) -> WeatherPattern:
    if pattern_id in PATTERN_REGISTRY:
        return PATTERN_REGISTRY[pattern_id]()
    raise ValueError(f"Unknown pattern id: {pattern_id}")


class ArcticWeatherComposite:
    def __init__(self, patterns: List[int], seed: int = 42, map_boundaries: Optional[Tuple[List[float], List[float]]] = None, padding: float = 10.0):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.state_machine = WeatherStateMachine(seed=self.seed)
        self.adapter = SpatialZoneAdapter(map_boundaries=map_boundaries, padding=padding)
        self.zlm = ZoneLifecycleManager(self.adapter)
        self.patterns = [create_pattern(pid) for pid in patterns]
        for p in self.patterns:
            p.setup(self.adapter, self.zlm, self.rng)

    def step(self) -> Dict[str, Any]:
        info = self.state_machine.step()
        if info["transition"]:
            for p in self.patterns:
                p.on_weather_transition(info["previous_state"], info["state"], self.adapter, self.zlm, self.rng)
        for p in self.patterns:
            p.on_tick(self.state_machine.total_ticks, info, self.adapter, self.zlm, self.rng)
        expired_zones = self.zlm.step()
        info["expired_zones"] = expired_zones
        return info

    def get_environment_rules(self) -> List[Dict[str, Any]]:
        rules = [self.state_machine.get_current_rule()]
        rules.extend(self.zlm.get_environment_rules())
        return rules


class WeatherAttachedScenario:
    def __init__(self, inner_scenario: Any, weather_scenario: Any):
        self.inner_scenario = inner_scenario
        self.weather_scenario = weather_scenario
        self.stage = getattr(inner_scenario, "stage", "peace")
        self.terminal_stages = getattr(inner_scenario, "terminal_stages", ["completed", "failed"])

    def setup(self):
        if self.inner_scenario and hasattr(self.inner_scenario, "setup"):
            self.inner_scenario.setup()
        if self.weather_scenario and hasattr(self.weather_scenario, "setup"):
            self.weather_scenario.setup()

    def tick(self, world):
        if self.inner_scenario and hasattr(self.inner_scenario, "tick"):
            self.inner_scenario.tick(world)
            self.stage = getattr(self.inner_scenario, "stage", self.stage)
        if self.weather_scenario and hasattr(self.weather_scenario, "tick"):
            self.weather_scenario.tick(world)

    def __getattr__(self, name: str) -> Any:
        if self.inner_scenario and hasattr(self.inner_scenario, name):
            return getattr(self.inner_scenario, name)
        if self.weather_scenario and hasattr(self.weather_scenario, name):
            return getattr(self.weather_scenario, name)
        raise AttributeError(f"Attribute '{name}' not found on WeatherAttachedScenario")


class ArcticWeatherScenario:
    patterns: Optional[List[int]] = None
    seed: int = 42

    def __init__(self, api: Any, auto_robot: bool = True, weather_config: Optional[Dict[str, Any]] = None):
        self.api = api
        self.world = getattr(api, "world", None)
        self.auto_robot = auto_robot
        self.stage = "peace"
        self.terminal_stages = ["completed", "failed"]
        self.weather_config = weather_config
        self.weather = None
        self.env = None

    def setup(self):
        patterns = self.patterns
        seed = self.seed
        enabled = True
        padding = 10.0
        if self.weather_config is not None:
            enabled = self.weather_config.get("enabled", True)
            patterns = self.weather_config.get("patterns", patterns)
            seed = self.weather_config.get("seed", seed)
            padding = self.weather_config.get("padding", padding)
        elif patterns is None and self.world and hasattr(self.world, "config") and isinstance(self.world.config, dict):
            cfg = self.world.config.get("weather", {})
            enabled = cfg.get("enabled", True)
            patterns = cfg.get("patterns", None)
            seed = cfg.get("seed", seed)
            padding = cfg.get("padding", padding)

        if enabled and patterns:
            bounds = getattr(self.world, "map_boundaries", None)
            self.weather = ArcticWeatherComposite(patterns=patterns, seed=seed, map_boundaries=bounds, padding=padding)
            self.env = EnvironmentManager(self.world, {"rules": []})
            self.env.rules = self.weather.get_environment_rules()
            self.env.process_environment()
            if hasattr(self.world, "robots") and isinstance(self.world.robots, dict):
                for r_id in list(self.world.robots.keys()):
                    self.env.register_robot(r_id)
                    if self.api:
                        self.env.sync_virtual_zones(self.api, r_id)

    def tick(self, world):
        if self.weather and self.env:
            self.weather.step()
            self.env.rules = self.weather.get_environment_rules()
            self.env.process_environment()
            if hasattr(self.world, "robots") and isinstance(self.world.robots, dict):
                for r_id in list(self.world.robots.keys()):
                    if r_id not in self.env._log_throttle:
                        self.env.register_robot(r_id)
                    if self.api:
                        self.env.sync_virtual_zones(self.api, r_id)


WeatherScenario = ArcticWeatherScenario


def create_weather_scenario(patterns: Optional[List[int]] = None, seed: int = 42):
    class ConfiguredArcticScenario(ArcticWeatherScenario):
        def __init__(self, api: Any, auto_robot: bool = True):
            cfg = {"enabled": bool(patterns), "patterns": patterns or [], "seed": seed}
            super().__init__(api, auto_robot=auto_robot, weather_config=cfg)
    return ConfiguredArcticScenario
