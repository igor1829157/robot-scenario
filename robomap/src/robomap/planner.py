import math
import heapq
from typing import List, Tuple, Any, Optional
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union, nearest_points
from shapely.prepared import prep

class PathPlanner:
    def __init__(self, movement_step: float, physical_radius: float):
        self.max_step: float = movement_step
        self.physical_radius: float = physical_radius
        
        if self.physical_radius > 0:
            self.grid_res: float = max(0.4, self.physical_radius)
        else:
            self.grid_res: float = 0.5
            
        self.current_path: List[Tuple[float, float]] = []      
        self.current_target: Optional[Tuple[float, float]] = None  
        self.current_speed: float = 0.0    
        
        self.strict_physics: Polygon = Polygon()
        self.physical_space: Polygon = Polygon()
        self.obstacle_space: Polygon = Polygon() 
        self.prepared_space: Any = None 
        self.prepared_smoothing_space: Any = None
        self.obs_bounds: Optional[Tuple[float, float, float, float]] = None

    def _build_obstacle_space(self, obstacles: List[Any]):
        polygons = []
        for obj in obstacles:
            if getattr(obj, 'x', None) is None or getattr(obj, 'y', None) is None:
                continue
            if hasattr(obj, 'get_polygon'):
                poly = obj.get_polygon()
                if poly and not poly.is_empty and poly.is_valid:
                    polygons.append(poly)

        if not polygons:
            self.strict_physics = Polygon()
            self.physical_space = Polygon()
            self.obstacle_space = Polygon()
            self.prepared_space = None
            self.prepared_smoothing_space = None
            self.obs_bounds = None
        else:
            clean_polys = [p.buffer(0) for p in polygons if p.is_valid]
            
            self.strict_physics = unary_union(clean_polys)
            self.physical_space = self.strict_physics
            self.obstacle_space = self.strict_physics.buffer(self.physical_radius + 0.15)
            self.prepared_space = prep(self.obstacle_space)
            self.obs_bounds = self.obstacle_space.bounds 
            smooth_space = self.strict_physics.buffer(self.physical_radius + 0.05)
            self.prepared_smoothing_space = prep(smooth_space)

    def _is_valid(self, x: float, y: float) -> bool:
        if not self.prepared_space or not self.obs_bounds:
            return True
        minx, miny, maxx, maxy = self.obs_bounds
        if x < minx or x > maxx or y < miny or y > maxy:
            return True
        return not self.prepared_space.intersects(Point(x, y))

    def _line_of_sight(self, p1: Tuple[float, float], p2: Tuple[float, float], use_smoothing_space: bool = False) -> bool:
        space = self.prepared_smoothing_space if use_smoothing_space else self.prepared_space
        if not space:
            return True
            
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        
        if dist <= 0.01:
            return True

        trim_dist = min(dist * 0.4, self.physical_radius * 0.4)
        start_x = p1[0] + (dx / dist) * trim_dist
        start_y = p1[1] + (dy / dist) * trim_dist
        
        line = LineString([(start_x, start_y), p2])
        return not space.intersects(line)

    def _raycast_to_target(self, current_pos: Tuple[float, float], target_x: float, target_y: float) -> Optional[Tuple[float, float]]:
        if self._is_valid(target_x, target_y):
            return target_x, target_y

        if not hasattr(self, 'physical_space') or self.physical_space.is_empty:
            return target_x, target_y

        line = LineString([current_pos, (target_x, target_y)])
        intersection = self.physical_space.intersection(line)
        
        if intersection.is_empty:
            return target_x, target_y
            
        _, hit_pt = nearest_points(Point(current_pos), intersection)
        
        hx, hy = hit_pt.x, hit_pt.y
        cx, cy = current_pos[0], current_pos[1]
        dist = math.hypot(hx - cx, hy - cy)
        
        retreat_dist = self.physical_radius + 0.20
        if dist > retreat_dist:
            ratio = (dist - retreat_dist) / dist
            return (cx + (hx - cx) * ratio, cy + (hy - cy) * ratio)
        else:
            return current_pos

    def _find_closest_grid_node(self, current_pos: Tuple[float, float], target_x: float, target_y: float) -> Optional[Tuple[float, float]]:
        if self._is_valid(target_x, target_y):
            return target_x, target_y
            
        def round_point(x_val, y_val):
            return round(x_val/self.grid_res)*self.grid_res, round(y_val/self.grid_res)*self.grid_res
            
        tx, ty = round_point(target_x, target_y)
        r = self.grid_res
        max_search_cells = int(8.0 / self.grid_res)
        
        valid_candidates = []
        for max_rad in range(1, max_search_cells + 1): 
            for ix in range(-max_rad, max_rad + 1):
                for iy in range(-max_rad, max_rad + 1):
                    if abs(ix) == max_rad or abs(iy) == max_rad:
                        cx = tx + ix * r
                        cy = ty + iy * r
                        if self._is_valid(cx, cy):
                            line = LineString([current_pos, (cx, cy)])
                            if not hasattr(self, 'strict_physics') or self.strict_physics.is_empty or not self.strict_physics.intersects(line):
                                valid_candidates.append((cx, cy))
            
            if valid_candidates:
                valid_candidates.sort(key=lambda p: math.hypot(p[0]-current_pos[0], p[1]-current_pos[1]))
                return valid_candidates[0]
                
        return None

    def _a_star(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1]) * 1.5

        def round_point(p):
            return (round(p[0]/self.grid_res)*self.grid_res, round(p[1]/self.grid_res)*self.grid_res)

        start_grid = round_point(start)
        goal_grid = round_point(goal)
        
        if not self._is_valid(start_grid[0], start_grid[1]):
            nearest = self._find_closest_grid_node(start, start_grid[0], start_grid[1])
            if not nearest: return []
            start_grid = round_point(nearest)

        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}

        valid_cache = {start_grid: True}
        closed_set = set()

        closest_node = start_grid
        min_dist_to_goal = math.hypot(start_grid[0] - goal_grid[0], start_grid[1] - goal_grid[1])

        neighbors = [(0, self.grid_res), (0, -self.grid_res), (self.grid_res, 0), (-self.grid_res, 0),
                     (self.grid_res, self.grid_res), (self.grid_res, -self.grid_res), 
                     (-self.grid_res, self.grid_res), (-self.grid_res, -self.grid_res)]

        nodes_expanded = 0
        max_nodes = 5000

        while open_set:
            nodes_expanded += 1
            if nodes_expanded > max_nodes:
                break

            _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            dist_to_goal = math.hypot(current[0] - goal_grid[0], current[1] - goal_grid[1])
            if dist_to_goal < min_dist_to_goal:
                min_dist_to_goal = dist_to_goal
                closest_node = current

            if dist_to_goal < self.grid_res * 1.5:
                path = [goal]
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dx, dy in neighbors:
                neighbor = (round(current[0] + dx, 3), round(current[1] + dy, 3))
                
                if neighbor in closed_set:
                    continue
                
                if neighbor not in valid_cache:
                    valid_cache[neighbor] = self._is_valid(neighbor[0], neighbor[1])
                
                if not valid_cache[neighbor]:
                    continue

                step_cost = math.hypot(dx, dy)
                tentative_g_score = g_score[current] + step_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score, neighbor))

        if closest_node == start_grid:
            return []

        path = [closest_node]
        current = closest_node
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def _smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(path) <= 2: return path
        
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            furthest_visible_idx = current_idx + 1
            for i in range(current_idx + 2, len(path)):
                if self._line_of_sight(smoothed[-1], path[i], use_smoothing_space=True):
                    furthest_visible_idx = i
                else:
                    break
            smoothed.append(path[furthest_visible_idx])
            current_idx = furthest_visible_idx
            
        return smoothed

    def _get_path_length(self, current_x: float, current_y: float) -> float:
        if not self.current_path: return 0.0
        dist = math.hypot(self.current_path[0][0] - current_x, self.current_path[0][1] - current_y)
        for i in range(len(self.current_path) - 1):
            dist += math.hypot(self.current_path[i+1][0] - self.current_path[i][0], 
                               self.current_path[i+1][1] - self.current_path[i][1])
        return dist

    def is_path_blocked(self, current_pos: Tuple[float, float], obstacles: List[Any]) -> bool:
        if not self.current_path:
            return False

        dist_to_end = math.hypot(self.current_path[-1][0] - current_pos[0], self.current_path[-1][1] - current_pos[1])
        if dist_to_end <= self.physical_radius * 1.5:
            return False

        next_point = self.current_path[0]
        dx = next_point[0] - current_pos[0]
        dy = next_point[1] - current_pos[1]
        dist = math.hypot(dx, dy)
        
        if dist > (self.physical_radius * 0.3):
            trim = self.physical_radius * 0.3
            check_start = (current_pos[0] + (dx/dist)*trim, current_pos[1] + (dy/dist)*trim)
        else:
            check_start = current_pos

        pts = [check_start] + self.current_path[:2]
        path_line = LineString(pts)
        
        collision_threshold = self.physical_radius + 0.02
        next_wp = Point(self.current_path[0])

        for obj in obstacles:
            if getattr(obj, 'x', None) is None: 
                continue
            if hasattr(obj, 'get_polygon'):
                poly = obj.get_polygon()
                if poly and not poly.is_empty and poly.is_valid:
                    if poly.distance(path_line) < collision_threshold:
                        buf_poly = poly.buffer(collision_threshold)
                        if buf_poly.contains(next_wp):
                            return True
                        inter = buf_poly.intersection(path_line)
                        if inter.length > 0.05:
                            return True

        return False

    def set_target(self, current_pos: Tuple[float, float], target_x: float, target_y: float, obstacles: List[Any]):
        self._build_obstacle_space(obstacles)
        
        safe_target = self._raycast_to_target(current_pos, target_x, target_y)
        final_target = safe_target if safe_target else (target_x, target_y)
        self.current_target = final_target
        
        if safe_target and self._line_of_sight(current_pos, safe_target, use_smoothing_space=True):
            self.current_path = [safe_target]
        else:
            raw_path = self._a_star(current_pos, final_target)
            self.current_path = self._smooth_path(raw_path)

    def get_next_step(self, current_x: float, current_y: float) -> Tuple[float, float, List[Tuple[float, float]]]:
        if not self.current_path:
            self.current_speed = 0.0
            return 0.0, 0.0, []

        while self.current_path:
            dist_to_start = math.hypot(self.current_path[0][0] - current_x, self.current_path[0][1] - current_y)
            if dist_to_start <= 0.05:
                self.current_path.pop(0)
            else:
                break

        if not self.current_path:
            self.current_speed = 0.0
            self.current_target = None
            return 0.0, 0.0, []

        dist_to_final = self._get_path_length(current_x, current_y)
        accel_step = self.max_step / 2.0
        
        if dist_to_final <= self.max_step:
            desired_speed = dist_to_final
        elif dist_to_final <= self.max_step * 2.0:
            desired_speed = self.max_step / 2.0
        else:
            desired_speed = self.max_step

        if self.current_speed < desired_speed:
            self.current_speed = min(self.current_speed + accel_step, desired_speed)
        elif self.current_speed > desired_speed:
            self.current_speed = max(self.current_speed - accel_step, desired_speed)

        next_wp = self.current_path[0]
        dist_to_wp = math.hypot(next_wp[0] - current_x, next_wp[1] - current_y)

        if self.current_speed >= dist_to_wp:
            dx = next_wp[0] - current_x
            dy = next_wp[1] - current_y
            self.current_path.pop(0) 
            
            if not self.current_path:
                self.current_target = None
                self.current_speed = 0.0
                
            return dx, dy, list(self.current_path)
            
        else:
            ratio = self.current_speed / dist_to_wp
            dx = (next_wp[0] - current_x) * ratio
            dy = (next_wp[1] - current_y) * ratio
            return dx, dy, list(self.current_path)