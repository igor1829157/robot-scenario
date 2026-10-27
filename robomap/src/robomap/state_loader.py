import json
import os
import copy
from robomap.world import World
from robomap.models import Robot, WorldObject, MemoryObject

import logging
logger = logging.getLogger(__name__)

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def deep_merge(base: dict, update: dict) -> dict:
    """Рекурсивно сливает два словаря. update накладывается поверх base."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_and_merge_configs(global_config_path: str, scenario_name: str = None) -> dict:
    """
    Загружает глобальный конфиг. Если указан сценарий, ищет его локальный 
    config.json и накладывает поверх глобального (Deep Merge).
    """
    abs_global = os.path.join(project_path, global_config_path) if not os.path.isabs(global_config_path) else global_config_path
    
    global_config = {}
    if os.path.exists(abs_global):
        with open(abs_global, 'r', encoding='utf-8') as f:
            global_config = json.load(f)
            
    if not scenario_name or scenario_name in ["empty", "sandbox", "none", ""]:
        return global_config
        
    scenarios_dir = global_config.get("paths", {}).get("scenarios_dir", "data/scenarios")
    scenario_config_path = os.path.join(project_path, scenarios_dir, scenario_name, "config.json")
    
    if os.path.exists(scenario_config_path):
        try:
            with open(scenario_config_path, 'r', encoding='utf-8') as f:
                scenario_config = json.load(f)
            
            merged = copy.deepcopy(global_config)
            merged = deep_merge(merged, scenario_config)
            logger.info("STATE LOADER | Загружен и применен локальный конфиг сценария: %s", scenario_name)
            return merged
        except Exception as e:
            logger.error("STATE LOADER | Ошибка слияния конфига сценария '%s': %s", scenario_name, str(e))
            
    return global_config


def enrich_object_data(obj_data: dict, objects_meta: dict) -> dict:
    obj_id = str(obj_data.get('id', ''))
    meta = objects_meta.get(obj_id, {})
    
    enriched = meta.copy()
    enriched.update(obj_data)
    
    if "inventory" in enriched and isinstance(enriched["inventory"], list):
        enriched_inventory = []
        for sub_item in enriched["inventory"]:
            if isinstance(sub_item, dict):
                enriched_inventory.append(enrich_object_data(sub_item, objects_meta))
        enriched["inventory"] = enriched_inventory
        
    return enriched


def enrich_memory_data(m_data: dict, objects_meta: dict) -> dict:
    real_id = str(m_data.get('real_id', m_data.get('id', '')))
    meta = objects_meta.get(real_id, {})
    
    enriched = meta.copy()
    enriched.update(m_data)
    
    if "inventory" in enriched and isinstance(enriched["inventory"], list):
        enriched_inventory = []
        for sub_mem_item in enriched["inventory"]:
            if isinstance(sub_mem_item, dict):
                enriched_inventory.append(enrich_memory_data(sub_mem_item, objects_meta))
        enriched["inventory"] = enriched_inventory
        
    return enriched


def load_state_to_world(world: World, state_filepath: str, objects_meta_filepath: str = "data/objects/objects.json"):
    """Загружает JSON-состояние в ООП-мир, подтягивая метаданные."""
    state_filepath = os.path.join(project_path, state_filepath) if not os.path.isabs(state_filepath) else state_filepath
    objects_meta_filepath = os.path.join(project_path, objects_meta_filepath) if not os.path.isabs(objects_meta_filepath) else objects_meta_filepath
    
    world.robots = {}
    eps = world.config.get("clustering", {}).get("eps", 0.0)

    if not state_filepath or not os.path.exists(state_filepath):
        logger.warning("STATE LOADER | Initial state file '%s' not found. Creating empty world.", state_filepath)
        world.robots["robot_1"] = Robot(obj_id="robot_1", x=0.0, y=0.0, vision_radius=5.0, eps=eps)
        return
        
    with open(state_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    objects_meta = {}
    if os.path.exists(objects_meta_filepath):
        with open(objects_meta_filepath, 'r', encoding='utf-8') as f:
            meta_list = json.load(f)
            objects_meta = {str(item.get('id')): item for item in meta_list}

    world.real_map_data = []
    for obj_data in data.get("real_map_data", []):
        enriched_data = enrich_object_data(obj_data, objects_meta)
        obj = WorldObject.from_dict(enriched_data)
        world.real_map_data.append(obj)

        if obj.x is not None and obj.y is not None:
            poly = obj.get_polygon()
            if world.check_collision(poly, ignore_ids=[obj.id]):
                logger.warning(
                    "MAP LOAD WARNING | Объект '%s:%s' на карте размещен с наложением на другие объекты в координатах (%.1f, %.1f)!",
                    obj.type_name, obj.id, obj.x, obj.y
                )

    robots_data_list = data.get("robots_data", [])

    for r_data in robots_data_list:
        r_id = str(r_data.get("id", f"robot_{len(world.robots)+1}"))
        
        robot = Robot(
            obj_id=r_id, 
            x=r_data.get('x', 0.0), 
            y=r_data.get('y', 0.0), 
            vision_radius=r_data.get('radius', 5.0),
            angle=r_data.get('angle', 0.0),
            fov=r_data.get('fov', 90.0),
            rotation_speed=r_data.get('rotation_speed', 15.0),
            eps=eps
        )

        if robot.x is not None and robot.y is not None:
            poly = robot.get_polygon()
            if world.check_collision(poly, ignore_ids=[robot.id]):
                logger.warning(
                    "MAP LOAD WARNING | Робот '%s' на карте размещен с наложением на другие объекты в координатах (%.1f, %.1f)!",
                    robot.id, robot.x, robot.y
                )
        
        robot.memory = []
        for m_data in r_data.get("memory_data", []):
            enriched_m_data = enrich_memory_data(m_data, objects_meta)
            mem_obj = MemoryObject.from_dict(enriched_m_data)
            
            if enriched_m_data.get('shape') or mem_obj.absolute_wkt:
                mem_obj.is_fully_explored = True

            if not mem_obj.absolute_wkt and mem_obj.x is not None and mem_obj.y is not None:
                shape_coords = enriched_m_data.get('shape', [])
                if shape_coords and len(shape_coords) >= 3:
                    try:
                        from shapely.geometry import Polygon
                        import math
                        
                        angle = float(enriched_m_data.get("angle", 0.0))
                        rad = math.radians(-angle)
                        cos_a = math.cos(rad)
                        sin_a = math.sin(rad)
                        
                        coords = []
                        for dx, dy in shape_coords:
                            rot_dx = dx * cos_a - dy * sin_a
                            rot_dy = dx * sin_a + dy * cos_a
                            coords.append((mem_obj.x + rot_dx, mem_obj.y + rot_dy))
                            
                        poly = Polygon(coords)
                        mem_obj.absolute_wkt = poly.wkt
                        mem_obj.absolute_shape = World._extract_coords_from_poly(poly)
                    except Exception:
                        pass
            
            robot.memory.append(mem_obj)
            
        world.robots[r_id] = robot

    world._calculate_boundaries()
    
    for robot in world.robots.values():
        world._update_robot_memory(robot)