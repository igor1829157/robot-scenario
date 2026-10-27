import math
from typing import List, Dict, Any, Optional, Tuple, Union
from shapely.ops import unary_union
from shapely.geometry import Polygon, Point, shape
import shapely.wkt

from robomap.planner import PathPlanner
from robomap.clustering import IncrementalObjectClusterer
from robomap.robot_logger import RobotLoggerMixin
import uuid

import logging
logger = logging.getLogger(__name__)


def _snapshot_mutable(value: Any) -> Any:
    """Копирует изменяемые контейнеры, пропуская скалярные значения.

    Дешёвая замена ``copy.deepcopy`` для результатов ``to_dict``: обходит
    только ``dict``/``list``/``tuple``, а неизменяемые скаляры (``str``,
    ``int``, ``float``, ``bool``, ``None``) переиспользует по ссылке.
    Нужна, чтобы записанные кадры истории не менялись задним числом при
    последующих мутациях живых объектов мира.

    Args:
        value (Any): Значение из сериализуемой структуры.

    Returns:
        Any: Значение, безопасное для долговременного хранения.
    """
    value_type = type(value)
    if value_type is dict:
        return {key: _snapshot_mutable(item) for key, item in value.items()}
    if value_type is list:
        return [_snapshot_mutable(item) for item in value]
    if value_type is tuple:
        return tuple(_snapshot_mutable(item) for item in value)
    return value


class Action:
    """
    Базовый класс для всех действий объектов.

    Атрибуты:
        is_background (bool): Флаг фонового действия. Фоновые действия
            выполняются параллельно с основным блокирующим действием.
    """
    is_background: bool = False

    def execute(self, entity: 'WorldObject', world: Any) -> bool:
        """
        Выполняет один шаг действия.

        Args:
            entity (WorldObject): Сущность, выполняющая действие.
            world (Any): Экземпляр симулируемого мира.

        Returns:
            bool: True, если действие полностью завершено, иначе False.

        Raises:
            NotImplementedError: Если метод не переопределен в дочернем классе.
        """
        raise NotImplementedError("Метод execute должен быть переопределен.")


class MoveAction(Action):
    """
    Действие перемещения сущности в заданную точку.

    Атрибуты:
        target_x (float): Координата X целевой точки.
        target_y (float): Координата Y целевой точки.
        is_path_set (bool): Флаг успешного построения маршрута.
    """

    def __init__(self, target_x: float, target_y: float, nav_mode: str = "hybrid"):
        """
        Инициализирует действие перемещения.

        Args:
            target_x (float): Координата X цели.
            target_y (float): Координата Y цели.
            nav_mode (str): Режим навигации ("hybrid", "memory_only", "notes_only").
        """
        self.target_x = target_x
        self.target_y = target_y
        self.nav_mode = str(nav_mode)
        self.is_path_set = False
        self.stuck_ticks = 0

    def execute(self, entity: 'WorldObject', world: Any) -> bool:
        """
        Выполняет один шаг движения.

        Анализирует наличие препятствий в зоне видимости, перестраивает
        маршрут при необходимости и вычисляет смещение на текущий кадр.

        Args:
            entity (WorldObject): Сущность, выполняющая перемещение.
            world (Any): Экземпляр симулируемого мира.

        Returns:
            bool: True, если цель достигнута или маршрут невозможен, иначе False.
        """
        if not entity.planner or entity.x is None or entity.y is None:
            entity.log_event(f"Robot cannot move inside. Exit outside first.", level="ERROR", world=world, action_code="move_cancel")
            entity.clear_actions()
            return True

        is_robot = isinstance(entity, Robot)
        
        if is_robot:
            raw_knowledge = entity.memory
        else:
            raw_knowledge = [obj for obj in world.real_map_data if obj.id != entity.id]
            if hasattr(world, 'robots'):
                for r in world.robots.values():
                    if r.id != entity.id and r.location_id is None:
                        raw_knowledge.append(r)

        global_knowledge = []
        for obj in raw_knowledge:
            is_virt = getattr(obj, 'is_virtual', False)
            is_obs = getattr(obj, 'is_obstacle', False)
            
            if self.nav_mode == "memory_only":
                if not is_virt: 
                    global_knowledge.append(obj)
            elif self.nav_mode == "notes_only":
                if is_virt and is_obs: 
                    global_knowledge.append(obj)
            else:
                if not is_virt or (is_virt and is_obs): 
                    global_knowledge.append(obj)

        if not self.is_path_set:
            entity.planner.set_target(
                current_pos=(entity.x, entity.y),
                target_x=self.target_x,
                target_y=self.target_y,
                obstacles=global_knowledge
            )
            self.is_path_set = True
            entity.log_event(f"Started moving to ({self.target_x}, {self.target_y}).", level="INFO", world=world, action_code="move_start")
        
        if not entity.planner.current_path:
                dist_to_target = math.hypot(self.target_x - entity.x, self.target_y - entity.y)
                if dist_to_target <= entity.physical_radius + 0.5:
                    entity.log_event(f"MoveAction completed: Reached target at ({entity.x:.2f}, {entity.y:.2f}).", level="SUCCESS", world=world, action_code="move_success")
                    return True
                else:
                    entity.log_event(f"MoveAction finished: Stopped at nearest reachable point ({entity.x:.2f}, {entity.y:.2f}).", level="WARNING", world=world, action_code="move_warn_stopped")
                    return True

        needs_replanning = False

        if is_robot:
            if hasattr(entity.planner, 'is_path_blocked'):
                needs_replanning = entity.planner.is_path_blocked((entity.x, entity.y), global_knowledge)
        else:
            local_obstacles = []
            vision_rad = entity.vision_radius if entity.vision_radius is not None else 15.0
            
            for obs in global_knowledge:
                if obs.x is None or obs.y is None:
                    continue
                dist = math.hypot(obs.x - entity.x, obs.y - entity.y)
                if dist <= vision_rad:
                    local_obstacles.append(obs)
            
            if hasattr(entity.planner, 'is_path_blocked'):
                needs_replanning = entity.planner.is_path_blocked((entity.x, entity.y), local_obstacles)

        if needs_replanning:
            entity.planner.set_target(
                current_pos=(entity.x, entity.y),
                target_x=self.target_x,
                target_y=self.target_y,
                obstacles=global_knowledge
            )
            if not getattr(self, '_replanned_recently', False):
                entity.log_event(f"Path blocked! Replanning route to ({self.target_x}, {self.target_y}).", level="WARNING", world=world, action_code="move_warn_obstacle")
                self._replanned_recently = True
        else:
            self._replanned_recently = False

        if is_robot and entity.planner.current_path:
            next_wp = entity.planner.current_path[0]
            dx_to_wp = next_wp[0] - entity.x
            dy_to_wp = next_wp[1] - entity.y
            
            if math.hypot(dx_to_wp, dy_to_wp) > 0.05:
                movement_angle = (-math.degrees(math.atan2(dy_to_wp, dx_to_wp))) % 360
                diff = (movement_angle - entity.angle + 180) % 360 - 180
                
                if abs(diff) > 0.5:
                    original_angle = entity.angle
                    
                    if abs(diff) <= entity.rotation_speed:
                        entity.angle = movement_angle
                    else:
                        if diff > 0:
                            entity.angle = (entity.angle + entity.rotation_speed) % 360
                        else:
                            entity.angle = (entity.angle - entity.rotation_speed) % 360
                            
                    predicted_rot_poly = entity.get_polygon().buffer(-0.1)
                    if world.check_collision(predicted_rot_poly, ignore_ids=[entity.id]):
                        entity.angle = original_angle
                        self.stuck_ticks += 1
                        if self.stuck_ticks > 3:
                            entity.log_event(
                                "Physical collision predicted during rotation. Emergency stop activated.", 
                                level="ERROR", world=world, action_code="move_fail_collision"
                            )
                            entity.clear_actions()
                            return True
                        return False
                        
                    return False

        dx, dy, remaining_path = entity.planner.get_next_step(entity.x, entity.y)
        
        if dx != 0.0 or dy != 0.0:
            original_x, original_y = entity.x, entity.y
            entity.x = original_x + dx
            entity.y = original_y + dy
            
            predicted_poly = entity.get_polygon().buffer(-0.1)
            
            entity.x, entity.y = original_x, original_y
            
            if world.check_collision(predicted_poly, ignore_ids=[entity.id]):
                self.stuck_ticks += 1
                
                if self.stuck_ticks > 3:
                    entity.log_event(
                        f"Robot is stuck! Physical collision prevents movement. Action canceled.", 
                        level="ERROR", 
                        world=world, 
                        action_code="move_fail_collision"
                    )
                    entity.clear_actions()
                    return True
                    
                entity.planner.current_path = []
                return False 
                
            self.stuck_ticks = 0
            entity.x += dx
            entity.y += dy

        if dx == 0.0 and dy == 0.0 and not remaining_path:
            entity.log_event(f"MoveAction completed: Reached nearest valid point at ({entity.x:.2f}, {entity.y:.2f}).", level="SUCCESS", world=world, action_code="move_success")
            return True

        return False
    

class RotateAction(Action):
    """
    Действие поворота объекта

    Поддерживает три режима:
    - "absolute": поворот к точному углу (например, 90 градусов).
    - "relative": поворот на заданный угол относительно текущего (например, +45 или -45).
    - "object": поворот лицом к указанному объекту (по ID из памяти робота). 
                Объект должен находиться в радиусе видимости.
    """

    def __init__(self, target: Any, mode: str = "relative"):
        self.target = target
        self.mode = mode
        self.target_angle = None

    def execute(self, entity: 'WorldObject', world: Any) -> bool:
        if not hasattr(entity, 'angle'):
            entity.log_event("RotateAction canceled: Entity cannot rotate.", level="ERROR", world=world, action_code="rotate_cancel_no_rotate")
            entity.clear_actions()
            return True

        if self.target_angle is None:
            if self.mode == "absolute":
                self.target_angle = float(self.target) % 360
                
            elif self.mode == "relative":
                self.target_angle = (entity.angle + float(self.target)) % 360
                
            elif self.mode == "object":
                target_id = str(self.target)
                target_x, target_y = None, None
                
                if hasattr(entity, 'memory'):
                    mem_obj = next((m for m in entity.memory if m.id == target_id), None)
                    if mem_obj and mem_obj.x is not None and mem_obj.y is not None:
                        target_x = mem_obj.x
                        target_y = mem_obj.y
                    else:
                        entity.log_event(f"RotateAction canceled: Target '{target_id}' not found in memory.", level="ERROR", world=world, action_code="rotate_fail_not_found_mem")
                        entity.clear_actions()
                        return True
                else:
                    obj = world.get_object_by_id(target_id)
                    if obj and obj.x is not None and obj.y is not None:
                        target_x, target_y = obj.x, obj.y
                    else:
                        entity.log_event(f"RotateAction canceled: Target '{target_id}' not found.", level="ERROR", world=world, action_code="rotate_fail_general")
                        entity.clear_actions()
                        return True

                if entity.x is None or entity.y is None:
                    entity.log_event("RotateAction canceled: Entity lacks coordinates.", level="ERROR", world=world, action_code="rotate_cancel_no_coords")
                    entity.clear_actions()
                    return True

                dist = math.hypot(target_x - entity.x, target_y - entity.y)
                if hasattr(entity, 'vision_radius') and entity.vision_radius is not None:
                    if dist > entity.vision_radius:
                        entity.log_event(f"RotateAction canceled: '{target_id}' is too far.", level="ERROR", world=world, action_code="rotate_fail_too_far")
                        entity.clear_actions()
                        return True

                dx = target_x - entity.x
                dy = target_y - entity.y
                self.target_angle = (-math.degrees(math.atan2(dy, dx))) % 360
                
            else:
                entity.log_event(f"RotateAction canceled: Unknown mode '{self.mode}'.", level="ERROR", world=world, action_code="rotate_fail_general")
                entity.clear_actions()
                return True

            original_angle = entity.angle
            entity.angle = self.target_angle
            predicted_poly = entity.get_polygon()
            entity.angle = original_angle

            if world.check_collision(predicted_poly, ignore_ids=[entity.id]):
                entity.log_event(f"RotateAction canceled: Not enough space to rotate to {self.target_angle:.1f} deg.", level="ERROR", world=world, action_code="rotate_fail_space")
                entity.clear_actions()
                return True

            entity.log_event(f"Started rotating to {self.target_angle:.1f} deg (Mode: {self.mode}).", level="INFO", world=world, action_code="rotate_start")

        diff = (self.target_angle - entity.angle + 180) % 360 - 180

        if abs(diff) <= entity.rotation_speed:
            entity.angle = self.target_angle
            entity.log_event(f"RotateAction completed: Reached angle {entity.angle:.1f} deg.", level="SUCCESS", world=world, action_code="rotate_success")
            return True

        if diff > 0:
            entity.angle = (entity.angle + entity.rotation_speed) % 360
        else:
            entity.angle = (entity.angle - entity.rotation_speed) % 360

        return False


class InteractRotateAction(Action):
    """
    Действие поворота целевого объекта.
    Поворачивает объект на заданный угол, если у него есть свойство 'rotateable'.
    """

    def __init__(self, target_id: str, angle_offset: float):
        self.target_id = str(target_id)
        self.angle_offset = float(angle_offset)

    def execute(self, entity: 'WorldObject', world: Any) -> bool:
        real_target_id = self.target_id

        if isinstance(entity, Robot):
            def _find_mem_obj(mem_list, t_id):
                for m in mem_list:
                    if m.id == t_id: return m
                    if m.inventory:
                        res = _find_mem_obj(m.inventory, t_id)
                        if res: return res
                return None
            
            mem_target = _find_mem_obj(entity.memory, self.target_id)
            if not mem_target:
                entity.log_event(f"InteractRotateAction: Цель '{self.target_id}' не найдена в памяти.", level="ERROR", world=world, action_code="interact_rotate_fail_not_found_mem")
                entity.clear_actions()
                return True
            real_target_id = mem_target.real_id

        target = world.get_object_by_id(real_target_id)
        
        if not target:
            entity.log_event(f"InteractRotateAction: Физическая цель '{self.target_id}' не найдена.", level="ERROR", world=world, action_code="interact_rotate_fail_not_found_real")
            entity.clear_actions()
            return True

        if "rotateable" not in target.affordance:
            entity.log_event(f"InteractRotateAction: Объект '{self.target_id}' нельзя повернуть (нет свойства 'rotateable').", level="ERROR", world=world, action_code="interact_rotate_fail_not_rotateable")
            entity.clear_actions()
            return True

        if entity.x is not None and entity.y is not None and target.x is not None and target.y is not None:
            dist = math.hypot(target.x - entity.x, target.y - entity.y)
            req_dist = entity.physical_radius + target.physical_radius + 1.5
            if dist > req_dist:
                entity.log_event(f"InteractRotateAction: Слишком далеко от '{self.target_id}'.", level="ERROR", world=world, action_code="interact_rotate_fail_too_far")
                entity.clear_actions()
                return True

        new_angle = (target.angle + self.angle_offset) % 360
        original_angle = target.angle
        
        target.angle = new_angle
        predicted_poly = target.get_polygon()
        target.angle = original_angle
        
        if world.check_collision(predicted_poly, ignore_ids=[target.id]):
            entity.log_event(f"InteractRotateAction canceled: Not enough space to rotate '{target.type_name}:{self.target_id}'.", level="ERROR", world=world, action_code="interact_rotate_fail_space")
            entity.clear_actions()
            return True

        target.angle = new_angle
        target.initialize_physics()

        entity.log_event(f"Повернул '{target.type_name}:{self.target_id}' на {self.angle_offset} градусов.", level="SUCCESS", world=world, action_code="interact_rotate_success")
        return True


class PickupAction(Action):
    """
    Действие подбора объекта и перемещения его в инвентарь сущности.

    Позволяет сущности забрать предмет с земли или из указанного контейнера.
    Учитывает вложенность инвентарей (контейнеры внутри контейнеров) и 
    расположение сущности (например, запрещает брать предметы из здания, 
    находясь на улице).

    Атрибуты:
        target_id (str): Идентификатор целевого объекта для подбора.
        container_path (List[str]): Иерархический путь из ID контейнеров 
            к целевому объекту (от корня до конечного хранилища).
    """

    def __init__(self, target_id: str, container_path: Optional[Union[str, List[str]]] = None):
        """
        Инициализирует действие подбора объекта.

        Args:
            target_id (str): Идентификатор объекта, который нужно подобрать.
            container_path (Optional[Union[str, List[str]]]): Путь к контейнеру, 
                в котором находится объект. Если None, поиск идет в корне текущей локации.
        """
        self.target_id = str(target_id)
        if container_path is None:
            self.container_path = []
        elif isinstance(container_path, str):
            self.container_path = [container_path]
        else:
            self.container_path = [str(c) for c in container_path]

    def execute(self, entity: 'WorldObject', world: Any) -> bool:
        """
        Выполняет попытку подбора объекта на текущем тике симуляции.

        Проверяет наличие инвентаря у сущности, физическое существование предмета, 
        дистанцию до него и наличие свойства (affordance) 'pickup'. 
        Для роботов дополнительно осуществляется синхронизация памяти 
        и реального мира через трансляцию ID.

        Args:
            entity (WorldObject): Сущность, выполняющая подбор.
            world (Any): Экземпляр симулируемого мира.

        Returns:
            bool: True (действие выполняется за один такт и сразу завершается).
        """
        if not entity.has_inventory:
            entity.log_event("PickupAction canceled: No inventory.", level="ERROR", world=world, action_code="pickup_fail_no_inventory", target_id=self.target_id, container_path=self.container_path)
            entity.clear_actions()
            return True 

        def _resolve_strict_path(root_list, path, item_id=None):
            curr_list = root_list
            root_owner = None
            for i, step_id in enumerate(path):
                step_obj = next((obj for obj in curr_list if obj.id == step_id), None)
                if not step_obj: return None, None, None
                if i == 0: root_owner = step_obj
                curr_list = step_obj.inventory
            
            if item_id:
                item_obj = next((obj for obj in curr_list if obj.id == item_id), None)
                return item_obj, curr_list, root_owner
            return None, curr_list, root_owner

        real_target_id = self.target_id
        real_container_path = list(self.container_path)
        mem_item = None

        if isinstance(entity, Robot):
            mem_root_list = entity.memory
            if entity.location_id:
                mem_parent = entity._find_mem_obj_recursively(entity.memory, entity.location_id, by_real_id=True)
                if mem_parent:
                    mem_root_list = mem_parent.inventory

            mem_item, mem_source_list, mem_root = _resolve_strict_path(mem_root_list, self.container_path, self.target_id)

            if not mem_item:
                entity.log_event(f"PickupAction: Target '{self.target_id}' not found in memory.", level="ERROR", world=world, action_code="pickup_fail_not_found_mem", target_id=self.target_id, container_path=self.container_path)
                entity.clear_actions()
                return True

            real_target_id = mem_item.real_id

            real_container_path = []
            curr_mem_list = mem_root_list
            for step_id in self.container_path:
                step_mem = next((m for m in curr_mem_list if m.id == step_id), None)
                real_container_path.append(step_mem.real_id)
                curr_mem_list = step_mem.inventory

        search_root_list = world.real_map_data
        if entity.location_id:
            parent_container = world.get_object_by_id(entity.location_id)
            if parent_container:
                search_root_list = parent_container.inventory

        target_item, source_list, target_root = _resolve_strict_path(search_root_list, real_container_path, real_target_id)
        
        if not target_item:
            entity.log_event(f"PickupAction canceled: Physical item '{self.target_id}' not found.", level="ERROR", world=world, action_code="pickup_fail_not_found_real", target_id=self.target_id, container_path=self.container_path)
            entity.clear_actions()
            return True

        target_dist_obj = target_root if target_root else target_item
        if target_dist_obj.id == entity.id:
            entity.log_event("PickupAction: Cannot pick up from yourself this way.", level="ERROR", world=world, action_code="pickup_fail_self", target_id=self.target_id, container_path=self.container_path)
            entity.clear_actions()
            return True

        if entity.location_id is None:
            if target_dist_obj and "enter" in target_dist_obj.affordance and real_container_path:
                entity.log_event(f"PickupAction canceled: Must enter container to access '{self.target_id}'.", level="ERROR", world=world, action_code="pickup_fail_enter_first", target_id=self.target_id, container_path=self.container_path)
                entity.clear_actions()
                return True
                
            if target_dist_obj.x is not None and entity.x is not None:
                dist = math.hypot(target_dist_obj.x - entity.x, target_dist_obj.y - entity.y)
                interaction_dist = entity.physical_radius + getattr(target_dist_obj, 'physical_radius', 0.5) + 0.5
                if dist > interaction_dist:
                    entity.log_event(f"PickupAction canceled: Target '{self.target_id}' is too far.", level="ERROR", world=world, action_code="pickup_fail_too_far", target_id=self.target_id, container_path=self.container_path)
                    entity.clear_actions()
                    return True
        else:
            pass 

        if isinstance(entity, Robot) and self.container_path:
            world._tactile_sync_path(
                robot=entity,
                container_path=self.container_path,
                real_container_path=real_container_path,
                start_real_list=search_root_list,
                start_mem_list=mem_root_list
            )

        if "pickup" not in target_item.affordance:
            entity.log_event("PickupAction canceled: Object cannot be picked up.", level="ERROR", world=world, action_code="pickup_fail_not_pickupable", target_id=self.target_id, container_path=self.container_path)
            entity.clear_actions()
            return True

        source_list.remove(target_item) 
        target_item.x, target_item.y = None, None
        target_item.location_id = entity.id
        entity.inventory.append(target_item) 

        if isinstance(entity, Robot) and mem_item:
            mem_source_list.remove(mem_item)
            mem_item.x, mem_item.y = None, None
            mem_item.absolute_wkt, mem_item.absolute_shape = "", []
            mem_item.location_id = entity.id
            entity.memory_inventory.append(mem_item)

        path_str = f" from {self.container_path}" if self.container_path else ""
        entity.log_event(f"Picked up '{target_item.type_name}:{self.target_id}'{path_str}.", level="SUCCESS", world=world, action_code="pickup_success", target_id=self.target_id, container_path=self.container_path)
        return True


class PutAction(Action):
    """
    Действие извлечения объекта из инвентаря и его размещения.

    Позволяет сущности выложить предмет из своего инвентаря. Объект может быть
    помещен в указанный контейнер, сброшен на землю (с автоматическим поиском
    свободного места для избежания коллизий) или оставлен внутри текущего здания.

    Атрибуты:
        item_id (str): Идентификатор объекта, который необходимо выложить.
        target_container_path (List[str]): Иерархический путь из ID контейнеров
            к целевому хранилищу. Пустой список означает сброс в текущую локацию.
    """

    def __init__(self, item_id: str, target_container_path: Optional[Union[str, List[str]]] = None):
        """
        Инициализирует действие размещения объекта.

        Args:
            item_id (str): Идентификатор размещаемого объекта из инвентаря.
            target_container_path (Optional[Union[str, List[str]]]): Путь к 
                целевому контейнеру. Если None, объект кладется на землю или 
                в инвентарь текущего помещения.
        """
        self.item_id = str(item_id)
        if target_container_path is None:
            self.target_container_path = []
        elif isinstance(target_container_path, str):
            self.target_container_path = [target_container_path]
        else:
            self.target_container_path = [str(c) for c in target_container_path]

    def _find_drop_location(self, entity: 'WorldObject', target_item: 'WorldObject', world: Any) -> Optional[Tuple[float, float, float]]:
        start_radius = entity.physical_radius + target_item.physical_radius + 0.1
        max_radius = start_radius + 1.0
        step = 0.2
        positions_angles = [i * math.pi / 4 for i in range(8)]

        original_angle = target_item.angle
        
        rotational_angles = [
            original_angle,
            (original_angle + 90) % 360,
            (original_angle + 180) % 360,
            (original_angle + 270) % 360
        ]

        current_r = start_radius
        while current_r <= max_radius:
            for pos_angle in positions_angles:
                cx = entity.x + current_r * math.cos(pos_angle)
                cy = entity.y + current_r * math.sin(pos_angle)
                
                for rot_angle in rotational_angles:
                    target_item.x = cx
                    target_item.y = cy
                    target_item.angle = rot_angle
                    
                    item_poly = target_item.get_polygon()
                    
                    if not world.check_collision(item_poly, ignore_ids=[entity.id]):
                        return cx, cy, rot_angle
                        
            current_r += step

        target_item.angle = original_angle
        return None

    def execute(self, entity: 'WorldObject', world: Any) -> bool:
        """
        Выполняет попытку размещения объекта на текущем тике симуляции.

        Проверяет фактическое наличие предмета в инвентаре. Если задан путь
        контейнера, проверяет дистанцию до него и права доступа (affordance 'enter').
        При сбросе на землю рассчитывает безопасные координаты. Для роботов
        автоматически обновляет дерево инвентарей в виртуальной памяти.

        Args:
            entity (WorldObject): Сущность, выполняющая действие.
            world (Any): Экземпляр симулируемого мира.

        Returns:
            bool: True (действие выполняется за один такт и завершается).
        """
        def _resolve_strict_path(root_list, path):
            curr_list = root_list
            root_owner = None
            target_cont = None
            for i, step_id in enumerate(path):
                step_obj = next((obj for obj in curr_list if obj.id == step_id), None)
                if not step_obj: return None, None, None
                if i == 0: root_owner = step_obj
                target_cont = step_obj
                curr_list = step_obj.inventory
            return target_cont, curr_list, root_owner

        def _find_item_in_own_inv(obj_list, t_id):
            for obj in obj_list:
                if obj.id == t_id: return obj, obj_list
                if getattr(obj, 'inventory', None):
                    r_obj, r_list = _find_item_in_own_inv(obj.inventory, t_id)
                    if r_obj: return r_obj, r_list
            return None, None

        real_item_id = self.item_id
        real_container_path = list(self.target_container_path)
        mem_item = None
        mem_cont = None

        if isinstance(entity, Robot):
            mem_item, mem_source_list = _find_item_in_own_inv(entity.memory_inventory, self.item_id)
            if not mem_item:
                entity.log_event(f"PutAction: Item '{self.item_id}' not in virtual inventory.", level="ERROR", world=world, action_code="put_fail_not_in_inv", item_id=self.item_id, target_container_path=self.target_container_path)
                entity.clear_actions()
                return True
            real_item_id = mem_item.real_id

            if self.target_container_path:
                mem_root_list = entity.memory
                if entity.location_id:
                    mem_parent = next((m for m in entity.memory if m.real_id == entity.location_id), None)
                    if mem_parent:
                        mem_root_list = mem_parent.inventory

                mem_cont, mem_dest_list, _ = _resolve_strict_path(mem_root_list, self.target_container_path)
                if not mem_cont:
                    entity.log_event(f"PutAction: Target path {self.target_container_path} not found in memory.", level="ERROR", world=world, action_code="put_fail_not_found_mem", item_id=self.item_id, target_container_path=self.target_container_path)
                    entity.clear_actions()
                    return True
                
                real_container_path = []
                curr_mem_list = mem_root_list
                for step_id in self.target_container_path:
                    step_mem = next((m for m in curr_mem_list if m.id == step_id), None)
                    real_container_path.append(step_mem.real_id)
                    curr_mem_list = step_mem.inventory

        target_item, source_list = _find_item_in_own_inv(entity.inventory, real_item_id)
        if not target_item:
            entity.log_event("PutAction: Physical item missing from inventory!", level="CRITICAL", world=world, action_code="put_fail_not_in_inv", item_id=self.item_id, target_container_path=self.target_container_path)
            entity.clear_actions()
            return True

        if real_container_path:
            search_root_list = world.real_map_data
            if entity.location_id:
                parent = world.get_object_by_id(entity.location_id)
                if parent: search_root_list = parent.inventory

            container, dest_list, root_owner = _resolve_strict_path(search_root_list, real_container_path)
            if not container or not container.has_inventory:
                entity.log_event("PutAction: Target container invalid.", level="ERROR", world=world, action_code="put_fail_invalid_container", item_id=self.item_id, target_container_path=self.target_container_path)
                entity.clear_actions()
                return True

            if entity.location_id is None:
                if "enter" in root_owner.affordance:
                    entity.log_event(f"PutAction: Must enter target location {self.target_container_path} first.", level="ERROR", world=world, action_code="put_fail_enter_first", item_id=self.item_id, target_container_path=self.target_container_path)
                    entity.clear_actions()
                    return True

                if root_owner.x is not None and entity.x is not None:
                    dist = math.hypot(root_owner.x - entity.x, root_owner.y - entity.y)
                    if dist > entity.physical_radius + getattr(root_owner, 'physical_radius', 0.5) + 0.5:
                        entity.log_event("PutAction: Target path is too far.", level="ERROR", world=world, action_code="put_fail_too_far", item_id=self.item_id, target_container_path=self.target_container_path)
                        entity.clear_actions()
                        return True

            if isinstance(entity, Robot) and self.target_container_path:
                world._tactile_sync_path(
                    robot=entity,
                    container_path=self.target_container_path,
                    real_container_path=real_container_path,
                    start_real_list=search_root_list,
                    start_mem_list=mem_root_list
                )

            source_list.remove(target_item)
            target_item.location_id = container.id
            dest_list.append(target_item)
            
            if isinstance(entity, Robot) and mem_item and mem_cont:
                mem_source_list.remove(mem_item)
                mem_item.location_id = mem_cont.id
                mem_dest_list.append(mem_item)
                
            entity.log_event(f"Placed '{target_item.type_name}' into {self.target_container_path}.", level="SUCCESS", world=world, action_code="put_success_container", item_id=self.item_id, target_container_path=self.target_container_path)

        else:
            source_list.remove(target_item)
            
            if entity.location_id is not None:
                parent_container = world.get_object_by_id(entity.location_id)
                target_item.location_id = parent_container.id
                target_item.x, target_item.y = None, None
                parent_container.inventory.append(target_item)
                
                log_loc = parent_container.id
                if isinstance(entity, Robot) and mem_item:
                    mem_source_list.remove(mem_item)
                    mem_item.location_id = parent_container.id
                    mem_item.x, mem_item.y = None, None
                    mem_parent = entity._find_mem_obj_recursively(entity.memory, parent_container.id, by_real_id=True)
                    if mem_parent: 
                        mem_parent.inventory.append(mem_item)
                        log_loc = mem_parent.id
                        
                entity.log_event(f"Dropped '{target_item.type_name}:{self.item_id}' inside '{log_loc}'.", level="SUCCESS", world=world, action_code="put_success_inside", item_id=self.item_id, target_container_path=self.target_container_path, location_id=log_loc)
            else:
                drop_data = self._find_drop_location(entity, target_item, world)
                
                if drop_data is None:
                    source_list.append(target_item)
                    target_item.x, target_item.y = None, None
                    
                    entity.log_event(f"PutAction canceled: Not enough space to drop '{target_item.type_name}:{self.item_id}' here.", level="ERROR", world=world, action_code="put_fail_no_space", item_id=self.item_id, target_container_path=self.target_container_path)
                    entity.clear_actions()
                    return True
                    
                drop_x, drop_y, drop_angle = drop_data
                
                target_item.x, target_item.y = drop_x, drop_y
                target_item.angle = drop_angle
                target_item.location_id = None
                
                target_item.initialize_physics()
                world.real_map_data.append(target_item)

                if isinstance(entity, Robot) and mem_item:
                    mem_source_list.remove(mem_item)
                    mem_item.x, mem_item.y = drop_x, drop_y
                    mem_item.location_id = None
                    
                    poly = target_item.get_polygon()
                    mem_item.absolute_wkt = poly.wkt
                    
                    coords = []
                    if poly.geom_type == 'Polygon':
                        coords.extend([[x, y] for x, y in poly.exterior.coords])
                    elif poly.geom_type in ['MultiPolygon', 'GeometryCollection']:
                        for g in getattr(poly, 'geoms', []):
                            if g.geom_type == 'Polygon':
                                coords.extend([[x, y] for x, y in g.exterior.coords])
                                coords.append([None, None])
                        if coords and coords[-1] == [None, None]:
                            coords.pop()
                            
                    mem_item.absolute_shape = coords
                    entity.memory.append(mem_item)

                entity.log_event(f"Dropped '{target_item.type_name}:{self.item_id}' on ground at ({drop_x:.1f}, {drop_y:.1f}) with angle {drop_angle:.1f} deg.", level="SUCCESS", world=world, action_code="put_success_ground", item_id=self.item_id, target_container_path=self.target_container_path, drop_x=drop_x, drop_y=drop_y, drop_angle=drop_angle)

        return True
    

class EnterAction(Action):
    """
    Действие перемещения сущности внутрь другого объекта.

    Позволяет сущности войти в объект, обладающий свойством 'enter' (здание,
    автомобиль, палатка). При успешном входе сущность удаляется с физической 
    карты и помещается в инвентарь целевого объекта, теряя абсолютные координаты.

    Атрибуты:
        target_id (str): Идентификатор объекта (входа).
    """

    def __init__(self, target_id: str):
        self.target_id = str(target_id)

    def execute(self, entity: 'WorldObject', world: Any) -> bool:
        """
        Выполняет проверку дистанции и перемещает сущность внутрь объекта.

        Для роботов осуществляется трансляция виртуального ID из памяти в
        физический ID мира. Проверяется наличие свойства 'enter' у цели.
        После входа координаты сущности обнуляются, а текущий маршрут 
        планировщика очищается.

        Args:
            entity (WorldObject): Сущность, выполняющая вход.
            world (Any): Экземпляр симулируемого мира.

        Returns:
            bool: True (вход выполняется за один такт и завершается).
        """
        real_target_id = self.target_id

        if isinstance(entity, Robot):
            def _find_mem_obj(mem_list, t_id):
                for m in mem_list:
                    if m.id == t_id: return m
                    if m.inventory:
                        res = _find_mem_obj(m.inventory, t_id)
                        if res: return res
                return None
            
            mem_target = _find_mem_obj(entity.memory, self.target_id)
            if not mem_target:
                entity.log_event(f"EnterAction: Target '{self.target_id}' not found in physical world.", level="ERROR", world=world, action_code="enter_fail_not_found")
                entity.clear_actions()
                return True
            
            real_target_id = mem_target.real_id

        target = world.get_object_by_id(real_target_id)
        
        if not target:
            entity.log_event(f"EnterAction: Target '{self.target_id}' not found in physical world.", level="ERROR", world=world, action_code="enter_fail_not_found")
            entity.clear_actions()
            return True

        if "enter" not in target.affordance:
            entity.log_event("EnterAction: Target cannot be entered (no 'enter' affordance).", level="ERROR", world=world, action_code="enter_fail_no_affordance")
            entity.clear_actions()
            return True

        if entity.location_id == target.id:
            entity.log_event(f"Already inside '{self.target_id}'.", level="WARNING", world=world, action_code="enter_warn_already_inside")
            return True

        if entity.location_id is None:
            if target.x is not None and target.y is not None and entity.x is not None and entity.y is not None:
                dist = math.hypot(target.x - entity.x, target.y - entity.y)
                req_dist = entity.physical_radius + target.physical_radius + 0.5
                if dist > req_dist:
                    entity.log_event(f"EnterAction: Too far from '{self.target_id}'.", level="ERROR", world=world, action_code="enter_fail_too_far")
                    entity.clear_actions()
                    return True

        if entity.location_id is None:
            if entity in world.real_map_data:
                world.real_map_data.remove(entity)
        else:
            parent = world.get_object_by_id(entity.location_id)
            if parent and entity in parent.inventory:
                parent.inventory.remove(entity)

        target.inventory.append(entity)
        entity.location_id = target.id
        
        entity.x, entity.y = None, None 
        if entity.planner:
            entity.planner.current_path = []

        log_target = self.target_id if isinstance(entity, Robot) else target.id
        entity.log_event(f"Entered '{target.type_name}:{log_target}'.", level="SUCCESS", world=world, action_code="enter_success")
        return True
    

class ExitAction(Action):
    """
    Действие выхода сущности из текущего местоположения на улицу.

    Извлекает сущность из инвентаря здания или транспорта и помещает ее 
    в локацию родительского объекта (обычно на глобальную карту), рассчитывая
    ближайшую свободную точку для предотвращения застревания в стенах.
    """

    def _find_exit_location(self, entity: 'WorldObject', parent: 'WorldObject', world: Any) -> Tuple[float, float]:
        target_list = world.real_map_data
        if parent.location_id is not None:
            grandparent = world.get_object_by_id(parent.location_id)
            if grandparent:
                target_list = grandparent.inventory
        else:
            if hasattr(world, 'robots'):
                target_list.extend([r for r in world.robots.values() if r.location_id is None])

        polygons = [parent.get_polygon()] 
        for obj in target_list:
            if obj.x is not None and obj.y is not None and obj.id != entity.id and obj.id != parent.id:
                polygons.append(obj.get_polygon())
        
        obstacle_space = unary_union(polygons)

        start_radius = parent.physical_radius + entity.physical_radius + 0.2
        max_radius = start_radius + 1.0
        step = 0.2
        angles = [i * math.pi / 4 for i in range(8)]

        current_r = start_radius
        while current_r <= max_radius:
            for angle in angles:
                cx = parent.x + current_r * math.cos(angle)
                cy = parent.y + current_r * math.sin(angle)
                
                test_point = Point(cx, cy).buffer(entity.physical_radius)
                if not obstacle_space.intersects(test_point):
                    return cx, cy
            current_r += step

        entity.log_event(f"No free space outside '{parent.id}'. Forcing exit.", level="WARNING", world=world, action_code="exit_warn_no_space")
        return parent.x + parent.physical_radius + 1.0, parent.y


    def execute(self, entity: 'WorldObject', world: Any) -> bool:
        """
        Выполняет извлечение сущности из контейнера и ее размещение в мире.

        Извлекает сущность из инвентаря здания, рассчитывает координаты снаружи,
        очищает текущий маршрут планировщика и добавляет сущность обратно 
        в массив объектов физического мира.

        Args:
            entity (WorldObject): Сущность, выходящая наружу.
            world (Any): Экземпляр симулируемого мира.

        Returns:
            bool: True (выход выполняется за один такт и завершается).
        """
        if entity.location_id is None:
            entity.log_event("ExitAction: Already outside.", level="ERROR", world=world, action_code="exit_fail_already_outside")
            entity.clear_actions()
            return True

        parent = world.get_object_by_id(entity.location_id)
        if not parent:
            entity.log_event(f"ExitAction: Parent location '{entity.location_id}' not found.", level="ERROR", world=world, action_code="exit_fail_parent_not_found")
            entity.location_id = None
            entity.clear_actions()
            return True

        if entity in parent.inventory:
            parent.inventory.remove(entity)

        if parent.location_id is None:
            exit_x, exit_y = self._find_exit_location(entity, parent, world)
            entity.x, entity.y = exit_x, exit_y
        else:
            entity.x, entity.y = None, None
        
        entity.location_id = parent.location_id

        if entity.location_id is None:
            if not isinstance(entity, Robot):
                world.real_map_data.append(entity)
        else:
            grandparent = world.get_object_by_id(entity.location_id)
            if grandparent:
                grandparent.inventory.append(entity)
            else:
                if not isinstance(entity, Robot):
                    world.real_map_data.append(entity)
                entity.location_id = None

        if entity.planner:
            entity.planner.current_path = []

        if entity.x is not None and entity.y is not None:
            entity.log_event(f"Exited '{parent.type_name}:{parent.id}' to ({entity.x:.1f}, {entity.y:.1f}).", level="SUCCESS", world=world, action_code="exit_success")
        else:
            entity.log_event(f"Exited '{parent.type_name}:{parent.id}' to the parent location container.", level="SUCCESS", world=world, action_code="exit_success")
            
        return True
    

class BreakAction(Action):
    """
    Действие разрушения или атаки целевого объекта.

    Процесс, занимающий несколько тиков симуляции. Скорость разрушения зависит 
    от наносимого сущностью урона (damage) и прочности цели (durability). 
    Объект считается сломанным, когда его прочность падает до нуля.

    Атрибуты:
        target_id (str): Идентификатор объекта, который подвергается атаке.
    """

    def __init__(self, target_id: str):
        """
        Инициализирует действие атаки или разрушения.

        Args:
            target_id (str): Идентификатор разрушаемого объекта.
        """
        self.target_id = str(target_id)

    def execute(self, entity: 'WorldObject', world: Any) -> bool:
        """
        Наносит урон объекту на текущем тике симуляции.

        Проверяет дистанцию до цели, нахождение сущностей в одном и том же
        пространстве (во избежание атак сквозь стены) и наличие свойства 'break'.
        Уменьшает прочность цели. Если прочность падает до нуля, объект
        переводится в состояние 'broken' и меняет тип для графического отображения.

        Args:
            entity (WorldObject): Атакующая сущность.
            world (Any): Экземпляр симулируемого мира.

        Returns:
            bool: False, если процесс разрушения не окончен (действие остается 
                в очереди на следующий тик). True, если объект сломан окончательно 
                или действие отменено из-за ошибки.
        """
        real_target_id = self.target_id

        if isinstance(entity, Robot):
            def _find_mem_obj(mem_list, t_id):
                for m in mem_list:
                    if m.id == t_id: return m
                    if m.inventory:
                        res = _find_mem_obj(m.inventory, t_id)
                        if res: return res
                return None
            
            mem_target = _find_mem_obj(entity.memory, self.target_id)
            if not mem_target:
                entity.log_event(f"BreakAction: Target '{self.target_id}' not found in memory.", level="ERROR", world=world, action_code="break_fail_not_found_mem", target_id=self.target_id)
                entity.clear_actions()
                return True
            real_target_id = mem_target.real_id

        target = world.get_object_by_id(real_target_id)
        
        if not target:
            entity.log_event(f"BreakAction: Physical target '{self.target_id}' not found.", level="ERROR", world=world, action_code="break_fail_not_found_real", target_id=self.target_id)
            return True

        if "break" not in target.affordance:
            entity.log_event(f"BreakAction: Target '{self.target_id}' cannot be broken (no 'break' affordance).", level="ERROR", world=world, action_code="break_fail_not_breakable", target_id=self.target_id)
            entity.clear_actions()
            return True

        if entity.damage is None or entity.damage <= 0:
            entity.log_event("BreakAction: Cannot break objects (no damage capability).", level="ERROR", world=world, action_code="break_fail_no_damage", target_id=self.target_id)
            entity.clear_actions()
            return True

        if target.state.get("broken", False):
            entity.log_event(f"BreakAction: Target '{self.target_id}' is already broken.", level="WARNING", world=world, action_code="break_fail_already_broken", target_id=self.target_id)
            entity.clear_actions()
            return True

        if entity.location_id != target.location_id:
            entity.log_event(f"BreakAction: Must be in the same location as target '{self.target_id}'.", level="ERROR", world=world, action_code="break_fail_same_location", target_id=self.target_id)
            entity.clear_actions()
            return True

        if entity.x is not None and entity.y is not None and target.x is not None and target.y is not None:
            dist = math.hypot(target.x - entity.x, target.y - entity.y)
            req_dist = entity.physical_radius + target.physical_radius + 0.5
            if dist > req_dist:
                entity.log_event(f"BreakAction: Too far from '{self.target_id}'.", level="ERROR", world=world, action_code="break_fail_too_far", target_id=self.target_id)
                entity.clear_actions()
                return True

        if target.durability is None:
            target.durability = 100.0

        target.durability -= entity.damage
        if target.durability > 0:
            entity.log_event(f"Hitting '{target.type_name}:{self.target_id}'. Durability left: {target.durability:.1f}", level="INFO", world=world, action_code="break_progress", target_id=self.target_id, durability_left=target.durability)
            return False
        
        target.durability = 0.0
        target.state["broken"] = True
        
        if not target.type_name.startswith("broken_"):
            target.type_name = f"broken_{target.type_name}"
        
        entity.log_event(f"Broke '{target.type_name}:{self.target_id}'!", level="SUCCESS", world=world, action_code="break_success", target_id=self.target_id)
        return True


class SpeakAction(Action):
    """
    Фоновое действие произнесения фразы.

    Устанавливает текстовое сообщение в состояние сущности на заданное количество
    тиков. Действие является фоновым, поэтому оно выполняется параллельно с 
    основными действиями (перемещение, атака и т.д.) и не блокирует очередь.

    Атрибуты:
        is_background (bool): Флаг, разрешающий параллельное выполнение (True).
        message (str): Текст произносимой фразы.
        duration (int): Оставшееся время отображения фразы (в тиках).
        started (bool): Флаг начала выполнения действия.
    """

    is_background = True

    def __init__(self, message: str, duration: int = 3):
        """
        Инициализирует фоновое действие речи.

        Args:
            message (str): Фраза, которую произносит сущность.
            duration (int): Длительность отображения фразы в тиках (минимум 1).
        """
        self.message = str(message)
        self.duration = max(1, duration)
        self.started = False

    def execute(self, entity: 'WorldObject', world: Any) -> bool:
        """
        Обновляет состояние речи сущности на текущем тике симуляции.

        При первом вызове добавляет фразу в состояние объекта. Каждый последующий
        тик уменьшает таймер. По истечении таймера фраза удаляется из состояния.

        Args:
            entity (WorldObject): Говорящая сущность.
            world (Any): Экземпляр симулируемого мира.

        Returns:
            bool: False, пока таймер речи активен. True, когда время вышло.
        """
        if not self.started:
            entity.log_event(f"Says: '{self.message}'", level="INFO", world=world, action_code="speak_success", message_text=self.message)
            self.started = True

        if world and hasattr(world, "active_sound_sources"):
            world.active_sound_sources[entity.id] = self.message

        if self.duration > 0:
            entity.state["speaking"] = self.message
            self.duration -= 1
            return False
        
        if "speaking" in entity.state:
            del entity.state["speaking"]
            
        return True

class InspectAction(Action):
    """
    Действие осмотра содержимого контейнера.
    Робот подходит к контейнеру и обновляет свою память о том, что лежит внутри.
    """
    def __init__(self, target_container_path: Optional[Union[str, List[str]]] = None):
        if isinstance(target_container_path, str):
            self.target_container_path = [target_container_path]
        else:
            self.target_container_path = [str(c) for c in target_container_path] if target_container_path else []

    def execute(self, entity: 'WorldObject', world: Any) -> bool:
        if not self.target_container_path:
            entity.log_event("InspectAction canceled: No container path provided.", level="ERROR", world=world, action_code="inspect_fail_no_path")
            entity.clear_actions()
            return True

        real_container_path = list(self.target_container_path)
        mem_root_list = []

        if isinstance(entity, Robot):
            mem_root_list = entity.memory
            if entity.location_id:
                mem_parent = entity._find_mem_obj_recursively(entity.memory, entity.location_id, by_real_id=True)
                if mem_parent: mem_root_list = mem_parent.inventory

            curr_mem_list = mem_root_list
            real_container_path = []
            for step_id in self.target_container_path:
                step_mem = next((m for m in curr_mem_list if m.id == step_id), None)
                if not step_mem:
                    entity.log_event(f"InspectAction: Path {self.target_container_path} not found in memory.", level="ERROR", world=world, action_code="inspect_fail_not_found_mem", target_container_path=self.target_container_path)
                    entity.clear_actions()
                    return True
                real_container_path.append(step_mem.real_id)
                curr_mem_list = step_mem.inventory

        search_root_list = world.real_map_data
        if entity.location_id:
            parent_container = world.get_object_by_id(entity.location_id)
            if parent_container: search_root_list = parent_container.inventory

        def _resolve_strict_path(root_list, path):
            curr_list = root_list
            root_owner = None
            target_cont = None
            for i, step_id in enumerate(path):
                step_obj = next((obj for obj in curr_list if obj.id == step_id), None)
                if not step_obj: return None, None
                if i == 0: root_owner = step_obj
                target_cont = step_obj
                curr_list = step_obj.inventory
            return target_cont, root_owner

        container, root_owner = _resolve_strict_path(search_root_list, real_container_path)
        if not container or not container.has_inventory:
            entity.log_event("InspectAction: Target is not a valid container.", level="ERROR", world=world, action_code="inspect_fail_invalid_container", target_container_path=self.target_container_path)
            entity.clear_actions()
            return True

        if entity.location_id is None:
            if "enter" in root_owner.affordance:
                entity.log_event(f"InspectAction: Must enter target location first.", level="ERROR", world=world, action_code="inspect_fail_enter_first", target_container_path=self.target_container_path)
                entity.clear_actions()
                return True

            if root_owner.x is not None and entity.x is not None:
                dist = math.hypot(root_owner.x - entity.x, root_owner.y - entity.y)
                if dist > entity.physical_radius + getattr(root_owner, 'physical_radius', 0.5) + 0.5:
                    entity.log_event("InspectAction: Target is too far.", level="ERROR", world=world, action_code="inspect_fail_too_far", target_container_path=self.target_container_path)
                    entity.clear_actions()
                    return True

        if isinstance(entity, Robot):
            world._tactile_sync_path(
                robot=entity,
                container_path=self.target_container_path,
                real_container_path=real_container_path,
                start_real_list=search_root_list,
                start_mem_list=mem_root_list
            )

        entity.log_event(f"Inspected contents of {self.target_container_path}.", level="SUCCESS", world=world, action_code="inspect_success", target_container_path=self.target_container_path)
        return True

class WorldObject:
    """
    Базовый класс для всех сущностей и объектов физического мира.

    Хранит геометрию, координаты, инвентарь и состояние объекта. 
    Динамические объекты (имеющие поведение 'move') автоматически инициализируют 
    планировщик маршрутов (PathPlanner).

    Атрибуты:
        id (str): Уникальный идентификатор объекта.
        type_name (str): Тип объекта (например, 'tree', 'bear', 'robot').
        description (Optional[str]): Текстовое описание объекта.
        shape_coords (List[List[float]]): Относительные координаты полигона.
        x (Optional[float]): Глобальная координата X. None, если объект в инвентаре.
        y (Optional[float]): Глобальная координата Y. None, если объект в инвентаре.
        location_id (Optional[str]): ID родительского объекта (контейнера/здания).
        dynamic (bool): Флаг подвижности/активности объекта.
        physical_radius (float): Вычисляемый радиус коллизии объекта.
        affordance (List[str]): Список доступных взаимодействий с объектом ('enter', 'break').
        behavior (List[str]): Список разрешенных действий самой сущности ('move', 'pickup').
        has_inventory (Optional[bool]): Флаг наличия инвентаря.
        inventory (List['WorldObject']): Список вложенных физических объектов.
        speed (Optional[float]): Максимальная дистанция перемещения за один тик.
        vision_radius (Optional[float]): Радиус обнаружения препятствий и объектов.
        durability (Optional[float]): Очки прочности (здоровье).
        damage (Optional[float]): Урон, наносимый при действии 'break'.
        state (Dict[str, Any]): Словарь динамических состояний (например, 'broken', 'speaking').
        action_queue (List['Action']): Очередь запланированных действий.
        planner (Optional[Any]): Экземпляр планировщика маршрутов (PathPlanner).
    """
    def __init__(self, obj_id: str, obj_type: str, 
                 x: Optional[float] = None, y: Optional[float] = None, 
                 dynamic: bool = False, behavior: Optional[List[str]] = None, **kwargs):
        """
        Инициализирует физический объект и применяет параметры по умолчанию.

        Args:
            obj_id (str): Уникальный идентификатор.
            obj_type (str): Строковой тип объекта.
            x (Optional[float]): Начальная координата X.
            y (Optional[float]): Начальная координата Y.
            dynamic (bool): Является ли объект активным/подвижным.
            behavior (Optional[List[str]]): Список доступных действий.
            **kwargs: Дополнительные параметры (durability, speed, shape и т.д.).
        """
        
        self.id: str = str(obj_id)
        self.type_name: str = obj_type
        self.description: Optional[str] = kwargs.get("description")
        self.shape_coords: List[List[float]] = kwargs.get("shape", [])
        
        self.x: Optional[float] = float(x) if x is not None else None
        self.y: Optional[float] = float(y) if y is not None else None

        self.location_id: Optional[str] = kwargs.get("location_id")
        
        self.dynamic: bool = dynamic
        self.physical_radius: float = 0.5   
        self.affordance: List[str] = kwargs.get("affordance", [])
        self.behavior: List[str] = behavior or []
        
        self.has_inventory: Optional[bool] = kwargs.get("has_inventory")
        self.inventory: List['WorldObject'] = []
        
        self.speed: Optional[float] = kwargs.get("speed")
        self.vision_radius: Optional[float] = kwargs.get("vision_radius")
        self.durability: Optional[float] = kwargs.get("durability")
        self.max_durability: Optional[float] = kwargs.get("max_durability", self.durability)
        self.damage: Optional[float] = kwargs.get("damage")
        
        for b in self.behavior:
            if b == "move":
                if self.speed is None: self.speed = 2.0
                if self.vision_radius is None: self.vision_radius = 15.0
            elif b == "break":
                if self.damage is None: self.damage = 10.0

        if "break" in self.affordance and self.durability is None:
            self.durability = 100.0

        self.angle: float = float(kwargs.get("angle", 0.0))
                        
        self.state: Dict[str, Any] = kwargs.get("state", {})
        self.action_queue: List['Action'] = []
        self.completed_actions_this_tick: List['Action'] = []
        self.planner: Optional[Any] = None
        self._cached_polygon: Optional[Polygon] = None
        self._polygon_cache_key: Optional[Tuple[Any, ...]] = None

    def log_event(self, message: str, level: str = "INFO", world: Any = None, action_code: Optional[str] = None, **kwargs):
        """Отправляет лог в консоль и сохраняет результат последнего действия в состоянии объекта."""
        
        if "feedback" not in self.state:
            self.state["feedback"] = []
        self.state["feedback"].append(message)

        if len(self.state["feedback"]) > 100:
            self.state["feedback"] = self.state["feedback"][-100:]

        tick_str = f"TICK {world.tick_count:04d}" if world else "QUEUE"
        full_msg = f"[{tick_str}] [{level}] [{self.type_name}:{self.id}] -> {message}"
        
        if level in ["ERROR", "CRITICAL"]:
            logger.error(full_msg)
        elif level == "WARNING":
            logger.warning(full_msg)
        else:
            logger.info(full_msg)

    def _calculate_physical_radius(self) -> float:
        """
        Вычисляет физический радиус коллизии на основе формы объекта.

        Returns:
            float: Максимальное расстояние от центра до вершин полигона. 
                Если форма не задана, возвращает значение по умолчанию (0.4).
        """
        if not self.shape_coords:
            return 0.4 
            
        max_sq_dist = 0.0
        for pt in self.shape_coords:
            if len(pt) >= 2 and pt[0] is not None:
                dist_sq = pt[0]**2 + pt[1]**2
                if dist_sq > max_sq_dist:
                    max_sq_dist = dist_sq
        return math.sqrt(max_sq_dist) if max_sq_dist > 0 else 0.4

    def initialize_physics(self):
        """
        Инициализирует физическую модель и навигацию объекта.

        Вычисляет радиус коллизии и, если объект динамический с правом
        на перемещение ('move'), создает экземпляр планировщика PathPlanner.
        """
        self.physical_radius = self._calculate_physical_radius()
        
        if self.dynamic and "move" in self.behavior:
            obj_speed = self.speed if self.speed is not None else 2.0
            self.planner = PathPlanner(
                movement_step=obj_speed, 
                physical_radius=self.physical_radius
            )

    def tick(self, world: Any):
        """
        Выполняет логику объекта и его инвентаря на текущем шаге симуляции.

        Итерация по очереди действий (`action_queue`) происходит по ее копии.
        Это защищает движок от сбоев, если во время выполнения действия 
        (например, при ошибке) очередь будет экстренно очищена. 
        Фоновые действия выполняются параллельно с одним блокирующим действием.
        Успешно завершенные действия безопасно удаляются из очереди.

        Args:
            world (Any): Экземпляр симулируемого мира.
        """
        if getattr(self, 'last_ticked_count', -1) == world.tick_count:
            return
        self.last_ticked_count = world.tick_count

        self.completed_actions_this_tick.clear()

        for item in list(self.inventory):
            item.tick(world)

        if not self.dynamic or not self.action_queue:
            return

        actions_to_remove = []
        blocking_action_executed = False

        for action in list(self.action_queue):
            if action not in self.action_queue:
                continue 

            if getattr(action, 'is_background', False):
                is_done = action.execute(self, world)
                if is_done:
                    if action in self.action_queue:
                        action.completion_tick = world.tick_count
                        actions_to_remove.append(action)
                        self.completed_actions_this_tick.append(action)
            else:
                if not blocking_action_executed:
                    is_done = action.execute(self, world)
                    blocking_action_executed = True
                    if is_done:
                        if action in self.action_queue:
                            action.completion_tick = world.tick_count
                            actions_to_remove.append(action)
                            self.completed_actions_this_tick.append(action)
        for a in actions_to_remove:
            if a in self.action_queue:
                self.action_queue.remove(a)

    def move(self, target_x: float, target_y: float, nav_mode: str = "hybrid"):
        """Добавляет команду перемещения в очередь."""
        if "move" not in self.behavior:
            self.log_event("Cannot queue 'move': Action not allowed by behavior.", level="ERROR")
            return
        self.action_queue.append(MoveAction(target_x, target_y, nav_mode))

    def rotate(self, target: Any, mode: str = "relative"):
        """Добавляет команду 'поворот' в очередь действий."""
        if "rotate" not in self.behavior:
            self.log_event("Cannot queue 'rotate': Action not allowed by behavior.", level="ERROR")
            return
        self.action_queue.append(RotateAction(target, mode))

    def interact_rotate(self, target_id: str, angle_offset: float):
        """Добавляет команду 'повернуть другой объект' в очередь действий."""
        if "interact_rotate" not in self.behavior:
            self.log_event("Cannot queue 'interact_rotate': Action not allowed by behavior.", level="ERROR")
            return
        self.action_queue.append(InteractRotateAction(target_id, angle_offset))

    def pickup(self, target_item_id: str, container_id: Optional[str] = None):
        """
        Добавляет команду 'подобрать предмет' в очередь действий.
        """
        if "pickup" not in self.behavior:
            self.log_event("Cannot queue 'pickup': Action not allowed by behavior.", level="ERROR")
            return
        self.action_queue.append(PickupAction(target_item_id, container_id))

    def put(self, item_id: str, target_container_id: Optional[str] = None):
        """
        Добавляет команду 'положить/выбросить предмет' в очередь действий.
        """
        if "put" not in self.behavior:
            self.log_event("Cannot queue 'put': Action not allowed by behavior.", level="ERROR")
            return
        self.action_queue.append(PutAction(item_id, target_container_id))

    def enter(self, target_id: str):
        """Добавляет команду 'войти в объект' в очередь действий."""
        if "enter" not in self.behavior:
            self.log_event("Cannot queue 'enter': Action not allowed by behavior.", level="ERROR")
            return
        self.action_queue.append(EnterAction(target_id))

    def exit(self):
        """Добавляет команду 'выйти наружу' в очередь действий."""
        if "exit" not in self.behavior:
            self.log_event("Cannot queue 'exit': Action not allowed by behavior.", level="ERROR")
            return
        self.action_queue.append(ExitAction())

    def break_obj(self, target_id: str):
        """Добавляет команду 'сломать/ударить объект' в очередь действий."""
        if "break" not in self.behavior:
            self.log_event("Cannot queue 'break': Action not allowed by behavior.", level="ERROR")
            return
        self.action_queue.append(BreakAction(target_id))

    def speak(self, message: str, duration: int = 3):
        """Добавляет команду 'говорить' в очередь действий."""
        if "speak" not in self.behavior:
            self.log_event("Cannot queue 'speak': Action not allowed by behavior.", level="ERROR")
            return
        self.action_queue.append(SpeakAction(message, duration))

    def inspect(self, container_path: List[str]):
        """Добавляет команду 'осмотреть контейнер' в очередь действий."""
        if "inspect" not in self.behavior:
            self.log_event("Cannot queue 'inspect': Action not allowed by behavior.", level="ERROR")
            return
        self.action_queue.append(InspectAction(container_path))

    def clear_actions(self):
        """
        Принудительно очищает очередь действий и сбрасывает планировщик.

        Удаляет все запланированные шаги, очищает построенный маршрут 
        и снимает временные флаги состояний (например, 'speaking').
        """
        self.action_queue.clear()
        self.completed_actions_this_tick.clear()
        if self.planner:
            self.planner.current_path = []
            self.planner.current_target = None
        if "speaking" in self.state:
            del self.state["speaking"]

    def _polygon_cache_signature(self) -> Tuple[Any, ...]:
        """Возвращает ключ кэша абсолютного полигона объекта.

        ``shape_coords`` в рантайме только переприсваивается, поэтому
        достаточно идентичности списка и его длины. ``physical_radius``
        входит в ключ, чтобы сброс физики не оставлял устаревший буфер.

        Returns:
            Tuple[Any, ...]: Сигнатура позы, формы и радиуса коллизии.
        """
        return (
            self.x,
            self.y,
            self.angle,
            self.physical_radius,
            id(self.shape_coords),
            len(self.shape_coords),
        )

    def get_polygon(self) -> Polygon:
        """
        Возвращает абсолютный полигон объекта в координатах мира.

        Результат кэшируется: статические объекты строят геометрию один раз,
        динамические — только после смены координат, угла, формы или радиуса.

        Returns:
            Polygon: Объект Shapely Polygon. Если глобальные координаты X, Y 
                отсутствуют (объект в инвентаре), возвращается пустой полигон.
        """
        cache_key = self._polygon_cache_signature()
        cached_polygon = self._cached_polygon
        if cached_polygon is not None and self._polygon_cache_key == cache_key:
            return cached_polygon

        if self.x is None or self.y is None:
            polygon = Polygon()
        elif len(self.shape_coords) >= 3:
            rad = math.radians(-self.angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            
            abs_coords = []
            for dx, dy in self.shape_coords:
                if dx is not None:
                    rot_dx = dx * cos_a - dy * sin_a
                    rot_dy = dx * sin_a + dy * cos_a
                    abs_coords.append((self.x + rot_dx, self.y + rot_dy))
                    
            if len(abs_coords) >= 3:
                polygon = Polygon(abs_coords).buffer(0)
            else:
                polygon = Point(self.x, self.y).buffer(self.physical_radius)
        else:
            polygon = Point(self.x, self.y).buffer(self.physical_radius)

        self._cached_polygon = polygon
        self._polygon_cache_key = cache_key
        return polygon
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализует объект в словарь для JSON. Поля со значением None не добавляются."""
        queue_repr = []
        for action in self.action_queue:
            params = {
                k: v for k, v in action.__dict__.items() 
                if not k.startswith('_') and k not in ('is_path_set', 'started', 'duration')
            }
            queue_repr.append({
                "type": action.__class__.__name__,
                "params": _snapshot_mutable(params)
            })

        data = {
            "id": self.id,
            "object": self.type_name,
            "name": self.type_name,
            "dynamic": self.dynamic,
            "shape": self.shape_coords,
            "inventory": [item.to_dict() for item in self.inventory],
            "affordance": self.affordance,
            "behavior": self.behavior,
            "state": _snapshot_mutable(self.state),
            "action_queue": queue_repr
        }
        
        if self.x is not None and self.y is not None:
            data["x"] = self.x
            data["y"] = self.y
            
        if self.description is not None: data["description"] = self.description
        if self.has_inventory is not None: data["has_inventory"] = self.has_inventory
        if self.angle != 0.0: data["angle"] = self.angle
        if self.speed is not None: data["speed"] = self.speed
        if self.vision_radius is not None: data["vision_radius"] = self.vision_radius
        if self.durability is not None: data["durability"] = self.durability
        if self.max_durability is not None: data["max_durability"] = self.max_durability
        if self.damage is not None: data["damage"] = self.damage
        if self.planner and self.planner.current_path:
            data["planned_path"] = _snapshot_mutable(self.planner.current_path)
        if self.location_id is not None: data["location_id"] = self.location_id
                
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorldObject':
        """Десериализует объект из JSON."""
        obj = cls(
            obj_id=data.get("id", "0"),
            obj_type=data.get("object", data.get("name", "unknown")),
            x=data.get("x"),
            y=data.get("y"),
            angle=data.get("angle", 0.0),
            dynamic=data.get("dynamic", False),
            behavior=data.get("behavior", []),
            description=data.get("description"),
            shape=data.get("shape", []),
            has_inventory=data.get("has_inventory"),
            affordance=data.get("affordance", []),
            speed=data.get("speed"),
            vision_radius=data.get("vision_radius"),
            durability=data.get("durability"),
            max_durability=data.get("max_durability"),
            damage=data.get("damage"),
            location_id=data.get("location_id"),
            state=data.get("state", {})
        )
        
        if "inventory" in data and isinstance(data["inventory"], list):
            obj.inventory = [WorldObject.from_dict(item) for item in data["inventory"]]
        obj.initialize_physics()
        return obj


class Robot(RobotLoggerMixin, WorldObject):
    """
    Класс Робота в симулируемом мире.

    Наследует свойства WorldObject, добавляя специализированные структуры 
    для хранения воспоминаний (виртуальной карты) о мире.

    Атрибуты:
        memory (List['MemoryObject']): Список объектов, увиденных и 
            запомненных роботом на глобальной карте.
        memory_inventory (List['MemoryObject']): Виртуальное представление 
            собственного инвентаря робота.
    """
    def __init__(self, obj_id: str, x: float, y: float, vision_radius: float = 15.0, angle: float = 0.0, fov: float = 90.0, rotation_speed: float = 15.0, eps: float = 0.0):
        """
        Инициализирует робота с полным набором поведений и инвентарем.

        Args:
            obj_id (str): Уникальный идентификатор робота.
            x (float): Начальная координата X.
            y (float): Начальная координата Y.
            vision_radius (float): Радиус обзора (дальность).
            angle (float): Текущий угол поворота в градусах (0 - смотрит вправо/на восток).
            fov (float): Угол сектора обзора в градусах.
            rotation_speed (float): Скорость поворота в градусах за один тик.
            eps (float): Радиус кластеризации. Если 0.0, кластеризация отключена.
        """
        super().__init__(obj_id=obj_id, obj_type="robot", x=x, y=y, dynamic=True, 
                         behavior=["move", "pickup", "put", "enter", "exit", "break", "speak", "rotate", "interact_rotate", "inspect"], vision_radius=vision_radius, has_inventory=True)
        self.memory: List['MemoryObject'] = []
        self.memory_inventory: List['MemoryObject'] = []
        self.currently_visible_objects: List[Dict[str, Any]] = []
        self.angle: float = float(angle)
        self.fov: float = float(fov)
        self.rotation_speed: float = float(rotation_speed)
        self.event_log: List[Dict[str, Any]] = []
        self.clusterer = None
        if eps > 0.0:
            self.clusterer = IncrementalObjectClusterer(eps=eps)
        self.previous_mem_ids: set = set()
        self.current_clusters: list = []
        self.initialize_physics()

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        if self.vision_radius is not None: data["radius"] = self.vision_radius
        data["angle"] = self.angle
        data["fov"] = self.fov
        data["rotation_speed"] = self.rotation_speed
        if hasattr(self, 'current_vision_wkt'):
            data["vision_wkt"] = self.current_vision_wkt
        data["inventory"] = [item.to_dict() for item in self.memory_inventory]
        data["visible_objects"] = getattr(self, "currently_visible_objects", [])
        if self.clusterer:
            data["clusters"] = _snapshot_mutable(self.current_clusters)
        return data
    
    def _resolve_coordinates(self, obj: Any, is_in_robot_inv: bool = False) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Ищет глобальные координаты объекта, поднимаясь вверх по дереву location_id.
        """
        if obj.x is not None and obj.y is not None:
            return round(obj.x, 2), round(obj.y, 2), False
            
        current_loc_id = obj.location_id
        while current_loc_id:
            if current_loc_id == self.id:
                break
                
            parent = self._find_mem_obj_recursively(self.memory, current_loc_id, by_real_id=False)
            if not parent:
                parent = self._find_mem_obj_recursively(self.memory, current_loc_id, by_real_id=True)
            if not parent:
                parent = self._find_mem_obj_recursively(self.memory_inventory, current_loc_id, by_real_id=False)
            if not parent:
                parent = self._find_mem_obj_recursively(self.memory_inventory, current_loc_id, by_real_id=True)
                
            if parent:
                if parent.x is not None and parent.y is not None:
                    return round(parent.x, 2), round(parent.y, 2), True
                current_loc_id = parent.location_id
            else:
                break
                
        if is_in_robot_inv or current_loc_id == self.id:
            if obj.id != self.id:
                rx, ry, _ = self._resolve_coordinates(self, False)
                return rx, ry, True

        return None, None, True

    def _format_mem_obj_for_llm(self, mem_obj: 'MemoryObject', current_tick: int, include_distance: bool = False, include_last_seen: bool = False, is_in_robot_inv: bool = False, include_geometry: bool = False) -> Dict[str, Any]:
        """
        Форматирует объект памяти.
        """
        resolved_x, resolved_y, is_derived = self._resolve_coordinates(mem_obj, is_in_robot_inv)
        
        loc_str = "outside"
        if is_in_robot_inv:
            loc_str = mem_obj.location_id if mem_obj.location_id else self.id
        elif mem_obj.location_id:
            loc_str = mem_obj.location_id

        pos_dict = {"location": loc_str}
        if is_derived:
            pos_dict["parent_x"] = resolved_x
            pos_dict["parent_y"] = resolved_y
        else:
            pos_dict["x"] = resolved_x
            pos_dict["y"] = resolved_y

        data = {
            "id": mem_obj.id,
            "object_type": mem_obj.type_name,
            "description": mem_obj.description if mem_obj.description else "",
            "position": pos_dict,
            "inventory": []
        }

        if mem_obj.inventory:
            data["inventory"] = [
                self._format_mem_obj_for_llm(item, current_tick, include_distance=False, include_last_seen=include_last_seen, is_in_robot_inv=is_in_robot_inv) 
                for item in mem_obj.inventory
                if item.id != self.id and item.real_id != self.id
            ]

        if include_distance:
            robot_x, robot_y, _ = self._resolve_coordinates(self)
            if robot_x is not None and robot_y is not None and resolved_x is not None and resolved_y is not None:
                dist = math.hypot(resolved_x - robot_x, resolved_y - robot_y)
                data["distance"] = round(dist, 2)
            else:
                data["distance"] = None

        if include_last_seen and not getattr(mem_obj, 'is_virtual', False):
            data["last_seen_tick"] = int(mem_obj.timestamp) if str(mem_obj.timestamp).isdigit() else 0

        if getattr(mem_obj, 'notes', ""):
            data["notes"] = mem_obj.notes
            
        if getattr(mem_obj, 'is_virtual', False):
            data["is_virtual"] = True
            data["is_obstacle"] = getattr(mem_obj, 'is_obstacle', False)
            data["creation_tick"] = int(mem_obj.timestamp) if str(mem_obj.timestamp).isdigit() else 0

        data["exploration_status"] = "Full" if getattr(mem_obj, 'is_fully_explored', False) else "Partial"
        
        if include_geometry and getattr(mem_obj, 'absolute_shape', []):
            data["geometry"] = mem_obj.absolute_shape

        return data
    
    def _format_text_md(self, context: Dict[str, Any]) -> str:
        lines = []
        
        def _render_item(item, indent=""):
            desc = f" - {item['description']}" if item.get('description') else ""
            lines.append(f"{indent}- [{item['id']}] ({item['object_type']}){desc}")
            
            pos = item.get('position', {})
            if pos:
                loc = pos.get('location', '')
                if loc: lines.append(f"{indent}  * Location: {loc}")
                if 'x' in pos and pos['x'] is not None:
                    lines.append(f"{indent}  * Position: x={pos['x']}, y={pos['y']}")
                elif 'parent_x' in pos and pos['parent_x'] is not None:
                    lines.append(f"{indent}  * Position: parent_x={pos['parent_x']}, parent_y={pos['parent_y']}")
                    
            if item.get('distance') is not None:
                lines.append(f"{indent}  * Distance: {item['distance']}m")
            if item.get('last_seen_tick') is not None:
                lines.append(f"{indent}  * Last seen at tick: {item['last_seen_tick']}")
            
            if item.get('is_virtual'):
                obs_str = "YES (Blocks navigation)" if item.get('is_obstacle') else "NO (Marker only)"
                lines.append(f"{indent}  * Navigation Obstacle: {obs_str}")
                if item.get('creation_tick') is not None:
                    lines.append(f"{indent}  * Created at tick: {item['creation_tick']}")

            if item.get('notes'):
                lines.append(f"{indent}  * Note: {item['notes']}")
                
            lines.append(f"{indent}  * Exploration: {item.get('exploration_status', 'Partial')}")
            if 'geometry' in item:
                lines.append(f"{indent}  * Geometry points: {item['geometry']}")
                
            inv = item.get('inventory', [])
            if inv:
                lines.append(f"{indent}  * Inventory:")
                for sub in inv:
                    _render_item(sub, indent + "    ")

        if "current_state" in context:
            lines.append("# CURRENT STATE")
            st = context["current_state"]
            lines.append(f"- Tick: {st['current_tick']}")
            pos = st.get('position', {})
            if pos:
                lines.append(f"- Location: {pos.get('location', '')}")
                if 'x' in pos and pos['x'] is not None:
                    lines.append(f"- Position: x={pos['x']}, y={pos['y']}")
                elif 'parent_x' in pos and pos['parent_x'] is not None:
                    lines.append(f"- Position: parent_x={pos['parent_x']}, parent_y={pos['parent_y']}")
            if st.get('angle') is not None:
                lines.append(f"- Angle: {st['angle']}°")
            if st.get('fov') is not None:
                lines.append(f"- FOV: {st['fov']}°")
            if st.get('speed') is not None:
                lines.append(f"- Speed: {st['speed']}")
            if st.get('vision_radius') is not None:
                lines.append(f"- Vision radius: {st['vision_radius']}")
            if st.get('state'):
                for k, v in st['state'].items():
                    lines.append(f"- State ({k}): {v}")
            lines.append("")

        for section in ["inventory", "visible_now", "memory", "virtual_map"]:
            if section in context and context[section]:
                header = "MAP NOTES (VIRTUAL)" if section == "virtual_map" else section.upper().replace('_', ' ')
                lines.append(f"# {header}")
                for item in context[section]:
                    _render_item(item)
                lines.append("")

        return "\n".join(lines).strip()

    def _format_text_yaml(self, context: Dict[str, Any]) -> str:
        lines = []
        
        def _render_item(item, indent=""):
            lines.append(f"{indent}- id: {item['id']}")
            lines.append(f"{indent}  type: {item['object_type']}")
            if item.get('description'):
                lines.append(f"{indent}  desc: \"{item['description']}\"")
                
            pos = item.get('position', {})
            if pos:
                if pos.get('location'): lines.append(f"{indent}  loc: {pos['location']}")
                if 'x' in pos and pos['x'] is not None:
                    lines.append(f"{indent}  pos: {pos['x']}, {pos['y']}")
                elif 'parent_x' in pos and pos['parent_x'] is not None:
                    lines.append(f"{indent}  parent_pos: {pos['parent_x']}, {pos['parent_y']}")
                    
            if item.get('distance') is not None:
                lines.append(f"{indent}  dist: {item['distance']}")
            if item.get('last_seen_tick') is not None:
                lines.append(f"{indent}  last_seen: {item['last_seen_tick']}")
                
            if item.get('is_virtual'):
                lines.append(f"{indent}  is_obstacle: {item.get('is_obstacle', False)}")
                if item.get('creation_tick') is not None:
                    lines.append(f"{indent}  creation_tick: {item['creation_tick']}")

            if item.get('notes'):
                safe_note = str(item['notes']).replace('"', '\\"')
                lines.append(f"{indent}  notes: \"{safe_note}\"")
                
            lines.append(f"{indent}  exploration: {item.get('exploration_status', 'Partial')}")
            if 'geometry' in item:
                lines.append(f"{indent}  geometry: {item['geometry']}")
                
            inv = item.get('inventory', [])
            if inv:
                lines.append(f"{indent}  inv:")
                for sub in inv:
                    _render_item(sub, indent + "    ")

        if "current_state" in context:
            lines.append("# CURRENT_STATE")
            st = context["current_state"]
            lines.append(f"tick: {st['current_tick']}")
            pos = st.get('position', {})
            if pos:
                if pos.get('location'): lines.append(f"loc: {pos['location']}")
                if 'x' in pos and pos['x'] is not None:
                    lines.append(f"pos: {pos['x']}, {pos['y']}")
                elif 'parent_x' in pos and pos['parent_x'] is not None:
                    lines.append(f"parent_pos: {pos['parent_x']}, {pos['parent_y']}")
            if st.get('angle') is not None:
                lines.append(f"angle: {st['angle']}")
            if st.get('fov') is not None:
                lines.append(f"fov: {st['fov']}")
            if st.get('speed') is not None:
                lines.append(f"speed: {st['speed']}")
            if st.get('vision_radius') is not None:
                lines.append(f"vision_radius: {st['vision_radius']}")
            if st.get('state'):
                lines.append("state:")
                for k, v in st['state'].items():
                    lines.append(f"  {k}: {v}")
            lines.append("")

        for section in ["inventory", "visible_now", "memory", "virtual_map"]:
            if section in context and context[section]:
                header = "MAP_NOTES_VIRTUAL" if section == "virtual_map" else section.upper()
                lines.append(f"# {header}")
                for item in context[section]:
                    _render_item(item)
                lines.append("")

        return "\n".join(lines).strip()

    def _format_text_narrative(self, context: Dict[str, Any]) -> str:
        lines = []
        
        def _render_item(item, indent=""):
            desc = f" - {item['description']}" if item.get('description') else ""
            pos = item.get('position', {})
            pos_str = ""
            if pos:
                loc = pos.get('location', '')
                loc_phrase = f"located outside" if loc == "outside" else f"located inside {loc}"
                if 'x' in pos and pos['x'] is not None:
                    pos_str = f" {loc_phrase} at ({pos['x']}, {pos['y']})"
                elif 'parent_x' in pos and pos['parent_x'] is not None:
                    pos_str = f" {loc_phrase} (parent coordinates: {pos['parent_x']}, {pos['parent_y']})"
            
            dist_str = f". Distance: {item['distance']}m" if item.get('distance') is not None else ""
            seen_str = f". Last seen at tick {item['last_seen_tick']}" if item.get('last_seen_tick') is not None else ""
            
            virt_str = ""
            if item.get('is_virtual'):
                obs_word = "an obstacle" if item.get('is_obstacle') else "a marker"
                virt_str = f". This is a virtual {obs_word}"
                if item.get('creation_tick') is not None:
                    virt_str += f" created at tick {item['creation_tick']}"
                    
            note_str = f". Note: {item['notes']}" if item.get('notes') else ""
            exp_str = ". Its full geometry is known" if item.get('exploration_status') == 'Full' else ". It is partially explored"
            geom_str = f". Geometry points: {item['geometry']}" if 'geometry' in item else ""
            
            inv = item.get('inventory', [])
            inv_str = ". It contains:" if inv else "."
            
            lines.append(f"{indent}- A {item['object_type']} (ID: {item['id']}){desc}{pos_str}{dist_str}{seen_str}{virt_str}{note_str}{exp_str}{geom_str}{inv_str}")
            for sub in inv:
                _render_item(sub, indent + "  ")

        if "current_state" in context:
            lines.append("[CURRENT STATE]")
            st = context["current_state"]
            pos = st.get('position', {})
            loc_str = pos.get('location', 'unknown location')
            if 'x' in pos and pos['x'] is not None:
                pos_info = f"at coordinates ({pos['x']}, {pos['y']})"
            elif 'parent_x' in pos and pos['parent_x'] is not None:
                pos_info = f"(parent coordinates: {pos['parent_x']}, {pos['parent_y']})"
            else:
                pos_info = ""
                
            angle_str = f", facing {st['angle']} degrees" if st.get('angle') is not None else ""
            lines.append(f"It is currently tick {st['current_tick']}. You are located {loc_str} {pos_info}{angle_str}.")
            extra_params = []
            if st.get('fov') is not None:
                extra_params.append(f"FOV is {st['fov']}°")
            if st.get('speed') is not None:
                extra_params.append(f"speed is {st['speed']}")
            if st.get('vision_radius') is not None:
                extra_params.append(f"vision radius is {st['vision_radius']}")
            if st.get('state'):
                state_str = ", ".join(f"{k}: {v}" for k, v in st['state'].items())
                extra_params.append(f"state: ({state_str})")
            if extra_params:
                lines.append(f"Status: {', '.join(extra_params)}.")
            lines.append("")

        if "inventory" in context and context["inventory"]:
            lines.append("[INVENTORY]")
            lines.append("You are carrying:")
            for item in context["inventory"]:
                _render_item(item)
            lines.append("")

        if "visible_now" in context and context["visible_now"]:
            lines.append("[VISIBLE NOW]")
            lines.append(f"You currently see {len(context['visible_now'])} objects:")
            for item in context["visible_now"]:
                _render_item(item)
            lines.append("")

        if "memory" in context and context["memory"]:
            lines.append("[MEMORY]")
            lines.append(f"You remember {len(context['memory'])} objects from the past:")
            for item in context["memory"]:
                _render_item(item)
            lines.append("")

        if "virtual_map" in context and context["virtual_map"]:
            lines.append("[MAP NOTES (VIRTUAL)]")
            lines.append(f"You have {len(context['virtual_map'])} virtual notes/markers on your map:")
            for item in context["virtual_map"]:
                _render_item(item)
            lines.append("")

        return "\n".join(lines).strip()

    def get_llm_context(self, current_tick: int, sections: Optional[List[str]] = None, format_style: str = "json", include_geometry: bool = False) -> Any:
        """
        Генерирует контекст для агента.
        """
        if not sections or "all" in sections:
            sections = ["current_state", "visible_now", "memory", "inventory", "virtual_map"]

        context = {}

        if "current_state" in sections:
            robot_x, robot_y, robot_derived = self._resolve_coordinates(self)
            loc_str = "outside"
            if self.location_id:
                loc_str = self._translate_real_to_mem_id(self.location_id)
            
            pos_dict = {"location": loc_str}
            
            if robot_derived:
                pos_dict["parent_x"] = robot_x
                pos_dict["parent_y"] = robot_y
            else:
                pos_dict["x"] = robot_x
                pos_dict["y"] = robot_y

            st_dict = {
                "current_tick": current_tick,
                "position": pos_dict,
                "angle": round(self.angle, 2) if getattr(self, 'angle', None) is not None else None,
                "fov": round(float(self.fov), 2) if getattr(self, 'fov', None) is not None else None,
                "speed": round(float(self.speed), 2) if getattr(self, 'speed', None) is not None else None,
                "vision_radius": round(float(self.vision_radius), 2) if getattr(self, 'vision_radius', None) is not None else None
            }
            if getattr(self, 'state', None) and isinstance(self.state, dict):
                clean_state = {k: round(v, 2) if isinstance(v, float) else v for k, v in self.state.items() if k not in ['feedback', 'last_action_result']}
                if clean_state:
                    st_dict["state"] = clean_state

            context["current_state"] = st_dict

        if "inventory" in sections:
            context["inventory"] = [
                self._format_mem_obj_for_llm(item, current_tick, include_distance=False, include_last_seen=False, is_in_robot_inv=True, include_geometry=include_geometry)
                for item in self.memory_inventory if not getattr(item, 'is_virtual', False)
            ]

        if "visible_now" in sections:
            visible_now = []
            if self.location_id:
                mem_parent = self._find_mem_obj_recursively(self.memory, self.location_id, by_real_id=True)
                if mem_parent:
                    for mem_obj in mem_parent.inventory:
                        if mem_obj.id != self.id and not getattr(mem_obj, 'is_virtual', False):
                            visible_now.append(self._format_mem_obj_for_llm(mem_obj, current_tick, include_distance=True, include_last_seen=False, is_in_robot_inv=False, include_geometry=include_geometry))
            else:
                for mem_obj in self.memory:
                    if not getattr(mem_obj, 'is_virtual', False):
                        obj_tick = int(mem_obj.timestamp) if str(mem_obj.timestamp).isdigit() else 0
                        if obj_tick == current_tick:
                            visible_now.append(self._format_mem_obj_for_llm(mem_obj, current_tick, include_distance=True, include_last_seen=False, is_in_robot_inv=False, include_geometry=include_geometry))
            context["visible_now"] = visible_now

        if "memory" in sections:
            memory = []
            for mem_obj in self.memory_inventory:
                if mem_obj.id != self.id and mem_obj.real_id != self.id and not getattr(mem_obj, 'is_virtual', False):
                    memory.append(self._format_mem_obj_for_llm(mem_obj, current_tick, include_distance=False, include_last_seen=True, is_in_robot_inv=True, include_geometry=include_geometry))
            for mem_obj in self.memory:
                if mem_obj.id != self.id and mem_obj.real_id != self.id and not getattr(mem_obj, 'is_virtual', False):
                    memory.append(self._format_mem_obj_for_llm(mem_obj, current_tick, include_distance=False, include_last_seen=True, is_in_robot_inv=False, include_geometry=include_geometry))
            context["memory"] = memory

        if "virtual_map" in sections:
            virtual_map = []
            for mem_obj in self.memory_inventory:
                if getattr(mem_obj, 'is_virtual', False) and not getattr(mem_obj, 'render_only', False):
                    virtual_map.append(self._format_mem_obj_for_llm(mem_obj, current_tick, include_distance=False, include_last_seen=False, is_in_robot_inv=True, include_geometry=include_geometry))
            for mem_obj in self.memory:
                if getattr(mem_obj, 'is_virtual', False) and not getattr(mem_obj, 'render_only', False):
                    virtual_map.append(self._format_mem_obj_for_llm(mem_obj, current_tick, include_distance=True, include_last_seen=False, is_in_robot_inv=False, include_geometry=include_geometry))
            context["virtual_map"] = virtual_map

        if format_style == "text-md":
            return self._format_text_md(context)
        elif format_style == "text-yaml":
            return self._format_text_yaml(context)
        elif format_style == "text-narrative":
            return self._format_text_narrative(context)

        return context

    def annotate_memory(self, mem_id: str, notes: str, world: Any):
        mem_obj = self._find_mem_obj_recursively(self.memory, mem_id)
        if not mem_obj:
            mem_obj = self._find_mem_obj_recursively(self.memory_inventory, mem_id)
            
        if not mem_obj:
            self.log_event(f"Cannot annotate: Object '{mem_id}' not found in memory.", level="ERROR", world=world, action_code="memory_edit_annotate_fail_not_found", mem_id=mem_id)
            return False
            
        mem_obj.notes = str(notes)
        self.log_event(f"Annotated '{mem_id}' with: {notes}", level="SUCCESS", world=world, action_code="memory_edit_annotate_success", mem_id=mem_id, notes=notes)
        return True

    def add_virtual_object(self, obj_type: str, x: Optional[float], y: Optional[float], is_obstacle: bool, radius: float, shape: list, notes: str, parent_id: Optional[str], inventory_data: list, world: Any, silent: bool = False, render_only: bool = False, color: Optional[str] = None):
        
        def _build_virtual_mem_obj(v_type, v_notes, v_is_obstacle, v_inv_data, v_parent_id, v_x=None, v_y=None, v_radius=0.5, v_shape=None, v_render_only=False, v_color=None):
            new_id = f"virt_{v_type}_{uuid.uuid4().hex[:4]}"
            mem_obj = MemoryObject(mem_id=new_id, real_id=new_id, obj_type=v_type, x=v_x, y=v_y)
            mem_obj.is_virtual = True
            mem_obj.is_obstacle = bool(v_is_obstacle)
            mem_obj.render_only = bool(v_render_only)
            mem_obj.shape = list(v_shape) if v_shape and isinstance(v_shape, list) and len(v_shape) >= 3 else []
            mem_obj.notes = str(v_notes)
            mem_obj.timestamp = str(world.tick_count)
            mem_obj.is_fully_explored = True
            mem_obj.location_id = v_parent_id
            if v_color:
                mem_obj.color = str(v_color)
                if not isinstance(mem_obj.state, dict):
                    mem_obj.state = {}
                mem_obj.state["color"] = mem_obj.color
            
            if v_x is not None and v_y is not None:
                if v_shape and isinstance(v_shape, list) and len(v_shape) >= 3:
                    try:
                        abs_coords = [(v_x + float(dx), v_y + float(dy)) for dx, dy in v_shape if dx is not None and dy is not None]
                        poly = Polygon(abs_coords).buffer(0)
                    except Exception:
                        poly = Point(v_x, v_y).buffer(v_radius)
                else:
                    poly = Point(v_x, v_y).buffer(v_radius)
                    
                mem_obj.absolute_wkt = poly.wkt
                mem_obj.absolute_shape = world._extract_coords_from_poly(poly)
            else:
                mem_obj.absolute_wkt = ""
                mem_obj.absolute_shape = []

            if v_inv_data:
                for sub_data in v_inv_data:
                    sub_obj = _build_virtual_mem_obj(
                        v_type=sub_data.get("obj_type", "unknown"),
                        v_notes=sub_data.get("notes", ""),
                        v_is_obstacle=sub_data.get("is_obstacle", False),
                        v_inv_data=sub_data.get("inventory", []),
                        v_parent_id=new_id,
                        v_x=None, v_y=None, v_radius=0.0,
                        v_render_only=sub_data.get("render_only", v_render_only)
                    )
                    mem_obj.inventory.append(sub_obj)
            return mem_obj

        target_list = self.memory
        if parent_id:
            parent_obj = self._find_mem_obj_recursively(self.memory, parent_id)
            if not parent_obj:
                parent_obj = self._find_mem_obj_recursively(self.memory_inventory, parent_id)
                
            if not parent_obj:
                if not silent:
                    self.log_event(f"Cannot add virtual object: Parent '{parent_id}' not found.", level="ERROR", world=world, action_code="memory_edit_add_fail_parent", obj_type=obj_type)
                return False
                
            target_list = parent_obj.inventory
            x, y = None, None

        new_mem_obj = _build_virtual_mem_obj(obj_type, notes, is_obstacle, inventory_data, parent_id, x, y, radius, shape, render_only, color)
        
        target_list.append(new_mem_obj)
        
        log_args = {
            "mem_id": new_mem_obj.id,
            "obj_type": obj_type,
            "x": x, "y": y,
            "is_obstacle": is_obstacle,
            "radius": radius,
            "shape": shape,
            "parent_id": parent_id,
            "notes": notes,
            "render_only": render_only
        }
        
        if not silent:
            self.log_event("Virtual object creation request.", level="SUCCESS", world=world, action_code="memory_edit_add_success", **log_args)
        return True

    def update_virtual_object(self, mem_id: str, updates: dict, world: Any, silent: bool = False):
        mem_obj = self._find_mem_obj_recursively(self.memory, mem_id)
        if not mem_obj:
            mem_obj = self._find_mem_obj_recursively(self.memory_inventory, mem_id)
            
        if not mem_obj:
            if not silent:
                self.log_event(f"Cannot update: Object '{mem_id}' not found.", level="ERROR", world=world, action_code="memory_edit_update_fail_not_found", mem_id=mem_id)
            return False
            
        if not getattr(mem_obj, 'is_virtual', False):
            if not silent:
                self.log_event(f"Cannot alter physical properties of real object '{mem_id}'. Use 'annotate' instead.", level="ERROR", world=world, action_code="memory_edit_update_fail_real", mem_id=mem_id)
            return False

        applied = {}

        if 'x' in updates: 
            mem_obj.x = float(updates['x'])
            applied['x'] = mem_obj.x
        if 'y' in updates: 
            mem_obj.y = float(updates['y'])
            applied['y'] = mem_obj.y
        if 'obj_type' in updates: 
            mem_obj.type_name = str(updates['obj_type'])
            applied['obj_type'] = mem_obj.type_name
        if 'is_obstacle' in updates: 
            mem_obj.is_obstacle = bool(updates['is_obstacle'])
            applied['is_obstacle'] = mem_obj.is_obstacle
        if 'render_only' in updates:
            mem_obj.render_only = bool(updates['render_only'])
            applied['render_only'] = mem_obj.render_only
        if 'notes' in updates: 
            mem_obj.notes = str(updates['notes'])
            applied['notes'] = mem_obj.notes
        if 'color' in updates:
            mem_obj.color = str(updates['color'])
            if not isinstance(mem_obj.state, dict):
                mem_obj.state = {}
            mem_obj.state['color'] = mem_obj.color
            applied['color'] = mem_obj.color
        if 'shape' in updates:
            shape_val = updates['shape']
            mem_obj.shape = list(shape_val) if shape_val and isinstance(shape_val, list) and len(shape_val) >= 3 else []
            applied['shape'] = mem_obj.shape
        if 'description' in updates: 
            mem_obj.description = str(updates['description'])
            applied['description'] = mem_obj.description

        if any(k in updates for k in ('x', 'y', 'radius', 'shape')):
            cx = mem_obj.x if mem_obj.x is not None else 0.0
            cy = mem_obj.y if mem_obj.y is not None else 0.0
            
            from shapely.geometry import Point, Polygon
            
            shape_data = updates.get('shape', [])
            if shape_data and isinstance(shape_data, list) and len(shape_data) >= 3:
                try:
                    abs_coords = [(cx + float(dx), cy + float(dy)) for dx, dy in shape_data if dx is not None and dy is not None]
                    poly = Polygon(abs_coords).buffer(0)
                    applied['shape'] = shape_data
                except Exception:
                    rad = float(updates.get('radius', 0.5))
                    poly = Point(cx, cy).buffer(rad)
                    if 'radius' in updates: applied['radius'] = rad
            else:
                rad = float(updates.get('radius', 0.5))
                poly = Point(cx, cy).buffer(rad)
                if 'radius' in updates: applied['radius'] = rad
                
            mem_obj.absolute_wkt = poly.wkt
            mem_obj.absolute_shape = world._extract_coords_from_poly(poly)

        if not silent:
            self.log_event("Virtual object updated.", level="SUCCESS", world=world, action_code="memory_edit_update_success", mem_id=mem_id, applied_updates=applied)
        return True

    def delete_virtual_object(self, mem_id: str, world: Any, silent: bool = False):
        def _remove(mem_list, target_id):
            for i, m in enumerate(mem_list):
                if m.id == target_id:
                    if not getattr(m, 'is_virtual', False):
                        return "real_object"
                    mem_list.pop(i)
                    return "success"
                if m.inventory:
                    res = _remove(m.inventory, target_id)
                    if res: return res
            return None

        res = _remove(self.memory, mem_id)
        if not res:
            res = _remove(self.memory_inventory, mem_id)

        if not res:
            if not silent:
                self.log_event(f"Cannot delete: Object '{mem_id}' not found.", level="ERROR", world=world, action_code="memory_edit_delete_fail_not_found", mem_id=mem_id)
            return False
        elif res == "real_object":
            if not silent:
                self.log_event(f"Cannot force delete real object '{mem_id}'.", level="ERROR", world=world, action_code="memory_edit_delete_fail_real", mem_id=mem_id)
            return False

        if not silent:
            self.log_event(f"Deleted virtual object '{mem_id}'.", level="SUCCESS", world=world, action_code="memory_edit_delete_success", mem_id=mem_id)
        return True

class MemoryObject:
    """
    Тень (воспоминание) физического объекта в виртуальной памяти робота.

    Хранит неполную или устаревшую информацию о реальных объектах мира.
    Вместо точных геометрических форм хранит WKT (Well-Known Text) полигонов,
    представляющих видимые части объекта, усеченные "туманом войны".

    Атрибуты:
        id (str): Уникальный виртуальный идентификатор воспоминания.
        real_id (str): ID реального физического объекта, с которым связано воспоминание.
        type_name (str): Тип увиденного объекта.
        x (Optional[float]): Запомненная координата X.
        y (Optional[float]): Запомненная координата Y.
        location_id (Optional[str]): Запомненное местоположение (внутри здания/контейнера).
        dynamic (bool): Флаг подвижности объекта на момент запоминания.
        timestamp (str): Тик симуляции, когда объект был замечен в последний раз.
        absolute_wkt (str): Строковое представление (WKT) видимого полигона объекта.
        absolute_shape (List): Список координат вершин видимого полигона.
        inventory (List['MemoryObject']): Запомненное содержимое инвентаря объекта.
    """
    def __init__(self, mem_id: str, real_id: str, obj_type: str, x: Optional[float] = None, y: Optional[float] = None):
        """
        Инициализирует объект памяти.

        Args:
            mem_id (str): Локальный ID в памяти.
            real_id (str): Ссылка на физический ID в мире.
            obj_type (str): Тип объекта.
            x (Optional[float]): Запомненная координата X.
            y (Optional[float]): Запомненная координата Y.
        """
        self.id: str = str(mem_id)
        self.real_id: str = str(real_id)
        self.type_name: str = obj_type
        self.x: Optional[float] = float(x) if x is not None else None
        self.y: Optional[float] = float(y) if y is not None else None

        self.location_id: Optional[str] = None
        
        self.timestamp: str = "0"
        self.absolute_wkt: str = ""
        self.absolute_shape: List = []
        self._cached_polygon: Optional[Polygon] = None
        self._polygon_cache_key: Optional[Tuple[Any, ...]] = None

        self.description: Optional[str] = None
        self.state: Dict[str, Any] = {}
        
        self.inventory: List['MemoryObject'] = []

        self.is_virtual: bool = False
        self.is_obstacle: bool = False
        self.render_only: bool = False
        self.shape: List = []
        self.notes: str = ""
        self.is_fully_explored: bool = False

    def get_polygon(self) -> Polygon:
        """
        Восстанавливает геометрический полигон объекта из памяти.

        Результат кэшируется по ``(x, y, absolute_wkt)``: повторный разбор
        одной и той же WKT-строки на соседних тиках не выполняется.

        Returns:
            Polygon: Объект Shapely Polygon, воссозданный из строки WKT. 
                Если WKT отсутствует или содержит ошибку, возвращает 
                точечный буфер (Point.buffer) по координатам X, Y.
        """
        cache_key = (self.x, self.y, self.absolute_wkt)
        cached_polygon = self._cached_polygon
        if cached_polygon is not None and self._polygon_cache_key == cache_key:
            return cached_polygon

        if self.x is None or self.y is None:
            polygon = Polygon()
        elif self.absolute_wkt:
            try:
                polygon = shapely.wkt.loads(self.absolute_wkt)
            except Exception:
                polygon = Point(self.x, self.y).buffer(0.4)
        else:
            polygon = Point(self.x, self.y).buffer(0.4)

        self._cached_polygon = polygon
        self._polygon_cache_key = cache_key
        return polygon

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация объектов из памяти."""
        data = {
            "id": self.id,
            "real_id": self.real_id,
            "object": self.type_name,
            "name": self.type_name,
            "timestamp": self.timestamp,
            "absolute_wkt": self.absolute_wkt,
            "absolute_shape": self.absolute_shape,
            "state": _snapshot_mutable(self.state),
            "inventory": [item.to_dict() for item in self.inventory]
        }
        if self.x is not None and self.y is not None:
            data["x"] = self.x
            data["y"] = self.y

        if self.location_id is not None: data["location_id"] = self.location_id
        if self.description is not None: data["description"] = self.description

        if self.is_virtual:
            data["is_virtual"] = self.is_virtual
            data["is_obstacle"] = self.is_obstacle
            if self.render_only:
                data["render_only"] = self.render_only
            if self.shape:
                data["shape"] = self.shape
        if self.notes:
            data["notes"] = self.notes
        if getattr(self, "color", None) or (isinstance(self.state, dict) and self.state.get("color")):
            data["color"] = getattr(self, "color", None) or self.state.get("color")
        data["is_fully_explored"] = self.is_fully_explored

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryObject':
        """Десериализация объектов из памяти."""
        mem_obj = cls(
            mem_id=data.get("id", "0"),
            real_id=data.get("real_id", "0"),
            obj_type=data.get("object", data.get("name", "unknown")),
            x=data.get("x"),
            y=data.get("y")
        )
        mem_obj.timestamp = data.get("timestamp", "0")
        mem_obj.absolute_wkt = data.get("absolute_wkt", "")
        mem_obj.absolute_shape = data.get("absolute_shape", [])
        mem_obj.location_id = data.get("location_id")
        mem_obj.description = data.get("description")
        mem_obj.state = data.get("state", {})

        mem_obj.is_virtual = data.get("is_virtual", False)
        mem_obj.is_obstacle = data.get("is_obstacle", False)
        mem_obj.render_only = data.get("render_only", False)
        mem_obj.shape = data.get("shape", [])
        mem_obj.notes = data.get("notes", "")
        mem_obj.is_fully_explored = data.get("is_fully_explored", False)
        
        if "inventory" in data and isinstance(data["inventory"], list):
            mem_obj.inventory = [MemoryObject.from_dict(item) for item in data["inventory"]]
            
        return mem_obj