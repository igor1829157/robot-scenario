import copy
import json
import os
import math
import uuid
from random import Random
from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Point, Polygon, LineString, MultiPoint
from shapely.strtree import STRtree

from robomap.models import WorldObject, Robot, MemoryObject

import logging
logger = logging.getLogger(__name__)

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",))

class World:
    """
    Класс Мира. Хранит реальную карту, агентов, управляет течением времени 
    и логикой записи реального мира в память робота.

    Атрибуты:
        config (Dict[str, Any]): Глобальная конфигурация движка.
        real_map_data (List[WorldObject]): Список всех физических объектов на верхнем уровне мира.
        robot (Optional[Robot]): Ссылка на робота.
        scenario_controller (Optional[Any]): Контроллер текущего сценария/плагина.
        tick_count (int): Текущий такт симуляции.
        map_boundaries (Tuple[List[float], List[float]]): Текущие границы карты [X_min, X_max], [Y_min, Y_max] для камеры.
        blueprints (Dict[str, Any]): Шаблоны характеристик объектов.
    """

    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Инициализирует пустое пространство мира и загружает конфигурацию.

        Args:
            config_dict (Dict[str, Any]): Собранный словарь настроек (глобальный + сценарий).
        """

        self.config: Dict[str, Any] = config_dict if config_dict is not None else {}
        
        self.real_map_data: List[WorldObject] = []
        self.robots: Dict[str, Robot] = {}

        self.active_sound_sources: Dict[str, str] = {}
        
        self.scenario_controller = None

        self.tick_count: int = 0

        self.map_boundaries: Tuple[List[float], List[float]] = ([-50, 50], [-50, 50])

        self.blueprints: Dict[str, Any] = {}
        self._collision_tree: Optional[STRtree] = None
        self._collision_tree_ids: List[str] = []
        self._collision_index_ids: Tuple[str, ...] = ()
        self._collision_index_keys: Dict[str, Tuple[Any, ...]] = {}
        self._load_blueprints()


    def tick(self):
        """
        Продвигает симуляцию на один логический такт вперед.

        Выполняется в строгом порядке:
        1. Отработка глобальных правил сценария (раздача команд NPC).
        2. Физический шаг всех динамических объектов (обработка очередей действий).
        3. Физический шаг робота.
        4. Обновление виртуальной памяти робота.
        5. Перерасчет границ камеры для рендера.
        """
        self.tick_count += 1
        self.active_sound_sources.clear()
        pre_tick_actions = {}
        for obj in list(self.real_map_data):
            if obj.dynamic and obj.action_queue:
                pre_tick_actions[obj.id] = obj.action_queue[0]

        if self.scenario_controller:
            self.scenario_controller.tick(self)

        for obj in list(self.real_map_data):
            obj.tick(self)
        
        for robot in list(self.robots.values()):
            robot.tick(self)
            robot.pre_tick_actions = pre_tick_actions

        for robot in list(self.robots.values()):
            self._update_robot_memory(robot)

        self._calculate_boundaries()

    def get_object_by_id(self, obj_id: str) -> Optional[WorldObject]:
        """
        Рекурсивно ищет физический объект в мире по его идентификатору.

        Поиск осуществляется как на глобальной карте, так и внутри 
        всех вложенных инвентарей (включая инвентарь самого робота).

        Args:
            obj_id (str): Уникальный идентификатор искомого объекта.

        Returns:
            Optional[WorldObject]: Найденный объект или None, если объект не существует.
        """
        def _search(obj_list):
            for obj in obj_list:
                if obj.id == str(obj_id): return obj
                if obj.inventory:
                    res = _search(obj.inventory)
                    if res: return res
            return None
            
        res = _search(self.real_map_data)
        if res: return res
        
        for robot in self.robots.values():
            if robot.id == str(obj_id): return robot
            res = _search(robot.inventory)
            if res: return res

        return None

    def add_object(self, obj_type: str, x: Optional[float], y: Optional[float], 
                   dynamic: bool = False, **kwargs) -> Optional[WorldObject]:
        """
        Создает и добавляет новый объект в мир на основе шаблонов (blueprints).

        Args:
            obj_type (str): Строковой тип объекта (например, 'tree', 'bear').
            x (Optional[float]): Глобальная координата X. Если None, объект 
                считается находящимся в абстрактном пространстве (инвентаре).
            y (Optional[float]): Глобальная координата Y.
            dynamic (bool): Определяет, может ли объект выполнять действия.
            **kwargs: Переопределение параметров (obj_id, shape, behavior и т.д.).

        Returns:
            Optional[WorldObject]: Экземпляр созданного объекта.
        """
        new_id = kwargs.pop('obj_id', f"{obj_type}_{uuid.uuid4().hex[:3]}")

        blueprint = self._get_blueprint(obj_type)
        is_dynamic = kwargs.get('dynamic', blueprint.get('dynamic', dynamic))
        new_angle = kwargs.get('angle', blueprint.get('angle', 0.0))
        
        new_obj = WorldObject(obj_id=new_id, obj_type=obj_type, x=x, y=y, dynamic=is_dynamic, angle=new_angle)
        
        new_obj.description = kwargs.get('description', blueprint.get('description', ''))
        new_obj.shape_coords = kwargs.get('shape', blueprint.get('shape', []))
        new_obj.behavior = kwargs.get('behavior', blueprint.get('behavior', []))
        new_obj.affordance = kwargs.get('affordance', blueprint.get('affordance', []))
        
        new_obj.initialize_physics()

        if x is not None and y is not None:
            poly = new_obj.get_polygon()
            if self.check_collision(poly, ignore_ids=[new_obj.id]):
                logger.warning(
                    "SPAWN WARNING | Объект '%s:%s' заспавнен с наложением на другие объекты в координатах (%.1f, %.1f)!",
                    obj_type, new_id, x, y
                )

        self.real_map_data.append(new_obj)
        logger.info("[%d] Добавлен объект %s (ID:%s) на (%s, %s)", self.tick_count, obj_type, new_id, x, y)
        
        return new_obj

    def update_object(self, obj_id: str, new_x: Optional[float] = None, new_y: Optional[float] = None, 
                      new_type: Optional[str] = None, **kwargs) -> bool:
        """
        Динамически обновляет свойства существующего физического объекта.

        Специальная обработка применяется для ключа 'shape_coords' (требует 
        переинициализации физики коллизий) и словаря 'state' (обновляется методом update).

        Args:
            obj_id (str): Идентификатор обновляемого объекта.
            new_x (Optional[float]): Новая координата X.
            new_y (Optional[float]): Новая координата Y.
            new_type (Optional[str]): Новый визуальный тип объекта.
            **kwargs: Любые другие атрибуты для изменения (durability, speed и т.д.).

        Returns:
            bool: True, если объект успешно найден и обновлен, иначе False.
        """
        obj = self.get_object_by_id(obj_id)
        if not obj:
            logger.error("Ошибка: Объект с ID %s не найден для обновления.", obj_id)
            return False

        if new_x is not None: obj.x = float(new_x)
        if new_y is not None: obj.y = float(new_y)
        if new_type is not None: obj.type_name = new_type

        for k, v in kwargs.items():
            if k == "shape_coords":
                obj.shape_coords = v
                obj.initialize_physics()
            elif k == "angle":
                obj.angle = float(v)
            elif k == "state" and isinstance(v, dict):
                obj.state.update(v)
            elif hasattr(obj, k):
                setattr(obj, k, v)
            else:
                obj.state[k] = v

        logger.info("[%d] Обновлен объект ID:%s", self.tick_count, obj_id)
        return True

    def delete_object(self, obj_id: str) -> bool:
        """
        Args:
            obj_id (str): Идентификатор удаляемого объекта.

        Returns:
            bool: True, если объект был найден и удален.
        """
        if obj_id in self.robots:
            del self.robots[obj_id]
            logger.info("[%d] Удален робот ID:%s", self.tick_count, obj_id)
            return True

        original_len = len(self.real_map_data)
        self.real_map_data = [obj for obj in self.real_map_data if obj.id != str(obj_id)]
        
        if len(self.real_map_data) < original_len:
            logger.info("[%d] Удален объект ID:%s", self.tick_count, obj_id)
            return True
        return False

    def _extract_mem_obj_by_real_id(self, target_real_id: str, search_list: List[MemoryObject]) -> Optional[MemoryObject]:
        """
        Ищет и извлекает (удаляет) объект из дерева виртуальной памяти.

        Используется для перемещения известных объектов между локациями в памяти 
        (например, с земли в инвентарь) без клонирования объектов.

        Args:
            target_real_id (str): Физический ID искомого объекта.
            search_list (List[MemoryObject]): Список объектов памяти для рекурсивного поиска.

        Returns:
            Optional[MemoryObject]: Извлеченный объект памяти или None.
        """
        for i, m in enumerate(search_list):
            if m.real_id == target_real_id:
                return search_list.pop(i)
            if m.inventory:
                found = self._extract_mem_obj_by_real_id(target_real_id, m.inventory)
                if found: return found
        return None

    def _extract_from_any_inventory(self, target_real_id: str, vision_circle: Polygon, robot: Robot) -> Optional[MemoryObject]:
        """
        Ищет объект памяти внутри контейнеров с учетом ограничений видимости.

        Реализует логику "Тумана войны" для инвентарей: объект может быть 
        извлечен из контейнера только в том случае, если сам контейнер 
        в данный момент находится в зоне видимости робота (или если робот 
        находится внутри этого же контейнера/здания).

        Args:
            target_real_id (str): Физический ID искомого объекта.
            vision_circle (Polygon): Текущий полигон зоны видимости робота.
            robot (Robot): Робот, для которого выполняется поиск.
        Returns:
            Optional[MemoryObject]: Извлеченный объект памяти или None.
        """
        found = self._extract_mem_obj_by_real_id(target_real_id, robot.memory_inventory)
        if found: return found

        if robot.location_id:
            mem_parent = next((m for m in robot.memory if m.real_id == robot.location_id), None)
            if mem_parent:
                found = self._extract_mem_obj_by_real_id(target_real_id, mem_parent.inventory)
                if found: return found

        for m in robot.memory:
            if m.inventory:
                try:
                    mem_poly = m.get_polygon()
                    if vision_circle.intersects(mem_poly):
                        found = self._extract_mem_obj_by_real_id(target_real_id, m.inventory)
                        if found: return found
                except Exception:
                    pass
        return None

    def _sync_inventory_to_memory(self, real_inventory: List[WorldObject], mem_inventory: List[MemoryObject], 
                                  vision_circle: Polygon, updated_ids: set, robot: Robot, parent_mem_id: Optional[str] = None,
                                  depth_limit: int = 0, scanned_containers: set = None):
        """
        Синхронизирует содержимое реального инвентаря с виртуальным инвентарем в памяти.

        Сопоставляет реальные объекты с объектами в памяти. Обновляет метки 
        времени (timestamps), обнуляет глобальные координаты вложенных объектов 
        и проставляет ссылки на родительский контейнер (`location_id`). 
        Специальное правило предотвращает рекурсивное дублирование инвентаря 
        самого робота в его собственной памяти.

        Args:
            real_inventory (List[WorldObject]): Эталонный физический инвентарь.
            mem_inventory (List[MemoryObject]): Обновляемый виртуальный инвентарь.
            vision_circle (Polygon): Полигон зоны видимости (для поиска перемещенных предметов).
            updated_ids (set): Множество ID, обновленных на текущем тике.
            robot (Robot): Робот, для которого выполняется синхронизация.
            parent_mem_id (Optional[str]): Идентификатор родительского контейнера в памяти.
        """
        if parent_mem_id and scanned_containers is not None:
            scanned_containers.add(parent_mem_id)

        for r_item in real_inventory:
            if robot and r_item.id == robot.id:
                existing_robot_mem = next((m for m in mem_inventory if m.id == r_item.id), None)
                if not existing_robot_mem:
                    existing_robot_mem = MemoryObject(mem_id=r_item.id, real_id=r_item.id, obj_type=r_item.type_name)
                    mem_inventory.append(existing_robot_mem)
                
                existing_robot_mem.timestamp = str(self.tick_count)
                existing_robot_mem.x, existing_robot_mem.y = None, None
                existing_robot_mem.absolute_wkt, existing_robot_mem.absolute_shape = "", []
                existing_robot_mem.location_id = parent_mem_id
                existing_robot_mem.description = r_item.description
                existing_robot_mem.state = r_item.state
                updated_ids.add(existing_robot_mem.id)
                continue

            existing_sub_mem = next((m for m in mem_inventory if m.real_id == r_item.id), None)
            is_new_addition = False
            
            if not existing_sub_mem:
                is_new_addition = True
                existing_sub_mem = self._extract_from_any_inventory(r_item.id, vision_circle, robot)
                
                if not existing_sub_mem:
                    for i, m in enumerate(robot.memory):
                        if m.real_id == r_item.id:
                            try:
                                mem_poly = m.get_polygon()
                                if vision_circle.intersects(mem_poly):
                                    existing_sub_mem = robot.memory.pop(i)
                                    break
                            except Exception:
                                pass
                
                if existing_sub_mem:
                    mem_inventory.append(existing_sub_mem)
            
            if existing_sub_mem:
                existing_sub_mem.timestamp = str(self.tick_count)
                existing_sub_mem.x, existing_sub_mem.y = None, None
                existing_sub_mem.absolute_wkt, existing_sub_mem.absolute_shape = "", []
                existing_sub_mem.location_id = parent_mem_id
                existing_sub_mem.type_name = r_item.type_name
                existing_sub_mem.description = r_item.description
                existing_sub_mem.state = r_item.state
                existing_sub_mem.is_fully_explored = True
                updated_ids.add(existing_sub_mem.id)
                
                if robot and hasattr(robot, "log_observation"):
                    robot.log_observation(self.tick_count, existing_sub_mem, r_item, world=self)
            else:
                new_sub_id = f"mem_{r_item.type_name}_{uuid.uuid4().hex[:3]}"
                new_sub_mem = MemoryObject(mem_id=new_sub_id, real_id=r_item.id, obj_type=r_item.type_name)
                new_sub_mem.timestamp = str(self.tick_count)
                new_sub_mem.location_id = parent_mem_id
                new_sub_mem.description = r_item.description
                new_sub_mem.state = r_item.state
                new_sub_mem.is_fully_explored = True
                mem_inventory.append(new_sub_mem)
                updated_ids.add(new_sub_mem.id)
                existing_sub_mem = new_sub_mem
                
                if robot and hasattr(robot, "log_observation"):
                    robot.log_observation(self.tick_count, existing_sub_mem, r_item)
            
            if is_new_addition and parent_mem_id and parent_mem_id != robot.id:
                if hasattr(robot, "log_inventory_change"):
                    robot.log_inventory_change(self.tick_count, "ADDED", parent_mem_id, existing_sub_mem)

            is_enterable = "enter" in r_item.affordance if r_item.affordance else False

            if r_item.has_inventory and depth_limit > 0 and not is_enterable:
                self._sync_inventory_to_memory(
                    r_item.inventory, 
                    existing_sub_mem.inventory, 
                    vision_circle, 
                    updated_ids, 
                    robot, 
                    existing_sub_mem.id,
                    depth_limit=depth_limit - 1,
                    scanned_containers=scanned_containers
                )

    def _cleanup_nested_inventory(self, mem_inv: List[MemoryObject], updated_ids: set, is_visible: bool, 
                                  robot: Robot = None, parent_mem_id: str = None, scanned_containers: set = None):
        """
        Рекурсивно удаляет пропавшие предметы из инвентаря в самом конце тика.
        
        Args:
            mem_inv (List[MemoryObject]): Список объектов памяти в контейнере.
            updated_ids (set): Множество актуальных ID на текущем тике.
            is_visible (bool): Находится ли родительский контейнер в поле зрения.
        """
        if scanned_containers is not None and parent_mem_id not in scanned_containers:
            return

        to_keep = []
        for sub in mem_inv:
            keep = False
            if getattr(sub, 'is_virtual', False):
                keep = True
            elif sub.id in updated_ids:
                keep = True
            elif not is_visible:
                keep = True
            
            if keep:
                if sub.inventory:
                    if not scanned_containers or sub.id in scanned_containers:
                        self._cleanup_nested_inventory(
                            sub.inventory, 
                            updated_ids, 
                            is_visible, 
                            robot, 
                            sub.id, 
                            scanned_containers
                        )
                to_keep.append(sub)
            else:
                if robot and parent_mem_id and parent_mem_id != robot.id:
                    if hasattr(robot, "log_inventory_change"):
                        robot.log_inventory_change(self.tick_count, "REMOVED", parent_mem_id, sub)
        
        mem_inv[:] = to_keep

    def _ensure_container_in_memory(self, robot: Robot, container_id: str) -> Any:
        mem_obj = robot._find_mem_obj_recursively(robot.memory, container_id, by_real_id=True)
        if mem_obj:
            return mem_obj
            
        container = self.get_object_by_id(container_id)
        if not container:
            return None
            
        mem_obj = MemoryObject(
            mem_id=f"mem_{container.type_name}_{uuid.uuid4().hex[:3]}", 
            real_id=container.id, 
            obj_type=container.type_name,
            x=container.x,
            y=container.y
        )
        mem_obj.location_id = None
        mem_obj.description = container.description
        mem_obj.state = container.state.copy() if container.state else {}
        
        if container.location_id:
            parent_mem = self._ensure_container_in_memory(robot, container.location_id)
            if parent_mem:
                mem_obj.location_id = parent_mem.id
                parent_mem.inventory.append(mem_obj)
            else:
                robot.memory.append(mem_obj)
        else:
            robot.memory.append(mem_obj)
            
        return mem_obj

    @staticmethod
    def _index_memory_by_real_id(
        memory: List[MemoryObject],
    ) -> Dict[str, List[MemoryObject]]:
        """Строит индекс верхнеуровневых воспоминаний по real_id."""
        index: Dict[str, List[MemoryObject]] = {}
        for mem_obj in memory:
            index.setdefault(mem_obj.real_id, []).append(mem_obj)
        return index

    def _update_robot_memory(self, robot: Robot):
        """
        Выполняет полный цикл обновления виртуальной карты (памяти) робота.

        Определяет, что робот "видит" на текущем тике. 
        Логика разделена на два режима:
        1. Внутри здания: Робот сканирует только содержимое помещения, 
           игнорируя глобальную карту (улицу).
        2. На улице: Вычисляется пересечение полигонов физических объектов 
           с кругом обзора робота. Обновляются формы объектов (усеченные 
           туманом войны). 
        В обоих режимах происходит полная синхронизация инвентарей видимых контейнеров.
        """
        if not robot: return

        self._extract_mem_obj_by_real_id(robot.id, robot.memory)

        if hasattr(robot, 'angle') and hasattr(robot, 'fov'):
            vision_polygon = self._create_vision_sector(
                robot.x or 0.0, 
                robot.y or 0.0, 
                robot.vision_radius, 
                robot.angle, 
                robot.fov
            )
        else:
            vision_polygon = Point(robot.x or 0.0, robot.y or 0.0).buffer(robot.vision_radius)
        memory_ids_updated_this_tick = set()
        scanned_containers = set()
        self._sync_inventory_to_memory(robot.inventory, robot.memory_inventory, vision_polygon, memory_ids_updated_this_tick, robot, robot.id, depth_limit=99, scanned_containers=scanned_containers)
        visible_now = []

        if robot.location_id:
            parent_container = self.get_object_by_id(robot.location_id)
            if parent_container:
                mem_parent = self._ensure_container_in_memory(robot, robot.location_id)
                
                self._sync_inventory_to_memory(
                    parent_container.inventory, 
                    mem_parent.inventory, 
                    vision_polygon, 
                    memory_ids_updated_this_tick, 
                    robot, 
                    mem_parent.id,
                    depth_limit=1,
                    scanned_containers=scanned_containers
                )
                mem_parent.timestamp = str(self.tick_count)
                memory_ids_updated_this_tick.add(mem_parent.id)
                
                self._cleanup_nested_inventory(
                    mem_parent.inventory, 
                    memory_ids_updated_this_tick, 
                    is_visible=True, 
                    robot=robot, 
                    parent_mem_id=mem_parent.id,
                    scanned_containers=scanned_containers
                )
                
                for item in parent_container.inventory:
                    if item.id != robot.id:
                        mem_item = next((m for m in mem_parent.inventory if m.real_id == item.id), None)
                        if mem_item:
                            mem_item.description = item.description
                            mem_item.state = item.state
                            visible_now.append(mem_item.to_dict())
            
            self._cleanup_nested_inventory(
                robot.memory_inventory, 
                memory_ids_updated_this_tick, 
                is_visible=True, 
                robot=robot, 
                parent_mem_id=robot.id,
                scanned_containers=scanned_containers
            )
            self._update_robot_hearing(robot, memory_ids_updated_this_tick)
            
            robot.currently_visible_objects = visible_now
            return

        if robot.x is None or robot.y is None: return

        if hasattr(robot, 'angle') and hasattr(robot, 'fov'):
            base_vision_polygon = self._create_vision_sector(
                robot.x, robot.y, robot.vision_radius, robot.angle, robot.fov
            )
        else:
            base_vision_polygon = Point(robot.x, robot.y).buffer(robot.vision_radius)

        top_level_objects = self.real_map_data.copy()
        for other_robot in self.robots.values():
            if other_robot.id != robot.id and other_robot.location_id is None:
                top_level_objects.append(other_robot)

        current_vision_polygon = base_vision_polygon

        objects_in_range = []
        for real_obj in top_level_objects:
            if real_obj.id == robot.id or real_obj.x is None or real_obj.y is None:
                continue
            dist = math.hypot(real_obj.x - robot.x, real_obj.y - robot.y)
            if dist <= robot.vision_radius + real_obj.physical_radius + 0.5:
                rescued_mem_obj = self._extract_from_any_inventory(
                    real_obj.id, current_vision_polygon, robot
                )
                if rescued_mem_obj:
                    robot.memory.append(rescued_mem_obj)
                objects_in_range.append((dist, real_obj))

        objects_in_range.sort(key=lambda item: item[0])
        memory_by_real_id = self._index_memory_by_real_id(robot.memory)

        for dist, real_obj in objects_in_range:
            real_poly = real_obj.get_polygon()
            
            if not current_vision_polygon.intersects(real_poly):
                continue
                            
            rescued_mem_obj = self._extract_from_any_inventory(real_obj.id, current_vision_polygon, robot)
            if rescued_mem_obj:
                robot.memory.append(rescued_mem_obj)
                memory_by_real_id.setdefault(rescued_mem_obj.real_id, []).append(
                    rescued_mem_obj
                )

            if real_obj.dynamic:
                seen_chunk = real_poly
            else:
                temp_shadow = self._create_shadow_polygon(robot.x, robot.y, real_poly, robot.vision_radius)
                
                if temp_shadow and not temp_shadow.is_empty:
                    stopped_vision = current_vision_polygon.difference(temp_shadow)
                    penetrating_vision = stopped_vision.buffer(0.3)
                    seen_chunk = real_poly.intersection(penetrating_vision)
                else:
                    seen_chunk = real_poly.intersection(current_vision_polygon)

                if seen_chunk.is_empty:
                    seen_chunk = real_poly.intersection(current_vision_polygon)

            matches = list(memory_by_real_id.get(real_obj.id, []))
            existing_mem_obj = None

            if matches:
                def match_weight(m):
                    if m.x is None: return -1.0
                    return math.hypot(m.x - real_obj.x, m.y - real_obj.y)
                matches.sort(key=match_weight)
                best_match = matches[0]
                if best_match.x is None or match_weight(best_match) <= robot.vision_radius + 0.5:
                    existing_mem_obj = best_match

            if not existing_mem_obj:
                proximity_matches = []
                r_desc = real_obj.description or ""
                
                for m in robot.memory:
                    if getattr(m, 'is_virtual', False):
                        continue
                    if m.x is not None and m.y is not None:
                        m_desc = m.description or ""
                        if m.type_name == real_obj.type_name and m_desc == r_desc:
                            dist = math.hypot(m.x - real_obj.x, m.y - real_obj.y)
                            if dist <= 0.5:
                                proximity_matches.append((dist, m))
                
                if proximity_matches:
                    proximity_matches.sort(key=lambda item: item[0])
                    best_prox_match = proximity_matches[0][1]
                    
                    best_prox_match.real_id = real_obj.id
                    existing_mem_obj = best_prox_match
                    memory_by_real_id.setdefault(real_obj.id, []).append(
                        best_prox_match
                    )

            if existing_mem_obj:
                if hasattr(robot, "log_perception_change"): robot.log_perception_change(self.tick_count, existing_mem_obj, real_obj)
                if hasattr(robot, "log_observation"): robot.log_observation(self.tick_count, existing_mem_obj, real_obj, world=self)

                existing_mem_obj.x, existing_mem_obj.y = real_obj.x, real_obj.y
                existing_mem_obj.dynamic = real_obj.dynamic
                existing_mem_obj.type_name = real_obj.type_name
                existing_mem_obj.timestamp = str(self.tick_count)
                existing_mem_obj.description = real_obj.description
                preserved_keys = {}
                for key, value in existing_mem_obj.state.items():
                    if key.startswith("completed_") or key == "last_observed_action":
                        preserved_keys[key] = value
                existing_mem_obj.state = copy.deepcopy(real_obj.state)
                existing_mem_obj.state.update(preserved_keys)
                if getattr(real_obj, "durability", None) is not None: 
                    existing_mem_obj.state["durability"] = real_obj.durability
                    existing_mem_obj.state["max_durability"] = getattr(real_obj, "max_durability", real_obj.durability)
                
                if real_obj.dynamic:
                    final_poly = seen_chunk
                else:
                    try:
                        if existing_mem_obj.absolute_wkt:
                            old_poly = existing_mem_obj.get_polygon()
                            visible_empty_space = current_vision_polygon.difference(real_poly)
                            
                            cut_area = old_poly.intersection(visible_empty_space).area if not old_poly.is_empty else 0.0
                            new_area = seen_chunk.difference(old_poly).area if not old_poly.is_empty else seen_chunk.area

                            if cut_area < 0.05 and new_area < 0.05:
                                final_poly = old_poly
                            else:
                                if cut_area >= 0.05:
                                    old_poly = old_poly.difference(visible_empty_space)
                                if new_area >= 0.05:
                                    old_poly = old_poly.union(seen_chunk)
                                
                                final_poly = old_poly.simplify(0.03, preserve_topology=True)

                                v_cnt = len(final_poly.exterior.coords) if hasattr(final_poly, "exterior") else sum(len(p.exterior.coords) for p in getattr(final_poly, "geoms", []))
                                if v_cnt > 50:
                                    final_poly = final_poly.simplify(0.08, preserve_topology=True)
                        else:
                            final_poly = seen_chunk.simplify(0.03, preserve_topology=True)
                    except Exception:
                        final_poly = seen_chunk

                old_explored = getattr(existing_mem_obj, 'is_fully_explored', False)
                new_explored = False
                
                if not real_obj.dynamic and real_poly.geom_type in ['Polygon', 'MultiPolygon']:
                    seen_perimeter = final_poly.intersection(real_poly.boundary).length
                    total_perimeter = real_poly.boundary.length
                    if total_perimeter > 0 and seen_perimeter >= total_perimeter * 0.85:
                        final_poly = real_poly
                        new_explored = True
                elif real_obj.dynamic:
                    new_explored = True
                    
                if old_explored != new_explored:
                    if hasattr(robot, "log_exploration_change"):
                        robot.log_exploration_change(self.tick_count, existing_mem_obj.id, new_explored)
                        
                existing_mem_obj.is_fully_explored = new_explored

                existing_mem_obj.absolute_wkt = final_poly.wkt
                existing_mem_obj.absolute_shape = self._extract_coords_from_poly(final_poly)

                if real_obj.has_inventory and "enter" not in real_obj.affordance:
                    self._sync_inventory_to_memory(
                        real_obj.inventory, 
                        existing_mem_obj.inventory,
                        current_vision_polygon, 
                        memory_ids_updated_this_tick, 
                        robot, 
                        existing_mem_obj.id,
                        depth_limit=0,
                        scanned_containers=scanned_containers
                    )
                memory_ids_updated_this_tick.add(existing_mem_obj.id)
                visible_now.append(existing_mem_obj.to_dict())

            else:
                new_mem_id = f"mem_{real_obj.type_name}_{uuid.uuid4().hex[:3]}"
                new_mem_obj = MemoryObject(
                    mem_id=new_mem_id, real_id=real_obj.id, obj_type=real_obj.type_name, x=real_obj.x, y=real_obj.y
                )
                new_mem_obj.dynamic = real_obj.dynamic
                new_mem_obj.timestamp = str(self.tick_count)
                new_mem_obj.description = real_obj.description
                new_mem_obj.state = copy.deepcopy(real_obj.state)
                if getattr(real_obj, "durability", None) is not None: 
                    new_mem_obj.state["durability"] = real_obj.durability
                    new_mem_obj.state["max_durability"] = getattr(real_obj, "max_durability", real_obj.durability)

                new_explored = False
                final_poly = seen_chunk
                if not real_obj.dynamic and real_poly.geom_type in ['Polygon', 'MultiPolygon']:
                    seen_perimeter = final_poly.intersection(real_poly.boundary).length
                    total_perimeter = real_poly.boundary.length
                    if total_perimeter > 0 and seen_perimeter >= total_perimeter * 0.85:
                        final_poly = real_poly
                        new_explored = True
                elif real_obj.dynamic:
                    new_explored = True
                    
                new_mem_obj.is_fully_explored = new_explored
                    
                new_mem_obj.absolute_wkt = final_poly.wkt
                new_mem_obj.absolute_shape = self._extract_coords_from_poly(final_poly)

                if hasattr(robot, "log_perception"):
                    robot.log_perception(self.tick_count, "NEW", new_mem_obj)
                if hasattr(robot, "log_observation"):
                    robot.log_observation(self.tick_count, new_mem_obj, real_obj, world=self)

                robot.memory.append(new_mem_obj)
                memory_by_real_id.setdefault(real_obj.id, []).append(new_mem_obj)
                 
                if real_obj.has_inventory and "enter" not in real_obj.affordance:
                    self._sync_inventory_to_memory(
                        real_obj.inventory, 
                        new_mem_obj.inventory,
                        current_vision_polygon, 
                        memory_ids_updated_this_tick, 
                        robot, 
                        new_mem_obj.id,
                        depth_limit=0,
                        scanned_containers=scanned_containers
                    )
                
                memory_ids_updated_this_tick.add(new_mem_id)
                visible_now.append(new_mem_obj.to_dict())

            if real_obj.physical_radius >= 0.4:
                shadow_poly = self._create_shadow_polygon(robot.x, robot.y, real_poly, robot.vision_radius)
                if shadow_poly:
                    current_vision_polygon = current_vision_polygon.difference(shadow_poly)

        self._cleanup_nested_inventory(
            robot.memory_inventory, 
            memory_ids_updated_this_tick, 
            is_visible=True, 
            robot=robot, 
            parent_mem_id=robot.id,
            scanned_containers=scanned_containers
        )
        mem_objects_to_keep = []
        for mem_obj in robot.memory:
            keep = False
            is_visible = False
            if getattr(mem_obj, 'is_virtual', False):
                keep = True
                is_visible = False
            elif mem_obj.id in memory_ids_updated_this_tick:
                keep = True
                is_visible = True
            else:
                try:
                    mem_poly = mem_obj.get_polygon()
                    if current_vision_polygon.intersects(mem_poly):
                        is_visible = True
                        erased_poly = mem_poly.difference(current_vision_polygon)
                        
                        if erased_poly.area > 0.05:
                            mem_obj.absolute_wkt = erased_poly.wkt
                            mem_obj.absolute_shape = self._extract_coords_from_poly(erased_poly)
                            keep = True
                        else:
                            keep = False
                except Exception:
                    pass
                
                if not is_visible:
                    keep = True
            
            if keep:
                if mem_obj.inventory:
                    self._cleanup_nested_inventory(
                        mem_obj.inventory, 
                        memory_ids_updated_this_tick, 
                        is_visible, 
                        robot=robot, 
                        parent_mem_id=mem_obj.id,
                        scanned_containers=scanned_containers
                    )
                mem_objects_to_keep.append(mem_obj)

        if hasattr(robot, "log_perception"):
            deleted_objs = [m for m in robot.memory if m not in mem_objects_to_keep]
            for d_obj in deleted_objs:
                robot.log_perception(self.tick_count, "LOST", d_obj)

        robot.current_vision_wkt = current_vision_polygon.wkt
        robot.memory = mem_objects_to_keep
        robot.currently_visible_objects = visible_now

        self._update_robot_hearing(robot, memory_ids_updated_this_tick)

        if hasattr(robot, "sort_current_tick_logs"):
            robot.sort_current_tick_logs(self.tick_count)

        def _remove_self_ghosts_recursively(mem_list, self_id, allowed_real_parent_id):
            for m in mem_list:
                if m.inventory:
                    if m.real_id != allowed_real_parent_id:
                        m.inventory = [sub for sub in m.inventory if sub.id != self_id]
                    _remove_self_ghosts_recursively(m.inventory, self_id, allowed_real_parent_id)

        _remove_self_ghosts_recursively(robot.memory, robot.id, robot.location_id)

        if robot.clusterer is not None:
            flat_memory_objects = []
            def _flatten_memory(mem_list):
                for m in mem_list:
                    flat_memory_objects.append(m)
                    if m.inventory:
                        _flatten_memory(m.inventory)

            _flatten_memory(robot.memory)

            current_mem_ids = {m.id: m for m in flat_memory_objects}

            newly_detected = [
                m.to_dict() for m_id, m in current_mem_ids.items() 
                if m_id not in robot.previous_mem_ids
            ]

            disappeared = [
                {"id": m_id} for m_id in robot.previous_mem_ids 
                if m_id not in current_mem_ids
            ]

            current_step_data = {
                "newly detected objects": newly_detected,
                "disappeared objects": disappeared,
                "full robot memory": [m.to_dict() for m in flat_memory_objects]
            }

            robot.clusterer.update_clusters(current_step_data)
            robot.current_clusters = robot.clusterer.get_clusters_with_data()
            
            robot.previous_mem_ids = set(current_mem_ids.keys())


    def _calculate_boundaries(self):
        """Вычисляет границы карты (для графиков Dash)."""
        padding = self.config.get('map_settings', {}).get('padding', 20.0)
        all_x, all_y = [], []
        
        for robot in self.robots.values():
            if robot.x is not None:
                all_x.append(robot.x)
                all_y.append(robot.y)

        for obj in self.real_map_data:
            if obj.x is not None: all_x.append(obj.x)
            if obj.y is not None: all_y.append(obj.y)

        if not all_x or not all_y:
            self.map_boundaries = ([-padding, padding], [-padding, padding])
        else:
            self.map_boundaries = (
                [min(all_x) - padding, max(all_x) + padding],
                [min(all_y) - padding, max(all_y) + padding]
            )

    @staticmethod
    def _extract_coords_from_poly(geom: Polygon) -> List[List[Optional[float]]]:
        """Превращает Shapely Полигон в список точек [x, y] для Plotly."""
        coords = []
        if geom.is_empty:
            return coords

        if geom.geom_type == 'Polygon':
            coords.extend([[x, y] for x, y in geom.exterior.coords])
        elif geom.geom_type in ['MultiPolygon', 'GeometryCollection']:
            polys = [g for g in getattr(geom, 'geoms', []) if g.geom_type == 'Polygon']
            for i, p in enumerate(polys):
                coords.extend([[x, y] for x, y in p.exterior.coords])
                if i < len(polys) - 1:
                    coords.append([None, None])
        return coords

    def _load_json(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _load_blueprints(self):
        """Загружает базу объектов (blueprints.json), чтобы знать их формы по умолчанию."""
        meta_path = self.config.get('paths', {}).get('blueprints_path', 'data/objects/blueprints.json')
        meta_list = self._load_json(os.path.join(project_path, meta_path))
        if isinstance(meta_list, list):
            self.blueprints = {item.get('object', 'unknown'): item for item in meta_list}

    def _get_blueprint(self, obj_type: str) -> Dict[str, Any]:
        """Возвращает настройки объекта, либо 'default', если тип неизвестен."""
        blueprint = self.blueprints.get(obj_type)
        if not blueprint:
            blueprint = self.blueprints.get('default', {})
        return blueprint
    
    @staticmethod
    def _create_vision_sector(x: float, y: float, radius: float, angle: float, fov: float) -> Polygon:
        """
        Создает полигон сектора видимости для робота.
        """
        if fov >= 360:
            return Point(x, y).buffer(radius)
            
        points = [(x, y)]
        start_angle = math.radians(-angle - fov / 2.0)
        end_angle = math.radians(-angle + fov / 2.0)
        
        num_segments = max(10, int(fov / 5))
        
        for i in range(num_segments + 1):
            theta = start_angle + (end_angle - start_angle) * (i / num_segments)
            points.append((
                x + radius * math.cos(theta),
                y + radius * math.sin(theta)
            ))
            
        return Polygon(points)
    
    @staticmethod
    def _create_shadow_polygon(origin_x: float, origin_y: float, obstacle_poly: Polygon, ray_length: float) -> Optional[Polygon]:
        """Создает бесконечный полигон тени, отбрасываемой препятствием."""
        if obstacle_poly.is_empty:
            return None
            
        if obstacle_poly.geom_type == 'Polygon':
            coords = list(obstacle_poly.exterior.coords)
        else:
            coords = list(obstacle_poly.convex_hull.exterior.coords)
            
        extended_points = []
        for px, py in coords:
            dx = px - origin_x
            dy = py - origin_y
            dist = math.hypot(dx, dy)
            if dist < 0.05:
                dist = 0.05
                
            ratio = (ray_length * 2.0) / dist
            extended_points.append((origin_x + dx * ratio, origin_y + dy * ratio))
            
        all_points = coords + extended_points
        return MultiPoint(all_points).convex_hull

    def get_real_map_data_dict(self) -> List[Dict[str, Any]]:
        """Сериализует объекты карты в список словарей.

        ``WorldObject.to_dict`` уже возвращает снимок, не разделяющий
        изменяемые контейнеры с живым объектом, поэтому результат можно
        безопасно хранить в истории кадров без дополнительного копирования.
        """
        return [obj.to_dict() for obj in self.real_map_data]

    def get_robots_memory_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Возвращает память роботов для графиков."""
        return {
            robot_id: [
                memory_object.to_dict()
                for memory_object in robot.memory
            ]
            for robot_id, robot in self.robots.items()
        }

    def _get_map_object_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Возвращает bbox объектов карты: (min_x, min_y, max_x, max_y)."""
        xs: List[float] = []
        ys: List[float] = []
        for obj in self.real_map_data:
            if obj.x is None or obj.y is None or obj.location_id is not None:
                continue
            xs.append(obj.x)
            ys.append(obj.y)
        if not xs or not ys:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    def place_robot_randomly(
        self,
        robot_id: str,
        rng: Random,
        max_attempts: int = 500,
    ) -> bool:
        """
        Размещает робота в случайной свободной точке внутри bbox карты.

        Использует планировщик робота для проверки коллизий с буфером,
        как при движении. При неудаче позиция из initial_state не меняется.

        Args:
            robot_id: ID робота в мире.
            rng: Генератор случайных чисел.
            max_attempts: Максимум попыток rejection sampling.

        Returns:
            True, если робот успешно перемещён; иначе False.
        """
        robot = self.robots.get(robot_id)
        if robot is None:
            logger.warning("RANDOM SPAWN | Робот '%s' не найден.", robot_id)
            return False

        bbox = self._get_map_object_bbox()
        if bbox is None:
            logger.warning("RANDOM SPAWN | Нет объектов на карте для определения bbox.")
            return False

        min_x, min_y, max_x, max_y = bbox
        obstacles = [
            obj for obj in self.real_map_data
            if obj.x is not None and obj.y is not None and obj.location_id is None
        ]

        if robot.planner is None:
            robot.initialize_physics()

        robot.planner._build_obstacle_space(obstacles)

        for _ in range(max_attempts):
            x = rng.uniform(min_x, max_x)
            y = rng.uniform(min_y, max_y)
            if robot.planner._is_valid(x, y):
                robot.x = x
                robot.y = y
                logger.info(
                    "RANDOM SPAWN | Робот '%s' размещён в (%.2f, %.2f).",
                    robot_id,
                    x,
                    y,
                )
                return True

        logger.warning(
            "RANDOM SPAWN | Не удалось найти свободную точку за %d попыток. "
            "Оставлена позиция из initial_state (%.2f, %.2f).",
            max_attempts,
            robot.x,
            robot.y,
        )
        return False

    @staticmethod
    def _is_world_obstacle(obj: WorldObject) -> bool:
        """True, если объект занимает место на верхней карте."""
        return obj.x is not None and obj.y is not None and obj.location_id is None

    @staticmethod
    def _overlaps_interior(probe: Polygon, obstacle: Polygon) -> bool:
        """True при наложении внутренностей, False при касании границ."""
        if obstacle is None or obstacle.is_empty:
            return False
        if not probe.intersects(obstacle):
            return False
        if probe.touches(obstacle):
            return False
        return True

    def _sync_collision_index(self) -> None:
        """Пересобирает STRtree только если поза или состав карты изменились.

        Роботы в индекс не входят: они почти каждый тик меняют ключ полигона.
        Предметы карты (в том числе брошенная лопата) попадают в дерево после
        смены ``x/y/angle`` одним перестроением, а не по флагу ``dynamic``.
        """
        bodies = [
            obj for obj in self.real_map_data if self._is_world_obstacle(obj)
        ]
        keys = {obj.id: obj._polygon_cache_signature() for obj in bodies}
        ids = tuple(obj.id for obj in bodies)
        if (
            self._collision_tree is not None
            and self._collision_index_ids == ids
            and self._collision_index_keys == keys
        ):
            return

        geometries: List[Polygon] = []
        tree_ids: List[str] = []
        for obj in bodies:
            poly = obj.get_polygon()
            if poly is not None and not poly.is_empty:
                geometries.append(poly)
                tree_ids.append(obj.id)

        self._collision_tree = STRtree(geometries)
        self._collision_tree_ids = tree_ids
        self._collision_index_ids = ids
        self._collision_index_keys = keys

    def check_collision(self, polygon: Polygon, ignore_ids: Optional[List[str]] = None) -> bool:
        """
        Проверяет, пересекается ли переданный полигон с какими-либо физическими объектами в мире.
        Разрешает касание границ (вплотную), но запрещает наложение внутренних площадей.
        """
        if not polygon or polygon.is_empty:
            return False

        if ignore_ids is None:
            ignore_set: set[str] = set()
        else:
            ignore_set = {str(item) for item in ignore_ids}

        try:
            self._sync_collision_index()
            if self._collision_tree is not None:
                for raw_index in self._collision_tree.query(polygon):
                    index = int(raw_index)
                    obj_id = self._collision_tree_ids[index]
                    if obj_id in ignore_set:
                        continue
                    obstacle = self._collision_tree.geometries[index]
                    if self._overlaps_interior(polygon, obstacle):
                        return True

            for robot_id, robot in self.robots.items():
                if robot_id in ignore_set:
                    continue
                if not self._is_world_obstacle(robot):
                    continue
                poly = robot.get_polygon()
                if self._overlaps_interior(polygon, poly):
                    return True
        except Exception as e:
            logger.warning("COLLISION CHECKER | Ошибка: %s", str(e))
            return False

        return False
    
    def _tactile_sync_path(self, robot: Any, container_path: List[str], real_container_path: List[str], 
                            start_real_list: List[Any], start_mem_list: List[Any]):
        if not robot or not hasattr(robot, "memory"):
            return

        curr_real_list = start_real_list
        curr_mem_list = start_mem_list
        
        scanned_containers = set()

        for i in range(len(container_path)):
            mem_id = container_path[i]
            real_id = real_container_path[i]

            real_cont = next((obj for obj in curr_real_list if obj.id == real_id), None)
            mem_cont = next((obj for obj in curr_mem_list if obj.id == mem_id), None)

            if real_cont and mem_cont:
                updated_ids = set()
                
                self._sync_inventory_to_memory(
                    real_inventory=real_cont.inventory,
                    mem_inventory=mem_cont.inventory,
                    vision_circle=None,
                    updated_ids=updated_ids,
                    robot=robot,
                    parent_mem_id=mem_cont.id,
                    depth_limit=0,
                    scanned_containers=scanned_containers
                )

                self._cleanup_nested_inventory(
                    mem_inv=mem_cont.inventory,
                    updated_ids=updated_ids,
                    is_visible=True,
                    robot=robot,
                    parent_mem_id=mem_cont.id,
                    scanned_containers=scanned_containers
                )

                curr_real_list = real_cont.inventory
                curr_mem_list = mem_cont.inventory
            else:
                break
    
    def _update_robot_hearing(self, robot: Robot, memory_ids_updated_this_tick: set):
        if self.active_sound_sources:
            for speaker_id, message_text in self.active_sound_sources.items():
                if speaker_id == robot.id:
                    continue

                speaker = self.get_object_by_id(speaker_id)
                if not speaker:
                    continue

                if robot.location_id != speaker.location_id:
                    continue

                is_visible = robot._translate_real_to_mem_id(speaker_id) in memory_ids_updated_this_tick
                dist_val = 0.0

                if robot.location_id is None:
                    if robot.x is not None and robot.y is not None and speaker.x is not None and speaker.y is not None:
                        dist_val = math.hypot(speaker.x - robot.x, speaker.y - robot.y)
                        if dist_val > robot.vision_radius:
                            continue
                    else:
                        continue

                if hasattr(robot, "log_hearing"):
                    robot.log_hearing(
                        tick=self.tick_count,
                        source_obj=speaker,
                        message=message_text,
                        is_visible=is_visible,
                        distance=dist_val,
                        angle_to_source=0.0
                    )