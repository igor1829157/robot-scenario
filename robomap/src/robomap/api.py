import threading
from typing import Any, Optional, Tuple

class API:
    """
    Предоставляет интерфейс для передачи команд сущностям от 
    внешних контроллеров.

    Атрибуты:
        world (Any): Экземпляр управляемого физического мира (World).
        lock (threading.Lock | threading.RLock): Мьютекс (блокировщик) сессии.
    """
    def __init__(self, world: Any, lock: threading.Lock):
        """
        Инициализирует API интерфейс.

        Args:
            world (Any): Экземпляр управляемого мира.
            lock (threading.Lock | threading.RLock): Объект блокировки потоков.
        """
        self.world = world
        self.lock = lock

    def command_move(self, entity_id: str, target_x: float = 0.0, target_y: float = 0.0, queue: bool = True, **kwargs):
        with self.lock:
            entity = self.world.get_object_by_id(entity_id)
            if not entity: return False
            
            try:
                target_x = float(target_x)
                target_y = float(target_y)
            except (ValueError, TypeError):
                entity.log_event("Cannot move: coordinates target_x and target_y must be numbers.", level="ERROR")
                return False
    
            nav_mode = str(kwargs.get("nav_mode", "hybrid"))
    
            if not queue:
                entity.clear_actions()
            entity.move(target_x, target_y, nav_mode=nav_mode)
            return True

    def command_action(self, entity_id: str, action: str, *args, **kwargs):
        """Универсальная отправка действий (pickup, put, speak, break)."""
        with self.lock:
            entity = self.world.get_object_by_id(entity_id)
            if not entity: return False
            
            if action in ["annotate", "add_virtual", "update_virtual", "delete_virtual"]:
                from robomap.models import Robot
                if not isinstance(entity, Robot):
                    entity.log_event(f"Action '{action}' is only supported for Robots.", level="ERROR")
                    return False
                
                silent = bool(kwargs.get("silent", False))

                if action == "annotate":
                    mem_id = str(kwargs.get("mem_id", args[0] if len(args) > 0 else ""))
                    notes = str(kwargs.get("notes", args[1] if len(args) > 1 else ""))
                    entity.annotate_memory(mem_id, notes, self.world)
                    
                elif action == "add_virtual":
                    obj_type = str(kwargs.get("obj_type", args[0] if len(args) > 0 else "marker"))
                    
                    x_val = kwargs.get("x", args[1] if len(args) > 1 else None)
                    x = float(x_val) if x_val is not None else None
                    
                    y_val = kwargs.get("y", args[2] if len(args) > 2 else None)
                    y = float(y_val) if y_val is not None else None
                    
                    is_obstacle = bool(kwargs.get("is_obstacle", args[3] if len(args) > 3 else False))
                    radius = float(kwargs.get("radius", args[4] if len(args) > 4 else 0.5))
                    
                    shape = kwargs.get("shape", [])
                    
                    notes = str(kwargs.get("notes", args[5] if len(args) > 5 else ""))
                    
                    p_val = kwargs.get("parent_id", args[6] if len(args) > 6 else None)
                    parent_id = str(p_val) if p_val else None
                    
                    inventory_data = kwargs.get("inventory_data", args[7] if len(args) > 7 else [])
                    render_only = bool(kwargs.get("render_only", args[8] if len(args) > 8 else False))
                    color = kwargs.get("color", args[9] if len(args) > 9 else None)
                    entity.add_virtual_object(obj_type, x, y, is_obstacle, radius, shape, notes, parent_id, inventory_data, self.world, silent=silent, render_only=render_only, color=color)
                    
                elif action == "update_virtual":
                    mem_id = str(kwargs.get("mem_id", args[0] if len(args) > 0 else ""))
                    updates = kwargs.get("updates", args[1] if len(args) > 1 else {})
                    entity.update_virtual_object(mem_id, updates, self.world, silent=silent)
                    
                elif action == "delete_virtual":
                    mem_id = str(kwargs.get("mem_id", args[0] if len(args) > 0 else ""))
                    entity.delete_virtual_object(mem_id, self.world, silent=silent)
            
            if action == "speak":
                message = str(kwargs.get("message", args[0] if len(args) > 0 else ""))
                duration = int(kwargs.get("duration", args[1] if len(args) > 1 else 3))
                entity.speak(message, duration)
                
            elif action == "put":
                item_id = str(kwargs.get("item_id", args[0] if len(args) > 0 else ""))
                target_container_path = kwargs.get("target_container_path", kwargs.get("target_container_id", args[1] if len(args) > 1 else None))
                entity.put(item_id, target_container_path)
                
            elif action == "pickup":
                target_item_id = str(kwargs.get("target_item_id", kwargs.get("target_id", args[0] if len(args) > 0 else "")))
                container_path = kwargs.get("container_path", kwargs.get("container_id", args[1] if len(args) > 1 else None))
                entity.pickup(target_item_id, container_path)
                
            elif action == "break":
                target_id = str(kwargs.get("target_id", args[0] if len(args) > 0 else ""))
                entity.break_obj(target_id)
                
            elif action == "clear":
                entity.clear_actions()
                
            elif action == "rotate":
                target = kwargs.get("target", args[0] if len(args) > 0 else 0.0)
                mode = str(kwargs.get("mode", args[1] if len(args) > 1 else "relative"))
                if mode in ["relative", "absolute"]:
                    try:
                        target = float(target)
                    except (ValueError, TypeError):
                        entity.log_event("Cannot rotate: target angle must be a number.", level="ERROR")
                        return False
                entity.rotate(target, mode)
                
            elif action == "interact_rotate":
                target_id = str(kwargs.get("target_id", args[0] if len(args) > 0 else ""))
                angle_offset = kwargs.get("angle_offset", args[1] if len(args) > 1 else 90.0)
                try:
                    angle_offset = float(angle_offset)
                except (ValueError, TypeError):
                    entity.log_event("Cannot rotate target object: angle_offset must be a number.", level="ERROR")
                    return False
                entity.interact_rotate(target_id, angle_offset)
                
            elif action == "enter":
                target_id = str(kwargs.get("target_id", args[0] if len(args) > 0 else ""))
                entity.enter(target_id)
                
            elif action == "exit":
                entity.exit()

            elif action == "inspect":
                container_path = kwargs.get("container_path", kwargs.get("container_id", args[0] if len(args) > 0 else []))
                entity.inspect(container_path)
                
            return True