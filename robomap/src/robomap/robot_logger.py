import math
from typing import Tuple, Optional, Any, Dict, List

class RobotLoggerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_log: List[Dict[str, Any]] = []

    def _get_first_blocking_action(self):
        for action in self.action_queue:
            if not getattr(action, 'is_background', False):
                return action
        return None

    def add_event_log(self, tick: int, category: str, action_type: str, message: str, status: str, **kwargs):
        log_entry = {
            "tick": int(tick),
            "robot_id": str(self.id),
            "category": str(category),
            "action_type": str(action_type),
            "message": str(message),
            "status": str(status),
            "args": kwargs
        }
        self.event_log.append(log_entry)

    def _get_location_path_str(self, world: Any) -> str:
        if not self.location_id:
            return "outside"
            
        curr_loc_id = self.location_id
        path = []
        
        while curr_loc_id:
            mem_id = self._translate_real_to_mem_id(curr_loc_id)
            path.insert(0, mem_id)
            
            parent_obj = world.get_object_by_id(curr_loc_id) if world else None
            curr_loc_id = parent_obj.location_id if parent_obj else None
            
        return " -> ".join(path)

    def sort_current_tick_logs(self, tick: int):
        if not self.event_log:
            return
            
        tick_count = int(tick)

        def sort_key(log):
            log_tick = int(log.get("tick", 0))
            
            if log_tick != tick_count:
                return (log_tick, 0, 0)
                
            status = log.get("status", "")
            category = log.get("category", "")
            
            priority = 3
            if status in ["NEW", "LOST"]: 
                priority = 0
            elif category == "OBSERVATION" or status == "HEAR": 
                priority = 1
            elif status == "CHANGED": 
                priority = 2
                
            return (log_tick, priority, 1)

        self.event_log.sort(key=sort_key)

        MAX_LOGS = 5000
        if len(self.event_log) > MAX_LOGS:
            del self.event_log[:-MAX_LOGS]

    def log_perception(self, tick: int, status: str, real_obj: Any):
        tick_count = int(tick)
        rx = round(real_obj.x, 2) if real_obj.x is not None else None
        ry = round(real_obj.y, 2) if real_obj.y is not None else None
        if status == "NEW":
            msg = f"Object {real_obj.type_name} '{real_obj.id}' is sighted at ({rx}, {ry})."
            self.add_event_log(
                tick=tick_count,
                category="PERCEPTION",
                action_type="vision",
                message=msg,
                status="NEW",
                observed_id=real_obj.id,
                observed_type=real_obj.type_name,
                observed_x=rx,
                observed_y=ry
            )
        elif status == "LOST":
            msg = f"Object {real_obj.type_name} '{real_obj.id}' disappeared. It was previously located at ({rx}, {ry})."
            self.add_event_log(
                tick=tick_count,
                category="PERCEPTION",
                action_type="vision",
                message=msg,
                status="LOST",
                observed_id=real_obj.id,
                observed_type=real_obj.type_name,
                observed_x=rx,
                observed_y=ry
            )

    def _get_durability_percentage(self, durability: Optional[float], max_durability: Optional[float]) -> Optional[int]:
        if durability is None:
            return None
        max_val = max_durability if max_durability is not None and max_durability > 0 else 100.0
        return max(0, min(100, int((float(durability) / float(max_val)) * 100)))

    def log_perception_change(self, tick: int, mem_obj: Any, real_obj: Any):
        tick_count = int(tick)
        rx = round(real_obj.x, 2) if real_obj.x is not None else None
        ry = round(real_obj.y, 2) if real_obj.y is not None else None
        mx = round(mem_obj.x, 2) if mem_obj.x is not None else None
        my = round(mem_obj.y, 2) if mem_obj.y is not None else None

        if rx is not None and ry is not None and mx is not None and my is not None:
            move_dist = math.hypot(rx - mx, ry - my)
            if move_dist > 0.01:
                msg = f"Object {real_obj.type_name} '{mem_obj.id}' moved from ({mx}, {my}) to ({rx}, {ry})."
                self.add_event_log(
                    tick=tick_count,
                    category="PERCEPTION",
                    action_type="vision",
                    message=msg,
                    status="CHANGED",
                    observed_id=mem_obj.id,
                    observed_type=real_obj.type_name,
                    observed_x=rx,
                    observed_y=ry
                )

        old_dur = mem_obj.state.get("durability") if "durability" in mem_obj.state else getattr(mem_obj, "durability", None)
        old_max = mem_obj.state.get("max_durability") if "max_durability" in mem_obj.state else getattr(mem_obj, "max_durability", None)
        new_dur = getattr(real_obj, "durability", None)
        new_max = getattr(real_obj, "max_durability", None)
        
        old_pct = self._get_durability_percentage(old_dur, old_max)
        new_pct = self._get_durability_percentage(new_dur, new_max)

        if old_pct is not None and new_pct is not None and old_pct != new_pct:
            if new_pct == 0:
                msg = f"Object {real_obj.type_name} '{mem_obj.id}' became broken (0%)."
            else:
                msg = f"Object {real_obj.type_name} '{mem_obj.id}' durability changed from {old_pct}% to {new_pct}%."

            self.add_event_log(
                tick=tick_count,
                category="PERCEPTION",
                action_type="vision",
                message=msg,
                status="CHANGED",
                observed_id=mem_obj.id,
                observed_type=real_obj.type_name,
                observed_x=rx,
                observed_y=ry
            )

    def log_exploration_change(self, tick: int, mem_id: str, is_fully_explored: bool):
        tick_count = int(tick)
        if is_fully_explored:
            msg = f"Robot fully mapped the geometry of '{mem_id}'."
        else:
            msg = f"Geometry of '{mem_id}' changed or view is obstructed. Mapping is now incomplete."
            
        self.add_event_log(
            tick=tick_count,
            category="PERCEPTION",
            action_type="exploration",
            message=msg,
            status="CHANGED",
            observed_id=mem_id
        )

    def log_inventory_change(self, tick: int, status: str, parent_id: str, item_obj: Any):
        tick_count = int(tick)
        parent_type = "container"
        parent_mem_id = parent_id

        parent_obj = self._find_mem_obj_recursively(self.memory, parent_id, by_real_id=False)
        if not parent_obj:
            parent_obj = self._find_mem_obj_recursively(self.memory_inventory, parent_id, by_real_id=False)

        if parent_obj:
            parent_type = parent_obj.type_name

        if status == "ADDED":
            msg = f"Object {item_obj.type_name} '{item_obj.id}' is now inside {parent_type} '{parent_mem_id}'."
            self.add_event_log(
                tick=tick_count,
                category="PERCEPTION",
                action_type="inventory",
                message=msg,
                status="CHANGED",
                observed_id=item_obj.id,
                observed_type=item_obj.type_name,
                observed_x=None,
                observed_y=None
            )
        elif status == "REMOVED":
            msg = f"Object {item_obj.type_name} '{item_obj.id}' is no longer inside {parent_type} '{parent_mem_id}'."
            self.add_event_log(
                tick=tick_count,
                category="PERCEPTION",
                action_type="inventory",
                message=msg,
                status="CHANGED",
                observed_id=item_obj.id,
                observed_type=item_obj.type_name,
                observed_x=None,
                observed_y=None
            )

    def log_hearing(self, tick: int, source_obj: Any, message: str, is_visible: bool, distance: float, angle_to_source: float):
        tick_count = int(tick)
        
        if is_visible:
            mem_id = self._translate_real_to_mem_id(source_obj.id)
            msg = f"Robot heard {source_obj.type_name} '{mem_id}' say: '{message}'."
            observed_id = mem_id
            observed_type = source_obj.type_name
            observed_x = round(source_obj.x, 2) if source_obj.x is not None else None
            observed_y = round(source_obj.y, 2) if source_obj.y is not None else None
        else:
            proximity_str = "nearby" if distance <= 5.0 else "in the distance"
            msg = f"Robot heard {source_obj.type_name} {proximity_str} say: '{message}'."
            observed_id = "unidentified"
            observed_type = source_obj.type_name
            observed_x = None
            observed_y = None

        self.add_event_log(
            tick=tick_count,
            category="HEAR",
            action_type="speak",
            message=msg,
            status="HEAR",
            observed_id=observed_id,
            observed_type=observed_type,
            observed_x=observed_x,
            observed_y=observed_y,
            message_text=message
        )

    def _find_mem_obj_recursively(self, items: List[Any], target_id: str, by_real_id: bool = False) -> Optional[Any]:
        for item in items:
            check_id = item.real_id if by_real_id else item.id
            if check_id == target_id:
                return item
            
            if getattr(item, 'inventory', None):
                found = self._find_mem_obj_recursively(item.inventory, target_id, by_real_id)
                if found:
                    return found
        return None
    
    def _translate_real_to_mem_id(self, real_id: str) -> str:
        if not real_id:
            return "unknown"
            
        found = self._find_mem_obj_recursively(self.memory, real_id, by_real_id=True)
        if found:
            return found.id
            
        found_own = self._find_mem_obj_recursively(self.memory_inventory, real_id, by_real_id=True)
        if found_own:
            return found_own.id
            
        return real_id
    
    def _translate_actor_id_to_observer_id(self, actor: Any, actor_target_id: str) -> str:

        from robomap.models import Robot 

        if not actor_target_id:
            return "unknown"
            
        real_id = actor_target_id
        
        if isinstance(actor, Robot):
            def _find_real_id(mem_list, t_id):
                for m in mem_list:
                    if m.id == t_id: return m.real_id
                    if m.inventory:
                        res = _find_real_id(m.inventory, t_id)
                        if res: return res
                return None
            
            found_real = _find_real_id(actor.memory, actor_target_id)
            if not found_real:
                found_real = _find_real_id(actor.memory_inventory, actor_target_id)
            if found_real:
                real_id = found_real

        mem_id = self._translate_real_to_mem_id(real_id)
        
        if mem_id == real_id:
            if real_id != self.id:
                return "unknown_object"
        return mem_id

    def log_observation(self, tick: int, mem_obj: Any, real_obj: Any, world: Any = None):
        tick_count = int(tick)
        rx = round(real_obj.x, 2) if real_obj.x is not None else None
        ry = round(real_obj.y, 2) if real_obj.y is not None else None

        if hasattr(real_obj, 'completed_actions_this_tick') and real_obj.completed_actions_this_tick:
            for act in real_obj.completed_actions_this_tick:
                act_name = act.__class__.__name__
                msg = ""
                
                if act_name == "MoveAction":
                    msg = f"Object {real_obj.type_name} '{mem_obj.id}' finished moving."
                elif act_name == "RotateAction":
                    msg = f"Object {real_obj.type_name} '{mem_obj.id}' finished turning."
                elif act_name == "InteractRotateAction":
                    t_mem_id = self._translate_actor_id_to_observer_id(real_obj, act.target_id)
                    msg = f"Object {real_obj.type_name} '{mem_obj.id}' rotated object '{t_mem_id}' by {round(act.angle_offset, 2)}°."
                elif act_name == "PickupAction":
                    t_mem_id = self._translate_actor_id_to_observer_id(real_obj, act.target_id)
                    msg = f"Object {real_obj.type_name} '{mem_obj.id}' picked up object '{t_mem_id}'."
                elif act_name == "PutAction":
                    t_mem_id = self._translate_actor_id_to_observer_id(real_obj, act.item_id)
                    if act.target_container_path:
                        cont_id = self._translate_actor_id_to_observer_id(real_obj, act.target_container_path[-1])
                        msg = f"Object {real_obj.type_name} '{mem_obj.id}' placed object '{t_mem_id}' into '{cont_id}'."
                    else:
                        dropped_item = None
                        if hasattr(act, "item_id") and world is not None:
                            dropped_item = next((item for item in world.real_map_data if item.id == act.item_id), None)
                        
                        if dropped_item and dropped_item.x is not None and dropped_item.y is not None:
                            msg = f"Object {real_obj.type_name} '{mem_obj.id}' dropped object '{t_mem_id}' at ({round(dropped_item.x, 2)}, {round(dropped_item.y, 2)})."
                        else:
                            msg = f"Object {real_obj.type_name} '{mem_obj.id}' dropped object '{t_mem_id}'."
                elif act_name == "EnterAction":
                    t_mem_id = self._translate_actor_id_to_observer_id(real_obj, act.target_id)
                    msg = f"Object {real_obj.type_name} '{mem_obj.id}' entered object '{t_mem_id}'."
                elif act_name == "ExitAction":
                    msg = f"Object {real_obj.type_name} '{mem_obj.id}' exited to the outside ground."
                elif act_name == "BreakAction":
                    t_mem_id = self._translate_actor_id_to_observer_id(real_obj, act.target_id)
                    msg = f"Object {real_obj.type_name} '{mem_obj.id}' broke object '{t_mem_id}'."

                if msg:
                    completion_tick = getattr(act, "completion_tick", tick_count)
                    observed_key = f"completed_{act_name}_{completion_tick}"
                    if hasattr(mem_obj, "state"):
                        if mem_obj.state.get(observed_key):
                            continue
                        mem_obj.state[observed_key] = True

                    self.add_event_log(
                        tick=tick_count,
                        category="OBSERVATION",
                        action_type=act_name.replace("Action", "").lower(),
                        message=msg,
                        status="OBSERVATION",
                        observed_id=mem_obj.id,
                        observed_type=real_obj.type_name,
                        observed_x=rx,
                        observed_y=ry
                    )

        if not getattr(real_obj, 'action_queue', None):
            if hasattr(mem_obj, "last_observed_action"):
                delattr(mem_obj, "last_observed_action")
            return

        act = real_obj.action_queue[0]
        act_name = act.__class__.__name__
        details = ""
        msg = ""

        if act_name == "MoveAction":
            details = "moving"
            msg = f"Object {real_obj.type_name} '{mem_obj.id}' started moving."
        elif act_name == "RotateAction":
            details = f"mode:{act.mode}, target:{act.target}"
            msg = f"Object {real_obj.type_name} '{mem_obj.id}' started turning (Mode: {act.mode}, Target: {act.target})."
        elif act_name == "InteractRotateAction":
            t_mem_id = self._translate_actor_id_to_observer_id(real_obj, act.target_id)
            details = f"target:{t_mem_id}, offset:{act.angle_offset}"
            msg = f"Object {real_obj.type_name} '{mem_obj.id}' is rotating object '{t_mem_id}' by {round(act.angle_offset, 2)}°."
        elif act_name == "PickupAction":
            t_mem_id = self._translate_actor_id_to_observer_id(real_obj, act.target_id)
            details = f"target:{t_mem_id}"
            msg = f"Object {real_obj.type_name} '{mem_obj.id}' is picking up object '{t_mem_id}'."
        elif act_name == "PutAction":
            t_mem_id = self._translate_actor_id_to_observer_id(real_obj, act.item_id)
            details = f"item:{t_mem_id}"
            msg = f"Object {real_obj.type_name} '{mem_obj.id}' is placing object '{t_mem_id}'."
        elif act_name == "EnterAction":
            t_mem_id = self._translate_actor_id_to_observer_id(real_obj, act.target_id)
            details = f"target:{t_mem_id}"
            msg = f"Object {real_obj.type_name} '{mem_obj.id}' is entering object '{t_mem_id}'."
        elif act_name == "ExitAction":
            details = "exit"
            msg = f"Object {real_obj.type_name} '{mem_obj.id}' is exiting to the outside ground."
        elif act_name == "BreakAction":
            t_mem_id = self._translate_actor_id_to_observer_id(real_obj, act.target_id)
            details = f"target:{t_mem_id}"
            msg = f"Object {real_obj.type_name} '{mem_obj.id}' is breaking object '{t_mem_id}'."
        else:
            return

        last_obs = getattr(mem_obj, "last_observed_action", None)
        current_obs = f"{act_name}_{details}"
        
        if last_obs == current_obs:
            return
            
        setattr(mem_obj, "last_observed_action", current_obs)
        
        self.add_event_log(
            tick=tick_count,
            category="OBSERVATION",
            action_type=act_name.replace("Action", "").lower(),
            message=msg,
            status="OBSERVATION",
            observed_id=mem_obj.id,
            observed_type=real_obj.type_name,
            observed_x=rx,
            observed_y=ry
        )

    def _get_active_move_target(self) -> Tuple[Optional[float], Optional[float]]:
        for action in self.action_queue:
            if action.__class__.__name__ == "MoveAction":
                return action.target_x, action.target_y
        return None, None

    def _get_active_rotation(self) -> Tuple[Optional[str], Optional[Any], Optional[float]]:
        for action in self.action_queue:
            if action.__class__.__name__ == "RotateAction":
                return action.mode, action.target, action.target_angle
        return None, None, None

    def _get_active_interact_rotate(self) -> Tuple[Optional[str], Optional[float]]:
        for action in self.action_queue:
            if action.__class__.__name__ == "InteractRotateAction":
                return action.target_id, action.angle_offset
        return None, None

    def _get_active_pickup(self) -> Tuple[Optional[str], List[str]]:
        for action in self.action_queue:
            if action.__class__.__name__ == "PickupAction":
                return action.target_id, action.container_path
        return None, []

    def _get_active_put(self) -> Tuple[Optional[str], List[str]]:
        for action in self.action_queue:
            if action.__class__.__name__ == "PutAction":
                return action.item_id, action.target_container_path
        return None, []

    def _get_active_enter(self) -> Optional[str]:
        for action in self.action_queue:
            if action.__class__.__name__ == "EnterAction":
                return action.target_id
        return None

    def _get_active_break(self) -> Optional[str]:
        for action in self.action_queue:
            if action.__class__.__name__ == "BreakAction":
                return action.target_id
        return None

    def _build_move_args(self, target_x: Optional[float], target_y: Optional[float]) -> Dict[str, float]:

        if self.x is None or self.y is None:
            return {
                "current_x": 0.0,
                "current_y": 0.0,
                "target_x": round(target_x, 2) if target_x is not None else 0.0,
                "target_y": round(target_y, 2) if target_y is not None else 0.0,
                "angle_to_target": 0.0
            }

        tx = round(target_x, 2) if target_x is not None else round(self.x, 2)
        ty = round(target_y, 2) if target_y is not None else round(self.y, 2)
        dx = tx - self.x
        dy = ty - self.y
        angle_to_target = (-math.degrees(math.atan2(dy, dx))) % 360 if math.hypot(dx, dy) > 0.001 else self.angle
        return {
            "current_x": round(self.x, 2),
            "current_y": round(self.y, 2),
            "target_x": tx,
            "target_y": ty,
            "angle_to_target": round(angle_to_target, 2)
        }

    def _build_rotate_args(self, mode: Optional[str], target: Optional[Any], target_angle: Optional[float]) -> Dict[str, Any]:
        t_angle = round(target_angle, 2) if target_angle is not None else round(self.angle, 2)
        t_mode = mode if mode else "unknown"
        t_target = target if target is not None else "unknown"
        if isinstance(t_target, (int, float)):
            t_target = round(float(t_target), 2)
        return {
            "current_angle": round(self.angle, 2),
            "target_angle": t_angle,
            "mode": t_mode,
            "target": t_target
        }

    def _build_interact_rotate_args(self, target_id: Optional[str], angle_offset: Optional[float]) -> Dict[str, Any]:
        return {
            "target_id": target_id if target_id else "unknown",
            "angle_offset": round(float(angle_offset), 2) if angle_offset is not None else 0.0
        }

    def _build_pickup_args(self, target_id: Optional[str], container_path: List[str]) -> Dict[str, Any]:
        return {
            "target_id": target_id if target_id else "unknown",
            "container_path": container_path if container_path else []
        }

    def _build_put_args(self, item_id: Optional[str], target_container_path: List[str]) -> Dict[str, Any]:
        return {
            "item_id": item_id if item_id else "unknown",
            "target_container_path": target_container_path if target_container_path else []
        }

    def _build_break_args(self, target_id: Optional[str]) -> Dict[str, Any]:
        return {
            "target_id": target_id if target_id else "unknown"
        }

    def log_event(self, message: str, level: str = "INFO", world: Any = None, action_code: str = None, **kwargs):
        super().log_event(message, level, world, action_code)
        tick_count = world.tick_count if world else 0

        if not action_code:
            return

        if action_code.startswith("interact_rotate"):
            target_id, angle_offset = self._get_active_interact_rotate()
            args = self._build_interact_rotate_args(target_id, angle_offset)
            t_offset = round(float(angle_offset), 2) if angle_offset is not None else 0.0

            if action_code == "interact_rotate_success":
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="interact_rotate",
                    message=f"Robot rotated object '{target_id}' by {t_offset}°.",
                    status="SUCCESS", **args
                )
            else:
                reason = "Unknown failure."
                if action_code == "interact_rotate_fail_not_found_mem":
                    reason = f"Target '{target_id}' not found in memory."
                elif action_code == "interact_rotate_fail_not_found_real":
                    reason = f"Target '{target_id}' not found in the physical world."
                elif action_code == "interact_rotate_fail_not_rotateable":
                    reason = f"Target '{target_id}' is not rotateable."
                elif action_code == "interact_rotate_fail_too_far":
                    reason = f"Target '{target_id}' is too far."
                elif action_code == "interact_rotate_fail_space":
                    reason = f"Not enough space to rotate target '{target_id}'."

                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="interact_rotate",
                    message=f"Failed to rotate object: {reason}", status="FAIL", **args
                )

        elif action_code.startswith("pickup"):
            target_id = kwargs.get("target_id", "unknown")
            container_path = kwargs.get("container_path", [])
            args = {"target_id": target_id, "container_path": container_path}
            path_str = f" from {container_path}" if container_path else ""

            if action_code == "pickup_success":
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="pickup",
                    message=f"Robot picked up object '{target_id}'{path_str}.",
                    status="SUCCESS", **args
                )
            else:
                reason = "Unknown failure."
                if action_code == "pickup_fail_no_inventory":
                    reason = "Robot lacks inventory capability."
                elif action_code == "pickup_fail_not_found_mem":
                    if not container_path:
                        reason = f"Target '{target_id}' not found at this level. If the item is inside a container, you MUST specify the full 'container_path' list."
                    else:
                        reason = f"Target '{target_id}' not found in memory at path {container_path}."
                elif action_code == "pickup_fail_not_found_real":
                    if not container_path:
                        reason = f"Target '{target_id}' not found in the physical world. Check if it requires a 'container_path'."
                    else:
                        reason = f"Target '{target_id}' not found in the physical world at path {container_path}."
                elif action_code == "pickup_fail_self":
                    reason = "Cannot pick up an item from yourself."
                elif action_code == "pickup_fail_enter_first":
                    reason = f"Must enter container {container_path} first."
                elif action_code == "pickup_fail_too_far":
                    reason = f"Target '{target_id}' is too far."
                elif action_code == "pickup_fail_not_pickupable":
                    reason = f"Target '{target_id}' is not pickable."

                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="pickup",
                    message=f"Failed to pick up: {reason}", status="FAIL", **args
                )

        elif action_code.startswith("put"):
            item_id = kwargs.get("item_id", "unknown")
            target_container_path = kwargs.get("target_container_path", [])
            args = {"item_id": item_id, "target_container_path": target_container_path}

            if action_code.startswith("put_success"):
                if action_code == "put_success_container":
                    msg_out = f"Robot placed object '{item_id}' into container {target_container_path}."
                elif action_code == "put_success_inside":
                    loc = kwargs.get("location_id", "unknown")
                    msg_out = f"Robot dropped object '{item_id}' inside location '{loc}'."
                elif action_code == "put_success_ground":
                    drop_x = kwargs.get("drop_x", 0.0)
                    drop_y = kwargs.get("drop_y", 0.0)
                    drop_angle = kwargs.get("drop_angle", 0.0)
                    msg_out = f"Robot dropped object '{item_id}' on ground at ({drop_x:.1f}, {drop_y:.1f}) at an angle of {drop_angle:.1f}°."
                
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="put",
                    message=msg_out, status="SUCCESS", **args
                )
            else:
                reason = "Unknown failure."
                if action_code == "put_fail_not_in_inv":
                    reason = f"Item '{item_id}' is missing from inventory."
                elif action_code == "put_fail_not_found_mem":
                    reason = f"Target path {target_container_path} not found in memory."
                elif action_code == "put_fail_invalid_container":
                    reason = "Target container is invalid."
                elif action_code == "put_fail_enter_first":
                    reason = f"Must enter target location {target_container_path} first."
                elif action_code == "put_fail_too_far":
                    reason = "Target container is too far."
                elif action_code == "put_fail_no_space":
                    reason = f"Not enough space on the ground near current position."

                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="put",
                    message=f"Failed to place: {reason}", status="FAIL", **args
                )

        elif action_code.startswith("break"):
            target_id = kwargs.get("target_id", "unknown")
            args = {"target_id": target_id}

            if action_code == "break_success":
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="break",
                    message=f"Robot broke object '{target_id}'.",
                    status="SUCCESS", **args
                )
            elif action_code == "break_progress":
                durability_left = kwargs.get("durability_left", 0.0)
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="break",
                    message=f"Hitting object '{target_id}'. Durability left: {durability_left:.1f}.",
                    status="PROGRESS", **args
                )
            else:
                reason = "Unknown failure."
                if action_code == "break_fail_not_found_mem":
                    reason = f"Target '{target_id}' not found in memory."
                elif action_code == "break_fail_not_found_real":
                    reason = f"Target '{target_id}' not found in the physical world."
                elif action_code == "break_fail_not_breakable":
                    reason = f"Target '{target_id}' cannot be broken."
                elif action_code == "break_fail_no_damage":
                    reason = "Robot lacks damage capability."
                elif action_code == "break_fail_already_broken":
                    reason = f"Target '{target_id}' is already broken."
                elif action_code == "break_fail_same_location":
                    reason = "Must be in the same location container."
                elif action_code == "break_fail_too_far":
                    reason = f"Target '{target_id}' is too far."

                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="break",
                    message=f"Failed to break: {reason}", status="FAIL", **args
                )

        elif action_code.startswith("enter"):
            target_id = self._get_active_enter()
            t_id = target_id if target_id else "unknown"
            args = {"target_id": t_id}

            if action_code == "enter_success":
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="enter",
                    message=f"Robot entered object '{t_id}'.",
                    status="SUCCESS", **args
                )
            elif action_code == "enter_warn_already_inside":
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="enter",
                    message=f"Robot is already inside object '{t_id}'.",
                    status="WARNING", **args
                )
            else:
                reason = "Unknown failure."
                if action_code == "enter_fail_not_found":
                    reason = f"Target '{t_id}' not found."
                elif action_code == "enter_fail_no_affordance":
                    reason = f"Target '{t_id}' cannot be entered."
                elif action_code == "enter_fail_too_far":
                    reason = f"Robot is too far from target '{t_id}'."

                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="enter",
                    message=f"Failed to enter: {reason}", status="FAIL", **args
                )

        elif action_code.startswith("exit"):
            if action_code == "exit_success":
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="exit",
                    message="Robot exited to the outside ground.",
                    status="SUCCESS"
                )
            elif action_code == "exit_warn_no_space":
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="exit",
                    message="Robot exited to the outside ground. Warning: No free space outside, exit was forced.",
                    status="SUCCESS"
                )
            else:
                reason = "Unknown failure."
                if action_code == "exit_fail_already_outside":
                    reason = "Robot is already outside."
                elif action_code == "exit_fail_parent_not_found":
                    reason = "Parent location container not found."

                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="exit",
                    message=f"Failed to exit: {reason}", status="FAIL"
                )
        
        elif action_code.startswith("inspect"):
            target_container_path = kwargs.get("target_container_path", [])
            args = {"target_container_path": target_container_path}

            if action_code == "inspect_success":
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="inspect",
                    message=f"Robot inspected contents of container {target_container_path}.",
                    status="SUCCESS", **args
                )
            else:
                reason = "Unknown failure."
                if action_code == "inspect_fail_no_path": reason = "No container path provided."
                elif action_code == "inspect_fail_not_found_mem": reason = f"Path {target_container_path} not found in memory."
                elif action_code == "inspect_fail_invalid_container": reason = "Target is not a valid container."
                elif action_code == "inspect_fail_enter_first": reason = "Must enter target location first."
                elif action_code == "inspect_fail_too_far": reason = "Target container is too far."

                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="inspect",
                    message=f"Failed to inspect: {reason}", status="FAIL", **args
                )

        elif action_code.startswith("memory_edit"):
            mem_id = kwargs.get("mem_id", "unknown")
            args = {"mem_id": mem_id}
            
            if "annotate" in action_code:
                notes = kwargs.get("notes", "")
                if action_code.endswith("success"):
                    self.add_event_log(tick=tick_count, category="SELF_ACTION", action_type="annotate",
                        message=f"Added note to '{mem_id}': {notes}", status="SUCCESS", **args, notes=notes)
                else:
                    reason = "Object not found." if "not_found" in action_code else "Unknown error."
                    self.add_event_log(tick=tick_count, category="SELF_ACTION", action_type="annotate",
                        message=f"Failed to annotate: {reason}", status="FAIL", **args)
                        
            elif "add" in action_code:
                obj_type = kwargs.get("obj_type", "unknown")
                if action_code.endswith("success"):
                    p_id = kwargs.get("parent_id")
                    x = kwargs.get("x")
                    y = kwargs.get("y")
                    rad = kwargs.get("radius", 0.0)
                    is_obs = kwargs.get("is_obstacle", False)
                    notes = kwargs.get("notes", "")
                    
                    if p_id:
                        loc_msg = f"inside '{p_id}'"
                    else:
                        shape = kwargs.get("shape", [])
                        if shape and len(shape) >= 3:
                            loc_msg = f"at ({x}, {y}) with polygon shape: {shape}"
                        else:
                            loc_msg = f"at ({x}, {y}) with radius {rad}"
                        
                    obs_msg = "[OBSTACLE]" if is_obs else "[MARKER]"
                    note_msg = f" Notes: '{notes}'" if notes else ""
                    
                    full_message = f"Created virtual object '{mem_id}' ({obj_type}) {obs_msg} {loc_msg}.{note_msg}"
                    
                    self.add_event_log(tick=tick_count, category="SELF_ACTION", action_type="add_virtual",
                        message=full_message, status="SUCCESS", **kwargs)
                else:
                    reason = "Parent container not found." if "fail_parent" in action_code else "Unknown error."
                    self.add_event_log(tick=tick_count, category="SELF_ACTION", action_type="add_virtual",
                        message=f"Failed to create virtual object: {reason}", status="FAIL", **kwargs)
                        
            elif "update" in action_code:
                if action_code.endswith("success"):
                    applied = kwargs.get("applied_updates", {})
                    
                    if applied:
                        changes_str = ", ".join([f"{k} -> {v}" for k, v in applied.items()])
                        msg = f"Updated virtual object '{mem_id}'. Changes: [{changes_str}]"
                    else:
                        msg = f"Updated virtual object '{mem_id}' (no changes applied)."
                        
                    self.add_event_log(tick=tick_count, category="SELF_ACTION", action_type="update_virtual",
                        message=msg, status="SUCCESS", **kwargs)
                else:
                    reason = "Object not found." if "not_found" in action_code else "Cannot alter physical properties of real objects." if "fail_real" in action_code else "Unknown error."
                    self.add_event_log(tick=tick_count, category="SELF_ACTION", action_type="update_virtual",
                        message=f"Failed to update '{mem_id}': {reason}", status="FAIL", **args)
                        
            elif "delete" in action_code:
                if action_code.endswith("success"):
                    self.add_event_log(tick=tick_count, category="SELF_ACTION", action_type="delete_virtual",
                        message=f"Deleted virtual object '{mem_id}'.", status="SUCCESS", **args)
                else:
                    reason = "Object not found." if "not_found" in action_code else "Cannot force delete real objects." if "fail_real" in action_code else "Unknown error."
                    self.add_event_log(tick=tick_count, category="SELF_ACTION", action_type="delete_virtual",
                        message=f"Failed to delete '{mem_id}': {reason}", status="FAIL", **args)

        elif action_code == "speak_success":
            msg_text = kwargs.get("message_text", "unknown")
            self.add_event_log(
                tick=tick_count, category="SELF_ACTION", action_type="speak",
                message=f"Robot said: '{msg_text}'.",
                status="SUCCESS", message_text=msg_text
            )

        elif action_code.startswith("move"):
            target_x, target_y = self._get_active_move_target()
            tx = round(target_x, 2) if target_x is not None else "unknown"
            ty = round(target_y, 2) if target_y is not None else "unknown"
            args = self._build_move_args(target_x, target_y)

            curr_x = round(self.x, 2) if self.x is not None else "unknown"
            curr_y = round(self.y, 2) if self.y is not None else "unknown"

            if action_code == "move_success":
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="move",
                    message=f"Robot reached target point at ({tx}, {ty}). Current position: ({curr_x}, {curr_y}).",
                    status="SUCCESS", **args
                )
            elif action_code == "move_start":
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="move",
                    message=f"Robot started moving to target at ({tx}, {ty}). Current position: ({curr_x}, {curr_y}).",
                    status="PROGRESS", **args
                )
            elif action_code == "move_warn_stopped":
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="move",
                    message=f"Robot stopped at the nearest reachable point to ({tx}, {ty}). Current position: ({curr_x}, {curr_y}).",
                    status="SUCCESS", **args
                )
            elif action_code == "move_warn_obstacle":
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="move",
                    message=f"Robot detected an obstacle on its path to ({tx}, {ty}). Current position: ({curr_x}, {curr_y}). Replanning route.",
                    status="WARNING", **args
                )
            elif action_code in ["move_fail", "move_cancel", "move_fail_collision"]:
                if action_code == "move_fail_collision":
                    reason = "Physical collision predicted. Emergency stop activated."
                elif action_code == "move_cancel":
                    reason = "Movement canceled by external command."
                else:
                    reason = "Unknown failure."
                    
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="move",
                    message=f"Robot failed on the way to target ({tx}, {ty}). Current position: ({curr_x}, {curr_y}). Reason: {reason}",
                    status="FAIL", **args
                )

        elif action_code.startswith("rotate"):
            mode, target, target_angle = self._get_active_rotation()
            args = self._build_rotate_args(mode, target, target_angle)
            t_angle = round(target_angle, 2) if target_angle is not None else 0.0
            t_target = target if target is not None else "unknown"

            if action_code == "rotate_success":
                if mode == "relative":
                    msg_out = f"Robot finished turning by {round(float(target), 2)}° (relative). Current angle: {round(self.angle, 2)}°."
                elif mode == "absolute":
                    msg_out = f"Robot finished turning to face absolute angle {round(float(target), 2)}°. Current angle: {round(self.angle, 2)}°."
                elif mode == "object":
                    msg_out = f"Robot finished turning to face object '{target}'. Current angle: {round(self.angle, 2)}°."
                else:
                    msg_out = f"Robot finished turning. Current angle: {round(self.angle, 2)}°."

                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="rotate",
                    message=msg_out, status="SUCCESS", **args
                )
            elif action_code == "rotate_start":
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="rotate",
                    message=f"Robot started turning to {t_angle} degrees.", 
                    status="PROGRESS", **args
                )
            else:
                reason = "Unknown rotation failure."
                if action_code == "rotate_fail_space":
                    reason = f"Rotation failed: Not enough space to rotate to {t_angle}°."
                elif action_code == "rotate_fail_not_found_mem":
                    reason = f"Rotation failed: Target object '{t_target}' not found in memory."
                elif action_code == "rotate_fail_too_far":
                    reason = f"Rotation failed: Target object '{t_target}' is too far."
                elif action_code in ["rotate_cancel_no_rotate", "rotate_cancel_no_coords", "rotate_fail_general"]:
                    reason = f"Rotation failed: Invalid target."

                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="rotate",
                    message=reason, status="FAIL", **args
                )

    def tick(self, world: Any):
        tick_count = world.tick_count if world else 0
        
        first_blocking_before = self._get_first_blocking_action()
        action_name_before = first_blocking_before.__class__.__name__ if first_blocking_before else None

        had_move_action_before = action_name_before == "MoveAction"
        had_interact_rotate_before = action_name_before == "InteractRotateAction"
        had_pickup_before = action_name_before == "PickupAction"
        had_put_before = action_name_before == "PutAction"
        had_enter_before = action_name_before == "EnterAction"
        had_exit_before = action_name_before == "ExitAction"
        had_break_before = action_name_before == "BreakAction"
        
        prev_x, prev_y, prev_angle = self.x, self.y, self.angle
        start_move_logged = False
        target_x, target_y = self._get_active_move_target()
        
        if had_move_action_before and target_x is not None and target_y is not None and self.x is not None and self.y is not None:
            move_act = first_blocking_before
            if not getattr(move_act, 'is_path_set', False):
                start_move_logged = True

        if had_interact_rotate_before:
            target_id, angle_offset = self._get_active_interact_rotate()
            args = self._build_interact_rotate_args(target_id, angle_offset)
            self.add_event_log(
                tick=tick_count, category="SELF_ACTION", action_type="interact_rotate",
                message=f"Robot started rotating object '{target_id}' by {round(float(angle_offset), 2)}°.",
                status="PROGRESS", **args
            )

        if had_pickup_before:
            target_id, container_path = self._get_active_pickup()
            args = self._build_pickup_args(target_id, container_path)
            path_str = f" from {container_path}" if container_path else ""
            self.add_event_log(
                tick=tick_count, category="SELF_ACTION", action_type="pickup",
                message=f"Robot started picking up object '{target_id}'{path_str}.",
                status="PROGRESS", **args
            )

        if had_put_before:
            item_id, target_container_path = self._get_active_put()
            args = self._build_put_args(item_id, target_container_path)
            path_str = f" into {target_container_path}" if target_container_path else " on the ground"
            self.add_event_log(
                tick=tick_count, category="SELF_ACTION", action_type="put",
                message=f"Robot started placing object '{item_id}'{path_str}.",
                status="PROGRESS", **args
            )

        if had_enter_before:
            target_id = self._get_active_enter()
            self.add_event_log(
                tick=tick_count, category="SELF_ACTION", action_type="enter",
                message=f"Robot started entering object '{target_id}'.",
                status="PROGRESS", target_id=target_id if target_id else "unknown"
            )

        if had_exit_before:
            self.add_event_log(
                tick=tick_count, category="SELF_ACTION", action_type="exit",
                message="Robot started exiting to the outside ground.",
                status="PROGRESS"
            )

        if had_break_before:
            target_id = self._get_active_break()
            args = self._build_break_args(target_id)
            break_act = first_blocking_before
            if not getattr(break_act, '_start_logged', False):
                setattr(break_act, '_start_logged', True)
                self.add_event_log(
                    tick=tick_count, category="SELF_ACTION", action_type="break",
                    message=f"Robot started breaking object '{target_id}'.",
                    status="PROGRESS", **args
                )

        super().tick(world)

        first_blocking_after = self._get_first_blocking_action()
        action_name_after = first_blocking_after.__class__.__name__ if first_blocking_after else None
        
        has_move_action_after = action_name_after == "MoveAction"
        has_rotate_action_after = action_name_after == "RotateAction"
        
        if has_move_action_after:
            curr_target_x, curr_target_y = self._get_active_move_target()
            if curr_target_x is not None and curr_target_y is not None and self.x is not None and self.y is not None:
                ctx, cty = round(curr_target_x, 2), round(curr_target_y, 2)
                angle_changed = abs(self.angle - prev_angle) > 0.01
                coords_changed = (self.x != prev_x or self.y != prev_y)
                args = self._build_move_args(curr_target_x, curr_target_y)
                
                if angle_changed:
                    self.add_event_log(
                        tick=tick_count, category="SELF_ACTION", action_type="move",
                        message=f"Robot is turning to face the target at ({ctx}, {cty}). Current position: ({round(self.x, 2)}, {round(self.y, 2)}). Current angle: {round(self.angle, 2)}°.",
                        status="PROGRESS", **args
                    )
                elif coords_changed and not start_move_logged:
                    self.add_event_log(
                        tick=tick_count, category="SELF_ACTION", action_type="move",
                        message=f"Robot is moving to target at ({ctx}, {cty}). Current position: ({round(self.x, 2)}, {round(self.y, 2)}).",
                        status="PROGRESS", **args
                    )

        if has_rotate_action_after:
            mode, target, target_angle = self._get_active_rotation()
            if target_angle is not None:
                rot_act = first_blocking_after
                args = self._build_rotate_args(mode, target, target_angle)
                if not getattr(rot_act, '_start_logged', False):
                    setattr(rot_act, '_start_logged', True)
                    if mode == "relative":
                        msg = f"Robot started turning by {round(float(target), 2)}° (relative). Target angle: {round(target_angle, 2)}°. Current angle: {round(prev_angle, 2)}°."
                    elif mode == "absolute":
                        msg = f"Robot started turning to face absolute angle {round(float(target), 2)}°. Current angle: {round(prev_angle, 2)}°."
                    elif mode == "object":
                        msg = f"Robot started turning to face object '{target}'. Target angle: {round(target_angle, 2)}°. Current angle: {round(prev_angle, 2)}°."
                    else:
                        msg = f"Robot started turning. Current angle: {round(prev_angle, 2)}°."
                        
                    self.add_event_log(
                        tick=tick_count, category="SELF_ACTION", action_type="rotate",
                        message=msg, status="PROGRESS", **args
                    )
                elif abs(self.angle - prev_angle) > 0.01:
                    if mode == "relative":
                        msg = f"Robot is turning by {round(float(target), 2)}° (relative). Target angle: {round(target_angle, 2)}°. Current angle: {round(self.angle, 2)}°."
                    elif mode == "absolute":
                        msg = f"Robot is turning to absolute angle {round(float(target), 2)}°. Current angle: {round(self.angle, 2)}°."
                    elif mode == "object":
                        msg = f"Robot is turning to face object '{target}'. Current angle: {round(self.angle, 2)}°."
                    else:
                        msg = f"Robot is turning. Current angle: {round(self.angle, 2)}°."
                        
                    self.add_event_log(
                        tick=tick_count, category="SELF_ACTION", action_type="rotate",
                        message=msg, status="PROGRESS", **args
                    )
    
    def get_formatted_logs(self, log_type: str = "all", format_style: str = "json", tick_start: int = 0, include_env_progress: bool = True) -> List[Any]:
        """
        Сборщик и форматировщик логов робота.
        
        Args:
            log_type: "all" (все), "self" (только свои), "others" (только внешние события), "env" (события погоды и окружения).
            format_style: "json" (словарь объектов), "text" (только сообщения), "brief" (краткий текстовый лог).
            tick_start: фильтр по стартовому тику симуляции.
            include_env_progress: включать ли промежуточные логи (in progress) для категории ENVIRONMENT (по умолчанию True).
        """
        logs = [log for log in self.event_log if log.get("tick", 0) >= tick_start]
        
        if not include_env_progress:
            logs = [
                log for log in logs
                if not (log.get("category") == "ENVIRONMENT" and log.get("args", {}).get("phase") == "progress")
            ]

        if log_type == "self":
            logs = [log for log in logs if log.get("category") == "SELF_ACTION"]
        elif log_type == "others":
            logs = [log for log in logs if log.get("category") in ["OBSERVATION", "PERCEPTION", "HEAR"]]
        elif log_type == "env":
            logs = [log for log in logs if log.get("category") == "ENVIRONMENT"]

        if format_style == "brief":
            brief_logs = []
            for log in logs:
                t_val = log.get("tick", 0)
                c_val = log.get("category", "unknown").upper().replace("_", " ")
                m_val = log.get("message", "")
                brief_logs.append(f"[TICK {t_val}] [{c_val}] {m_val}")
            return brief_logs
            
        elif format_style == "text":
            return [log.get("message", "") for log in logs]
            
        else:
            return logs