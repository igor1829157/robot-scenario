import math
import logging
from typing import Dict, Any, Tuple, List, Optional
from shapely.geometry import Point, Polygon

from robomap.world import World

logger = logging.getLogger(__name__)

class EnvironmentManager:
    """
    Декларативный движок правил для окружения.
    """
    def __init__(self, world: World, config: Dict[str, Any]):
        self.world = world
        self.rules: List[Dict[str, Any]] = config.get("rules", [])
        self._base_attributes: Dict[Tuple[str, str], Any] = {}
        self._previous_positions: Dict[str, Tuple[float, float]] = {}
        self._log_throttle: Dict[str, Dict[str, int]] = {}
        self._tracked_virtual_zones: Dict[str, str] = {}
        self._active_rule_targets: Dict[Tuple[str, str], Tuple[Any, Dict[str, Any]]] = {}

    @staticmethod
    def _get_entity_attribute(entity: Any, attr: str, default: Any = 0.0) -> Tuple[Any, bool]:
        if "." in attr:
            parts = attr.split(".")
            curr = entity
            for part in parts[:-1]:
                if isinstance(curr, dict):
                    curr = curr.get(part, {})
                else:
                    curr = getattr(curr, part, {})
            last_key = parts[-1]
            if isinstance(curr, dict) and last_key in curr:
                return curr[last_key], True
            if hasattr(curr, last_key):
                return getattr(curr, last_key), True
            return default, False

        if hasattr(entity, attr):
            return getattr(entity, attr), True
        if hasattr(entity, "state") and isinstance(entity.state, dict) and attr in entity.state:
            return entity.state[attr], True

        return default, False

    @staticmethod
    def _set_entity_attribute(entity: Any, attr: str, value: Any):
        if "." in attr:
            parts = attr.split(".")
            curr = entity
            for part in parts[:-1]:
                if isinstance(curr, dict):
                    if part not in curr:
                        curr[part] = {}
                    curr = curr[part]
                else:
                    if not hasattr(curr, part):
                        setattr(curr, part, {})
                    curr = getattr(curr, part)
            last_key = parts[-1]
            if isinstance(curr, dict):
                curr[last_key] = value
            else:
                setattr(curr, last_key, value)
            return

        if hasattr(entity, attr):
            setattr(entity, attr, value)
        elif hasattr(entity, "state") and isinstance(entity.state, dict):
            entity.state[attr] = value
        else:
            setattr(entity, attr, value)

    @staticmethod
    def _parse_polygon(coords: Any) -> Optional[Polygon]:
        if isinstance(coords, Polygon):
            return coords
        if coords and isinstance(coords, (list, tuple)) and len(coords) >= 3:
            try:
                poly = Polygon([(float(p[0]), float(p[1])) for p in coords])
                if poly.is_valid and not poly.is_empty:
                    return poly
                return poly.buffer(0)
            except Exception:
                return None
        return None

    @classmethod
    def _interpolate_polygon(cls, poly1: Polygon, poly2: Polygon, alpha: float) -> Polygon:
        cx1, cy1 = poly1.centroid.x, poly1.centroid.y
        cx2, cy2 = poly2.centroid.x, poly2.centroid.y
        cx = (1.0 - alpha) * cx1 + alpha * cx2
        cy = (1.0 - alpha) * cy1 + alpha * cy2

        scale = 1.0 + alpha * (math.sqrt(poly2.area / max(0.001, poly1.area)) - 1.0)
        c1 = list(poly1.exterior.coords)[:-1]
        
        interp = [(cx + scale * (x - cx1), cy + scale * (y - cy1)) for x, y in c1]
        res = Polygon(interp).buffer(0)
        return res if res.is_valid else poly1

    def is_rule_active(self, rule: Dict[str, Any], current_tick: int) -> bool:
        active_ticks = rule.get("active_ticks")
        if active_ticks:
            start = active_ticks[0] if active_ticks[0] is not None else 0
            if current_tick < start:
                return False
            if len(active_ticks) >= 2 and active_ticks[1] is not None:
                if current_tick > active_ticks[1]:
                    return False
        return True

    def is_global_rule(self, rule: Dict[str, Any]) -> bool:
        return "zone" not in rule

    def get_zone_polygon(self, rule: Dict[str, Any], current_tick: int) -> Optional[Polygon]:
        if self.is_global_rule(rule) or not self.is_rule_active(rule, current_tick):
            return None

        zone = rule["zone"]
        start_tick = rule.get("active_ticks", [0])[0] or 0
        
        freeze_tick = zone.get("freeze_tick")
        effective_tick = min(current_tick, freeze_tick) if freeze_tick is not None else current_tick
        dt = max(0, effective_tick - start_tick)

        if "keyframes" in zone:
            keyframes = sorted(zone["keyframes"], key=lambda k: k["tick"])
            if current_tick <= keyframes[0]["tick"]:
                return self._parse_polygon(keyframes[0]["shape"])
            if current_tick >= keyframes[-1]["tick"]:
                return self._parse_polygon(keyframes[-1]["shape"])

            for i in range(len(keyframes) - 1):
                kf1, kf2 = keyframes[i], keyframes[i+1]
                if kf1["tick"] <= current_tick <= kf2["tick"]:
                    alpha = (current_tick - kf1["tick"]) / float(kf2["tick"] - kf1["tick"])
                    p1 = self._parse_polygon(kf1["shape"])
                    p2 = self._parse_polygon(kf2["shape"])
                    return self._interpolate_polygon(p1, p2, alpha)

        base_poly = self._parse_polygon(zone.get("shape"))
        if not base_poly:
            return None

        vel = zone.get("velocity", [0.0, 0.0])
        exp_rate = zone.get("expansion_rate", 0.0)

        if vel != [0.0, 0.0] or exp_rate != 0.0:
            cx, cy = base_poly.centroid.x, base_poly.centroid.y
            dx, dy = vel[0] * dt, vel[1] * dt
            scale = max(0.001, 1.0 + exp_rate * dt)
            
            coords = list(base_poly.exterior.coords)[:-1]
            new_coords = [(cx + dx + scale * (x - cx), cy + dy + scale * (y - cy)) for x, y in coords]
            return Polygon(new_coords).buffer(0)

        return base_poly

    def get_active_zones(self, current_tick: int) -> Dict[str, Dict[str, Any]]:
        active = {}
        for rule in self.rules:
            poly = self.get_zone_polygon(rule, current_tick)
            if poly:
                active[rule["name"]] = {
                    "polygon": poly,
                    "centroid": (float(poly.centroid.x), float(poly.centroid.y)),
                    "rule": rule
                }
        return active

    def _collect_targets(self, rule: Dict[str, Any]) -> List[Any]:
        target_type = rule.get("target_type", "all")
        cond_attr = rule.get("condition_attribute")
        
        targets = []
        objects_to_check = list(self.world.robots.values()) + getattr(self.world, "real_map_data", [])
        
        for obj in objects_to_check:
            is_robot = getattr(obj, "type_name", "") == "robot" or type(obj).__name__ == "Robot"
            
            is_match = False
            if target_type == "all": is_match = True
            elif target_type == "robot" and is_robot: is_match = True
            elif getattr(obj, "type_name", "") == target_type: is_match = True
                
            if is_match and cond_attr:
                val, exists = self._get_entity_attribute(obj, cond_attr["attribute"], 0.0)
                if not exists:
                    is_match = False
                else:
                    if "min_value" in cond_attr and val < cond_attr["min_value"]: is_match = False
                    if "max_value" in cond_attr and val > cond_attr["max_value"]: is_match = False
            
            if is_match:
                targets.append(obj)
                
            if getattr(obj, "has_inventory", False) and getattr(obj, "inventory", None):
                for item in obj.inventory:
                    item_match = False
                    if target_type == "all" or getattr(item, "type_name", "") == target_type: item_match = True
                    
                    if item_match and cond_attr:
                        val, exists = self._get_entity_attribute(item, cond_attr["attribute"], 0.0)
                        if not exists: item_match = False
                        else:
                            if "min_value" in cond_attr and val < cond_attr["min_value"]: item_match = False
                            if "max_value" in cond_attr and val > cond_attr["max_value"]: item_match = False
                    
                    if item_match:
                        targets.append(item)
        return targets

    def register_robot(self, robot_id: str):
        self._log_throttle[robot_id] = {}

    def sync_virtual_zones(self, api: Any, robot_id: str):
        """
        Опциональный метод для сценариев: синхронизирует динамические зоны с
        виртуальной памятью робота (для визуализации/демонстрации).
        - Появление зоны -> add_virtual
        - Движение/расширение -> update_virtual
        - Исчезновение -> delete_virtual
        """
        current_tick = self.world.tick_count
        active_zones = self.get_active_zones(current_tick)
        robot = self.world.robots.get(robot_id)
        if not robot:
            return

        if not hasattr(self, "_tracked_robot_zones"):
            self._tracked_robot_zones = {}
            if isinstance(getattr(self, "_tracked_virtual_zones", None), dict):
                for k, v in self._tracked_virtual_zones.items():
                    if isinstance(v, str):
                        self._tracked_robot_zones.setdefault(robot_id, {})[k] = v

        robot_tracked = self._tracked_robot_zones.setdefault(robot_id, {})

        for z_name in list(robot_tracked.keys()):
            if z_name not in active_zones:
                mem_id = robot_tracked.pop(z_name)
                api.command_action(robot_id, "delete_virtual", mem_id=mem_id, silent=True)

        for z_name, zone_info in active_zones.items():
            rule = zone_info.get("rule", {})
            zone_cfg = rule.get("zone", {}) if isinstance(rule, dict) else {}
            is_obstacle = bool(zone_cfg.get("is_obstacle", rule.get("is_obstacle", False)))
            expose_to_agent = bool(zone_cfg.get("expose_to_agent", rule.get("expose_to_agent", False)))
            render_only = bool(zone_cfg.get("render_only", rule.get("render_only", not expose_to_agent)))

            poly = zone_info["polygon"]
            cx, cy = float(poly.centroid.x), float(poly.centroid.y)
            coords = list(poly.exterior.coords)[:-1]
            rel_shape = [[round(p[0] - cx, 3), round(p[1] - cy, 3)] for p in coords]

            effects = rule.get("effects", [])
            attrs = set()
            effects_desc = []
            for eff in effects:
                a = eff.get("attribute", "")
                eff_t = eff.get("type", "")
                op = eff.get("operation", "")
                v = eff.get("value", "")
                if eff_t == "physics" and op == "drift":
                    attrs.add("drift")
                    effects_desc.append(f"снос: {v}")
                elif a == "speed":
                    attrs.add("speed")
                    effects_desc.append(f"скорость: x{v}" if eff_t == "multiply" else f"скорость: {v}")
                elif a == "vision_radius":
                    attrs.add("vision")
                    effects_desc.append(f"видимость: x{v}" if eff_t == "multiply" else f"видимость: {v}")
                elif a:
                    attrs.add(a)
                    effects_desc.append(f"{a}: {v}")

            if "speed" in attrs and "vision" in attrs and "drift" in attrs:
                modifier_type = "speed_vision_drift"
                mod_color = "#8E44AD"
                mod_label = "Скорость + Видимость + Дрейф"
            elif "speed" in attrs and "vision" in attrs:
                modifier_type = "speed_vision"
                mod_color = "#E74C3C"
                mod_label = "Скорость + Видимость"
            elif "speed" in attrs and "drift" in attrs:
                modifier_type = "speed_drift"
                mod_color = "#2980B9"
                mod_label = "Скорость + Дрейф"
            elif "speed" in attrs:
                modifier_type = "speed"
                mod_color = "#E67E22"
                mod_label = "Скорость"
            elif "vision" in attrs:
                modifier_type = "vision"
                mod_color = "#9B59B6"
                mod_label = "Видимость"
            elif "drift" in attrs:
                modifier_type = "drift"
                mod_color = "#1ABC9C"
                mod_label = "Дрейф (ветер)"
            else:
                modifier_type = "other"
                mod_color = "#7F8C8D"
                mod_label = "Среда"

            effects_str = ", ".join(effects_desc) if effects_desc else "пассив"
            notes_str = f"[ENV] {z_name} | {mod_label}: {effects_str}"

            existing_mem_id = robot_tracked.get(z_name)
            has_mem = False
            if existing_mem_id:
                has_mem = any(m.id == existing_mem_id for m in robot.memory)

            if not has_mem:
                prev_ids = {m.id for m in robot.memory}
                api.command_action(
                    robot_id, "add_virtual",
                    obj_type=z_name, x=cx, y=cy,
                    is_obstacle=is_obstacle, radius=0, shape=rel_shape,
                    notes=notes_str,
                    silent=True,
                    render_only=render_only,
                    color=mod_color
                )
                new_mem = next((m for m in reversed(robot.memory) if m.id not in prev_ids and getattr(m, "is_virtual", False) and m.type_name == z_name), None)
                if new_mem:
                    robot_tracked[z_name] = new_mem.id
                    new_mem.color = mod_color
                    if not isinstance(new_mem.state, dict):
                        new_mem.state = {}
                    new_mem.state["color"] = mod_color
                    new_mem.state["modifier_type"] = modifier_type
                    new_mem.state["modifier_label"] = mod_label
                    new_mem.state["effects_str"] = effects_str
                elif robot.memory:
                    robot_tracked[z_name] = robot.memory[-1].id
            else:
                api.command_action(
                    robot_id, "update_virtual",
                    mem_id=existing_mem_id,
                    updates={"x": cx, "y": cy, "shape": rel_shape, "is_obstacle": is_obstacle, "render_only": render_only, "notes": notes_str, "color": mod_color},
                    silent=True
                )
                mem_obj = next((m for m in robot.memory if m.id == existing_mem_id), None)
                if mem_obj:
                    mem_obj.notes = notes_str
                    mem_obj.color = mod_color
                    if not isinstance(mem_obj.state, dict):
                        mem_obj.state = {}
                    mem_obj.state["color"] = mod_color
                    mem_obj.state["modifier_type"] = modifier_type
                    mem_obj.state["modifier_label"] = mod_label
                    mem_obj.state["effects_str"] = effects_str

    def process_environment(self):
        current_tick = self.world.tick_count
        modifiers_buffer = {}
        rule_carrier_map = {}
        rule_effects = {}
        drain_records = {}

        for (entity_id, attr), base_val in list(self._base_attributes.items()):
            entity = self.world.robots.get(entity_id) or self.world.get_object_by_id(entity_id)
            if entity:
                self._set_entity_attribute(entity, attr, base_val)

        for rule in self.rules:
            if not self.is_rule_active(rule, current_tick):
                continue
            
            poly = self.get_zone_polygon(rule, current_tick)
            is_global = self.is_global_rule(rule)
            if not is_global and poly is None:
                continue

            targets = self._collect_targets(rule)
            for target in targets:
                target_id = getattr(target, "id", None)
                if not target_id:
                    continue

                carrier = target
                if target.x is None and target.location_id:
                    carrier = self.world.get_object_by_id(target.location_id) or target
                
                if not is_global and poly:
                    if carrier.x is None or carrier.y is None:
                        continue
                    if not poly.intersects(Point(carrier.x, carrier.y)):
                        continue

                key = (target_id, rule["name"])
                if key not in rule_carrier_map:
                    rule_carrier_map[key] = (carrier, rule)

                for effect in rule.get("effects", []):
                    attr = effect["attribute"]
                    eff_type = effect["type"]
                    val = effect["value"]

                    if (target_id, attr) not in self._base_attributes:
                        b_val, exists = self._get_entity_attribute(target, attr, 0.0)
                        if exists:
                            self._base_attributes[(target_id, attr)] = b_val

                    if eff_type == "drain":
                        curr_val, _ = self._get_entity_attribute(target, attr, 0.0)
                        if curr_val is None:
                            curr_val = 0.0
                        new_val = max(0.0, curr_val - float(val))
                        if "max_value" in effect:
                            new_val = min(float(effect["max_value"]), new_val)
                        self._set_entity_attribute(target, attr, new_val)
                        self._base_attributes[(target_id, attr)] = new_val
                        drain_records.setdefault(target_id, {}).setdefault(attr, []).append({
                            "rule_name": rule["name"],
                            "drain": float(val),
                            "before": float(curr_val),
                            "after": float(new_val)
                        })
                        continue

                    if eff_type == "physics" and effect.get("operation") == "drift":
                        self._handle_drift(carrier, float(val))
                        rule_effects.setdefault(target_id, {}).setdefault(rule["name"], []).append({
                            "attribute": attr,
                            "type": "physics",
                            "operation": effect.get("operation", "drift"),
                            "value": float(val)
                        })
                        continue

                    if target_id not in modifiers_buffer:
                        modifiers_buffer[target_id] = {}
                    if attr not in modifiers_buffer[target_id]:
                        modifiers_buffer[target_id][attr] = {'add': [], 'mult': [], 'set': None, 'rules': []}

                    modifiers_buffer[target_id][attr]['rules'].append({
                        'rule_name': rule['name'],
                        'effect': effect
                    })

                    if eff_type == "multiply":
                        modifiers_buffer[target_id][attr]['mult'].append(float(val))
                    elif eff_type == "add":
                        modifiers_buffer[target_id][attr]['add'].append(float(val))
                    elif eff_type == "interpolate":
                        start_t = rule.get("active_ticks", [0, 100])[0]
                        end_t = rule.get("active_ticks", [0, 100])[1]
                        progress = max(0.0, min(1.0, (current_tick - start_t) / max(1.0, float(end_t - start_t))))
                        interp_val = val[0] + (val[1] - val[0]) * progress
                        modifiers_buffer[target_id][attr]['set'] = interp_val

        for ent_id, attrs in modifiers_buffer.items():
            target = self.world.get_object_by_id(ent_id) or self.world.robots.get(ent_id)
            if not target:
                continue

            for attr, mods in attrs.items():
                if (ent_id, attr) not in self._base_attributes:
                    base_val, exists = self._get_entity_attribute(target, attr, 0.0)
                    if exists:
                        self._base_attributes[(ent_id, attr)] = base_val
                
                base = mods['set'] if mods['set'] is not None else self._base_attributes.get((ent_id, attr), 0.0)
                if base is None:
                    base = 0.0
                sum_adds = sum(mods['add'])
                prod_mults = math.prod(mods['mult']) if mods['mult'] else 1.0
                
                final_val = (base + sum_adds) * prod_mults
                self._set_entity_attribute(target, attr, final_val)

                all_rule_names = [r['rule_name'] for r in mods['rules']]
                for r in mods['rules']:
                    r_name = r['rule_name']
                    eff = r['effect']
                    eff_t = eff['type']
                    co_rules = [x for x in all_rule_names if x != r_name]
                    if eff_t == "multiply":
                        rule_effects.setdefault(ent_id, {}).setdefault(r_name, []).append({
                            "attribute": attr,
                            "type": "multiply",
                            "rule_multiplier": float(eff["value"]),
                            "base_value": round(float(base), 4),
                            "final_value": round(float(final_val), 4),
                            "delta": round(float(final_val - base), 4),
                            "combined_multiplier": round(float(prod_mults), 4),
                            "co_occurring_rules": co_rules
                        })
                    elif eff_t == "add":
                        rule_effects.setdefault(ent_id, {}).setdefault(r_name, []).append({
                            "attribute": attr,
                            "type": "add",
                            "value": float(eff["value"]),
                            "base_value": round(float(base), 4),
                            "final_value": round(float(final_val), 4),
                            "delta": round(float(final_val - base), 4),
                            "co_occurring_rules": co_rules
                        })
                    elif eff_t == "interpolate":
                        orig_base = self._base_attributes.get((ent_id, attr), 0.0)
                        rule_effects.setdefault(ent_id, {}).setdefault(r_name, []).append({
                            "attribute": attr,
                            "type": "interpolate",
                            "rule_value": round(float(mods['set']), 4),
                            "base_value": round(float(orig_base), 4),
                            "final_value": round(float(final_val), 4),
                            "delta": round(float(final_val - orig_base), 4),
                            "co_occurring_rules": co_rules
                        })

        for ent_id, attr_drains in drain_records.items():
            for attr, records in attr_drains.items():
                total_drain = sum(r['drain'] for r in records)
                all_drain_rules = [r['rule_name'] for r in records]
                for r in records:
                    r_name = r['rule_name']
                    co_rules = [x for x in all_drain_rules if x != r_name]
                    rule_effects.setdefault(ent_id, {}).setdefault(r_name, []).append({
                        "attribute": attr,
                        "type": "drain",
                        "rule_drain": float(r['drain']),
                        "before": round(float(r['before']), 4),
                        "after": round(float(r['after']), 4),
                        "delta": round(float(r['after'] - r['before']), 4),
                        "total_tick_drain": round(float(total_drain), 4),
                        "co_occurring_rules": co_rules
                    })

        active_now = set()
        for (target_id, rule_name), (carrier, rule) in rule_carrier_map.items():
            active_now.add((target_id, rule_name))
            log_cfg = rule.get("log_message")
            if log_cfg is not False and not rule.get("silent", False):
                if not log_cfg:
                    log_cfg = {
                        "message": rule["name"].replace("_", " "),
                        "level": "INFO",
                        "log_every_tick": rule.get("log_every_tick", True)
                    }
                log_every_tick = log_cfg.get("log_every_tick", rule.get("log_every_tick", True))
                was_active = (target_id, rule_name) in self._active_rule_targets
                phase = "progress" if was_active else "start"
                phase_label = "in progress" if was_active else "start"

                should_log = False
                if log_every_tick:
                    should_log = True
                elif not was_active:
                    should_log = True

                if should_log:
                    base_msg = log_cfg["message"].rstrip(". ")
                    level = log_cfg.get("level", "WARNING")
                    effects_data = rule_effects.get(target_id, {}).get(rule["name"], [])
                    details = []
                    for eff in effects_data:
                        attr = eff.get("attribute")
                        if "final_value" in eff:
                            details.append(f"{attr} changed to {eff['final_value']}")
                        elif "after" in eff:
                            details.append(f"{attr} changed to {eff['after']}")
                        elif eff.get("type") == "physics":
                            details.append(f"{attr} drift applied")

                    if details:
                        msg = f"{base_msg} ({phase_label}). {', '.join(details)}."
                    else:
                        msg = f"{base_msg} ({phase_label})."

                    log_args = {
                        "rule_name": rule["name"],
                        "level": level,
                        "phase": phase,
                        "effects": effects_data
                    }
                    if hasattr(carrier, "add_event_log"):
                        carrier.add_event_log(
                            tick=current_tick,
                            category="ENVIRONMENT",
                            action_type="weather_alert",
                            message=msg,
                            status=level,
                            **log_args
                        )
                    if hasattr(carrier, "log_event"):
                        carrier.log_event(msg, level=level, world=self.world)

        for (target_id, rule_name), (carrier, rule) in list(self._active_rule_targets.items()):
            if (target_id, rule_name) not in active_now:
                log_cfg = rule.get("log_message")
                if log_cfg is not False and not rule.get("silent", False):
                    if not log_cfg:
                        log_cfg = {
                            "message": rule["name"].replace("_", " "),
                            "level": "INFO"
                        }
                    exit_msg_base = log_cfg.get("exit_message")
                    if not exit_msg_base:
                        base_msg = log_cfg["message"].rstrip(". ")
                        exit_msg_base = f"{base_msg} (end)"
                    else:
                        exit_msg_base = f"{exit_msg_base.rstrip('. ')} (end)"

                    exit_details = []
                    exit_effects = []
                    for eff in rule.get("effects", []):
                        attr = eff.get("attribute")
                        eff_t = eff.get("type")
                        curr_val, exists = self._get_entity_attribute(carrier, attr, None)
                        if exists and isinstance(curr_val, (int, float)):
                            val_rounded = round(float(curr_val), 4)
                            exit_details.append(f"{attr} changed to {val_rounded}")
                            exit_effects.append({
                                "attribute": attr,
                                "type": eff_t,
                                "final_value": val_rounded,
                                "base_value": round(float(self._base_attributes.get((target_id, attr), val_rounded)), 4)
                            })
                        elif eff_t == "physics":
                            exit_details.append(f"{attr} normal physics restored")
                            exit_effects.append({
                                "attribute": attr,
                                "type": eff_t,
                                "status": "normal"
                            })

                    if exit_details:
                        exit_msg = f"{exit_msg_base}. {', '.join(exit_details)}."
                    else:
                        exit_msg = f"{exit_msg_base}."

                    log_args = {
                        "rule_name": rule["name"],
                        "level": "INFO",
                        "phase": "end",
                        "effects": exit_effects
                    }
                    if hasattr(carrier, "add_event_log"):
                        carrier.add_event_log(
                            tick=current_tick,
                            category="ENVIRONMENT",
                            action_type="weather_alert",
                            message=exit_msg,
                            status="INFO",
                            **log_args
                        )
                    if hasattr(carrier, "log_event"):
                        carrier.log_event(exit_msg, level="INFO", world=self.world)

        self._active_rule_targets = {k: rule_carrier_map[k] for k in active_now}
        self.latest_rule_effects = rule_effects

        all_movers = list(self.world.robots.values()) + [o for o in getattr(self.world, "real_map_data", []) if getattr(o, "dynamic", False)]
        for mover in all_movers:
            if mover.x is not None and mover.y is not None:
                if hasattr(mover, "planner") and mover.planner and hasattr(mover, "speed"):
                    mover.planner.max_step = getattr(mover, "speed", 2.0)
                self._previous_positions[mover.id] = (mover.x, mover.y)

    def get_active_modifiers(self, entity_id: str) -> List[Dict[str, Any]]:
        """Возвращает список активных модификаторов, действующих на сущность."""
        res = []
        ent_effects = getattr(self, "latest_rule_effects", {}).get(entity_id, {})
        for r_name, eff_list in ent_effects.items():
            for eff in eff_list:
                attr = eff.get("attribute", "")
                eff_type = eff.get("type", "")
                val = eff.get("rule_multiplier", eff.get("value", eff.get("rule_drain", "")))

                color = "#E67E22"
                if attr == "speed":
                    color = "#E67E22"
                elif attr == "vision_radius":
                    color = "#9B59B6"
                elif attr == "drift" or eff_type == "physics":
                    color = "#1ABC9C"
                elif attr == "temperature":
                    color = "#3498DB"
                elif attr == "battery":
                    color = "#E74C3C"

                res.append({
                    "rule": r_name,
                    "attribute": attr,
                    "type": eff_type,
                    "value": val,
                    "base_value": eff.get("base_value"),
                    "final_value": eff.get("final_value"),
                    "color": color
                })
        return res

    def _handle_drift(self, entity: Any, slip_factor: float):
        """Физика скольжения и заноса при потере сцепления."""
        prev_pos = self._previous_positions.get(entity.id)
        if not prev_pos or entity.x is None:
            return

        dx, dy = entity.x - prev_pos[0], entity.y - prev_pos[1]
        if math.hypot(dx, dy) > 0.05:
            wobble = math.sin(self.world.tick_count * 0.8) * 1.5
            slip_dx = (dx * slip_factor) + (-dy * wobble)
            slip_dy = (dy * slip_factor) + (dx * wobble)

            orig_x, orig_y = entity.x, entity.y
            entity.x += slip_dx
            entity.y += slip_dy

            if hasattr(entity, "get_polygon"):
                pred_poly = entity.get_polygon().buffer(-0.1)
                entity.x, entity.y = orig_x, orig_y
                if not self.world.check_collision(pred_poly, ignore_ids=[entity.id]):
                    entity.x += slip_dx
                    entity.y += slip_dy