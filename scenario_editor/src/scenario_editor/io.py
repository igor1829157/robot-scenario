"""Загрузка и сохранение данных сценария."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def get_repo_root() -> Path:
    """Корень репозитория (родитель каталога robomap)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "robomap" / "data" / "config.json").exists():
            return parent
    raise FileNotFoundError("Не найден robomap/data/config.json относительно репозитория")


def get_robomap_data_root() -> Path:
    """Каталог robomap/data."""
    return get_repo_root() / "robomap" / "data"


def get_scenarios_dir() -> Path:
    """Каталог data/scenarios."""
    return get_robomap_data_root() / "scenarios"


@dataclass
class MapEntity:
    """Объект или робот на карте."""

    entity_id: str
    object_type: str
    x: float
    y: float
    angle: float = 0.0
    shape: list[list[float]] = field(default_factory=list)
    is_robot: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioBundle:
    """Загруженный сценарий для редактирования."""

    source_name: str
    source_dir: Path
    initial_state: dict[str, Any]
    objects_catalog: list[dict[str, Any]]
    script_text: str
    entities: list[MapEntity] = field(default_factory=list)
    type_templates: dict[str, dict[str, Any]] = field(default_factory=dict)
    object_types: list[str] = field(default_factory=list)
    initial_map_ids: set[str] = field(default_factory=set)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_state_path(scenario_name: str, config: dict[str, Any]) -> Path:
    """Путь к initial_state.json (как в session_manager._get_state_path)."""
    data_root = get_robomap_data_root()
    scenarios_dir = config.get("paths", {}).get("scenarios_dir", "data/scenarios")
    custom_rel = f"{scenarios_dir}/{scenario_name}/initial_state.json"
    custom_abs = data_root.parent / custom_rel
    if custom_abs.exists():
        return custom_abs

    default_rel = config.get("paths", {}).get(
        "default_state_path",
        "data/init_state/initial_state.json",
    )
    return data_root.parent / default_rel


def _resolve_objects_path(config: dict[str, Any]) -> Path:
    """Путь к objects.json."""
    rel = config.get("paths", {}).get("objects_meta_path", "data/objects/objects.json")
    return get_robomap_data_root().parent / rel


def _merge_configs(scenario_name: str) -> dict[str, Any]:
    """Глобальный config + локальный config сценария."""
    global_cfg = _load_json(get_robomap_data_root() / "config.json")
    scenario_cfg_path = get_scenarios_dir() / scenario_name / "config.json"
    if not scenario_cfg_path.exists():
        return global_cfg

    scenario_cfg = _load_json(scenario_cfg_path)
    merged = copy.deepcopy(global_cfg)
    for key, value in scenario_cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def _build_type_templates(catalog: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Шаблоны по типу object и отсортированный список типов."""
    templates: dict[str, dict[str, Any]] = {}
    for item in catalog:
        obj_type = item.get("object") or item.get("name", "")
        if obj_type and obj_type not in templates:
            templates[obj_type] = item
    return templates, sorted(templates.keys())


def _meta_for_entity(entity_id: str, catalog: list[dict[str, Any]]) -> dict[str, Any]:
    for item in catalog:
        if str(item.get("id")) == entity_id:
            return item
    return {}


def _shape_from_meta(meta: dict[str, Any]) -> list[list[float]]:
    shape = meta.get("shape", [])
    if isinstance(shape, list):
        return shape
    return []


def load_scenario(scenario_name: str) -> ScenarioBundle:
    """Загружает сценарий для редактирования."""
    source_dir = get_scenarios_dir() / scenario_name
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Сценарий не найден: {source_dir}")

    config = _merge_configs(scenario_name)
    state_path = _resolve_state_path(scenario_name, config)
    objects_path = _resolve_objects_path(config)

    if not state_path.exists():
        raise FileNotFoundError(f"initial_state не найден: {state_path}")
    if not objects_path.exists():
        raise FileNotFoundError(f"objects.json не найден: {objects_path}")

    initial_state = _load_json(state_path)
    objects_catalog = _load_json(objects_path)
    script_path = source_dir / "script.py"
    script_text = script_path.read_text(encoding="utf-8") if script_path.exists() else ""

    type_templates, object_types = _build_type_templates(objects_catalog)
    entities: list[MapEntity] = []

    for item in initial_state.get("real_map_data", []):
        entity_id = str(item.get("id", ""))
        meta = _meta_for_entity(entity_id, objects_catalog)
        entities.append(
            MapEntity(
                entity_id=entity_id,
                object_type=str(meta.get("object") or meta.get("name") or entity_id),
                x=float(item.get("x", 0.0)),
                y=float(item.get("y", 0.0)),
                angle=float(item.get("angle", 0.0)),
                shape=_shape_from_meta(meta),
                is_robot=False,
                meta=copy.deepcopy(meta),
            )
        )

    initial_map_ids = {entity.entity_id for entity in entities if not entity.is_robot}

    for item in initial_state.get("robots_data", []):
        entity_id = str(item.get("id", "robot_1"))
        entities.append(
            MapEntity(
                entity_id=entity_id,
                object_type="robot",
                x=float(item.get("x", 0.0)),
                y=float(item.get("y", 0.0)),
                angle=float(item.get("angle", 0.0)),
                shape=[],
                is_robot=True,
                meta=copy.deepcopy(item),
            )
        )

    return ScenarioBundle(
        source_name=scenario_name,
        source_dir=source_dir,
        initial_state=initial_state,
        objects_catalog=objects_catalog,
        script_text=script_text,
        entities=entities,
        type_templates=type_templates,
        object_types=object_types,
        initial_map_ids=initial_map_ids,
    )


def absolute_polygon(
    x: float,
    y: float,
    shape: list[list[float]],
    angle: float,
) -> list[tuple[float, float]]:
    """Абсолютные координаты полигона (как WorldObject.get_polygon)."""
    if len(shape) < 3:
        return []

    rad = math.radians(-angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    coords: list[tuple[float, float]] = []
    for pt in shape:
        if len(pt) < 2 or pt[0] is None or pt[1] is None:
            continue
        dx, dy = float(pt[0]), float(pt[1])
        rot_dx = dx * cos_a - dy * sin_a
        rot_dy = dx * sin_a + dy * cos_a
        coords.append((x + rot_dx, y + rot_dy))
    return coords


def physical_radius(shape: list[list[float]]) -> float:
    """Максимальное расстояние от центра до вершин shape (как WorldObject)."""
    if not shape:
        return 0.4

    max_sq_dist = 0.0
    for pt in shape:
        if len(pt) >= 2 and pt[0] is not None:
            dist_sq = float(pt[0]) ** 2 + float(pt[1]) ** 2
            if dist_sq > max_sq_dist:
                max_sq_dist = dist_sq
    return math.sqrt(max_sq_dist) if max_sq_dist > 0 else 0.4


def pick_radius(shape: list[list[float]], min_radius: float = 2.5) -> float:
    """Радиус захвата объекта мышью."""
    return max(min_radius, physical_radius(shape))


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Проверка попадания точки в полигон (ray casting)."""
    if len(polygon) < 3:
        return False

    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _collect_numeric_ids(catalog: list[dict[str, Any]], entities: list[MapEntity]) -> list[int]:
    """Собирает числовые id и суффиксы для генерации нового id."""
    numbers: list[int] = []

    def scan_id(raw_id: str) -> None:
        if raw_id.isdigit():
            numbers.append(int(raw_id))
            return
        match = re.search(r"_(\d+)$", raw_id)
        if match:
            numbers.append(int(match.group(1)))

    for item in catalog:
        scan_id(str(item.get("id", "")))
    for entity in entities:
        scan_id(entity.entity_id)
    return numbers


def next_entity_id(object_type: str, catalog: list[dict[str, Any]], entities: list[MapEntity]) -> str:
    """Новый уникальный id вида {тип}_{n}."""
    numbers = _collect_numeric_ids(catalog, entities)
    next_num = (max(numbers) + 1) if numbers else 1
    candidate = f"{object_type}_{next_num}"
    existing = {str(item.get("id")) for item in catalog}
    existing.update(entity.entity_id for entity in entities)
    while candidate in existing:
        next_num += 1
        candidate = f"{object_type}_{next_num}"
    return candidate


def make_object_from_template(template: dict[str, Any], entity_id: str) -> dict[str, Any]:
    """Копия шаблона objects.json с новым id и пустым inventory."""
    obj = copy.deepcopy(template)
    obj["id"] = entity_id
    obj["object"] = template.get("object") or template.get("name", entity_id)
    obj["inventory"] = []
    if obj.get("has_inventory") is None:
        obj["has_inventory"] = bool(template.get("has_inventory", False))
    return obj


def build_initial_state(bundle: ScenarioBundle) -> dict[str, Any]:
    """Сериализует текущее состояние карты в initial_state.json."""
    real_map_data: list[dict[str, Any]] = []
    robots_data: list[dict[str, Any]] = []

    for entity in bundle.entities:
        if entity.is_robot:
            robot_entry = copy.deepcopy(entity.meta)
            robot_entry["id"] = entity.entity_id
            robot_entry["x"] = entity.x
            robot_entry["y"] = entity.y
            if entity.angle != 0.0:
                robot_entry["angle"] = entity.angle
            elif "angle" in robot_entry:
                robot_entry.pop("angle")
            robots_data.append(robot_entry)
            continue

        entry: dict[str, Any] = {
            "id": entity.entity_id,
            "x": entity.x,
            "y": entity.y,
            "timestamp": "0",
        }
        if entity.angle != 0.0:
            entry["angle"] = entity.angle
        real_map_data.append(entry)

    return {
        "robots_data": robots_data,
        "real_map_data": real_map_data,
    }


def build_objects_catalog(bundle: ScenarioBundle) -> list[dict[str, Any]]:
    """objects.json: полный каталог с добавлениями и удалениями объектов карты."""
    on_map_ids = {entity.entity_id for entity in bundle.entities if not entity.is_robot}
    map_entities = {
        entity.entity_id: entity
        for entity in bundle.entities
        if not entity.is_robot
    }
    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in bundle.objects_catalog:
        item_id = str(item.get("id", ""))
        if item_id in on_map_ids:
            entity = map_entities[item_id]
            meta = copy.deepcopy(entity.meta)
            meta["id"] = item_id
            meta["object"] = entity.object_type
            result.append(meta)
            seen.add(item_id)
        elif item_id in bundle.initial_map_ids:
            continue
        else:
            result.append(copy.deepcopy(item))
            seen.add(item_id)

    for entity in bundle.entities:
        if entity.is_robot or entity.entity_id in seen:
            continue
        meta = copy.deepcopy(entity.meta)
        meta["id"] = entity.entity_id
        meta["object"] = entity.object_type
        result.append(meta)

    return result


def save_scenario(bundle: ScenarioBundle, new_name: str) -> Path:
    """Сохраняет полную копию сценария в robomap/data/scenarios/{new_name}."""
    if not new_name or not re.fullmatch(r"[A-Za-z0-9_-]+", new_name):
        raise ValueError("Имя сценария: только буквы, цифры, _ и -")

    out_dir = get_scenarios_dir() / new_name
    if out_dir.exists():
        raise FileExistsError(f"Сценарий уже существует: {out_dir}")

    out_dir.mkdir(parents=True)

    initial_state = build_initial_state(bundle)
    objects_catalog = build_objects_catalog(bundle)

    config = {
        "paths": {
            "default_state_path": f"data/scenarios/{new_name}/initial_state.json",
            "objects_meta_path": f"data/scenarios/{new_name}/objects.json",
        }
    }

    with (out_dir / "initial_state.json").open("w", encoding="utf-8") as handle:
        json.dump(initial_state, handle, ensure_ascii=False, indent=4)
        handle.write("\n")

    with (out_dir / "objects.json").open("w", encoding="utf-8") as handle:
        json.dump(objects_catalog, handle, ensure_ascii=False, indent=4)
        handle.write("\n")

    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    script_src = bundle.source_dir / "script.py"
    if script_src.exists():
        shutil.copy2(script_src, out_dir / "script.py")
    else:
        (out_dir / "script.py").write_text(bundle.script_text, encoding="utf-8")

    return out_dir
