"""Конвертация CSV с 3D-объектами в objects.json и initial_state.json."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "DemoScene_top_objects.csv"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent.parent
    / "robomap"
    / "data"
    / "scenarios"
    / "3dmap"
)
MAX_DIMENSION_THRESHOLD = 100.0
BREAK_SIZE_THRESHOLD = 6.0
INVENTORY_BOUNDS_EPSILON = 0.55

NON_BREAKABLE_KEYWORDS = (
    "building",
    "concrete",
    "tower",
    "chimney",
    "rail",
    "helipad",
    "pylon",
    "antenna",
    "radar",
    "platform",
    "stairs",
    "hangar",
    "snowpile",
    "electricpylon",
    "dressingroom",
    "terrain",
)

STRUCTURAL_BREAK_EXCEPTIONS = (
    "door",
    "window",
)

CONTAINER_KEYWORDS = (
    "building",
    "hangar",
    "container",
    "box",
    "cabinet",
    "locker",
    "fridge",
    "table",
    "desk",
    "shelf",
    "rack",
    "bed",
    "chair",
    "sofa",
    "platform",
    "pallet",
    "trolley",
    "trashbin",
    "bucket",
    "tank",
    "barrel",
    "case",
    "suitecase",
    "kennel",
    "shower",
    "toilet",
    "sink",
    "alcohol",
    "bottle",
    "medpack",
    "healt",
    "medical",
    "corridoor",
)

ENTER_EXIT_KEYWORDS = (
    "building",
    "door",
    "hangar",
    "container",
    "kennel",
    "shower",
    "toilet",
    "gate",
)

PICKUP_PUT_KEYWORDS = (
    "axe",
    "hammer",
    "wrench",
    "screwdriver",
    "knife",
    "saw",
    "crowbar",
    "battery",
    "radio",
    "phone",
    "tablet",
    "map",
    "bottle",
    "cannedfood",
    "medpack",
    "vaccine",
    "syringe",
    "antibiotic",
    "alcohol",
    "match",
    "compass",
    "fork",
    "spoon",
    "plate",
    "coffepot",
    "torchlamp",
    "tool",
    "pot",
    "food",
    "medic",
    "healt",
    "poison",
    "oil",
    "glass",
    "clock",
    "keyboard",
    "monitor",
    "penholder",
    "paperholder",
    "iceaxe",
    "shovel",
    "snowshovel",
    "flare",
    "trashbag",
    "padlock",
    "chain",
    "flag",
    "speaker",
    "intercom",
    "modem",
    "alarm",
    "detector",
    "switch",
    "connector",
    "automat",
    "coffe",
    "curtain",
    "tarp",
    "ladder",
    "sign",
    "gascylinder",
    "dispenser",
    "heather",
)

BREAK_KEYWORDS = (
    "wood",
    "glass",
    "box",
    "door",
    "fence",
    "pipe",
    "container",
    "chair",
    "table",
    "lamp",
    "fridge",
    "cabinet",
    "plank",
    "barrel",
    "tank",
    "monitor",
    "television",
    "microwave",
    "trashbag",
    "wallclock",
    "trashbin",
    "bed",
    "sofa",
    "gate",
    "drain",
    "satellite",
    "pallet",
    "locker",
    "rack",
    "relux",
    "tree",
    "rock",
    "intercom",
    "speaker",
    "switch",
    "connector",
    "alarm",
    "detector",
    "sign",
    "ladder",
    "automat",
    "coffe",
    "dispenser",
    "cylinder",
    "flag",
    "padlock",
    "chain",
    "curtain",
    "tarp",
    "modem",
    "heather",
    "sink",
    "holder",
    "trolley",
    "kennel",
    "shower",
    "toilet",
)

DYNAMIC_KEYWORDS = (
    "human",
    "person",
    "man",
    "woman",
    "bear",
    "wolf",
    "dog",
    "car",
    "truck",
    "vehicle",
    "robot",
)

HUMAN_KEYWORDS = ("human", "person", "man", "woman")
ANIMAL_KEYWORDS = ("bear", "wolf", "dog")
VEHICLE_KEYWORDS = ("car", "truck", "vehicle")


@dataclass
class ParsedObject:
    """Промежуточное представление объекта при конвертации."""

    obj_id: str
    name: str
    normalized_name: str
    pos_x: float
    pos_y: float
    pos_z: float
    rot_y: float
    size_x: float
    size_y: float
    size_z: float
    bounds_min_x: float
    bounds_min_y: float
    bounds_min_z: float
    bounds_max_x: float
    bounds_max_y: float
    bounds_max_z: float
    shape: list[list[float]]
    dynamic: bool = False
    affordance: list[str] = field(default_factory=list)
    behavior: list[str] = field(default_factory=list)
    has_inventory: bool = False
    inventory: list[ParsedObject] = field(default_factory=list)
    localized_x: float = 0.0
    localized_y: float = 0.0


def normalize_name(name: str) -> str:
    """Убрать суффикс в скобках и привести имя к нижнему регистру для правил."""
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    return cleaned.lower()


def display_name(name: str) -> str:
    """Имя объекта для вывода в objects.json (как в CSV, со скобками)."""
    return name.strip()


def name_contains_any(normalized_name: str, keywords: tuple[str, ...]) -> bool:
    """Проверить, содержит ли имя одно из ключевых слов."""
    return any(keyword in normalized_name for keyword in keywords)


EXCLUDED_NAME_PATTERNS = (
    r"sm_rail\d",
    r"sm_antenna\d",
    r"sm_satellite$",
    r"sm_satelliteholder\d",
    r".*roofrail$",
    r"sm_acbox\d",
    r"sm_chimney\d",
    r"sm_bigbox\d",
    r"sm_radar02$",
    r"sm_box01$",
    r"sm_stairs\d",
    r"sm_fence\d",
    r"sm_smalllampflashing$",
)

EXCLUDED_WITHOUT_CASCADE_PATTERNS = (
    r"sm_platform$",
    r"sm_concrete\d",
    r"sm_hangar\d+_backwall$",
    r".*walldoor$",
    r"sm_interiordoor\d",
    r"^sm_door\d+$",
)


def _matches_name_patterns(normalized_name: str, patterns: tuple[str, ...]) -> bool:
    """Проверить, подходит ли нормализованное имя под один из regex-паттернов."""
    return any(re.match(pattern, normalized_name) for pattern in patterns)


def is_excluded_object(raw_name: str) -> bool:
    """Проверить, нужно ли исключить объект из экспорта (Rail, Antenna, Satellite и т.п.)."""
    normalized = normalize_name(raw_name)
    return (
        _matches_name_patterns(normalized, EXCLUDED_NAME_PATTERNS)
        or _matches_name_patterns(normalized, EXCLUDED_WITHOUT_CASCADE_PATTERNS)
    )


def is_cascade_excluded_object(raw_name: str) -> bool:
    """Проверить, должно ли исключение объекта запускать каскадное удаление содержимого."""
    normalized = normalize_name(raw_name)
    return _matches_name_patterns(normalized, EXCLUDED_NAME_PATTERNS)


def build_shape(size_x: float, size_z: float) -> list[list[float]]:
    """Построить прямоугольный полигон по размерам SizeX и SizeZ."""
    half_x = size_x / 2.0
    half_z = size_z / 2.0
    return [
        [-half_x, -half_z],
        [half_x, -half_z],
        [half_x, half_z],
        [-half_x, half_z],
    ]


def build_shape_from_bounds(
    pos_x: float,
    pos_z: float,
    bounds_min_x: float,
    bounds_min_z: float,
    bounds_max_x: float,
    bounds_max_z: float,
) -> list[list[float]]:
    """Построить полигон по BoundsMin/BoundsMax относительно PosX/PosZ."""
    return [
        [bounds_min_x - pos_x, bounds_min_z - pos_z],
        [bounds_max_x - pos_x, bounds_min_z - pos_z],
        [bounds_max_x - pos_x, bounds_max_z - pos_z],
        [bounds_min_x - pos_x, bounds_max_z - pos_z],
    ]


def resolve_dynamic(normalized_name: str) -> bool:
    """Определить, является ли объект динамическим."""
    return name_contains_any(normalized_name, DYNAMIC_KEYWORDS)


def resolve_has_inventory(normalized_name: str) -> bool:
    """Определить, может ли объект хранить другие объекты."""
    return name_contains_any(normalized_name, CONTAINER_KEYWORDS)


def can_break(
    normalized_name: str,
    size_x: float,
    size_z: float,
    max_dimension: float,
) -> bool:
    """Проверить, можно ли назначить affordance break."""
    footprint = max(size_x, size_z, max_dimension)
    if footprint > BREAK_SIZE_THRESHOLD:
        if not name_contains_any(normalized_name, STRUCTURAL_BREAK_EXCEPTIONS):
            return False

    if name_contains_any(normalized_name, NON_BREAKABLE_KEYWORDS):
        if name_contains_any(normalized_name, STRUCTURAL_BREAK_EXCEPTIONS):
            return footprint <= BREAK_SIZE_THRESHOLD
        return False

    return True


def resolve_affordance(
    normalized_name: str,
    size_x: float,
    size_z: float,
    max_dimension: float,
) -> list[str]:
    """Подобрать affordance по имени объекта."""
    affordance: set[str] = set()

    if name_contains_any(normalized_name, ENTER_EXIT_KEYWORDS):
        affordance.update(("enter", "exit"))

    if name_contains_any(normalized_name, PICKUP_PUT_KEYWORDS):
        affordance.update(("pickup", "put"))
        if can_break(normalized_name, size_x, size_z, max_dimension):
            affordance.add("break")

    elif name_contains_any(normalized_name, BREAK_KEYWORDS) and can_break(
        normalized_name,
        size_x,
        size_z,
        max_dimension,
    ):
        affordance.add("break")

    if name_contains_any(normalized_name, HUMAN_KEYWORDS):
        affordance.add("speak")

    return sorted(affordance)


def resolve_behavior(normalized_name: str, dynamic: bool) -> list[str]:
    """Подобрать behavior по имени объекта."""
    if not dynamic:
        return []

    if name_contains_any(normalized_name, HUMAN_KEYWORDS):
        return ["break", "enter", "exit", "move", "pickup", "put", "speak"]

    if name_contains_any(normalized_name, ANIMAL_KEYWORDS):
        return ["break", "enter", "exit", "move", "pickup", "put"]

    if name_contains_any(normalized_name, VEHICLE_KEYWORDS):
        return ["break", "enter", "exit", "move"]

    return []


def aabb_volume(obj: ParsedObject) -> float:
    """Объём мирового AABB объекта из BoundsMin/BoundsMax."""
    return (
        (obj.bounds_max_x - obj.bounds_min_x)
        * (obj.bounds_max_y - obj.bounds_min_y)
        * (obj.bounds_max_z - obj.bounds_min_z)
    )


def is_fully_inside_container(
    child: ParsedObject,
    container: ParsedObject,
    epsilon: float = INVENTORY_BOUNDS_EPSILON,
) -> bool:
    """Проверить попадание AABB дочернего объекта внутрь AABB контейнера с допуском."""
    return (
        child.bounds_min_x >= container.bounds_min_x - epsilon
        and child.bounds_max_x <= container.bounds_max_x + epsilon
        and child.bounds_min_y >= container.bounds_min_y - epsilon
        and child.bounds_max_y <= container.bounds_max_y + epsilon
        and child.bounds_min_z >= container.bounds_min_z - epsilon
        and child.bounds_max_z <= container.bounds_max_z + epsilon
    )


def assign_inventory(objects: list[ParsedObject]) -> None:
    """Распределить вложенные объекты по inventory контейнеров.

    Объект может лежать внутри другого (например, шкаф в ангаре, а лекарства
    в шкафу). Конейнером считается только объект с has_inventory; для каждого
    предмета выбирается наименьший по объёму подходящий контейнер.
    """
    containers = [obj for obj in objects if obj.has_inventory]
    for child in objects:
        candidates: list[ParsedObject] = []
        for container in containers:
            if container.obj_id == child.obj_id:
                continue
            if is_fully_inside_container(container, child):
                continue
            if is_fully_inside_container(child, container):
                candidates.append(container)

        if not candidates:
            continue

        best_container = min(candidates, key=aabb_volume)
        best_container.inventory.append(child)
        best_container.has_inventory = True


def is_row_fully_inside_row(child: dict[str, str], container: dict[str, str]) -> bool:
    """Проверить полное попадание AABB строки CSV внутрь AABB другой строки."""
    return (
        float(child["BoundsMinX"]) >= float(container["BoundsMinX"])
        and float(child["BoundsMaxX"]) <= float(container["BoundsMaxX"])
        and float(child["BoundsMinY"]) >= float(container["BoundsMinY"])
        and float(child["BoundsMaxY"]) <= float(container["BoundsMaxY"])
        and float(child["BoundsMinZ"]) >= float(container["BoundsMinZ"])
        and float(child["BoundsMaxZ"]) <= float(container["BoundsMaxZ"])
    )


def collect_cascade_excluded_row_indices(rows: list[dict[str, str]]) -> set[int]:
    """Собрать индексы строк для каскадного исключения по EXCLUDED_NAME_PATTERNS."""
    excluded_indices: set[int] = {
        index
        for index, row in enumerate(rows)
        if is_excluded_object(row["Name"])
    }
    cascade_containers: set[int] = {
        index
        for index, row in enumerate(rows)
        if is_cascade_excluded_object(row["Name"])
    }

    changed = True
    while changed:
        changed = False
        for child_index, child_row in enumerate(rows):
            if child_index in excluded_indices:
                continue
            for container_index in cascade_containers:
                if is_row_fully_inside_row(child_row, rows[container_index]):
                    excluded_indices.add(child_index)
                    cascade_containers.add(child_index)
                    changed = True
                    break

    return excluded_indices


def read_csv_rows(csv_path: Path, max_dimension: float) -> list[dict[str, str]]:
    """Прочитать CSV и отфильтровать слишком большие объекты."""
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            if float(row["MaxDimension"]) > max_dimension:
                continue
            rows.append(row)

    cascade_excluded = collect_cascade_excluded_row_indices(rows)
    return [
        row
        for index, row in enumerate(rows)
        if index not in cascade_excluded
    ]


def build_objects(rows: list[dict[str, str]]) -> list[ParsedObject]:
    """Собрать список объектов из строк CSV."""
    if not rows:
        return []

    positions_x = [float(row["PosX"]) for row in rows]
    positions_z = [float(row["PosZ"]) for row in rows]
    center_x = (min(positions_x) + max(positions_x)) / 2.0
    center_z = (min(positions_z) + max(positions_z)) / 2.0

    objects: list[ParsedObject] = []
    for index, row in enumerate(rows, start=1):
        raw_name = row["Name"]
        normalized = normalize_name(raw_name)
        size_x = float(row["SizeX"])
        size_y = float(row["SizeY"])
        size_z = float(row["SizeZ"])
        max_dimension = float(row["MaxDimension"])
        dynamic = resolve_dynamic(normalized)

        parsed = ParsedObject(
            obj_id=str(index),
            name=display_name(raw_name),
            normalized_name=normalized,
            pos_x=float(row["PosX"]),
            pos_y=float(row["PosY"]),
            pos_z=float(row["PosZ"]),
            rot_y=float(row["RotY"]),
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            bounds_min_x=float(row["BoundsMinX"]),
            bounds_min_y=float(row["BoundsMinY"]),
            bounds_min_z=float(row["BoundsMinZ"]),
            bounds_max_x=float(row["BoundsMaxX"]),
            bounds_max_y=float(row["BoundsMaxY"]),
            bounds_max_z=float(row["BoundsMaxZ"]),
            shape=build_shape_from_bounds(
                float(row["PosX"]),
                float(row["PosZ"]),
                float(row["BoundsMinX"]),
                float(row["BoundsMinZ"]),
                float(row["BoundsMaxX"]),
                float(row["BoundsMaxZ"]),
            ),
            dynamic=dynamic,
            affordance=resolve_affordance(normalized, size_x, size_z, max_dimension),
            behavior=resolve_behavior(normalized, dynamic),
            has_inventory=resolve_has_inventory(normalized),
            localized_x=float(row["PosX"]) - center_x,
            localized_y=float(row["PosZ"]) - center_z,
        )
        objects.append(parsed)

    assign_inventory(objects)
    return objects


def parsed_object_to_dict(item: ParsedObject) -> dict[str, Any]:
    """Сериализовать объект в dict для objects.json с рекурсивным inventory."""
    return {
        "id": item.obj_id,
        "object": item.name,
        "shape": item.shape,
        "dynamic": item.dynamic,
        "affordance": item.affordance,
        "behavior": item.behavior,
        "has_inventory": item.has_inventory,
        "inventory": [parsed_object_to_dict(child) for child in item.inventory],
    }


def objects_to_json(objects: list[ParsedObject]) -> list[dict[str, Any]]:
    """Сформировать структуру objects.json."""
    return [parsed_object_to_dict(item) for item in objects]


def _collect_nested_ids(objects: list[ParsedObject]) -> set[str]:
    """Собрать id всех объектов, вложенных в inventory (рекурсивно)."""
    nested_ids: set[str] = set()

    def _walk(item: ParsedObject) -> None:
        for child in item.inventory:
            nested_ids.add(child.obj_id)
            _walk(child)

    for obj in objects:
        _walk(obj)

    return nested_ids


def initial_state_to_json(objects: list[ParsedObject]) -> dict[str, Any]:
    """Сформировать структуру initial_state.json."""
    nested_ids = _collect_nested_ids(objects)
    return {
        "robots_data": [],
        "real_map_data": [
            {
                "id": item.obj_id,
                "x": round(item.localized_x, 6),
                "y": round(item.localized_y, 6),
                "timestamp": "0",
            }
            for item in objects
            if item.obj_id not in nested_ids
        ],
    }


def write_json(path: Path, data: Any) -> None:
    """Записать JSON-файл с читаемым форматированием."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
        json_file.write("\n")


def convert(
    csv_path: Path,
    output_dir: Path,
    max_dimension: float = MAX_DIMENSION_THRESHOLD,
) -> tuple[int, int]:
    """Выполнить конвертацию CSV в два JSON-файла."""
    rows = read_csv_rows(csv_path, max_dimension)
    objects = build_objects(rows)

    objects_path = output_dir / "objects.json"
    initial_state_path = output_dir / "initial_state.json"

    write_json(objects_path, objects_to_json(objects))
    write_json(initial_state_path, initial_state_to_json(objects))

    return len(objects), len(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Конвертация 3D CSV объектов в objects.json и initial_state.json.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Путь к входному CSV-файлу.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Папка для выходных JSON-файлов.",
    )
    parser.add_argument(
        "--max-dimension",
        type=float,
        default=MAX_DIMENSION_THRESHOLD,
        help="Порог MaxDimension для фильтрации крупных объектов.",
    )
    return parser


def main() -> None:
    """Точка входа CLI."""
    args = build_arg_parser().parse_args()
    object_count, _ = convert(args.csv, args.output, args.max_dimension)
    print(f"Создано объектов: {object_count}")
    print(f"objects.json -> {args.output / 'objects.json'}")
    print(f"initial_state.json -> {args.output / 'initial_state.json'}")


if __name__ == "__main__":
    main()
