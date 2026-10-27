"""Валидация objects.json для 3d2dmap."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

Json = Any

REQUIRED_FIELDS = (
    "id",
    "object",
    "shape",
    "dynamic",
    "affordance",
    "behavior",
    "has_inventory",
    "inventory",
)

DEFAULT_OBJECTS_PATH = (
    Path(__file__).resolve().parent.parent / "output" / "objects.json"
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """Одна найденная проблема в структуре."""

    message: str
    object_name: str | None = None

    def format(self) -> str:
        """Вернуть строковое представление проблемы для печати."""
        if self.object_name is None:
            return self.message
        return f"{self.object_name}: {self.message}"


def _is_valid_shape(value: object) -> bool:
    """Проверить, что shape — список вершин [[x, y], ...]."""
    if not isinstance(value, list) or len(value) < 3:
        return False
    for vertex in value:
        if not isinstance(vertex, list) or len(vertex) != 2:
            return False
        if not all(isinstance(coord, (int, float)) for coord in vertex):
            return False
    return True


def _is_string_list(value: object) -> bool:
    """Проверить, что значение — список непустых строк."""
    if not isinstance(value, list):
        return False
    return all(isinstance(item, str) and item.strip() for item in value)


def _validate_inventory_item(
    item: object,
    object_name: str,
    issues: list[ValidationIssue],
) -> bool:
    """Проверить один элемент inventory как полный dict объекта."""
    if not isinstance(item, dict):
        issues.append(
            ValidationIssue(
                "Элемент 'inventory' должен быть dict объекта.",
                object_name=object_name,
            )
        )
        return False

    child_id = item.get("id")
    if not isinstance(child_id, str) or not child_id.strip():
        issues.append(
            ValidationIssue(
                "В элементе inventory отсутствует/некорректное поле 'id'.",
                object_name=object_name,
            )
        )
        return False

    child_label = f"{object_name} -> {child_id}"
    valid = True

    for field in REQUIRED_FIELDS:
        if field not in item:
            issues.append(
                ValidationIssue(
                    f"В inventory отсутствует обязательное поле '{field}'.",
                    object_name=child_label,
                )
            )
            valid = False

    object_type = item.get("object")
    if not isinstance(object_type, str) or not object_type.strip():
        issues.append(
            ValidationIssue(
                "В inventory отсутствует/некорректное поле 'object' (имя).",
                object_name=child_label,
            )
        )
        valid = False

    shape = item.get("shape")
    if not _is_valid_shape(shape):
        issues.append(
            ValidationIssue(
                "В inventory поле 'shape' должно быть списком вершин [[x, y], ...].",
                object_name=child_label,
            )
        )
        valid = False

    dynamic = item.get("dynamic")
    if not isinstance(dynamic, bool):
        issues.append(
            ValidationIssue(
                "В inventory поле 'dynamic' должно быть bool.",
                object_name=child_label,
            )
        )
        valid = False

    has_inventory = item.get("has_inventory")
    if not isinstance(has_inventory, bool):
        issues.append(
            ValidationIssue(
                "В inventory поле 'has_inventory' должно быть bool.",
                object_name=child_label,
            )
        )
        valid = False

    affordance = item.get("affordance", [])
    if affordance is None:
        affordance = []
    if not _is_string_list(affordance):
        issues.append(
            ValidationIssue(
                "В inventory поле 'affordance' должно быть списком строк.",
                object_name=child_label,
            )
        )
        valid = False

    behavior = item.get("behavior", [])
    if behavior is None:
        behavior = []
    if not _is_string_list(behavior):
        issues.append(
            ValidationIssue(
                "В inventory поле 'behavior' должно быть списком строк.",
                object_name=child_label,
            )
        )
        valid = False

    nested = item.get("inventory", [])
    if nested is None:
        nested = []
    if not isinstance(nested, list):
        issues.append(
            ValidationIssue(
                "В inventory поле 'inventory' должно быть списком.",
                object_name=child_label,
            )
        )
        valid = False
    else:
        for nested_item in nested:
            if not _validate_inventory_item(nested_item, child_label, issues):
                valid = False

    return valid


def _is_inventory_list(
    value: object,
    object_name: str,
    issues: list[ValidationIssue],
) -> bool:
    """Проверить, что inventory — список полных dict-объектов."""
    if not isinstance(value, list):
        return False

    valid = True
    for item in value:
        if not _validate_inventory_item(item, object_name, issues):
            valid = False
    return valid


def load_objects(path: str | Path) -> list[dict[str, Json]]:
    """Загрузить список объектов из JSON-файла.

    Ожидаемый формат: [ { "id": "...", "object": "...", ... }, ... ]
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Ожидался JSON-массив верхнего уровня (list).")

    objects: list[dict[str, Json]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"objects[{idx}] должен быть dict.")
        objects.append(item)  # type: ignore[assignment]
    return objects


def _object_label(obj_id: str, obj: Mapping[str, Json]) -> str:
    """Сформировать подпись объекта для вывода дерева."""
    name = obj.get("object")
    if isinstance(name, str) and name.strip():
        return f"{obj_id} {name}"
    return obj_id


def index_objects(
    objects: Sequence[Mapping[str, Json]],
) -> tuple[dict[str, Mapping[str, Json]], list[ValidationIssue]]:
    """Построить индекс id -> объект и вернуть проблемы индексации."""
    issues: list[ValidationIssue] = []
    by_id: dict[str, Mapping[str, Json]] = {}

    for index, obj in enumerate(objects):
        label = f"objects[{index}]"
        obj_id = obj.get("id")
        if not isinstance(obj_id, str) or not obj_id.strip():
            issues.append(
                ValidationIssue("Отсутствует/некорректное поле 'id'.", object_name=label)
            )
            continue

        for field in REQUIRED_FIELDS:
            if field not in obj:
                issues.append(
                    ValidationIssue(
                        f"Отсутствует обязательное поле '{field}'.",
                        object_name=obj_id,
                    )
                )

        object_name = obj.get("object")
        if not isinstance(object_name, str) or not object_name.strip():
            issues.append(
                ValidationIssue(
                    "Отсутствует/некорректное поле 'object' (имя).",
                    object_name=obj_id,
                )
            )

        shape = obj.get("shape")
        if not _is_valid_shape(shape):
            issues.append(
                ValidationIssue(
                    "Поле 'shape' должно быть списком вершин [[x, y], ...].",
                    object_name=obj_id,
                )
            )

        dynamic = obj.get("dynamic")
        if not isinstance(dynamic, bool):
            issues.append(
                ValidationIssue("Поле 'dynamic' должно быть bool.", object_name=obj_id)
            )

        has_inventory = obj.get("has_inventory")
        if not isinstance(has_inventory, bool):
            issues.append(
                ValidationIssue(
                    "Поле 'has_inventory' должно быть bool.",
                    object_name=obj_id,
                )
            )

        affordance = obj.get("affordance", [])
        if affordance is None:
            affordance = []
        if not _is_string_list(affordance):
            issues.append(
                ValidationIssue(
                    "Поле 'affordance' должно быть списком строк.",
                    object_name=obj_id,
                )
            )

        behavior = obj.get("behavior", [])
        if behavior is None:
            behavior = []
        if not _is_string_list(behavior):
            issues.append(
                ValidationIssue(
                    "Поле 'behavior' должно быть списком строк.",
                    object_name=obj_id,
                )
            )

        inventory = obj.get("inventory", [])
        if inventory is None:
            inventory = []
        if not _is_inventory_list(inventory, obj_id, issues):
            if isinstance(inventory, list) and not inventory:
                pass
            elif not isinstance(inventory, list):
                issues.append(
                    ValidationIssue(
                        "Поле 'inventory' должно быть списком dict-объектов.",
                        object_name=obj_id,
                    )
                )

        if obj_id in by_id:
            issues.append(
                ValidationIssue("Дублирующийся id объекта.", object_name=obj_id)
            )
            continue

        by_id[obj_id] = obj

    return by_id, issues


def _register_inventory_children(
    parent_id: str,
    inventory: list[Json],
    parent_of: dict[str, str],
    children_of: dict[str, list[str]],
    issues: list[ValidationIssue],
) -> list[str]:
    """Зарегистрировать связи parent/child по вложенным dict в inventory."""
    children: list[str] = []

    for child in inventory:
        if not isinstance(child, dict):
            continue

        child_id = child.get("id")
        if not isinstance(child_id, str) or not child_id.strip():
            issues.append(
                ValidationIssue(
                    "В 'inventory' у dict-объекта должно быть непустое поле 'id'.",
                    object_name=parent_id,
                )
            )
            continue

        if child_id in parent_of and parent_of[child_id] != parent_id:
            issues.append(
                ValidationIssue(
                    "Объект находится в инвентаре нескольких объектов (нарушение вложенности).",
                    object_name=child_id,
                )
            )
            continue

        parent_of[child_id] = parent_id
        children.append(child_id)

        nested = child.get("inventory", [])
        if nested is None:
            nested = []
        if isinstance(nested, list) and nested:
            nested_children = _register_inventory_children(
                child_id,
                nested,
                parent_of,
                children_of,
                issues,
            )
            if nested_children:
                children_of[child_id] = nested_children

    return children


def build_parent_map(
    objects_by_id: Mapping[str, Mapping[str, Json]],
) -> tuple[dict[str, str], dict[str, list[str]], list[ValidationIssue]]:
    """Построить карту родителя для каждого объекта по inventory."""
    issues: list[ValidationIssue] = []
    parent_of: dict[str, str] = {}
    children_of: dict[str, list[str]] = {}

    for parent_id, parent_obj in objects_by_id.items():
        inventory = parent_obj.get("inventory", [])
        if inventory is None:
            inventory = []
        if not isinstance(inventory, list):
            issues.append(
                ValidationIssue(
                    "Поле 'inventory' должно быть списком.",
                    object_name=parent_id,
                )
            )
            continue

        children = _register_inventory_children(
            parent_id,
            list(inventory),
            parent_of,
            children_of,
            issues,
        )
        if children:
            children_of[parent_id] = children

    return parent_of, children_of, issues


def _detect_inventory_cycles(parent_of: Mapping[str, str]) -> list[list[str]]:
    """Найти циклы в отношениях child->parent."""
    cycles: list[list[str]] = []
    visited_global: set[str] = set()

    for node in parent_of.keys():
        if node in visited_global:
            continue

        path: list[str] = []
        index_in_path: dict[str, int] = {}
        current: str | None = node

        while current is not None and current in parent_of:
            if current in index_in_path:
                cycle = path[index_in_path[current] :] + [current]
                cycles.append(cycle)
                break
            if current in visited_global:
                break
            index_in_path[current] = len(path)
            path.append(current)
            current = parent_of.get(current)

        visited_global.update(path)

    return cycles


def validate_objects(objects_path: str | Path) -> list[ValidationIssue]:
    """Проверить структурную целостность файла objects.json."""
    objects = load_objects(objects_path)
    objects_by_id, issues = index_objects(objects)

    parent_of, children_of, inv_issues = build_parent_map(objects_by_id)
    issues.extend(inv_issues)

    for cycle in _detect_inventory_cycles(parent_of):
        pretty = " -> ".join(cycle)
        issues.append(ValidationIssue(f"Найден цикл вложенности inventory: {pretty}"))

    for obj_id, obj in objects_by_id.items():
        affordance = obj.get("affordance", [])
        behavior = obj.get("behavior", [])

        affordance_set: set[str] = set()
        if isinstance(affordance, list):
            affordance_set = {item for item in affordance if isinstance(item, str)}

        behavior_set: set[str] = set()
        if isinstance(behavior, list):
            behavior_set = {item for item in behavior if isinstance(item, str)}

        if "move" in affordance_set:
            issues.append(
                ValidationIssue(
                    "Действие 'move' допустимо только в поле behavior.",
                    object_name=obj_id,
                )
            )

        if ("enter" in affordance_set) != ("exit" in affordance_set):
            issues.append(
                ValidationIssue(
                    "Если есть 'enter', должен быть 'exit' и наоборот.",
                    object_name=obj_id,
                )
            )

        if ("pickup" in affordance_set) != ("put" in affordance_set):
            issues.append(
                ValidationIssue(
                    "Если есть 'pickup', должен быть 'put' и наоборот.",
                    object_name=obj_id,
                )
            )

        forbidden = {"use", "rotateable"}
        for action in forbidden:
            if action in affordance_set:
                issues.append(
                    ValidationIssue(
                        f"Действие '{action}' не должно присутствовать в affordance.",
                        object_name=obj_id,
                    )
                )
            if action in behavior_set:
                issues.append(
                    ValidationIssue(
                        f"Действие '{action}' не должно присутствовать в behavior.",
                        object_name=obj_id,
                    )
                )

        dynamic = obj.get("dynamic", False)
        if isinstance(dynamic, bool) and dynamic and "move" not in behavior_set:
            issues.append(
                ValidationIssue(
                    "Динамический объект должен иметь behavior 'move'.",
                    object_name=obj_id,
                )
            )

    for obj_id, obj in objects_by_id.items():
        has_inventory = obj.get("has_inventory", False)
        inventory = obj.get("inventory", [])
        if inventory is None:
            inventory = []
        if isinstance(has_inventory, bool) and not has_inventory:
            if isinstance(inventory, list) and len(inventory) > 0:
                issues.append(
                    ValidationIssue(
                        "has_inventory=false, но inventory не пустой.",
                        object_name=obj_id,
                    )
                )

    for parent_id, children in children_of.items():
        seen: set[str] = set()
        for child_id in children:
            if child_id in seen:
                issues.append(
                    ValidationIssue(
                        "Дублирующаяся запись в inventory.",
                        object_name=f"{parent_id} -> {child_id}",
                    )
                )
            seen.add(child_id)

    return issues


def _print_tree(
    root: str,
    children_of: Mapping[str, Sequence[str]],
    labels: Mapping[str, str],
    *,
    indent: str = "  ",
    _prefix: str = "",
    _ancestors: frozenset[str] = frozenset(),
) -> None:
    """Распечатать дерево вложенности inventory для одного корня."""
    print(f"{_prefix}- {labels.get(root, root)}")
    if root in _ancestors:
        print(f"{_prefix}{indent}(cycle detected, truncated)")
        return

    next_prefix = _prefix + indent
    next_ancestors = _ancestors | {root}
    for child in children_of.get(root, ()):
        _print_tree(
            child,
            children_of,
            labels,
            indent=indent,
            _prefix=next_prefix,
            _ancestors=next_ancestors,
        )


def print_inventory_hierarchy(objects_path: str | Path) -> None:
    """Вывести общее количество объектов и иерархию вложенности по inventory."""
    objects = load_objects(objects_path)
    objects_by_id, index_issues = index_objects(objects)
    parent_of, children_of, inv_issues = build_parent_map(objects_by_id)

    labels = {
        obj_id: _object_label(obj_id, obj) for obj_id, obj in objects_by_id.items()
    }

    print(f"Всего объектов: {len(objects_by_id)}")

    if index_issues or inv_issues:
        print("Примечание: структура имеет проблемы, дерево может быть неполным.")

    roots = sorted(obj_id for obj_id in objects_by_id.keys() if obj_id not in parent_of)
    print("Иерархия inventory (корневые объекты):")
    for root in roots:
        _print_tree(root, children_of, labels)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI-точка входа для проверки objects.json."""
    parser = argparse.ArgumentParser(
        description="Проверка целостности objects.json и печать дерева inventory.",
    )
    parser.add_argument(
        "objects_path",
        type=str,
        nargs="?",
        default=str(DEFAULT_OBJECTS_PATH),
        help="Путь до objects.json",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    objects_path = Path(args.objects_path)
    issues = validate_objects(objects_path)
    print_inventory_hierarchy(objects_path)

    if issues:
        print("\nОшибки валидации:")
        for issue in issues:
            print(f"- {issue.format()}")
        return 1

    print("\nOK: нарушений не найдено.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
