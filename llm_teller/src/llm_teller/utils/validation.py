from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


Json = Any


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


def _is_valid_pos(value: object) -> bool:
    """Проверить, что pos имеет вид [x, y] (два числа)."""

    if not isinstance(value, list):
        return False
    if len(value) != 2:
        return False
    return all(isinstance(v, (int, float)) for v in value)


def _has_any_pos(value: object) -> bool:
    """Проверить, что pos задан (не пустой) и имеет корректный формат."""

    return _is_valid_pos(value)


def load_objects(path: str | Path) -> list[dict[str, Json]]:
    """Загрузить список объектов из JSON файла.

    Ожидаемый формат:
    {
      "objects": [ { "object": "<имя>", ... }, ... ]
    }
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Ожидался JSON-объект верхнего уровня (dict).")
    raw_objects = data.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("Ожидался ключ 'objects' со значением-списком.")

    objects: list[dict[str, Json]] = []
    for idx, item in enumerate(raw_objects):
        if not isinstance(item, dict):
            raise ValueError(f"objects[{idx}] должен быть dict.")
        objects.append(item)  # type: ignore[assignment]
    return objects


def index_objects(objects: Sequence[Mapping[str, Json]]) -> tuple[dict[str, Mapping[str, Json]], list[ValidationIssue]]:
    """Построить индекс `имя -> объект` и вернуть проблемы (например, дубликаты)."""

    issues: list[ValidationIssue] = []
    by_name: dict[str, Mapping[str, Json]] = {}
    for obj in objects:
        name = obj.get("object")
        if not isinstance(name, str) or not name.strip():
            issues.append(ValidationIssue("Отсутствует/некорректное поле 'object' (имя)."))
            continue
        if name in by_name:
            issues.append(ValidationIssue("Дублирующееся имя объекта.", object_name=name))
            continue
        by_name[name] = obj
    return by_name, issues


def build_parent_map(
    objects_by_name: Mapping[str, Mapping[str, Json]],
) -> tuple[dict[str, str], dict[str, list[str]], list[ValidationIssue]]:
    """Построить карту родителя для каждого объекта по inventory.

    Возвращает:
    - parent_of: child -> parent
    - children_of: parent -> [child, ...] (в порядке как в inventory)
    - issues: ссылки на неизвестные объекты, множественные родители и т.п.
    """

    issues: list[ValidationIssue] = []
    parent_of: dict[str, str] = {}
    children_of: dict[str, list[str]] = {}

    for parent_name, parent_obj in objects_by_name.items():
        inventory = parent_obj.get("inventory", [])
        if inventory is None:
            inventory = []
        if not isinstance(inventory, list):
            issues.append(
                ValidationIssue("Поле 'inventory' должно быть списком.", object_name=parent_name)
            )
            continue

        children: list[str] = []
        for child in inventory:
            if not isinstance(child, str) or not child.strip():
                issues.append(
                    ValidationIssue("В 'inventory' должны быть непустые строки (имена).", object_name=parent_name)
                )
                continue
            if child not in objects_by_name:
                issues.append(
                    ValidationIssue("Ссылка в inventory на неизвестный объект.", object_name=f"{parent_name} -> {child}")
                )
                continue
            if child in parent_of and parent_of[child] != parent_name:
                issues.append(
                    ValidationIssue(
                        "Объект находится в инвентаре нескольких объектов (нарушение вложенности).",
                        object_name=child,
                    )
                )
                continue
            parent_of[child] = parent_name
            children.append(child)
        if children:
            children_of[parent_name] = children

    return parent_of, children_of, issues


def _detect_inventory_cycles(parent_of: Mapping[str, str]) -> list[list[str]]:
    """Найти циклы в отношениях child->parent (если они есть)."""

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
    """Проверить структурную целостность файла objects.json по заданным правилам."""

    objects = load_objects(objects_path)
    objects_by_name, issues = index_objects(objects)

    parent_of, children_of, inv_issues = build_parent_map(objects_by_name)
    issues.extend(inv_issues)

    # Проверка циклов
    for cycle in _detect_inventory_cycles(parent_of):
        pretty = " -> ".join(cycle)
        issues.append(ValidationIssue(f"Найден цикл вложенности inventory: {pretty}"))

    # pos: корневые объекты должны иметь pos; вложенные не должны.
    for name, obj in objects_by_name.items():
        in_inventory = name in parent_of
        pos_value = obj.get("pos", None)
        affordance = obj.get("affordance", [])
        behavior = obj.get("behavior", [])

        affordance_set: set[str] = set()
        if isinstance(affordance, list):
            affordance_set = {item for item in affordance if isinstance(item, str)}

        behavior_set: set[str] = set()
        if isinstance(behavior, list):
            behavior_set = {item for item in behavior if isinstance(item, str)}

        if in_inventory:
            if _has_any_pos(pos_value):
                issues.append(
                    ValidationIssue(
                        "Объект находится в инвентаре, но имеет координаты pos (должно быть пусто).",
                        object_name=name,
                    )
                )
        else:
            if not _has_any_pos(pos_value):
                issues.append(
                    ValidationIssue(
                        "Объект не находится в инвентаре других объектов и должен иметь координаты pos [x, y].",
                        object_name=name,
                    )
                )

        if "move" in affordance_set:
            issues.append(
                ValidationIssue("Действие 'move' допустимо только в поле behavior.", object_name=name)
            )

        if ("enter" in affordance_set) != ("exit" in affordance_set):
            issues.append(
                ValidationIssue("Если есть 'enter', должен быть 'exit' и наоборот.", object_name=name)
            )

        if ("pickup" in affordance_set) != ("put" in affordance_set):
            issues.append(
                ValidationIssue("Если есть 'pickup', должен быть 'put' и наоборот.", object_name=name)
            )

        if "use" in affordance_set:
            issues.append(
                ValidationIssue("Действие 'use' не должно присутствовать в affordance.", object_name=name)
            )

        # dynamic -> behavior includes "move"
        dynamic = obj.get("dynamic", False)
        if isinstance(dynamic, bool) and dynamic:
            if "move" not in behavior_set:
                issues.append(
                    ValidationIssue("Динамический объект должен иметь behavior 'move'.", object_name=name)
                )

        durability = obj.get("durability")
        if durability is None:
            issues.append(
                ValidationIssue("Поле 'durability' не должно быть null.", object_name=name)
            )

    # Дополнительная проверка согласованности has_inventory vs inventory
    for name, obj in objects_by_name.items():
        has_inventory = obj.get("has_inventory", False)
        inventory = obj.get("inventory", [])
        if inventory is None:
            inventory = []
        if isinstance(has_inventory, bool) and not has_inventory:
            if isinstance(inventory, list) and len(inventory) > 0:
                issues.append(
                    ValidationIssue(
                        "has_inventory=false, но inventory не пустой.",
                        object_name=name,
                    )
                )

    # Проверка «вложенность объектов в инвентаре» как единственный родитель уже сделана,
    # но добавим явное сообщение о пустых/дублирующихся записях в inventory.
    for parent, children in children_of.items():
        seen: set[str] = set()
        for child in children:
            if child in seen:
                issues.append(
                    ValidationIssue("Дублирующаяся запись в inventory.", object_name=f"{parent} -> {child}")
                )
            seen.add(child)

    return issues


def _print_tree(
    root: str,
    children_of: Mapping[str, Sequence[str]],
    *,
    indent: str = "  ",
    _prefix: str = "",
    _ancestors: frozenset[str] = frozenset(),
) -> None:
    """Распечатать дерево вложенности inventory для одного корня."""

    print(f"{_prefix}- {root}")
    if root in _ancestors:
        print(f"{_prefix}{indent}(cycle detected, truncated)")
        return

    next_prefix = _prefix + indent
    next_ancestors = _ancestors | {root}
    for child in children_of.get(root, ()):
        _print_tree(
            child,
            children_of,
            indent=indent,
            _prefix=next_prefix,
            _ancestors=next_ancestors,
        )


def print_inventory_hierarchy(objects_path: str | Path) -> None:
    """Вывести общее количество объектов и иерархию вложенности по inventory."""

    objects = load_objects(objects_path)
    objects_by_name, index_issues = index_objects(objects)
    parent_of, children_of, inv_issues = build_parent_map(objects_by_name)

    print(f"Всего объектов: {len(objects_by_name)}")

    if index_issues or inv_issues:
        print("Примечание: структура имеет проблемы, дерево может быть неполным.")

    roots = sorted([name for name in objects_by_name.keys() if name not in parent_of])
    print("Иерархия inventory (корневые объекты):")
    for root in roots:
        _print_tree(root, children_of)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI-точка входа для проверки objects.json."""

    parser = argparse.ArgumentParser(description="Проверка целостности objects.json и печать дерева inventory.")
    parser.add_argument(
        "objects_path",
        type=str,
        nargs="?",
        default="llm_teller/dummy-debug/objects.json",
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
