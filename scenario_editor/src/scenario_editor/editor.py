"""Окно редактора карты сценария (tkinter + matplotlib)."""

from __future__ import annotations

import copy
import math
from typing import Any, Optional

import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import messagebox, ttk

from scenario_editor.io import (
    MapEntity,
    ScenarioBundle,
    absolute_polygon,
    make_object_from_template,
    next_entity_id,
    physical_radius,
    pick_radius,
    point_in_polygon,
    save_scenario,
)


class ScenarioEditorApp:
    """GUI-редактор карты сценария."""

    ROTATE_STEP = 5.0
    ROTATE_FINE_STEP = 1.0
    ROBOT_RADIUS = 2.0
    MIN_PICK_RADIUS = 2.5
    MIN_DRAW_RADIUS = 0.4
    MAP_HALF = 100.0
    MARQUEE_PIXEL_THRESHOLD = 5.0
    UNDO_LIMIT = 50

    def __init__(self, bundle: ScenarioBundle, default_out_name: Optional[str] = None) -> None:
        self.bundle = bundle
        self.default_out_name = default_out_name

        self.selected_ids: set[str] = set()
        self.selected_type: Optional[str] = None
        self.drag_mode: Optional[str] = None
        self.drag_anchor: Optional[tuple[float, float]] = None
        self.drag_origins: dict[str, tuple[float, float]] = {}
        self.drag_undo_snapshot: Optional[list[dict[str, Any]]] = None
        self.undo_stack: list[list[dict[str, Any]]] = []
        self.marquee_start: Optional[tuple[float, float]] = None
        self.marquee_end: Optional[tuple[float, float]] = None
        self.press_pixel: Optional[tuple[float, float]] = None
        self.filtered_types = list(bundle.object_types)

        self.root = tk.Tk()
        self.root.title(f"Редактор сценария — {bundle.source_name}")
        self.root.geometry("1200x800")

        self._build_layout()
        self._bind_events()
        self._refresh_type_list()
        self._redraw()

    def _build_layout(self) -> None:
        """Собирает панели окна."""
        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, width=280)
        right = ttk.Frame(main)
        main.add(left, weight=0)
        main.add(right, weight=1)

        ttk.Label(left, text="Типы объектов").pack(anchor=tk.W, padx=8, pady=(8, 4))
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self._refresh_type_list())
        ttk.Entry(left, textvariable=self.filter_var).pack(fill=tk.X, padx=8, pady=(0, 4))

        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.type_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            exportselection=False,
        )
        scrollbar.config(command=self.type_listbox.yview)
        self.type_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.type_listbox.bind("<<ListboxSelect>>", self._on_type_selected)

        type_controls = ttk.Frame(left)
        type_controls.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Button(
            type_controls,
            text="Режим выбора",
            command=self._clear_placement_mode,
        ).pack(fill=tk.X)

        save_frame = ttk.LabelFrame(left, text="Сохранение")
        save_frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(save_frame, text="Имя сценария:").pack(anchor=tk.W, padx=8, pady=(4, 0))
        self.scenario_name_var = tk.StringVar(value=self.default_out_name or "")
        ttk.Entry(save_frame, textvariable=self.scenario_name_var).pack(
            fill=tk.X, padx=8, pady=4
        )
        buttons_row = ttk.Frame(save_frame)
        buttons_row.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(buttons_row, text="Сохранить", command=self._save_scenario).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(buttons_row, text="Выход", command=self._quit).pack(side=tk.LEFT)

        help_text = (
            "ЛКМ — выбрать / тащить\n"
            "Ctrl+ЛКМ — добавить / снять\n"
            "Рамка на пустом — несколько\n"
            "Тип + клик — поставить объект\n"
            "Esc / «Режим выбора» — стоп\n"
            "Колесо — поворот\n"
            "[ ] — поворот ±5°\n"
            "Delete — удалить\n"
            "Ctrl+Z — отменить"
        )
        ttk.Label(left, text=help_text, justify=tk.LEFT).pack(anchor=tk.W, padx=8, pady=8)

        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(left, textvariable=self.status_var, wraplength=250).pack(
            anchor=tk.W, padx=8, pady=(0, 8)
        )

        self.figure = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()

    def _bind_events(self) -> None:
        """Подключает обработчики мыши и клавиатуры."""
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.root.bind("<KeyPress-bracketleft>", self._on_rotate_left)
        self.root.bind("<KeyPress-bracketright>", self._on_rotate_right)
        self.root.bind("<KeyPress-Delete>", self._on_delete)
        self.root.bind("<KeyPress-BackSpace>", self._on_delete)
        self.root.bind("<Control-z>", self._on_undo)
        self.root.bind("<Control-Z>", self._on_undo)
        self.root.bind("<Escape>", self._on_escape)

    def _clear_placement_mode(self) -> None:
        """Выходит из режима постановки объектов."""
        self.selected_type = None
        self.type_listbox.selection_clear(0, tk.END)
        self._update_status()
        self._redraw()

    def _on_escape(self, _event) -> None:
        if self._is_text_input_focused():
            return
        self._clear_placement_mode()
        self.selected_ids.clear()
        self._update_status()
        self._redraw()

    def _capture_state(self) -> list[dict[str, Any]]:
        """Полный снимок карты для undo."""
        records: list[dict[str, Any]] = []
        for entity in self.bundle.entities:
            records.append(
                {
                    "entity_id": entity.entity_id,
                    "object_type": entity.object_type,
                    "x": entity.x,
                    "y": entity.y,
                    "angle": entity.angle,
                    "shape": copy.deepcopy(entity.shape),
                    "is_robot": entity.is_robot,
                    "meta": copy.deepcopy(entity.meta),
                }
            )
        return records

    def _restore_state(self, records: list[dict[str, Any]]) -> None:
        """Восстанавливает карту из снимка."""
        self.bundle.entities = [
            MapEntity(
                entity_id=record["entity_id"],
                object_type=record["object_type"],
                x=record["x"],
                y=record["y"],
                angle=record.get("angle", 0.0),
                shape=copy.deepcopy(record.get("shape", [])),
                is_robot=record.get("is_robot", False),
                meta=copy.deepcopy(record.get("meta", {})),
            )
            for record in records
        ]
        valid_ids = {entity.entity_id for entity in self.bundle.entities}
        self.selected_ids &= valid_ids

    def _states_equal(
        self,
        left: list[dict[str, Any]],
        right: list[dict[str, Any]],
    ) -> bool:
        """Сравнивает два снимка карты."""
        if len(left) != len(right):
            return False

        def _key(record: dict[str, Any]) -> str:
            return str(record["entity_id"])

        left_map = {_key(record): record for record in left}
        right_map = {_key(record): record for record in right}
        if left_map.keys() != right_map.keys():
            return False

        for entity_id, left_record in left_map.items():
            right_record = right_map[entity_id]
            for field in ("x", "y", "angle", "object_type", "is_robot"):
                if left_record.get(field) != right_record.get(field):
                    return False
            if left_record.get("shape") != right_record.get("shape"):
                return False
            if left_record.get("meta") != right_record.get("meta"):
                return False
        return True

    def _record_undo(self) -> None:
        """Запоминает текущее состояние перед изменением."""
        self._push_undo(self._capture_state())

    def _push_undo(self, snapshot: list[dict[str, Any]]) -> None:
        """Добавляет снимок в стек отмены."""
        if not snapshot:
            return
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > self.UNDO_LIMIT:
            self.undo_stack.pop(0)

    def _commit_drag_undo(self) -> None:
        """Сохраняет состояние до перетаскивания для Ctrl+Z."""
        if self.drag_undo_snapshot is None:
            return
        current_state = self._capture_state()
        if not self._states_equal(self.drag_undo_snapshot, current_state):
            self._push_undo(self.drag_undo_snapshot)
        self.drag_undo_snapshot = None

    def _on_undo(self, _event) -> None:
        """Отменяет последнее действие."""
        if self._is_text_input_focused():
            return
        if not self.undo_stack:
            return

        snapshot = self.undo_stack.pop()
        self._restore_state(snapshot)
        self._update_status()
        self._redraw()

    def _is_text_input_focused(self) -> bool:
        focused = self.root.focus_get()
        return isinstance(focused, (tk.Entry, ttk.Entry))

    def _refresh_type_list(self) -> None:
        """Обновляет список типов с учётом фильтра."""
        query = self.filter_var.get().strip().lower()
        if query:
            self.filtered_types = [
                name for name in self.bundle.object_types if query in name.lower()
            ]
        else:
            self.filtered_types = list(self.bundle.object_types)

        self.type_listbox.delete(0, tk.END)
        for name in self.filtered_types:
            self.type_listbox.insert(tk.END, name)

    def _on_type_selected(self, _event: object) -> None:
        """Выбор типа для постановки объекта."""
        selection = self.type_listbox.curselection()
        if not selection:
            self.selected_type = None
            return
        index = selection[0]
        self.selected_type = self.filtered_types[index]
        self._update_status()

    def _entity_pick_radius(self, entity: MapEntity) -> float:
        """Радиус захвата объекта."""
        if entity.is_robot:
            return self.ROBOT_RADIUS
        return pick_radius(entity.shape, self.MIN_PICK_RADIUS)

    def _entity_at(self, x: float, y: float) -> Optional[MapEntity]:
        """Объект под точкой (сверху вниз)."""
        for entity in reversed(self.bundle.entities):
            radius = self._entity_pick_radius(entity)
            dist = math.hypot(entity.x - x, entity.y - y)

            if entity.is_robot:
                if dist <= radius:
                    return entity
                continue

            polygon = absolute_polygon(entity.x, entity.y, entity.shape, entity.angle)
            if polygon and point_in_polygon(x, y, polygon):
                return entity
            if dist <= radius:
                return entity
        return None

    def _is_ctrl_pressed(self, event) -> bool:
        """Проверяет, зажат ли Ctrl при событии мыши."""
        key = (event.key or "").lower()
        if "control" in key or key == "ctrl":
            return True
        gui_event = getattr(event, "guiEvent", None)
        if gui_event is not None:
            state = getattr(gui_event, "state", 0)
            return bool(state & 0x0004)
        return False

    def _start_drag(self, event) -> None:
        """Начинает перетаскивание одного или нескольких объектов."""
        self.drag_mode = "multi" if len(self.selected_ids) > 1 else "single"
        self.drag_anchor = (event.xdata, event.ydata)
        self.drag_origins = {}
        for entity_id in self.selected_ids:
            entity = self._get_entity(entity_id)
            if entity is not None:
                self.drag_origins[entity_id] = (entity.x, entity.y)
        self.drag_undo_snapshot = self._capture_state()

    def _entities_in_marquee(self) -> set[str]:
        """Объекты, центр которых попадает в рамку выделения."""
        if self.marquee_start is None or self.marquee_end is None:
            return set()

        x1, x2 = sorted((self.marquee_start[0], self.marquee_end[0]))
        y1, y2 = sorted((self.marquee_start[1], self.marquee_end[1]))
        selected: set[str] = set()
        for entity in self.bundle.entities:
            if x1 <= entity.x <= x2 and y1 <= entity.y <= y2:
                selected.add(entity.entity_id)
        return selected

    def _on_press(self, event) -> None:
        """Начало перетаскивания, рамки или постановки объекта."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        if self.selected_type:
            self._place_object(event.xdata, event.ydata)
            return

        entity = self._entity_at(event.xdata, event.ydata)
        if entity is not None:
            if self._is_ctrl_pressed(event):
                if entity.entity_id in self.selected_ids:
                    self.selected_ids.discard(entity.entity_id)
                else:
                    self.selected_ids.add(entity.entity_id)
                self._update_status()
                self._redraw()
                return

            if entity.entity_id not in self.selected_ids:
                self.selected_ids = {entity.entity_id}
            self._start_drag(event)
            self._update_status()
            self._redraw()
            return

        self.marquee_start = (event.xdata, event.ydata)
        self.marquee_end = (event.xdata, event.ydata)
        self.press_pixel = (event.x, event.y)
        self.drag_mode = "marquee"
        self._redraw()

    def _on_motion(self, event) -> None:
        """Перетаскивание или рамка выделения."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        if self.drag_mode == "marquee":
            self.marquee_end = (event.xdata, event.ydata)
            self._redraw()
            return

        if self.drag_mode not in {"single", "multi"} or self.drag_anchor is None:
            return

        dx = event.xdata - self.drag_anchor[0]
        dy = event.ydata - self.drag_anchor[1]
        for entity_id, (origin_x, origin_y) in self.drag_origins.items():
            entity = self._get_entity(entity_id)
            if entity is None:
                continue
            entity.x = origin_x + dx
            entity.y = origin_y + dy
        self._redraw()

    def _on_release(self, event) -> None:
        """Завершение перетаскивания или рамки."""
        if self.drag_mode == "marquee":
            if (
                self.press_pixel is not None
                and event.x is not None
                and event.y is not None
                and math.hypot(
                    event.x - self.press_pixel[0],
                    event.y - self.press_pixel[1],
                )
                < self.MARQUEE_PIXEL_THRESHOLD
            ):
                self.selected_ids.clear()
            else:
                self.selected_ids = self._entities_in_marquee()
            self.marquee_start = None
            self.marquee_end = None
            self.press_pixel = None
            self.drag_mode = None
            self._update_status()
            self._redraw()
            return

        if self.drag_mode in {"single", "multi"}:
            self._commit_drag_undo()

        self.drag_mode = None
        self.drag_anchor = None
        self.drag_origins = {}

    def _selection_center(self, entity_ids: set[str]) -> Optional[tuple[float, float]]:
        """Центр масс выбранных объектов (без роботов)."""
        points: list[tuple[float, float]] = []
        for entity_id in entity_ids:
            entity = self._get_entity(entity_id)
            if entity is not None and not entity.is_robot:
                points.append((entity.x, entity.y))
        if not points:
            return None
        cx = sum(point[0] for point in points) / len(points)
        cy = sum(point[1] for point in points) / len(points)
        return cx, cy

    def _rotate_entities(self, entity_ids: set[str], delta: float) -> bool:
        """Поворачивает объекты: один — вокруг себя, несколько — вокруг общего центра."""
        rotatable = [
            entity
            for entity_id in entity_ids
            if (entity := self._get_entity(entity_id)) is not None and not entity.is_robot
        ]
        if not rotatable:
            return False

        if len(rotatable) == 1:
            self._record_undo()
            rotatable[0].angle = (rotatable[0].angle + delta) % 360.0
            return True

        center = self._selection_center(entity_ids)
        if center is None:
            return False

        self._record_undo()

        cx, cy = center
        rad = math.radians(-delta)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        for entity in rotatable:
            dx = entity.x - cx
            dy = entity.y - cy
            entity.x = cx + dx * cos_a - dy * sin_a
            entity.y = cy + dx * sin_a + dy * cos_a
            entity.angle = (entity.angle + delta) % 360.0
        return True

    def _on_scroll(self, event) -> None:
        """Поворот колёсиком мыши."""
        if event.inaxes != self.ax:
            return

        step = self.ROTATE_FINE_STEP if event.key == "shift" else self.ROTATE_STEP
        delta = step if event.button == "up" else -step

        target_ids = set(self.selected_ids)
        if event.xdata is not None and event.ydata is not None:
            entity = self._entity_at(event.xdata, event.ydata)
            if entity is not None:
                if len(self.selected_ids) > 1 and entity.entity_id in self.selected_ids:
                    target_ids = set(self.selected_ids)
                else:
                    target_ids = {entity.entity_id}

        if not self._rotate_entities(target_ids, delta):
            return

        if len(target_ids) == 1:
            self.selected_ids = target_ids
        self._update_status()
        self._redraw()

    def _on_rotate_left(self, _event) -> None:
        self._rotate_selected(-self.ROTATE_STEP)

    def _on_rotate_right(self, _event) -> None:
        self._rotate_selected(self.ROTATE_STEP)

    def _rotate_selected(self, delta: float) -> None:
        if not self._rotate_entities(self.selected_ids, delta):
            return
        self._update_status()
        self._redraw()

    def _on_delete(self, _event) -> None:
        if self._is_text_input_focused():
            return

        if not self.selected_ids:
            return

        delete_ids = {
            entity_id
            for entity_id in self.selected_ids
            if (entity := self._get_entity(entity_id)) is not None and not entity.is_robot
        }
        if not delete_ids:
            return

        self._record_undo()
        self.bundle.entities = [
            item for item in self.bundle.entities if item.entity_id not in delete_ids
        ]
        self.selected_ids -= delete_ids
        self._update_status()
        self._redraw()

    def _place_object(self, x: float, y: float) -> None:
        """Ставит новый объект выбранного типа."""
        if not self.selected_type:
            return

        template = self.bundle.type_templates.get(self.selected_type)
        if template is None:
            messagebox.showerror("Ошибка", f"Шаблон не найден: {self.selected_type}")
            return

        self._record_undo()

        new_id = next_entity_id(
            self.selected_type,
            self.bundle.objects_catalog,
            self.bundle.entities,
        )
        meta = make_object_from_template(template, new_id)
        entity = MapEntity(
            entity_id=new_id,
            object_type=self.selected_type,
            x=x,
            y=y,
            angle=0.0,
            shape=_shape_from_meta(meta),
            is_robot=False,
            meta=meta,
        )
        self.bundle.entities.append(entity)
        self.selected_ids.clear()
        self._update_status()
        self._redraw()

    def _save_scenario(self) -> None:
        """Сохраняет сценарий."""
        out_name = self.scenario_name_var.get().strip()
        if not out_name:
            messagebox.showwarning(
                "Сохранение",
                "Введите имя сценария.",
                parent=self.root,
            )
            return

        try:
            path = save_scenario(self.bundle, out_name)
        except Exception as exc:
            messagebox.showerror("Ошибка сохранения", str(exc), parent=self.root)
            return

        messagebox.showinfo("Сохранено", f"Сценарий сохранён:\n{path}", parent=self.root)
        self.status_var.set(f"Сохранено: {path}")

    def _quit(self) -> None:
        self.root.quit()

    def _get_entity(self, entity_id: Optional[str]) -> Optional[MapEntity]:
        if entity_id is None:
            return None
        for entity in self.bundle.entities:
            if entity.entity_id == entity_id:
                return entity
        return None

    def _update_status(self) -> None:
        if len(self.selected_ids) == 1:
            entity = self._get_entity(next(iter(self.selected_ids)))
            if entity:
                kind = "робот" if entity.is_robot else entity.object_type
                self.status_var.set(
                    f"Выбран: {entity.entity_id} ({kind}), угол {entity.angle:.0f}°"
                )
                return
        elif len(self.selected_ids) > 1:
            self.status_var.set(f"Выбрано объектов: {len(self.selected_ids)}")
            return
        elif self.selected_type:
            self.status_var.set(
                f"Постановка: {self.selected_type} — клик по карте (Esc — стоп)"
            )
            return
        self.status_var.set("Готово")

    def _draw_radius(self, entity: MapEntity) -> float:
        """Визуальный радиус объекта (реальный размер, без увеличения для клика)."""
        if entity.is_robot:
            return self.ROBOT_RADIUS
        return max(physical_radius(entity.shape), self.MIN_DRAW_RADIUS)

    def _draw_entity(self, entity: MapEntity) -> None:
        """Рисует один объект на карте."""
        is_selected = entity.entity_id in self.selected_ids
        if entity.is_robot:
            color = "blue" if is_selected else "cyan"
            patch = mpatches.Circle(
                (entity.x, entity.y),
                radius=self.ROBOT_RADIUS,
                facecolor=color,
                edgecolor="black",
                alpha=0.8,
            )
            self.ax.add_patch(patch)
            self.ax.text(
                entity.x, entity.y, entity.entity_id,
                ha="center", va="center", fontsize=7,
            )
            return

        color = "orange" if is_selected else "lightgray"
        polygon = absolute_polygon(entity.x, entity.y, entity.shape, entity.angle)

        if polygon:
            patch = mpatches.Polygon(
                polygon,
                closed=True,
                facecolor=color,
                edgecolor="black",
                alpha=0.7,
                linewidth=0.8,
            )
            self.ax.add_patch(patch)
        else:
            patch = mpatches.Circle(
                (entity.x, entity.y),
                radius=self._draw_radius(entity),
                facecolor=color,
                edgecolor="black",
                alpha=0.7,
            )
            self.ax.add_patch(patch)

    def _redraw(self) -> None:
        """Перерисовывает карту."""
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)

        border = mpatches.Rectangle(
            (-self.MAP_HALF, -self.MAP_HALF),
            self.MAP_HALF * 2,
            self.MAP_HALF * 2,
            fill=False,
            edgecolor="gray",
            linewidth=1.0,
            linestyle="--",
        )
        self.ax.add_patch(border)

        for entity in self.bundle.entities:
            self._draw_entity(entity)

        if self.marquee_start is not None and self.marquee_end is not None:
            x1, x2 = sorted((self.marquee_start[0], self.marquee_end[0]))
            y1, y2 = sorted((self.marquee_start[1], self.marquee_end[1]))
            marquee = mpatches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=True,
                facecolor="dodgerblue",
                edgecolor="dodgerblue",
                alpha=0.15,
                linewidth=1.5,
                linestyle="--",
            )
            self.ax.add_patch(marquee)

        self.ax.set_xlim(-self.MAP_HALF, self.MAP_HALF)
        self.ax.set_ylim(-self.MAP_HALF, self.MAP_HALF)

        title = f"Сценарий: {self.bundle.source_name}"
        if len(self.selected_ids) == 1:
            title += f" | выбран: {next(iter(self.selected_ids))}"
        elif self.selected_ids:
            title += f" | выбрано: {len(self.selected_ids)}"
        self.ax.set_title(title)
        self.canvas.draw_idle()

    def run(self) -> None:
        """Запускает главный цикл."""
        self.root.mainloop()


def _shape_from_meta(meta: dict) -> list[list[float]]:
    shape = meta.get("shape", [])
    return shape if isinstance(shape, list) else []


def run_editor(bundle: ScenarioBundle, default_out_name: Optional[str] = None) -> None:
    """Открывает редактор для загруженного сценария."""
    app = ScenarioEditorApp(bundle, default_out_name=default_out_name)
    app.run()
