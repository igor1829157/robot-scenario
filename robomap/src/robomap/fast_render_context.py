"""Overlay accumulator, bounds, RenderContext и рендер одного кадра."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from robomap.fast_render_drawing import (
    canvas_to_frame_np,
    real_panel_size,
    render_map_panel,
    resolve_shape_coords,
)
from robomap.fast_render_static_cache import (
    build_static_panel_image,
    get_static_panel_from_png,
    partition_real_map_layers,
)
from robomap.headless_utils import _get_cyrillic_font


@dataclass
class FrameOverlayState:
    """Предвычисленные overlay-данные для одного кадра."""

    trail: List[Tuple[float, float]]
    commanded_targets: List[Tuple[float, float]]


class FrameHistoryAccumulator:
    """Инкрементально накапливает trail и commanded route за один проход."""

    def __init__(
        self,
        robot_id: str,
        trail_length: Optional[int] = None,
    ) -> None:
        self.robot_id = robot_id
        self.trail_length = trail_length
        self.trail: List[Tuple[float, float]] = []
        self.commanded_targets: List[Tuple[float, float]] = []
        self._seen_command_keys: set[Tuple[float, float]] = set()

    def step(self, frame: Dict[str, Any]) -> FrameOverlayState:
        """Обрабатывает один кадр и возвращает текущее состояние overlay."""
        robots_state = frame.get('robots_state', {})
        robot_data = robots_state.get(self.robot_id, {})
        rx = robot_data.get('x')
        ry = robot_data.get('y')
        if rx is not None and ry is not None:
            self.trail.append((float(rx), float(ry)))

        for action in robot_data.get('action_queue', []):
            if action.get('type') != 'MoveAction':
                continue
            params = action.get('params', {})
            tx = params.get('target_x')
            ty = params.get('target_y')
            if tx is None or ty is None:
                continue
            key = (round(float(tx), 2), round(float(ty), 2))
            if key in self._seen_command_keys:
                continue
            self._seen_command_keys.add(key)
            self.commanded_targets.append((float(tx), float(ty)))

        trail_for_draw = list(self.trail)
        if self.trail_length is not None and self.trail_length > 0:
            trail_for_draw = trail_for_draw[-self.trail_length:]

        return FrameOverlayState(
            trail=trail_for_draw,
            commanded_targets=list(self.commanded_targets),
        )


def resolve_robot_id(
    history_frames: List[Dict[str, Any]],
    target_robot_id: str,
) -> str:
    """Определяет robot_id для рендера с fallback на первого робота."""
    for frame in history_frames:
        robots_state = frame.get('robots_state', {})
        if target_robot_id in robots_state:
            return target_robot_id
        if robots_state:
            return next(iter(robots_state.keys()))
    return target_robot_id


def precompute_frame_overlays(
    history_frames: List[Dict[str, Any]],
    robot_id: str,
    trail_length: Optional[int],
) -> List[FrameOverlayState]:
    """За O(n) предвычисляет trail и commanded route для всех кадров."""
    accumulator = FrameHistoryAccumulator(robot_id, trail_length)
    return [accumulator.step(frame) for frame in history_frames]


def collect_coords_from_map(
    map_data: List[Dict[str, Any]],
    all_x: List[float],
    all_y: List[float],
) -> None:
    """Добавляет координаты объектов карты в списки all_x/all_y."""
    for obj in map_data:
        x_val = obj.get('x')
        y_val = obj.get('y')
        if x_val is not None and y_val is not None:
            all_x.append(float(x_val))
            all_y.append(float(y_val))
        for px, py in resolve_shape_coords(obj):
            all_x.append(px)
            all_y.append(py)


def compute_frame_boundaries(
    map_data: List[Dict[str, Any]],
    robots_state: Dict[str, Any],
    robot_trail: Optional[List[Tuple[float, float]]],
    commanded_targets: Optional[List[Tuple[float, float]]],
    fallback_boundaries: Tuple[List[float], List[float]],
    padding_ratio: float = 0.1,
    min_padding: float = 20.0,
) -> Tuple[List[float], List[float]]:
    """Вычисляет границы кадра по всему видимому контенту."""
    all_x: List[float] = []
    all_y: List[float] = []

    collect_coords_from_map(map_data, all_x, all_y)

    for robot_state in (robots_state or {}).values():
        rx = robot_state.get('x')
        ry = robot_state.get('y')
        if rx is not None and ry is not None:
            all_x.append(float(rx))
            all_y.append(float(ry))

    for px, py in robot_trail or []:
        all_x.append(px)
        all_y.append(py)

    for tx, ty in commanded_targets or []:
        all_x.append(tx)
        all_y.append(ty)

    if not all_x or not all_y:
        return fallback_boundaries

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    x_span = max(max_x - min_x, 1.0)
    y_span = max(max_y - min_y, 1.0)
    x_pad = max(x_span * padding_ratio, min_padding)
    y_pad = max(y_span * padding_ratio, min_padding)

    return [min_x - x_pad, max_x + x_pad], [min_y - y_pad, max_y + y_pad]


def bounds_from_coords(
    all_x: List[float],
    all_y: List[float],
    fallback_boundaries: Tuple[List[float], List[float]],
    padding_ratio: float = 0.1,
    min_padding: float = 20.0,
) -> Tuple[List[float], List[float]]:
    """Строит границы по спискам координат."""
    if not all_x or not all_y:
        return fallback_boundaries

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    x_span = max(max_x - min_x, 1.0)
    y_span = max(max_y - min_y, 1.0)
    x_pad = max(x_span * padding_ratio, min_padding)
    y_pad = max(y_span * padding_ratio, min_padding)
    return [min_x - x_pad, max_x + x_pad], [min_y - y_pad, max_y + y_pad]


def compute_global_real_bounds(
    history_frames: List[Dict[str, Any]],
    overlays: List[FrameOverlayState],
    fallback_boundaries: Tuple[List[float], List[float]],
    show_robot_trail: bool,
) -> Tuple[List[float], List[float]]:
    """Фиксирует камеру real-панели по всему прогону."""
    all_x: List[float] = []
    all_y: List[float] = []

    for frame_index, frame in enumerate(history_frames):
        overlay = overlays[frame_index]
        real_data = frame['real_map']
        robots_state = frame.get('robots_state', {})
        trail = overlay.trail if show_robot_trail else None
        commanded = overlay.commanded_targets

        collect_coords_from_map(real_data, all_x, all_y)
        for robot_state in (robots_state or {}).values():
            rx = robot_state.get('x')
            ry = robot_state.get('y')
            if rx is not None and ry is not None:
                all_x.append(float(rx))
                all_y.append(float(ry))
        for px, py in trail or []:
            all_x.append(px)
            all_y.append(py)
        for tx, ty in commanded or []:
            all_x.append(tx)
            all_y.append(ty)

    return bounds_from_coords(all_x, all_y, fallback_boundaries)


def precompute_memory_bounds(
    history_frames: List[Dict[str, Any]],
    resolved_robot_id: str,
    fallback_boundaries: Tuple[List[float], List[float]],
) -> List[Tuple[List[float], List[float]]]:
    """Предвычисляет границы memory-панели для каждого кадра."""
    bounds: List[Tuple[List[float], List[float]]] = []
    for frame in history_frames:
        robots_state = frame.get('robots_state', {})
        robots_memory = frame.get('robots_memory', {})
        target_robot_data = robots_state.get(resolved_robot_id, {})
        target_memory_data = robots_memory.get(resolved_robot_id, [])
        mem_robots_state = (
            {resolved_robot_id: target_robot_data} if target_robot_data else {}
        )
        bounds.append(
            compute_frame_boundaries(
                target_memory_data,
                mem_robots_state,
                None,
                None,
                fallback_boundaries,
            )
        )
    return bounds


@dataclass
class RenderContext:
    """Общий контекст рендера, переиспользуемый между кадрами и чанками."""

    effective_layout: str
    width: int
    height: int
    scale: float
    font_size: int
    resolved_robot_id: str
    show_robot_trail: bool
    trail_length: Optional[int]
    trail_dash_length: int
    trail_gap_length: int
    vis_config: Dict[str, Any]
    global_real_x_range: Tuple[float, float]
    global_real_y_range: Tuple[float, float]
    static_real_panel_png: Optional[bytes]
    use_static_cache: bool


def build_render_context(
    history_frames: List[Dict[str, Any]],
    effective_layout: str,
    width: int,
    height: int,
    scale: float,
    resolved_robot_id: str,
    show_robot_trail: bool,
    trail_length: Optional[int],
    trail_dash_length: int,
    trail_gap_length: int,
    vis_config: Dict[str, Any],
    fallback_boundaries: Tuple[List[float], List[float]],
    overlays: List[FrameOverlayState],
) -> Tuple[RenderContext, List[List[Dict[str, Any]]]]:
    """Строит общий контекст рендера и dynamic-слои real_map."""
    global_real_x_range, global_real_y_range = compute_global_real_bounds(
        history_frames,
        overlays,
        fallback_boundaries,
        show_robot_trail,
    )

    use_static_cache, static_objects, dynamic_per_frame = partition_real_map_layers(
        history_frames
    )

    static_real_panel_png: Optional[bytes] = None
    if use_static_cache and effective_layout != 'memory_only':
        panel_w, panel_h = real_panel_size(effective_layout, width, height)
        static_panel = build_static_panel_image(
            static_objects,
            global_real_x_range,
            global_real_y_range,
            panel_w,
            panel_h,
            vis_config,
            scale,
        )
        buffer = io.BytesIO()
        static_panel.save(buffer, format='PNG')
        static_real_panel_png = buffer.getvalue()

    ctx = RenderContext(
        effective_layout=effective_layout,
        width=width,
        height=height,
        scale=scale,
        font_size=int(52 * scale),
        resolved_robot_id=resolved_robot_id,
        show_robot_trail=show_robot_trail,
        trail_length=trail_length,
        trail_dash_length=trail_dash_length,
        trail_gap_length=trail_gap_length,
        vis_config=vis_config,
        global_real_x_range=global_real_x_range,
        global_real_y_range=global_real_y_range,
        static_real_panel_png=static_real_panel_png,
        use_static_cache=use_static_cache,
    )
    return ctx, dynamic_per_frame


def render_single_frame(
    frame: Dict[str, Any],
    overlay: FrameOverlayState,
    memory_bounds: Tuple[List[float], List[float]],
    dynamic_real_objects: List[Dict[str, Any]],
    ctx: RenderContext,
    trail_layer: Optional[Image.Image] = None,
) -> np.ndarray:
    """Рендерит один кадр в numpy RGB."""
    tick = frame['tick']
    events = frame.get('events', [])
    real_data = frame['real_map']
    robots_state = frame.get('robots_state', {})
    robots_memory = frame.get('robots_memory', {})

    target_robot_data = robots_state.get(ctx.resolved_robot_id, {})
    target_memory_data = robots_memory.get(ctx.resolved_robot_id, [])
    mem_robots_state = (
        {ctx.resolved_robot_id: target_robot_data} if target_robot_data else {}
    )

    mem_x_range, mem_y_range = memory_bounds
    canvas = Image.new('RGBA', (ctx.width, ctx.height), (255, 255, 255, 255))

    real_map_to_draw = dynamic_real_objects if ctx.use_static_cache else real_data
    static_panel: Optional[Image.Image] = None
    if ctx.use_static_cache and ctx.static_real_panel_png:
        static_panel = get_static_panel_from_png(ctx.static_real_panel_png)

    real_trail_layer = trail_layer if ctx.show_robot_trail else None

    if ctx.effective_layout == 'real_only':
        render_map_panel(
            canvas,
            real_map_to_draw,
            robots_state,
            events,
            ctx.global_real_x_range,
            ctx.global_real_y_range,
            0,
            0,
            ctx.width,
            ctx.height,
            ctx.vis_config,
            ctx.scale,
            show_robot_trail=ctx.show_robot_trail,
            trail_layer=real_trail_layer,
            commanded_targets=overlay.commanded_targets,
            target_robot_id=ctx.resolved_robot_id,
            static_background=static_panel,
        )

    elif ctx.effective_layout == 'memory_only':
        render_map_panel(
            canvas,
            target_memory_data,
            mem_robots_state,
            [],
            mem_x_range,
            mem_y_range,
            0,
            0,
            ctx.width,
            ctx.height,
            ctx.vis_config,
            ctx.scale,
            show_robot_trail=False,
        )

    elif ctx.effective_layout == 'side_by_side':
        half_width = ctx.width // 2
        render_map_panel(
            canvas,
            real_map_to_draw,
            robots_state,
            events,
            ctx.global_real_x_range,
            ctx.global_real_y_range,
            0,
            0,
            half_width,
            ctx.height,
            ctx.vis_config,
            ctx.scale,
            show_robot_trail=ctx.show_robot_trail,
            trail_layer=real_trail_layer,
            commanded_targets=overlay.commanded_targets,
            target_robot_id=ctx.resolved_robot_id,
            static_background=static_panel,
        )
        render_map_panel(
            canvas,
            target_memory_data,
            mem_robots_state,
            [],
            mem_x_range,
            mem_y_range,
            half_width,
            0,
            half_width,
            ctx.height,
            ctx.vis_config,
            ctx.scale,
            show_robot_trail=False,
        )
        divider = ImageDraw.Draw(canvas)
        divider.line(
            [(half_width, 0), (half_width, ctx.height)],
            fill='black',
            width=max(2, int(6 * ctx.scale)),
        )

    elif ctx.effective_layout == 'pip_real_main':
        render_map_panel(
            canvas,
            real_map_to_draw,
            robots_state,
            events,
            ctx.global_real_x_range,
            ctx.global_real_y_range,
            0,
            0,
            ctx.width,
            ctx.height,
            ctx.vis_config,
            ctx.scale,
            show_robot_trail=ctx.show_robot_trail,
            trail_layer=real_trail_layer,
            commanded_targets=overlay.commanded_targets,
            target_robot_id=ctx.resolved_robot_id,
            static_background=static_panel,
        )

        mini_size = int(450 * ctx.scale)
        rx = target_robot_data.get('x') or 0
        ry = target_robot_data.get('y') or 0
        mini_x_range = (float(rx) - 15, float(rx) + 15)
        mini_y_range = (float(ry) - 15, float(ry) + 15)
        padding = int(40 * ctx.scale)
        mini_x = ctx.width - mini_size - padding
        mini_y = ctx.height - mini_size - padding

        render_map_panel(
            canvas,
            target_memory_data,
            mem_robots_state,
            [],
            mini_x_range,
            mini_y_range,
            mini_x,
            mini_y,
            mini_size,
            mini_size,
            ctx.vis_config,
            ctx.scale,
            show_robot_trail=False,
        )
        mini_border = ImageDraw.Draw(canvas)
        mini_border.rectangle(
            [mini_x, mini_y, mini_x + mini_size - 1, mini_y + mini_size - 1],
            outline='gray',
            width=max(2, int(4 * ctx.scale)),
        )

    font_large = _get_cyrillic_font(ctx.font_size)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (int(40 * ctx.scale), int(40 * ctx.scale)),
        f'TICK: {tick:04d}',
        font=font_large,
        fill='black',
    )

    return canvas_to_frame_np(canvas)
