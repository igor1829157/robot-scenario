"""PIL-примитивы, отрисовка панелей и инкрементальный trail layer."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from robomap.visualization import get_object_color

COMMANDED_ROUTE_COLOR = '#FF0000'

VALID_LAYOUT_TYPES = frozenset({
    'real_only',
    'memory_only',
    'side_by_side',
    'pip_real_main',
})


def pad_ranges_to_aspect_ratio(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    target_ratio: float,
) -> Tuple[List[float], List[float]]:
    """Расширяет диапазоны координат до целевого соотношения сторон."""
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    if y_span == 0:
        y_span = 1

    current_ratio = x_span / y_span
    if current_ratio < target_ratio:
        new_x_span = y_span * target_ratio
        pad = (new_x_span - x_span) / 2
        return [x_range[0] - pad, x_range[1] + pad], list(y_range)

    new_y_span = x_span / target_ratio
    pad = (new_y_span - y_span) / 2
    return list(x_range), [y_range[0] - pad, y_range[1] + pad]


class CoordinateTransformer:
    """Преобразование мировых координат в пиксели canvas."""

    def __init__(
        self,
        x_range: List[float],
        y_range: List[float],
        width: int,
        height: int,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> None:
        self.x_min = x_range[0]
        self.x_max = x_range[1]
        self.y_min = y_range[0]
        self.y_max = y_range[1]
        self.width = width
        self.height = height
        self.offset_x = offset_x
        self.offset_y = offset_y

    def to_pixel(self, wx: float, wy: float) -> Tuple[int, int]:
        """Переводит мировые координаты в пиксели."""
        x_span = self.x_max - self.x_min
        y_span = self.y_max - self.y_min
        if x_span == 0:
            x_span = 1
        if y_span == 0:
            y_span = 1

        px = int(round((wx - self.x_min) / x_span * (self.width - 1)))
        py = int(round((1.0 - (wy - self.y_min) / y_span) * (self.height - 1)))
        return self.offset_x + px, self.offset_y + py


def safe_float(value: Any, default: float = 0.0) -> float:
    """Безопасно преобразует значение в float, подставляя default для None."""
    if value is None:
        return default
    return float(value)


def resolve_shape_coords(obj: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Возвращает абсолютные координаты полигона объекта."""
    base_x = safe_float(obj.get('x'))
    base_y = safe_float(obj.get('y'))

    coords = obj.get('absolute_shape', [])
    if not coords:
        rel_shape = obj.get('shape', [])
        if rel_shape and len(rel_shape) >= 3:
            obj_angle = safe_float(obj.get('angle'))
            rad = math.radians(-obj_angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            coords = []
            for dx, dy in rel_shape:
                if dx is not None and dy is not None:
                    rot_dx = dx * cos_a - dy * sin_a
                    rot_dy = dx * sin_a + dy * cos_a
                    coords.append([base_x + rot_dx, base_y + rot_dy])

    result: List[Tuple[float, float]] = []
    for pt in coords:
        if pt is None or len(pt) < 2:
            continue
        px, py = pt[0], pt[1]
        if px is None or py is None:
            continue
        result.append((float(px), float(py)))
    return result


def draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: List[Tuple[int, int]],
    color: str,
    width: int,
    dash_length: int,
    gap_length: int,
) -> None:
    """Рисует пунктирную ломаную линию."""
    if len(points) < 2:
        return

    for start, end in zip(points[:-1], points[1:]):
        x0, y0 = start
        x1, y1 = end
        segment_len = math.hypot(x1 - x0, y1 - y0)
        if segment_len == 0:
            continue

        dx = (x1 - x0) / segment_len
        dy = (y1 - y0) / segment_len
        pos = 0.0
        drawing = True

        while pos < segment_len:
            step = dash_length if drawing else gap_length
            next_pos = min(pos + step, segment_len)
            if drawing:
                sx = int(x0 + dx * pos)
                sy = int(y0 + dy * pos)
                ex = int(x0 + dx * next_pos)
                ey = int(y0 + dy * next_pos)
                draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
            pos = next_pos
            drawing = not drawing


def real_panel_size(layout: str, width: int, height: int) -> Tuple[int, int]:
    """Возвращает размер real-панели для layout."""
    if layout == 'side_by_side':
        return width // 2, height
    return width, height


@dataclass
class TrailLayerState:
    """Состояние инкрементального trail-слоя для real-панели."""

    layer: Image.Image
    baked_trail: Tuple[Tuple[float, float], ...]


@dataclass
class TrailChunkPrefix:
    """PNG-prefix trail-слоя на границе parallel chunk."""

    layer_png: bytes
    baked_trail: Tuple[Tuple[float, float], ...]
    start_index: int


def create_empty_trail_layer(panel_w: int, panel_h: int) -> TrailLayerState:
    """Создаёт пустой прозрачный trail-слой."""
    return TrailLayerState(
        layer=Image.new('RGBA', (panel_w, panel_h), (0, 0, 0, 0)),
        baked_trail=(),
    )


def decode_trail_layer_png(png_bytes: bytes) -> Image.Image:
    """Декодирует PNG trail-слоя."""
    return Image.open(io.BytesIO(png_bytes)).convert('RGBA')


def encode_trail_layer_png(layer: Image.Image) -> bytes:
    """Кодирует trail-слой в PNG bytes."""
    buffer = io.BytesIO()
    layer.save(buffer, format='PNG')
    return buffer.getvalue()


def real_panel_transformer(
    ctx: Any,
    panel_w: int,
    panel_h: int,
) -> CoordinateTransformer:
    """Строит transformer для real-панели без panel offset (локальные координаты)."""
    adj_x, adj_y = pad_ranges_to_aspect_ratio(
        ctx.global_real_x_range,
        ctx.global_real_y_range,
        panel_w / panel_h,
    )
    return CoordinateTransformer(adj_x, adj_y, panel_w, panel_h)


def draw_robot_trail(
    draw: ImageDraw.ImageDraw,
    transformer: CoordinateTransformer,
    trail_points: List[Tuple[float, float]],
    vis_config: Dict[str, Any],
    scale: float,
    dash_length: int,
    gap_length: int,
) -> None:
    """Рисует пунктирный след робота."""
    if len(trail_points) < 2:
        return

    pixel_points = [transformer.to_pixel(px, py) for px, py in trail_points]
    trail_color = get_object_color('robot', vis_config)
    draw_dashed_polyline(
        draw,
        pixel_points,
        color=trail_color,
        width=1,
        dash_length=max(4, int(dash_length * scale)),
        gap_length=max(3, int(gap_length * scale)),
    )


def draw_robot_trail_segment(
    draw: ImageDraw.ImageDraw,
    transformer: CoordinateTransformer,
    start: Tuple[float, float],
    end: Tuple[float, float],
    vis_config: Dict[str, Any],
    scale: float,
    dash_length: int,
    gap_length: int,
) -> None:
    """Рисует один пунктирный сегмент trail между двумя точками."""
    pixel_points = [
        transformer.to_pixel(start[0], start[1]),
        transformer.to_pixel(end[0], end[1]),
    ]
    trail_color = get_object_color('robot', vis_config)
    draw_dashed_polyline(
        draw,
        pixel_points,
        color=trail_color,
        width=1,
        dash_length=max(4, int(dash_length * scale)),
        gap_length=max(3, int(gap_length * scale)),
    )


def rasterize_trail_layer(
    trail: Tuple[Tuple[float, float], ...],
    ctx: Any,
) -> Image.Image:
    """Полностью перестраивает trail-слой для заданного списка точек."""
    panel_w, panel_h = real_panel_size(ctx.effective_layout, ctx.width, ctx.height)
    layer = Image.new('RGBA', (panel_w, panel_h), (0, 0, 0, 0))
    if len(trail) < 2:
        return layer

    transformer = real_panel_transformer(ctx, panel_w, panel_h)
    draw = ImageDraw.Draw(layer)
    draw_robot_trail(
        draw,
        transformer,
        list(trail),
        ctx.vis_config,
        ctx.scale,
        dash_length=ctx.trail_dash_length,
        gap_length=ctx.trail_gap_length,
    )
    return layer


def trail_layer_step(
    state: TrailLayerState,
    prev_trail: Tuple[Tuple[float, float], ...],
    curr_trail: Tuple[Tuple[float, float], ...],
    ctx: Any,
) -> TrailLayerState:
    """Инкрементально обновляет trail-слой; при sliding window — полная перестройка."""
    if not ctx.show_robot_trail:
        return state

    if curr_trail == prev_trail:
        return state

    panel_w, panel_h = real_panel_size(ctx.effective_layout, ctx.width, ctx.height)

    if ctx.trail_length is not None and ctx.trail_length > 0:
        layer = rasterize_trail_layer(curr_trail, ctx)
        return TrailLayerState(layer=layer, baked_trail=curr_trail)

    if (
        len(curr_trail) == len(prev_trail) + 1
        and curr_trail[:-1] == prev_trail
        and len(curr_trail) >= 2
    ):
        transformer = real_panel_transformer(ctx, panel_w, panel_h)
        draw = ImageDraw.Draw(state.layer)
        draw_robot_trail_segment(
            draw,
            transformer,
            curr_trail[-2],
            curr_trail[-1],
            ctx.vis_config,
            ctx.scale,
            dash_length=ctx.trail_dash_length,
            gap_length=ctx.trail_gap_length,
        )
        return TrailLayerState(layer=state.layer, baked_trail=curr_trail)

    layer = rasterize_trail_layer(curr_trail, ctx)
    return TrailLayerState(layer=layer, baked_trail=curr_trail)


def draw_commanded_route(
    draw: ImageDraw.ImageDraw,
    transformer: CoordinateTransformer,
    robot_position: Tuple[float, float],
    commanded_targets: List[Tuple[float, float]],
    scale: float,
) -> None:
    """Рисует тонкий пунктирный маршрут move и красные маркеры целей."""
    if not commanded_targets:
        return

    route_points = [transformer.to_pixel(robot_position[0], robot_position[1])]
    for tx, ty in commanded_targets:
        route_points.append(transformer.to_pixel(tx, ty))

    if len(route_points) >= 2:
        draw_dashed_polyline(
            draw,
            route_points,
            color=COMMANDED_ROUTE_COLOR,
            width=1,
            dash_length=3,
            gap_length=10,
        )

    marker_radius = max(2, int(8 * scale / 3))
    for tx, ty in commanded_targets:
        center = transformer.to_pixel(tx, ty)
        draw.ellipse(
            [
                center[0] - marker_radius,
                center[1] - marker_radius,
                center[0] + marker_radius,
                center[1] + marker_radius,
            ],
            fill=COMMANDED_ROUTE_COLOR,
            outline='black',
            width=1,
        )


def draw_map_objects(
    draw: ImageDraw.ImageDraw,
    transformer: CoordinateTransformer,
    map_data: List[Dict[str, Any]],
    vis_config: Dict[str, Any],
    scale: float,
) -> None:
    """Рисует объекты карты: полигоны и точки."""
    for obj in map_data:
        x_val = obj.get('x')
        y_val = obj.get('y')
        if x_val is None or y_val is None:
            continue

        obj_type = obj.get('object', obj.get('name', 'unknown'))
        color = get_object_color(obj_type, vis_config)
        shape_coords = resolve_shape_coords(obj)

        if len(shape_coords) >= 3:
            pixel_coords = [transformer.to_pixel(px, py) for px, py in shape_coords]
            outline_width = max(2, int(2 * scale))
            draw.polygon(
                pixel_coords,
                fill=color,
                outline='black',
                width=outline_width,
            )
        else:
            px, py = transformer.to_pixel(float(x_val), float(y_val))
            radius = max(6, int(10 * scale))
            draw.ellipse(
                [px - radius, py - radius, px + radius, py + radius],
                fill=color,
                outline='black',
                width=1,
            )


def draw_path_waypoints(
    draw: ImageDraw.ImageDraw,
    transformer: CoordinateTransformer,
    waypoints: List[Any],
    color: str,
    scale: float,
) -> None:
    """Рисует маркеры точек маршрута робота."""
    marker_radius = max(3, int(5 * scale))
    for point in waypoints:
        if not point or len(point) < 2:
            continue
        px, py = point[0], point[1]
        if px is None or py is None:
            continue
        center = transformer.to_pixel(float(px), float(py))
        draw.ellipse(
            [
                center[0] - marker_radius,
                center[1] - marker_radius,
                center[0] + marker_radius,
                center[1] + marker_radius,
            ],
            fill=color,
            outline='black',
            width=1,
        )


def draw_robot(
    draw: ImageDraw.ImageDraw,
    transformer: CoordinateTransformer,
    robot_id: str,
    robot_state: Dict[str, Any],
    vis_config: Dict[str, Any],
    scale: float,
    has_commanded_route: bool = False,
) -> None:
    """Рисует робота, направление и запланированный путь."""
    rx = robot_state.get('x')
    ry = robot_state.get('y')
    if rx is None or ry is None:
        return

    robot_color = get_object_color('robot', vis_config)
    center = transformer.to_pixel(float(rx), float(ry))
    radius = max(4, int(8 * scale))
    draw.ellipse(
        [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
        fill=robot_color,
        outline='black',
        width=max(1, int(2 * scale)),
    )

    if not has_commanded_route:
        planned_path = robot_state.get('planned_path', [])
        if planned_path:
            path_points = [transformer.to_pixel(float(rx), float(ry))]
            for point in planned_path:
                if not point or len(point) < 2:
                    continue
                px, py = point[0], point[1]
                if px is None or py is None:
                    continue
                path_points.append(transformer.to_pixel(float(px), float(py)))
            if len(path_points) >= 2:
                draw_dashed_polyline(
                    draw,
                    path_points,
                    color=robot_color,
                    width=max(1, int(2 * scale)),
                    dash_length=max(3, int(6 * scale)),
                    gap_length=max(2, int(5 * scale)),
                )
            draw_path_waypoints(draw, transformer, planned_path, robot_color, scale)

    angle = robot_state.get('angle')
    if angle is not None:
        dir_length = 2.5
        rad_angle = math.radians(-float(angle))
        end_x = float(rx) + dir_length * math.cos(rad_angle)
        end_y = float(ry) + dir_length * math.sin(rad_angle)
        start_px = transformer.to_pixel(float(rx), float(ry))
        end_px = transformer.to_pixel(end_x, end_y)
        draw.line([start_px, end_px], fill='red', width=max(2, int(3 * scale)))

    label = robot_id
    draw.text((center[0] + radius + 2, center[1] - radius), label, fill='black')


def draw_events(
    draw: ImageDraw.ImageDraw,
    events: List[str],
    width: int,
    scale: float,
) -> None:
    """Рисует события текущего тика поверх карты."""
    if not events:
        return

    y_offset = int(80 * scale)
    for event_text in events[:5]:
        text = f"{event_text}"
        text_bbox = draw.textbbox((0, 0), text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x_pos = max(0, (width - text_width) // 2)
        draw.rectangle(
            [
                x_pos - 8,
                y_offset - 4,
                x_pos + text_width + 8,
                y_offset + text_height + 4,
            ],
            fill=(0, 0, 0, 153),
        )
        draw.text((x_pos, y_offset), text, fill='yellow')
        y_offset += text_height + int(12 * scale)


def render_map_panel(
    canvas: Image.Image,
    map_data: List[Dict[str, Any]],
    robots_state: Dict[str, Any],
    events: List[str],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    panel_x: int,
    panel_y: int,
    panel_w: int,
    panel_h: int,
    vis_config: Dict[str, Any],
    scale: float,
    show_robot_trail: bool = True,
    trail_layer: Optional[Image.Image] = None,
    commanded_targets: Optional[List[Tuple[float, float]]] = None,
    target_robot_id: Optional[str] = None,
    static_background: Optional[Image.Image] = None,
) -> None:
    """Рендерит одну панель карты на общий canvas."""
    adj_x, adj_y = pad_ranges_to_aspect_ratio(x_range, y_range, panel_w / panel_h)
    transformer = CoordinateTransformer(
        adj_x,
        adj_y,
        panel_w,
        panel_h,
        offset_x=panel_x,
        offset_y=panel_y,
    )
    draw = ImageDraw.Draw(canvas)

    if static_background is not None:
        canvas.paste(static_background, (panel_x, panel_y), static_background)
    else:
        draw.rectangle(
            [panel_x, panel_y, panel_x + panel_w - 1, panel_y + panel_h - 1],
            fill='white',
            outline='#cccccc',
            width=max(1, int(2 * scale)),
        )
        draw_map_objects(draw, transformer, map_data, vis_config, scale)

    if map_data and static_background is not None:
        draw_map_objects(draw, transformer, map_data, vis_config, scale)

    if show_robot_trail and trail_layer is not None:
        canvas.paste(trail_layer, (panel_x, panel_y), trail_layer)

    has_commanded_route = bool(commanded_targets)
    robot_position: Optional[Tuple[float, float]] = None
    if target_robot_id and target_robot_id in (robots_state or {}):
        rx = robots_state[target_robot_id].get('x')
        ry = robots_state[target_robot_id].get('y')
        if rx is not None and ry is not None:
            robot_position = (float(rx), float(ry))

    if commanded_targets and robot_position:
        draw_commanded_route(
            draw,
            transformer,
            robot_position,
            commanded_targets,
            scale,
        )

    for robot_id, robot_state in (robots_state or {}).items():
        draw_robot(
            draw,
            transformer,
            robot_id,
            robot_state,
            vis_config,
            scale,
            has_commanded_route=has_commanded_route and robot_id == target_robot_id,
        )

    draw_events(draw, events, panel_w, scale)


def canvas_to_frame_np(canvas: Image.Image) -> np.ndarray:
    """Конвертирует PIL canvas в numpy RGB с чётными размерами для x264."""
    final_rgb = canvas.convert('RGB')
    frame_np = np.array(final_rgb)
    if frame_np.shape[0] % 2 != 0:
        frame_np = frame_np[:-1, :, :]
    if frame_np.shape[1] % 2 != 0:
        frame_np = frame_np[:, :-1, :]
    return frame_np
