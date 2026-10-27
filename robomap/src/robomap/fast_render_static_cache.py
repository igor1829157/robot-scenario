"""Кэш статичного слоя real_map для fast-render."""

from __future__ import annotations

import copy
import io
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

from robomap.fast_render_drawing import (
    CoordinateTransformer,
    draw_map_objects,
    pad_ranges_to_aspect_ratio,
    safe_float,
)
from robomap.visualization import get_object_color


def object_render_signature(obj: Dict[str, Any]) -> Tuple[Any, ...]:
    """Возвращает сигнатуру объекта для сравнения между кадрами."""
    obj_id = str(obj.get('id', ''))
    shape = obj.get('shape') or obj.get('absolute_shape') or []
    return (
        obj_id,
        round(safe_float(obj.get('x')), 4),
        round(safe_float(obj.get('y')), 4),
        round(safe_float(obj.get('angle')), 4),
        bool(obj.get('dynamic', False)),
        tuple(tuple(round(safe_float(c), 4) for c in pt[:2]) for pt in shape if pt),
    )


def partition_real_map_layers(
    history_frames: List[Dict[str, Any]],
) -> Tuple[bool, List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Определяет, можно ли кэшировать real_map, и разделяет static/dynamic слои.

    Returns:
        use_static_cache, static_objects, dynamic_objects_per_frame
    """
    if not history_frames:
        return False, [], []

    baseline = history_frames[0]['real_map']
    baseline_by_id = {str(obj.get('id', idx)): obj for idx, obj in enumerate(baseline)}

    static_ids: set[str] = set()
    for obj_id, obj in baseline_by_id.items():
        if obj.get('dynamic', False):
            continue
        signature = object_render_signature(obj)
        is_static = True
        for frame in history_frames[1:]:
            frame_by_id = {str(o.get('id', i)): o for i, o in enumerate(frame['real_map'])}
            other = frame_by_id.get(obj_id)
            if other is None or object_render_signature(other) != signature:
                is_static = False
                break
        if is_static:
            static_ids.add(obj_id)

    use_cache = bool(static_ids)
    static_objects = [baseline_by_id[obj_id] for obj_id in baseline_by_id if obj_id in static_ids]

    dynamic_per_frame: List[List[Dict[str, Any]]] = []
    for frame in history_frames:
        frame_by_id = {str(o.get('id', i)): o for i, o in enumerate(frame['real_map'])}
        dynamic_objects = [
            frame_by_id[obj_id]
            for obj_id in frame_by_id
            if obj_id not in static_ids
        ]
        dynamic_per_frame.append(dynamic_objects)

    return use_cache, static_objects, dynamic_per_frame


def freeze_vis_config(
    vis_config: Dict[str, Any],
    map_objects: List[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Копирует vis_config и прогревает цвета объектов для детерминизма."""
    frozen = copy.deepcopy(vis_config)
    seen_types: set[str] = set()
    for objects in map_objects:
        for obj in objects:
            obj_type = obj.get('object', obj.get('name', 'unknown'))
            if obj_type in seen_types:
                continue
            seen_types.add(obj_type)
            get_object_color(obj_type, frozen)
    get_object_color('robot', frozen)
    return frozen


def build_static_panel_image(
    static_objects: List[Dict[str, Any]],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    panel_w: int,
    panel_h: int,
    vis_config: Dict[str, Any],
    scale: float,
) -> Image.Image:
    """Растеризует статичный слой real_map для фиксированной камеры."""
    canvas = Image.new('RGBA', (panel_w, panel_h), (255, 255, 255, 255))
    adj_x, adj_y = pad_ranges_to_aspect_ratio(x_range, y_range, panel_w / panel_h)
    transformer = CoordinateTransformer(adj_x, adj_y, panel_w, panel_h)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [0, 0, panel_w - 1, panel_h - 1],
        fill='white',
        outline='#cccccc',
        width=max(1, int(2 * scale)),
    )
    draw_map_objects(draw, transformer, static_objects, vis_config, scale)
    return canvas


def get_static_panel_from_png(png_bytes: bytes) -> Image.Image:
    """Декодирует PNG static panel с кэшем внутри процесса."""
    if not hasattr(get_static_panel_from_png, '_cache'):
        get_static_panel_from_png._cache = {}
    cache = get_static_panel_from_png._cache
    if png_bytes not in cache:
        cache[png_bytes] = Image.open(io.BytesIO(png_bytes)).convert('RGBA')
    return cache[png_bytes]
