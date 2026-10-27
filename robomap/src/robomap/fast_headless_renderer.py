"""Быстрый headless-рендер MP4 без Plotly/Kaleido."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from robomap.cpu_limit import detect_cpu_limit, resolve_worker_count
from robomap.fast_render_context import (
    build_render_context,
    precompute_frame_overlays,
    precompute_memory_bounds,
    resolve_robot_id,
)
from robomap.fast_render_drawing import VALID_LAYOUT_TYPES
from robomap.fast_render_parallel import (
    render_history_parallel,
    render_history_sequential,
)
from robomap.fast_render_static_cache import freeze_vis_config

logger = logging.getLogger(__name__)


def render_history_to_fast_video(
    history_frames: List[Dict[str, Any]],
    output_path: str,
    vis_config: Dict[str, Any],
    layout_type: str = 'pip_real_main',
    map_boundaries: Tuple[List[float], List[float]] = ([-50, 50], [-50, 50]),
    fps: int = 10,
    width: int = 1920,
    height: int = 1080,
    target_robot_id: str = 'robot_1',
    show_robot_trail: bool = True,
    trail_length: Optional[int] = None,
    trail_dash_length: int = 10,
    trail_gap_length: int = 8,
    workers: Optional[int] = None,
) -> None:
    """Рендерит MP4 из истории кадров без Plotly."""
    if not history_frames:
        logger.warning('FAST RENDERER | No history frames to render. Aborting video creation.')
        return

    fallback_boundaries = map_boundaries
    scale = max(0.5, height / 1080.0)

    effective_layout = layout_type
    if layout_type not in VALID_LAYOUT_TYPES:
        logger.warning(
            "FAST RENDERER | Unknown layout_type '%s'. Falling back to 'real_only'.",
            layout_type,
        )
        effective_layout = 'real_only'

    resolved_robot_id = resolve_robot_id(history_frames, target_robot_id)
    overlays = precompute_frame_overlays(history_frames, resolved_robot_id, trail_length)
    memory_bounds = precompute_memory_bounds(
        history_frames,
        resolved_robot_id,
        fallback_boundaries,
    )

    map_layers = [frame['real_map'] for frame in history_frames]
    map_layers.append(history_frames[0]['real_map'])
    frozen_vis_config = freeze_vis_config(vis_config, map_layers)

    ctx, dynamic_per_frame = build_render_context(
        history_frames=history_frames,
        effective_layout=effective_layout,
        width=width,
        height=height,
        scale=scale,
        resolved_robot_id=resolved_robot_id,
        show_robot_trail=show_robot_trail,
        trail_length=trail_length,
        trail_dash_length=trail_dash_length,
        trail_gap_length=trail_gap_length,
        vis_config=frozen_vis_config,
        fallback_boundaries=fallback_boundaries,
        overlays=overlays,
    )

    worker_count = resolve_worker_count(workers)
    cpu_limit = detect_cpu_limit()
    logger.info(
        'FAST RENDERER | Starting video rendering (%d frames) | Layout: %s | '
        'Resolution: %dx%d @ %dFPS | cpu_limit=%d workers=%d | static_cache=%s',
        len(history_frames),
        layout_type,
        width,
        height,
        fps,
        cpu_limit,
        worker_count,
        ctx.use_static_cache,
    )

    if worker_count <= 1:
        render_history_sequential(
            history_frames,
            output_path,
            ctx,
            overlays,
            memory_bounds,
            dynamic_per_frame,
            fps,
        )
    else:
        render_history_parallel(
            history_frames,
            output_path,
            ctx,
            overlays,
            memory_bounds,
            dynamic_per_frame,
            fps,
            worker_count,
        )

    logger.info('FAST RENDERER | Video rendering completed successfully.')
