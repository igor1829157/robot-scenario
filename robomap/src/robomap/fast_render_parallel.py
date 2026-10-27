"""Последовательный и параллельный рендер fast-render с trail prefixes."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import imageio
from tqdm import tqdm

from robomap.fast_render_context import (
    FrameOverlayState,
    RenderContext,
    render_single_frame,
)
from robomap.fast_render_drawing import (
    TrailChunkPrefix,
    TrailLayerState,
    create_empty_trail_layer,
    decode_trail_layer_png,
    encode_trail_layer_png,
    rasterize_trail_layer,
    real_panel_size,
    trail_layer_step,
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkRenderSpec:
    """Спецификация рендера одного чанка кадров."""

    chunk_index: int
    start_index: int
    end_index: int
    output_chunk_path: str
    fps: int
    context: RenderContext
    frames: List[Dict[str, Any]]
    overlays: List[FrameOverlayState]
    memory_bounds: List[Tuple[List[float], List[float]]]
    dynamic_real_objects: List[List[Dict[str, Any]]]
    trail_prefix: TrailChunkPrefix


def precompute_trail_chunk_prefixes(
    overlays: List[FrameOverlayState],
    ctx: RenderContext,
    ranges: List[Tuple[int, int]],
) -> List[TrailChunkPrefix]:
    """Готовит PNG-prefix trail-слоя на границах parallel chunks."""
    prefixes: List[TrailChunkPrefix] = []
    panel_w, panel_h = real_panel_size(ctx.effective_layout, ctx.width, ctx.height)

    for start, _end in ranges:
        if start == 0:
            empty = create_empty_trail_layer(panel_w, panel_h)
            prefixes.append(
                TrailChunkPrefix(
                    layer_png=encode_trail_layer_png(empty.layer),
                    baked_trail=(),
                    start_index=0,
                )
            )
            continue

        trail = tuple(overlays[start - 1].trail)
        layer = rasterize_trail_layer(trail, ctx)
        prefixes.append(
            TrailChunkPrefix(
                layer_png=encode_trail_layer_png(layer),
                baked_trail=trail,
                start_index=start,
            )
        )
    return prefixes


def _init_trail_state_from_prefix(
    prefix: TrailChunkPrefix,
    ctx: RenderContext,
) -> TrailLayerState:
    """Восстанавливает trail state из chunk prefix."""
    panel_w, panel_h = real_panel_size(ctx.effective_layout, ctx.width, ctx.height)
    if not prefix.baked_trail:
        return create_empty_trail_layer(panel_w, panel_h)
    layer = decode_trail_layer_png(prefix.layer_png)
    return TrailLayerState(layer=layer, baked_trail=prefix.baked_trail)


def render_chunk_worker(spec: ChunkRenderSpec) -> str:
    """Worker: рендерит непрерывный чанк кадров во временный MP4."""
    writer = imageio.get_writer(
        spec.output_chunk_path,
        fps=spec.fps,
        codec='libx264',
        format='FFMPEG',
        macro_block_size=2,
        ffmpeg_params=['-threads', '1'],
    )
    try:
        trail_state = _init_trail_state_from_prefix(spec.trail_prefix, spec.context)
        prev_trail = trail_state.baked_trail

        for local_index, frame in enumerate(spec.frames):
            overlay = spec.overlays[local_index]
            curr_trail = tuple(overlay.trail)
            trail_state = trail_layer_step(
                trail_state,
                prev_trail,
                curr_trail,
                spec.context,
            )
            prev_trail = trail_state.baked_trail

            trail_layer = (
                trail_state.layer if spec.context.show_robot_trail else None
            )
            frame_np = render_single_frame(
                frame,
                overlay,
                spec.memory_bounds[local_index],
                spec.dynamic_real_objects[local_index],
                spec.context,
                trail_layer=trail_layer,
            )
            writer.append_data(frame_np)
        writer.close()
        return spec.output_chunk_path
    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        if os.path.isfile(spec.output_chunk_path):
            os.remove(spec.output_chunk_path)
        raise


def get_ffmpeg_executable() -> str:
    """Возвращает путь к ffmpeg."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def concat_mp4_chunks(chunk_paths: List[str], output_path: str) -> None:
    """Склеивает MP4-чанки через ffmpeg concat demuxer."""
    if not chunk_paths:
        raise ValueError('No chunk videos to concatenate.')

    if len(chunk_paths) == 1:
        shutil.move(chunk_paths[0], output_path)
        return

    ffmpeg_exe = get_ffmpeg_executable()
    list_path = f'{output_path}.concat.txt'
    with open(list_path, 'w', encoding='utf-8') as handle:
        for chunk_path in chunk_paths:
            escaped = chunk_path.replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")

    cmd = [
        ffmpeg_exe,
        '-y',
        '-f',
        'concat',
        '-safe',
        '0',
        '-i',
        list_path,
        '-c',
        'copy',
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    os.remove(list_path)
    if result.returncode != 0:
        raise RuntimeError(
            f'ffmpeg concat failed: {result.stderr.strip() or result.stdout.strip()}'
        )


def split_frame_ranges(total_frames: int, num_workers: int) -> List[Tuple[int, int]]:
    """Делит диапазон кадров на num_workers непрерывных чанков."""
    if total_frames <= 0:
        return []
    num_workers = min(num_workers, total_frames)
    base = total_frames // num_workers
    remainder = total_frames % num_workers
    ranges: List[Tuple[int, int]] = []
    start = 0
    for chunk_idx in range(num_workers):
        size = base + (1 if chunk_idx < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def render_history_sequential(
    history_frames: List[Dict[str, Any]],
    output_path: str,
    ctx: RenderContext,
    overlays: List[FrameOverlayState],
    memory_bounds: List[Tuple[List[float], List[float]]],
    dynamic_per_frame: List[List[Dict[str, Any]]],
    fps: int,
) -> None:
    """Последовательный рендер всех кадров с инкрементальным trail."""
    tmp_path = f'{output_path}.tmp.mp4'
    writer = imageio.get_writer(
        tmp_path,
        fps=fps,
        codec='libx264',
        format='FFMPEG',
        macro_block_size=2,
        ffmpeg_params=['-threads', '1'],
    )

    panel_w, panel_h = real_panel_size(ctx.effective_layout, ctx.width, ctx.height)
    trail_state = create_empty_trail_layer(panel_w, panel_h)
    prev_trail: Tuple[Tuple[float, float], ...] = ()

    try:
        for frame_index, frame in enumerate(
            tqdm(history_frames, desc='Быстрый рендеринг', unit='тик')
        ):
            overlay = overlays[frame_index]
            curr_trail = tuple(overlay.trail)
            trail_state = trail_layer_step(trail_state, prev_trail, curr_trail, ctx)
            prev_trail = trail_state.baked_trail

            trail_layer = trail_state.layer if ctx.show_robot_trail else None
            frame_np = render_single_frame(
                frame,
                overlay,
                memory_bounds[frame_index],
                dynamic_per_frame[frame_index],
                ctx,
                trail_layer=trail_layer,
            )
            writer.append_data(frame_np)
        writer.close()
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise


def render_history_parallel(
    history_frames: List[Dict[str, Any]],
    output_path: str,
    ctx: RenderContext,
    overlays: List[FrameOverlayState],
    memory_bounds: List[Tuple[List[float], List[float]]],
    dynamic_per_frame: List[List[Dict[str, Any]]],
    fps: int,
    workers: int,
) -> None:
    """Параллельный рендер чанков с ordered concat и trail prefixes."""
    total_frames = len(history_frames)
    ranges = split_frame_ranges(total_frames, workers)
    trail_prefixes = precompute_trail_chunk_prefixes(overlays, ctx, ranges)
    temp_dir = tempfile.mkdtemp(prefix='robomap_fast_render_')
    tmp_output = f'{output_path}.tmp.mp4'
    chunk_paths: List[str] = []

    try:
        specs: List[ChunkRenderSpec] = []
        for chunk_index, (start, end) in enumerate(ranges):
            chunk_path = os.path.join(temp_dir, f'chunk_{chunk_index:04d}.mp4')
            specs.append(
                ChunkRenderSpec(
                    chunk_index=chunk_index,
                    start_index=start,
                    end_index=end,
                    output_chunk_path=chunk_path,
                    fps=fps,
                    context=ctx,
                    frames=history_frames[start:end],
                    overlays=overlays[start:end],
                    memory_bounds=memory_bounds[start:end],
                    dynamic_real_objects=dynamic_per_frame[start:end],
                    trail_prefix=trail_prefixes[chunk_index],
                )
            )

        results: Dict[int, str] = {}
        chunk_sizes = {
            spec.chunk_index: spec.end_index - spec.start_index for spec in specs
        }
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(render_chunk_worker, spec): spec.chunk_index
                for spec in specs
            }
            with tqdm(
                total=total_frames,
                desc='Быстрый рендеринг',
                unit='тик',
            ) as progress:
                for future in as_completed(future_map):
                    chunk_index = future_map[future]
                    results[chunk_index] = future.result()
                    frames_done = chunk_sizes[chunk_index]
                    progress.update(frames_done)
                    logger.info(
                        'FAST RENDERER | chunk %d/%d done (%d frames)',
                        chunk_index + 1,
                        len(specs),
                        frames_done,
                    )

        chunk_paths = [results[index] for index in sorted(results.keys())]
        concat_mp4_chunks(chunk_paths, tmp_output)
        os.replace(tmp_output, output_path)
    except Exception:
        if os.path.isfile(tmp_output):
            os.remove(tmp_output)
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
