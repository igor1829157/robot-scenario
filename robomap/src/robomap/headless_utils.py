import os
import io
import base64
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import urllib.request
from typing import Optional

from robomap.visualization import create_map_figure

import logging
logger = logging.getLogger(__name__)

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def _get_cyrillic_font(size: int) -> ImageFont.FreeTypeFont:
    font_path = "data/assets/roboto.ttf"
    os.makedirs(os.path.dirname(font_path), exist_ok=True)
    
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Regular.ttf"
        try:
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            return ImageFont.load_default()
            
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()

def get_asset_url_getter(vis_config: dict):
    def _mock_asset_url_getter(icon_path: str) -> str | None:
        if not icon_path: return None
        base_dir = vis_config.get("asset_base_path", "data/assets")
        local_path = os.path.join(project_path, base_dir, icon_path)
        try:
            with open(local_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode()
            return f"data:image/svg+xml;base64,{encoded}"
        except Exception:
            return None
    return _mock_asset_url_getter

def _pad_ranges_to_aspect_ratio(x_range: tuple, y_range: tuple, target_ratio: float):
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    if y_span == 0: y_span = 1
    
    current_ratio = x_span / y_span
    if current_ratio < target_ratio:
        new_x_span = y_span * target_ratio
        pad = (new_x_span - x_span) / 2
        return [x_range[0] - pad, x_range[1] + pad], y_range
    else:
        new_y_span = x_span / target_ratio
        pad = (new_y_span - y_span) / 2
        return x_range, [y_range[0] - pad, y_range[1] + pad]

def _render_plotly_to_pil(map_data: list, robots_states: dict, events: list, 
                          x_range: tuple, y_range: tuple, width: int, height: int, is_minimap: bool, vis_config: dict, clusters_data: Optional[list] = None) -> Image.Image:
    
    all_types = sorted(list(set(obj.get('object', obj.get('name', 'unknown')) for obj in map_data)))
    adj_x, adj_y = _pad_ranges_to_aspect_ratio(x_range, y_range, width / height)

    fig = create_map_figure(
        map_data=map_data,
        robots_states=robots_states,
        title=None,
        asset_url_getter_func=get_asset_url_getter(vis_config),
        current_visible_types=None, 
        all_legend_types_sorted=all_types, 
        overall_map_x_range=adj_x,
        overall_map_y_range=adj_y,
        vis_config=vis_config,
        show_legend=False,
        events=events,
        clusters_data=clusters_data,
        is_minimap=True
    )
    
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    img_bytes = fig.to_image(format="png", width=width, height=height)
    return Image.open(io.BytesIO(img_bytes)).convert("RGBA")

def render_history_to_video(
        history_frames: list, 
        output_path: str,
        vis_config: dict,
        layout_type: str = "pip_real_main", 
        map_boundaries: tuple = ([-50, 50], [-50, 50]),
        fps: int = 10,
        width: int = 1920,
        height: int = 1080,
        target_robot_id: str = "robot_1"
):
    if not history_frames:
        logger.warning("HEADLESS UTILS | No history frames to render. Aborting video creation.")
        return

    W, H = width, height
    x_range, y_range = map_boundaries
    scale = max(0.5, H / 1080.0) 
    font_large = _get_cyrillic_font(int(52 * scale))  

    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', format='FFMPEG', macro_block_size=2)
    logger.info("HEADLESS UTILS | Starting video rendering (%d frames) | Layout: %s | Resolution: %dx%d @ %dFPS", len(history_frames), layout_type, W, H, fps)
    
    for frame in tqdm(history_frames, desc="Рендеринг", unit="тик"):
        tick = frame['tick']
        events = frame.get('events', [])
        real_data = frame['real_map']
        robots_state = frame.get('robots_state', {})
        robots_memory = frame.get('robots_memory', {})

        if target_robot_id not in robots_state and robots_state:
            target_robot_id = list(robots_state.keys())[0]

        target_robot_data = robots_state.get(target_robot_id, {})
        target_memory_data = robots_memory.get(target_robot_id, [])
        mem_robots_state = {target_robot_id: target_robot_data} if target_robot_data else {}
        clusters_data = target_robot_data.get("clusters", None)

        virtual_weather = []
        source_mem = target_memory_data or (list(robots_memory.values())[0] if robots_memory else [])
        for m in source_mem:
            if m.get("is_virtual") and (m.get("render_only") or str(m.get("notes", "")).startswith("[ENV]")):
                virtual_weather.append(m)
        if virtual_weather:
            real_data = list(real_data) + virtual_weather

        canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

        if layout_type == "real_only":
            img = _render_plotly_to_pil(real_data, robots_state, events, x_range, y_range, W, H, False, vis_config)
            canvas.paste(img, (0, 0))

        elif layout_type == "memory_only":
            img = _render_plotly_to_pil(target_memory_data, mem_robots_state, [], x_range, y_range, W, H, False, vis_config, clusters_data=clusters_data)
            canvas.paste(img, (0, 0))

        elif layout_type == "side_by_side":
            img_real = _render_plotly_to_pil(real_data, robots_state, events, x_range, y_range, W//2, H, False, vis_config)
            img_mem = _render_plotly_to_pil(target_memory_data, mem_robots_state, [], x_range, y_range, W//2, H, False, vis_config, clusters_data=clusters_data)
        
            canvas.paste(img_real, (0, 0))
            canvas.paste(img_mem, (W//2, 0))
        
            draw = ImageDraw.Draw(canvas)
            draw.line([(W//2, 0), (W//2, H)], fill="black", width=int(6 * scale))

        elif layout_type == "pip_real_main":
            img_main = _render_plotly_to_pil(real_data, robots_state, events, x_range, y_range, W, H, False, vis_config)
            canvas.paste(img_main, (0, 0))
        
            MINI_SIZE = int(450 * scale)
            rx = target_robot_data.get('x', 0)
            ry = target_robot_data.get('y', 0)
            mini_x_range = [rx - 15, rx + 15]
            mini_y_range = [ry - 15, ry + 15]
        
            img_mini = _render_plotly_to_pil(target_memory_data, mem_robots_state, [], mini_x_range, mini_y_range, MINI_SIZE, MINI_SIZE, True, vis_config, clusters_data=clusters_data)
            draw_mini = ImageDraw.Draw(img_mini)
            draw_mini.rectangle([0, 0, MINI_SIZE-1, MINI_SIZE-1], outline="gray", width=int(4 * scale))
        
            padding = int(40 * scale)
            pos = (W - MINI_SIZE - padding, H - MINI_SIZE - padding)
            canvas.paste(img_mini, pos, img_mini)

        draw = ImageDraw.Draw(canvas)
        draw.text((int(40 * scale), int(40 * scale)), f"TICK: {tick:04d}", font=font_large, fill="black")

        final_rgb = canvas.convert("RGB")
        frame_np = np.array(final_rgb)
        
        if frame_np.shape[0] % 2 != 0: frame_np = frame_np[:-1, :, :]
        if frame_np.shape[1] % 2 != 0: frame_np = frame_np[:, :-1, :]
        
        writer.append_data(frame_np)

    writer.close()
    logger.info("HEADLESS UTILS | Video rendering completed successfully.")
