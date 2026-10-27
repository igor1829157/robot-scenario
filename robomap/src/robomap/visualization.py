import os
from typing import Callable, List, Dict, Any, Optional, Tuple
from collections import defaultdict

import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull

import zlib
import colorsys
import textwrap

import logging
logger = logging.getLogger(__name__)

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def _discover_icons_in_assets(vis_config: Dict[str, Any]) -> Dict[str, str]:
    discovered_map = {}
    icons_dir = vis_config.get("icon_path", "data/assets/icons")
    base_dir = vis_config.get("asset_base_path", "data/assets")
    
    icons_dir = os.path.join(project_path, icons_dir) if not os.path.isabs(icons_dir) else icons_dir
    base_dir = os.path.join(project_path, base_dir) if not os.path.isabs(base_dir) else base_dir

    if not os.path.isdir(icons_dir):
        return {}

    try:
        rel_dir = os.path.relpath(icons_dir, base_dir)
    except ValueError:
        rel_dir = "icons" 

    for filename in os.listdir(icons_dir):
        if filename.lower().endswith(('.svg', '.png', '.jpg', '.jpeg')):
            name_key = os.path.splitext(filename)[0]
            config_style_path = f"{rel_dir}/{filename}" if rel_dir != "." else filename
            discovered_map[name_key] = config_style_path

    return discovered_map

def get_draw_order_weight(obj_name: str, vis_config: Dict[str, Any]) -> int:
    draw_order = vis_config.get("object_draw_order", [])
    try:
        return draw_order.index(obj_name)
    except ValueError:
        return len(draw_order)

MODIFIER_COLORS = {
    "speed": "#E67E22",                # Оранжевый - Скорость
    "vision_radius": "#9B59B6",         # Фиолетовый - Видимость
    "vision": "#9B59B6",
    "drift": "#1ABC9C",                # Бирюзовый - Дрейф / Ветер
    "position": "#1ABC9C",
    "speed_vision": "#E74C3C",         # Красный - Скорость + Видимость
    "speed_drift": "#2980B9",          # Синий - Скорость + Дрейф
    "speed_vision_drift": "#8E44AD",   # Темно-бордовый - Комплексный шторм
}

def get_object_color(obj_type: str, vis_config: Dict[str, Any]) -> str:
    colors_dict = vis_config.get("object_colors", {})
    
    if obj_type in colors_dict:
        return colors_dict[obj_type]

    # Семантические цвета погодных модификаторов
    # 1. Скорость: 1, 2, 4, 5, 6-ridge, 11, 13-ridge
    if any(obj_type.startswith(p) for p in ["p1_", "p2_", "p4_", "p5_", "p11_"]) or any(k in obj_type for k in ["frozen_stream", "sastrugi", "channel", "ridge", "muskeg"]):
        return MODIFIER_COLORS["speed"]
    # 2. Видимость: 7, 8, 9
    if any(obj_type.startswith(p) for p in ["p7_", "p8_", "p9_"]) or any(k in obj_type for k in ["whiteout", "fog", "twilight", "diamond_dust"]):
        return MODIFIER_COLORS["vision_radius"]
    # 3. Дрейф / Ветер: 3, 10, 12, 13-creep
    if any(obj_type.startswith(p) for p in ["p3_", "p10_", "p12_"]) or any(k in obj_type for k in ["drift", "gorge", "crack", "creep", "surge"]):
        return MODIFIER_COLORS["drift"]
    # 4. Комплексные штормы / шквалы: 14, squall, blizzard
    if obj_type.startswith("p14_") or any(k in obj_type for k in ["squall", "blizzard"]):
        return MODIFIER_COLORS["speed_vision"]
        
    if "_procedural_cache" not in vis_config:
        vis_config["_procedural_cache"] = {}

    if obj_type in vis_config["_procedural_cache"]:
        return vis_config["_procedural_cache"][obj_type]
    
    occupied_colors = list(colors_dict.values()) + list(vis_config["_procedural_cache"].values())
    generated_color = _generate_procedural_color(obj_type, occupied_colors)
    vis_config["_procedural_cache"][obj_type] = generated_color
    
    return generated_color

def _build_hover_text(obj_dict: Dict[str, Any]) -> str:
    html_parts = []
    
    is_virtual = obj_dict.get('is_virtual', False)
    is_obstacle = obj_dict.get('is_obstacle', False)
    
    if is_virtual:
        if is_obstacle:
            html_parts.append("<span style='color:#ffaaaa;'><b>[VIRTUAL OBSTACLE]</b></span>")
        elif obj_dict.get('render_only', False):
            c = obj_dict.get('color') or obj_dict.get('state', {}).get('color', '#aaaaff')
            mod_lbl = obj_dict.get('state', {}).get('modifier_label', 'Среда')
            html_parts.append(f"<span style='color:{c};'><b>[ЗОНА: {mod_lbl.upper()}]</b></span>")
        else:
            html_parts.append("<span style='color:#aaffaa;'><b>[VIRTUAL MARKER]</b></span>")
            
    base_id_str = f"<b>ID:</b> {obj_dict.get('id', 'N/A')}"
    if 'real_id' in obj_dict and obj_dict['real_id'] and not is_virtual:
        base_id_str += f" <i>(Real: {obj_dict['real_id']})</i>"
    html_parts.append(base_id_str)
    
    base_x = float(obj_dict.get('x', 0.0)) if obj_dict.get('x') is not None else 0.0
    base_y = float(obj_dict.get('y', 0.0)) if obj_dict.get('y') is not None else 0.0
    
    html_parts.append(f"<b>Type:</b> {obj_dict.get('object', obj_dict.get('name', 'N/A'))}")
    html_parts.append(f"<b>X:</b> {base_x:.2f} | <b>Y:</b> {base_y:.2f}")
    
    notes = obj_dict.get('notes', "")
    if notes:
        wrapped_notes = textwrap.fill(str(notes), width=45).replace('\n', '<br>')
        html_parts.append(f"<br><b>Notes:</b><br><i>{wrapped_notes}</i><br>")
        
    reserved_keys = {
        'id', 'real_id', 'object', 'name', 'x', 'y', 'timestamp', 
        'shape', 'absolute_shape', 'absolute_wkt', 'original_x', 'original_y',
        'is_virtual', 'is_obstacle', 'render_only', 'notes', 'creation_tick',
        'inventory', 'action_queue', 'planned_path', 'visible_objects', 
        'clusters', 'feedback'
    }
    
    for k, v in obj_dict.items():
        if k not in reserved_keys:
            if v is None or (isinstance(v, list) and len(v) == 0) or (isinstance(v, dict) and len(v) == 0): 
                continue
            if isinstance(v, float) and np.isnan(v): 
                continue
                
            if isinstance(v, dict):
                dict_str = ", ".join([f"{dk}: {dv}" for dk, dv in v.items()])
                wrapped_dict = textwrap.fill(dict_str, width=45).replace('\n', '<br>')
                html_parts.append(f"<b>{str(k).capitalize()}:</b> {wrapped_dict}")
            elif isinstance(v, str):
                wrapped_v = textwrap.fill(v, width=45).replace('\n', '<br>')
                html_parts.append(f"<b>{str(k).capitalize()}:</b> {wrapped_v}")
            else:
                html_parts.append(f"<b>{str(k).capitalize()}:</b> {v}")

    inv = obj_dict.get('inventory', [])
    if inv:
        inv_tree = _build_inventory_tree_html(inv)
        if inv_tree:
            html_parts.append(f"<br><b>Inventory:</b><br>{inv_tree}")
            
    return "<br>".join(html_parts)

def _build_inventory_tree_html(inventory_list: List[Dict[str, Any]], indent_level: int = 1) -> str:
    if not inventory_list:
        return ""
        
    html_parts = []
    counts = {}
    for item in inventory_list:
        obj_name = item.get('object', item.get('name', 'unknown'))
        item_id = item.get('id', '')
        
        sub_inv = item.get('inventory', [])
        sub_html = _build_inventory_tree_html(sub_inv, indent_level + 1)
        
        key = (obj_name, sub_html)
        if key not in counts:
            counts[key] = {'count': 0, 'ids': []}
        counts[key]['count'] += 1
        counts[key]['ids'].append(item_id)
        
    indent = "&nbsp;" * (indent_level * 4)
    
    for (obj_name, sub_html), data in counts.items():
        count = data['count']
        count_str = f" <i>(x{count})</i>" if count > 1 else ""
        
        id_str = f" <span style='font-size: 10px; color: #ddd;'>[{data['ids'][0]}]</span>" if count == 1 else ""
        
        html_parts.append(f"{indent}- {obj_name}{count_str}{id_str}")
        if sub_html:
            html_parts.append(sub_html)
            
    return "<br>".join(html_parts)


def _get_cluster_color(cluster_id: int, alpha: float = 0.2) -> str:

    ratio = 0.618033988749895
    h = (cluster_id * ratio) % 1.0
    s, l = 0.85, 0.50
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})"


def _add_cluster_traces(fig: go.Figure, clusters_data: List[Dict[str, Any]], map_data: List[Dict[str, Any]], visible_cluster_names: Optional[List[str]]) -> None:
    if not clusters_data:
        return

    mem_lookup = {}
    def _build_lookup(obj_list):
        for obj in obj_list:
            mem_lookup[str(obj.get('id'))] = obj
            if 'inventory' in obj and isinstance(obj['inventory'], list):
                _build_lookup(obj['inventory'])
                
    _build_lookup(map_data)

    for i, cluster in enumerate(clusters_data):
        try:
            points_list = []
            object_ids_in_cluster = cluster.get('object_ids', [])
            
            for obj_id in object_ids_in_cluster:
                obj = mem_lookup.get(str(obj_id))
                if obj:
                    x_val = obj.get('x')
                    y_val = obj.get('y')
                    if x_val is not None and y_val is not None:
                        points_list.append([float(x_val), float(y_val)])

            if len(points_list) < 2: 
                continue

            points = np.array(points_list)
            cluster_id = cluster.get('id', i)
            color = _get_cluster_color(cluster_id, alpha=0.18)
            line_color = _get_cluster_color(cluster_id, alpha=0.60)

            cluster_name = cluster.get('name', f"Cluster {cluster_id}")
            trace_visible_property = True if (visible_cluster_names is None or cluster_name in visible_cluster_names) else 'legendonly'

            hover_text = (
                f"<b>{cluster_name}</b><br>"
                f"Objects: {cluster.get('num_objects', 0)}<br>"
                f"Types: {', '.join(cluster.get('object_types', []))}"
            )

            unique_points = np.unique(points, axis=0)
            if len(unique_points) < 3:
                min_idx, max_idx = np.argmin(unique_points[:, 0]), np.argmax(unique_points[:, 0])
                if unique_points[min_idx, 0] == unique_points[max_idx, 0]:
                    min_idx, max_idx = np.argmin(unique_points[:, 1]), np.argmax(unique_points[:, 1])

                line_points = np.array([unique_points[min_idx], unique_points[max_idx]])
                x_coords = [float(x) for x in line_points[:, 0]]
                y_coords = [float(y) for y in line_points[:, 1]]

                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords, mode='lines',
                    line=dict(color=line_color, width=3, dash='dot'),
                    hoverinfo='text', text=hover_text, name=cluster_name,
                    showlegend=True, visible=trace_visible_property
                ))
                continue

            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                
                x_raw = np.append(hull_points[:, 0], hull_points[0, 0])
                y_raw = np.append(hull_points[:, 1], hull_points[0, 1])
                x_coords = [float(x) for x in x_raw]
                y_coords = [float(y) for y in y_raw]

                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines', fill='toself', fillcolor=color,
                    line=dict(color=line_color, width=1.5, dash='dot'),
                    hoverinfo='text', text=hover_text, name=cluster_name,
                    showlegend=True, visible=trace_visible_property
                ))
            except Exception:
                min_idx, max_idx = np.argmin(points[:, 0]), np.argmax(points[:, 0])
                if points[min_idx, 0] == points[max_idx, 0]:
                    min_idx, max_idx = np.argmin(points[:, 1]), np.argmax(points[:, 1])

                line_points = np.array([points[min_idx], points[max_idx]])
                x_coords = [float(x) for x in line_points[:, 0]]
                y_coords = [float(y) for y in line_points[:, 1]]

                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords, mode='lines',
                    line=dict(color=line_color, width=3, dash='dot'),
                    hoverinfo='text', text=hover_text, name=cluster_name,
                    showlegend=True, visible=trace_visible_property
                ))

        except Exception:
            pass

def create_map_figure(
    map_data: List[Dict[str, Any]], 
    robots_states: Dict[str, Dict[str, Any]],
    title: str, 
    asset_url_getter_func: Callable[[str], Optional[str]], 
    current_visible_types: Optional[List[str]], 
    all_legend_types_sorted: List[str],
    overall_map_x_range: Tuple[float, float], 
    overall_map_y_range: Tuple[float, float], 
    vis_config: Dict[str, Any],
    show_legend: bool = True,
    clusters_data: Optional[List[Dict[str, Any]]] = None, 
    visible_cluster_names: Optional[List[str]] = None,
    events: Optional[List[str]] = None,
    is_minimap: bool = False
) -> go.Figure:
    
    fig = go.Figure()

    icon_mapping = vis_config.get("icon_mapping", {}).copy()
    discovered_icons = _discover_icons_in_assets(vis_config)
    for name, path in discovered_icons.items():
        if name not in icon_mapping:
            icon_mapping[name] = path

    if clusters_data:
        _add_cluster_traces(fig, clusters_data, map_data, visible_cluster_names)

    grouped_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for obj in map_data:
        if obj.get('x') is not None and obj.get('y') is not None:
            obj_type = obj.get('object', obj.get('name', 'unknown'))
            grouped_data[obj_type].append(obj)

    layout_images = []
    drawable_icon_types = set()
    base_dir = vis_config.get("asset_base_path", "data/assets")
    base_dir = os.path.join(project_path, base_dir) if not os.path.isabs(base_dir) else base_dir

    for obj_type in all_legend_types_sorted:
        is_visible = True if (current_visible_types is None or obj_type in current_visible_types) else 'legendonly'
        obj_color = get_object_color(obj_type, vis_config)
        objects_of_type = grouped_data.get(obj_type, [])
        for o in objects_of_type:
            c = o.get("color") or o.get("state", {}).get("color")
            if c:
                obj_color = c
                break
        
        real_poly_x, real_poly_y, real_poly_cd = [], [], []
        virt_poly_x, virt_poly_y, virt_poly_cd = [], [], []
        
        marker_x, marker_y, marker_cd = [], [], []
        marker_symbols, marker_opacities = [], []

        for obj in objects_of_type:
            hover_html = _build_hover_text(obj)
            is_virtual = obj.get('is_virtual', False)
            is_obstacle = obj.get('is_obstacle', False)
            base_x, base_y = float(obj['x']), float(obj['y'])
            
            marker_x.append(base_x)
            marker_y.append(base_y)
            marker_cd.append([hover_html])
            
            is_virt_point_marker = is_virtual and not is_obstacle and not obj.get('render_only', False) and not obj.get('shape')
            if is_virt_point_marker:
                marker_symbols.append('x')
                marker_opacities.append(1.0)
            else:
                marker_symbols.append('circle')
                marker_opacities.append(0.01 if obj_type in icon_mapping else 1.0)

            if obj.get('planned_path'):
                path_x = [base_x] + [p[0] for p in obj['planned_path']]
                path_y = [base_y] + [p[1] for p in obj['planned_path']]
                fig.add_trace(go.Scatter(
                    x=path_x, y=path_y, mode='lines',
                    line=dict(color=obj_color, width=2, dash='dot'),
                    hoverinfo='skip', showlegend=False,
                    visible=is_visible
                ))

            coords = obj.get('absolute_shape', [])
            if not coords:
                rel_shape = obj.get('shape', [])
                if rel_shape and len(rel_shape) >= 3:
                    import math
                    obj_angle = float(obj.get('angle', 0.0))
                    rad = math.radians(-obj_angle)
                    cos_a = math.cos(rad)
                    sin_a = math.sin(rad)
                    coords = []
                    for dx, dy in rel_shape:
                        if dx is not None:
                            rot_dx = dx * cos_a - dy * sin_a
                            rot_dy = dx * sin_a + dy * cos_a
                            coords.append([base_x + rot_dx, base_y + rot_dy])

            if coords and not is_virt_point_marker:
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                    
                target_x_list = virt_poly_x if is_virtual else real_poly_x
                target_y_list = virt_poly_y if is_virtual else real_poly_y
                target_cd_list = virt_poly_cd if is_virtual else real_poly_cd
                
                for pt in coords:
                    target_x_list.append(pt[0])
                    target_y_list.append(pt[1])
                    target_cd_list.append([hover_html])
                target_x_list.append(None)
                target_y_list.append(None)
                target_cd_list.append(None)

            if obj_type in icon_mapping and is_visible is True and not is_virtual:
                icon_path = icon_mapping[obj_type]
                full_path = os.path.join(base_dir, icon_path)
                if os.path.exists(full_path):
                    resolved_source = asset_url_getter_func(icon_path)
                    if resolved_source:
                        if obj_type == "robot":
                            icon_sx = vis_config.get("robot_size_x", vis_config.get("icon_size_x", 5.0) * 0.5)
                            icon_sy = vis_config.get("robot_size_y", vis_config.get("icon_size_y", 5.0) * 0.5)
                        else:
                            icon_sx = vis_config.get("icon_size_x", 5.0)
                            icon_sy = vis_config.get("icon_size_y", 5.0)

                        layout_images.append(dict(
                            source=resolved_source, xref="x", yref="y",
                            x=base_x, y=base_y, sizex=icon_sx, sizey=icon_sy,
                            xanchor="center", yanchor=vis_config.get("icon_y_anchor", "middle"), layer="above"
                        ))
                        drawable_icon_types.add(obj_type)

        if real_poly_x and real_poly_y:
            fig.add_trace(go.Scatter(
                x=real_poly_x, y=real_poly_y, mode='lines', fill='toself',
                fillcolor=obj_color, line=dict(color=obj_color, width=1.5), opacity=0.6,
                name=f"{obj_type} shapes", legendgroup=obj_type, showlegend=False,
                customdata=real_poly_cd, hovertemplate="%{customdata[0]}<extra></extra>",
                visible=is_visible
            ))

        if virt_poly_x and virt_poly_y:
            fig.add_trace(go.Scatter(
                x=virt_poly_x, y=virt_poly_y, mode='lines', fill='toself',
                fillcolor=obj_color, line=dict(color=obj_color, width=2, dash='dot'), opacity=0.2,
                name=f"{obj_type} virtual", legendgroup=obj_type, showlegend=False,
                customdata=virt_poly_cd, hovertemplate="%{customdata[0]}<extra></extra>",
                visible=is_visible
            ))

        htemplate = "%{customdata[0]}<extra></extra>" if marker_cd else f"<b>{obj_type}</b><extra></extra>"
        
        fig.add_trace(go.Scatter(
            x=marker_x, y=marker_y, mode='markers', 
            marker=dict(color=obj_color, size=10, opacity=marker_opacities, symbol=marker_symbols), 
            name=obj_type, legendgroup=obj_type,
            customdata=marker_cd, hovertemplate=htemplate,
            visible=is_visible
        ))

    robot_color = get_object_color("robot", vis_config)
    robot_icon_path = icon_mapping.get("robot", "icons/robot.svg")
    resolved_robot_icon = asset_url_getter_func(robot_icon_path) if robot_icon_path else None

    for r_id, r_state in (robots_states or {}).items():
        r_x = r_state.get('x')
        r_y = r_state.get('y')
        if r_x is None or r_y is None: continue

        r_radius = r_state.get('radius')
        r_timestamp = str(r_state.get('timestamp', 'N/A'))
        r_path = r_state.get('planned_path', [])

        if r_path and len(r_path) > 0:
            path_x = [r_x] + [p[0] for p in r_path]
            path_y = [r_y] + [p[1] for p in r_path]
            fig.add_trace(go.Scatter(
                x=path_x, y=path_y, mode='lines+markers',
                line=dict(color=robot_color, width=3, dash='dot'),
                marker=dict(color=robot_color, size=6, symbol='circle'),
                name=f'Path ({r_id})', hoverinfo='skip', showlegend=False
            ))

        if resolved_robot_icon:
            layout_images.append(dict(
                source=resolved_robot_icon, xref="x", yref="y",
                x=r_x, y=r_y,
                sizex=vis_config.get("robot_size_x", vis_config.get("icon_size_x", 5.0) * 0.5), 
                sizey=vis_config.get("robot_size_y", vis_config.get("icon_size_y", 5.0) * 0.5),
                xanchor="center", yanchor="middle", layer="above"
            ))

        fig.add_trace(go.Scatter(
            x=[r_x], y=[r_y], mode='markers', 
            marker=dict(color=robot_color, size=12, symbol='cross', opacity=0.01 if resolved_robot_icon else 1.0), 
            name=r_id, legendgroup="robots",
            customdata=[[f"<b>ID:</b> {r_id}<br><b>Type:</b> Robot<br><b>X:</b> {r_x:.2f} | <b>Y:</b> {r_y:.2f}<br><b>Time:</b> {r_timestamp}"]],
            hovertemplate="%{customdata[0]}<extra></extra>", showlegend=show_legend
        ))
        
        vision_wkt = r_state.get('vision_wkt')
        if vision_wkt:
            import shapely.wkt
            try:
                poly = shapely.wkt.loads(vision_wkt)
                poly_x, poly_y = [], []
                
                if poly.geom_type == 'Polygon':
                    poly_x.extend([p[0] for p in poly.exterior.coords])
                    poly_y.extend([p[1] for p in poly.exterior.coords])
                elif poly.geom_type in ['MultiPolygon', 'GeometryCollection']:
                    for g in getattr(poly, 'geoms', []):
                        if g.geom_type == 'Polygon':
                            poly_x.extend([p[0] for p in g.exterior.coords])
                            poly_y.extend([p[1] for p in g.exterior.coords])
                            poly_x.append(None)
                            poly_y.append(None)

                if poly_x and poly_y:
                    fig.add_trace(go.Scatter(
                        x=poly_x, y=poly_y, mode='lines', fill='toself',
                        fillcolor=robot_color, line=dict(color=robot_color, width=1, dash='dot'),
                        opacity=0.15, hoverinfo='skip', showlegend=False
                    ))
            except Exception as e:
                pass
                
        r_angle = r_state.get('angle')
        if r_angle is not None and r_radius is not None:
            import math
            dir_length = 2.5
            rad_angle = math.radians(-r_angle)
            end_x = r_x + dir_length * math.cos(rad_angle)
            end_y = r_y + dir_length * math.sin(rad_angle)
            fig.add_trace(go.Scatter(
                x=[r_x, end_x], y=[r_y, end_y], mode='lines',
                line=dict(color='red', width=3), hoverinfo='skip', showlegend=False
            ))

    layout_annotations = []
    if events:
        for i, ev_text in enumerate(events):
            layout_annotations.append(dict(
                text=f"<b>{ev_text}</b>", x=0.5, y=0.95 - (i * 0.05),
                xref="paper", yref="paper", showarrow=False,
                font=dict(size=18, color="yellow"),
                bgcolor="rgba(0,0,0,0.6)", bordercolor="white", borderwidth=1, borderpad=4
            ))

    fig.update_layout(
        title=title if not is_minimap else None, 
        xaxis_title="X Coordinate" if not is_minimap else None, 
        yaxis_title="Y Coordinate" if not is_minimap else None,
        legend_title="Object Types" if not is_minimap else None, 
        margin=dict(l=0, r=0, t=0, b=0) if is_minimap else dict(l=20, r=20, t=40, b=20),
        images=layout_images, 
        showlegend=show_legend and not is_minimap,
        annotations=layout_annotations,
        hoverlabel=dict(align="left", font_size=13, font_family="Arial, sans-serif")
    )

    if overall_map_x_range and overall_map_y_range:
        fig.update_xaxes(range=overall_map_x_range, autorange=False, visible=not is_minimap)
        fig.update_yaxes(range=overall_map_y_range, autorange=False, scaleanchor="x", scaleratio=1, visible=not is_minimap)
    else:
        fig.update_layout(yaxis_scaleanchor="x", yaxis_scaleratio=1)
        if is_minimap:
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)

    return fig

def _hex_to_hsl(hex_str: str) -> Tuple[float, float, float]:
    """Конвертирует HEX-цвет (#RRGGBB) в HSL (значения от 0.0 до 1.0)."""
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 6:
        return 0.0, 0.5, 0.0
    try:
        r, g, b = tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return h, s, l
    except Exception:
        return 0.0, 0.5, 0.0

def _hsl_distance(h1: float, s1: float, l1: float, h2: float, s2: float, l2: float) -> float:
    """Вычисляет расстояние между двумя цветами в HSL."""
    dh = min(abs(h1 - h2), 1.0 - abs(h1 - h2))
    ds = s1 - s2
    dl = l1 - l2
    return (dh * 2.0) ** 2 + ds ** 2 + dl ** 2

def _generate_procedural_color(obj_type: str, occupied_hex_colors: List[str]) -> str:
    """
    Генерирует уникальный, цвет для типа объекта по хэшу от типа объекта, 
    минимизируя схожесть с уже занятыми цветами.
    """
    occupied_hsl = [_hex_to_hsl(c) for c in occupied_hex_colors if c]

    s_levels = [0.45, 0.70, 0.95]
    l_levels = [0.35, 0.55, 0.75]

    hash_val = zlib.adler32(obj_type.encode('utf-8'))

    h_deg = hash_val % 360
    s_idx = (hash_val // 360) % 3
    l_idx = (hash_val // (360 * 3)) % 3

    collision_threshold = 0.18
    safety_counter = 0

    while safety_counter < 120:
        h = h_deg / 360.0
        s = s_levels[s_idx]
        l = l_levels[l_idx]

        collision = False
        for oh, os, ol in occupied_hsl:
            if _hsl_distance(h, s, l, oh, os, ol) < collision_threshold:
                collision = True
                break

        if not collision:
            break

        h_deg = (h_deg + 30) % 360
        safety_counter += 1
        
        if safety_counter % 12 == 0:
            s_idx = (s_idx + 1) % 3
        if safety_counter % 36 == 0:
            l_idx = (l_idx + 1) % 3

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"