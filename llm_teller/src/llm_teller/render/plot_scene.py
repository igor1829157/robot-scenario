"""Визуализация сцены из ListObject (JSON) через Plotly: hover с именем объекта.

Запуск из каталога ``llm-teller/src``::

    python -m render.plot_scene

    python -m render.plot_scene path/to/scene.json

Сохранить интерактивный HTML::

    python -m render.plot_scene -o scene.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Union

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

from llm_teller.core.schemes import ListObject, SceneObject

PathLike = Union[str, Path]

_QUAL_COLORS: list[str] = (
    list(pc.qualitative.Plotly)
    + list(pc.qualitative.Dark24)
    + list(pc.qualitative.Light24)
)


def _shape_to_xy(shape: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Преобразует плоский список координат вершин в массивы x и y.

    Ожидается чётное число элементов: пары (x, y) в локальной системе объекта.

    Args:
        shape: Список чисел вида [x0, y0, x1, y1, ...].

    Returns:
        Кортеж (xs, ys) как одномерные массивы numpy. Пустой shape даёт пустые массивы.
    """
    if not shape:
        return np.array([]), np.array([])
    arr = np.asarray(shape, dtype=float)
    if arr.size % 2 != 0:
        arr = arr[:-1]
    if arr.size < 2:
        return np.array([]), np.array([])
    xs = arr[0::2]
    ys = arr[1::2]
    return xs, ys


def _world_xy(obj: SceneObject) -> tuple[np.ndarray, np.ndarray]:
    """Локальные вершины ``shape`` в мировых координатах (с учётом ``pos``)."""
    px = float(obj.pos[0]) if len(obj.pos) > 0 else 0.0
    py = float(obj.pos[1]) if len(obj.pos) > 1 else 0.0
    lx, ly = _shape_to_xy(obj.shape)
    return lx + px, ly + py


def _is_placed_on_scene(obj: SceneObject) -> bool:
    """Объект с координатами на сцене (не только в инвентаре другого — тогда ``pos`` пустой).

    См. описание поля ``pos`` в :class:`~core.schemes.SceneObject`.
    """
    return len(obj.pos) >= 2


def _scene_xy_bounds(scene: ListObject) -> tuple[float, float, float, float]:
    """Минимальный прямоугольник по объектам, у которых задано положение ``pos`` на сцене."""
    xs: list[float] = []
    ys: list[float] = []
    for obj in scene.objects:
        if not _is_placed_on_scene(obj):
            continue
        wx, wy = _world_xy(obj)
        if wx.size == 0:
            px = float(obj.pos[0]) if len(obj.pos) > 0 else 0.0
            py = float(obj.pos[1]) if len(obj.pos) > 1 else 0.0
            xs.append(px)
            ys.append(py)
        else:
            xs.extend(float(t) for t in wx)
            ys.extend(float(t) for t in wy)
    if not xs:
        return -1.0, 1.0, -1.0, 1.0
    return min(xs), max(xs), min(ys), max(ys)


def _pad_limits(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    *,
    pad_ratio: float = 0.06,
) -> tuple[float, float, float, float]:
    """Расширяет пределы на долю размаха (не схлопывается при нулевом размахе)."""
    dx = xmax - xmin
    dy = ymax - ymin
    if dx < 1e-9:
        dx = 1.0
    if dy < 1e-9:
        dy = 1.0
    px = dx * pad_ratio
    py = dy * pad_ratio
    return xmin - px, xmax + px, ymin - py, ymax + py


def load_list_object_from_json(source: PathLike | dict[str, Any]) -> ListObject:
    """Загружает :class:`ListObject` из пути к .json или из уже распарсенного dict.

    Args:
        source: Путь к файлу JSON или словарь с ключом ``objects`` (как у модели).

    Returns:
        Валидированная модель :class:`ListObject`.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = source
    return ListObject.model_validate(data)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Строка ``rgba(...)`` для Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _colors_for_index(i: int) -> tuple[str, str]:
    """Заливка с прозрачностью и цвет линии."""
    hex_c = _QUAL_COLORS[i % len(_QUAL_COLORS)]
    return _hex_to_rgba(hex_c, 0.5), f"#{hex_c.lstrip('#')}"


def _add_object_trace(fig: go.Figure, obj: SceneObject, i: int) -> None:
    """Добавляет один trace с hover-текстом (имя объекта)."""
    px = float(obj.pos[0]) if len(obj.pos) > 0 else 0.0
    py = float(obj.pos[1]) if len(obj.pos) > 1 else 0.0
    wx, wy = _world_xy(obj)
    fill_c, line_c = _colors_for_index(i)
    name = obj.object
    hover = f"<b>{name}</b>"

    if wx.size == 0:
        fig.add_trace(
            go.Scatter(
                x=[px],
                y=[py],
                mode="markers",
                marker=dict(size=11, color=line_c, line=dict(width=1, color="black")),
                name=name,
                hovertemplate=hover + "<extra></extra>",
                showlegend=True,
            )
        )
        return

    if wx.size == 1:
        fig.add_trace(
            go.Scatter(
                x=[float(wx[0])],
                y=[float(wy[0])],
                mode="markers",
                marker=dict(size=11, color=line_c, line=dict(width=1, color="black")),
                name=name,
                hovertemplate=hover + "<extra></extra>",
                showlegend=True,
            )
        )
        return

    if wx.size == 2:
        fig.add_trace(
            go.Scatter(
                x=wx.tolist(),
                y=wy.tolist(),
                mode="lines",
                line=dict(width=3, color=line_c),
                name=name,
                hovertemplate=hover + "<extra></extra>",
                showlegend=True,
            )
        )
        return

    x_closed = np.append(wx, wx[0]).tolist()
    y_closed = np.append(wy, wy[0]).tolist()
    fig.add_trace(
        go.Scatter(
            x=x_closed,
            y=y_closed,
            fill="toself",
            mode="lines",
            fillcolor=fill_c,
            line=dict(width=1, color="black"),
            name=name,
            hoveron="fills",
            hovertemplate=hover + "<extra></extra>",
            showlegend=True,
        )
    )


def figure_list_object(
    data: ListObject | PathLike | dict[str, Any],
    *,
    title: str | None = "Сцена (ListObject)",
) -> go.Figure:
    """Строит фигуру Plotly: наведение на полигон/точку/линию показывает имя объекта.

    Объекты без пары координат в ``pos`` (предметы только во вложенных инвентарях) не
    рисуются.

    Args:
        data: :class:`ListObject`, путь к JSON или dict.
        title: Заголовок; ``None`` — без заголовка.

    Returns:
        Объект :class:`plotly.graph_objects.Figure`.
    """
    if isinstance(data, ListObject):
        scene = data
    else:
        scene = load_list_object_from_json(data)

    fig = go.Figure()
    drawable = [o for o in scene.objects if _is_placed_on_scene(o)]
    for i, obj in enumerate(drawable):
        _add_object_trace(fig, obj, i)

    xmin, xmax, ymin, ymax = _scene_xy_bounds(scene)
    xmin, xmax, ymin, ymax = _pad_limits(xmin, xmax, ymin, ymax)

    layout_kwargs: dict[str, Any] = dict(
        xaxis_title="X",
        yaxis_title="Y",
        xaxis=dict(range=[xmin, xmax], showgrid=True, zeroline=False),
        yaxis=dict(
            range=[ymin, ymax],
            showgrid=True,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        hovermode="closest",
        dragmode="pan",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(l=50, r=160, t=60 if title else 40, b=50),
    )
    if title:
        layout_kwargs["title"] = dict(text=title)

    fig.update_layout(**layout_kwargs)
    return fig


def open_interactive(
    json_path: Path,
    *,
    title: str | None,
    write_html: Path | None,
) -> int:
    """Загружает JSON, показывает интерактивную сцену или пишет HTML.

    Args:
        json_path: Путь к ``.json``.
        title: Заголовок окна/графика.
        write_html: Если задан — сохранить интерактивный HTML; иначе открыть браузер.

    Returns:
        Код выхода (0 при успехе).
    """
    path = json_path.resolve()
    if not path.is_file():
        print(f"Файл не найден: {path}", file=sys.stderr)
        return 1

    fig = figure_list_object(path, title=title)
    if write_html is not None:
        out = write_html.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out), include_plotlyjs="cdn", config={"scrollZoom": True})
        print(f"Сохранено: {out}")
    else:
        fig.show(config={"scrollZoom": True})
    return 0


def _default_dummy_debug_json() -> Path:
    """Путь к JSON по умолчанию: ``llm_teller/dummy-debug/objects.json``.

    Файл лежит рядом с корнем пакета, поэтому корректнее искать его по
    родительским директориям, а не полагаться на фиксированное число `parents`.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "dummy-debug" / "objects.json"
        if candidate.is_file():
            return candidate

    # Фоллбек на случай, если структура проекта изменится.
    return current.parents[3] / "dummy-debug" / "objects.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Парсит аргументы CLI для точки входа ``__main__``."""
    parser = argparse.ArgumentParser(
        description="Сцена ListObject в Plotly (hover — имя объекта).",
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        default=_default_dummy_debug_json(),
        help="Путь к .json со схемой ListObject (по умолчанию dummy-debug/objects.json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="FILE.html",
        help="Сохранить интерактивный HTML (без -o откроется браузер).",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Не подставлять заголовок.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа: загрузка JSON и показ сцены.

    Args:
        argv: Список аргументов (без имени программы). ``None`` — из ``sys.argv``.

    Returns:
        Код выхода процесса (0 при успехе).
    """
    args = _parse_args(argv)
    path = args.json_path.resolve()
    title = None if args.no_title else f"Сцена: {path.name}"
    return open_interactive(path, title=title, write_html=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
