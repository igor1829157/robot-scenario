import urllib.request
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

# ID сессии, которую мы создали
session_id = "sess_9fa955"
url = f"http://127.0.0.1:8000/api/v1/sessions/{session_id}/state"

# Получаем состояние
data = json.loads(urllib.request.urlopen(url).read())

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect("equal")
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.grid(True, alpha=0.3)
ax.set_title(f"Мир Robomap | Сессия: {session_id}")

# Рисуем объекты (здания, деревья и т.д.)
for obj in data.get("real_map", []):
    x, y = obj.get("x", 0), obj.get("y", 0)
    name = obj.get("name", obj.get("object", "?"))

    # Рисуем кружок для объекта
    circle = mpatches.Circle((x, y), radius=4, color="gray", alpha=0.5)
    ax.add_patch(circle)
    ax.text(x, y, name, ha="center", va="center", fontsize=6, color="black")

# Рисуем робота
robots = data.get("robots_state", {})
if "robot_1" in robots:
    r = robots["robot_1"]
    rx, ry = r["x"], r["y"]
    angle = r.get("angle", 0)

    # Кружок робота
    robot_patch = mpatches.Circle((rx, ry), radius=3, color="blue", zorder=10)
    ax.add_patch(robot_patch)

    # Стрелка направления
    dx = 8 * math.cos(math.radians(angle))
    dy = 8 * math.sin(math.radians(angle))
    ax.arrow(rx, ry, dx, dy, head_width=3, head_length=3, fc="red", ec="red", zorder=11)

plt.show()