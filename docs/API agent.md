# API для агента
Автор: Владислав Латалин


**Endpoint:** POST /api/v1/sessions/{session_id}/command
Отправляет действие в очередь команд для определенной сущности.

Поддерживает как именованные аргументы (kwargs), так и позиционные в виде массива (args).
**Приоритет:** Если параметр передан в kwargs, система возьмет его. Если в kwargs его нет, система попытается взять его по индексу из args.

Request body schema:
{
  "token": "string",
  "entity_id": "string",
  "action": "string",
  "args": \[\],
  "kwargs": {}
}

token - зарегистрированный токен для сущности (token="admin" может управлять всеми сущностями)
entity_id - глобальное id (real_id) управляемой сущности
action - действие
args - аргументы действия
kwargs - именованные аргументы действия

## 1.  Навигация и Перемещение
### `move`
Перемещение к указанным координатам с использованием алгоритма A*.
- **target_x** (float) - kwargs\["target_x"\] или args\[0\]. Целевая координата X.
- **target_y** (float) - kwargs\["target_y"\] или args\[1\]. Целевая координата Y.
- **queue** (bool) - kwargs\["queue"\] или args\[2\]. Если false, сбрасывает текущую очередь и немедленно начинает новое движение. По умолчанию: true.
- **nav_mode** (str) - **\[только kwargs\]** kwargs\["nav_mode"\]. Режим навигации и интерпретации карты планировщиком.
	- "hybrid" (по умолчанию) - Планировщик строит маршрут, обходя и реальные объекты из памяти, и виртуальные преграды (с атрибутом is_obstacle=true). Обычные метки игнорируются.
	- "memory_only" - Планировщик учитывает только физические объекты. Все виртуальные препятствия будут полностью проигнорированы (робот пойдет сквозь них).
	- "notes_only" - Планировщик игнорирует физическую память и строит маршрут исключительно вокруг виртуальных преград (is_obstacle=true). Если робот физически врежется в реальную стену, сработает аварийная остановка.
```JSON
{
  "action": "move",
  "kwargs": { "target_x": 15.0, "target_y": -10.0, "nav_mode": "hybrid" }
}
```
### `rotate`
Поворачивает самого робота.
- **target** (float | str) - kwargs\["target"\] или args\[0\]. Цель поворота (угол в градусах или mem_id объекта).
- **mode** (str) — kwargs\["mode"\] или args\[1\].
	- "relative" (по умолчанию) - Повернуться на указанный угол относительно текущего положения. (Например, target: 90 повернет робота на 90 градусов от того, куда он смотрит сейчас).
	- "absolute" - Повернуться к указанному глобальному градусу.
	- "object" - Повернуться лицом к объекту. В поле target передается строковый mem_id объекта из памяти робота. Робот сам высчитает нужный угол и повернется к нему.
```JSON
{
  "action": "rotate",
  "kwargs": { "target": "tree_5", "mode": "object" }
}
```
```JSON
{
  "action": "rotate",
  "kwargs": { "target": 45.0, "mode": "relative" }
}
```
### `clear`
Мгновенно очищает очередь действий робота.
- Не принимает аргументов.

## 2.  Взаимодействие с объектами
### `inspect`
Заглянуть внутрь инвентаря. Узнать что лежит в объекте.
- **container_path** (list\[str\]) - kwargs\["container_path"\] или args\[0\]. Путь к контейнеру.
```JSON
{
  "action": "inspect",
  "kwargs": { "container_path": ["box_big", "box_small"] }
}
```
### `pickup`
Поднять предмет с земли или из контейнера в свой инвентарь.
-  **target_item_id** (str) - kwargs\["target_item_id"\] или args\[0\]. id предмета, который нужно взять (mem_id).
- **container_path** (list\[str\]) - kwargs\["container_path"\] или args\[1\]. Путь к контейнеру.
```JSON
{
  "action": "pickup",
  "kwargs": { "target_item_id": "apple_1", "container_path": ["box_big", "box_small"] }
}
```
```JSON
{
  "action": "pickup",
  "kwargs": { "target_item_id": "apple_1", "container_path": [] }
}
```
### `put`
Положить предмет из инвентаря.
- **item_id** (str) - kwargs\["item_id"\] или args\[0\]. id предмета из инвентаря робота.
- **target_container_path** (list\[str\]) - kwargs\["target_container_path"\] или args\[1\]. Путь к контейнеру. Если \[\] - бросить на землю/пол если в здании.
```JSON
{
  "action": "put",
  "kwargs": { "target_item_id": "apple_1", "container_path": ["box_big", "box_small"] }
}
```
```JSON
{
  "action": "put",
  "kwargs": { "target_item_id": "apple_1", "container_path": [] }
}
```
### `interact_rotate`
Повернуть другой физический объект. (Лучше пока не использовать, добавил чтобы потом доработать)
- **target_id** (str) - kwargs\["target_id"\] или args\[0\]. id целевого объекта.
- **angle_offset** (float) - kwargs\["angle_offset"\] или args\[1\]. На сколько градусов повернуть (по умолчанию 90.0).
### `break`
Нанести урон объекту. Отнимает от durability целевого объекта значение равное damage. 
- **target_id** (str) - kwargs\["target_id"\] или args\[0\]. id объекта для атаки.
### `enter`
Войти внутрь объекта.
- **target_id** (str) - kwargs\["target_id"\] или args\[0\]. id объекта в который нужно войти.
### `exit`
Выйти из текущего объекта на уровень выше.
- Не принимает аргументов.
### `speak`
Фоновое действие (не блокирует очередь действий). Произнести фразу.
- **message** (str) - kwargs\["message"\] или args\[0\]. Текст фразы.
- **duration** (int) - kwargs\["duration"\] или args\[1\]. Сколько тиков отображать фразу (по умолчанию 3).
## 3. Управление памятью и виртуальными заметками на карте памяти
### `annotate`
Добавляет или перезаписывает текстовое примечание к любому объекту.
- **mem_id** (str) - kwargs\["mem_id"\] или args\[0\]. id объекта в памяти робота.
- **notes** (str) — kwargs\["notes"\] или args\[1\]. Текст примечания.
```JSON
{
  "action": "annotate",
  "kwargs": { "mem_id": "door_1", "notes": "Заперто, нужен ржавый ключ." }
}
```
### `add_virtual`
Создает виртуальный объект на карте памяти, как заметку с ней нельзя взаимодействовать командами из 2.
- **obj_type** (str) - kwargs\["obj_type"\] или args\[0\]. Тип (напр. "лед", "сугроб", "дом".
- **x** (float) - kwargs\["x"\] или args\[1\]. Координата X.
- **y** (float) — kwargs\["y"\] или args\[2\]. Координата Y.
- **is_obstacle** (bool) - kwargs\["is_obstacle"\] или args\[3\]. true = планировщик объезжает зону. false = только метка. По умолчанию false.
- **radius** (float) - kwargs\["radius"\] или args\[4\]. Радиус зоны. По умолчанию 0.5.
- **notes** (str) - kwargs\["notes"\] или args\[5\]. Текст примечания.
- **parent_id** (str) - kwargs\["parent_id"\] или args\[6\]. Использовать, если нужно положить метку внутрь контейнера.
- **inventory_data** (list\[dict\]) - kwargs\["inventory_data"\] или args\[7\]. Список словарей для создания вложенных виртуальных предметов.
- **shape** (list\[list\[float\]\]) - \[Только kwargs\] kwargs\["shape"\]. Относительный полигон, напр. \[\[-2,-2\], \[2,-2\], \[2,2\], \[-2,2\]\]. Перебивает radius.
```JSON
{
  "action": "add_virtual",
  "kwargs": {
    "obj_type": "radiation_zone",
    "x": 20.0, "y": -5.0,
    "is_obstacle": true,
    "shape": [[-2, -2], [2, -2], [2, 2], [-2, 2]],
    "notes": "Опасная зона! Запрещено для входа."
  }
}
```
### `update_virtual`
Изменяет параметры существующего виртуального объекта.
- **mem_id** (str) - kwargs\["mem_id"\] или args\[0\]. id виртуального объекта.
- **updates** (dict) - kwargs\["updates"\] или args\[1\]. Словарь ключей для обновления
### `delete_virtual`
Удаляет виртуальный объект из памяти.
- **mem_id** (str) - kwargs\["mem_id"\] или args\[0\]. id виртуального объекта.
## 5. Посмотреть логи
**Endpoint:** GET /api/v1/sessions/{session_id}/logs
Возвращает логи робота с поддержкой инкрементального скачивания и кумулятивной фильтрации.

**Parameters**
token - токен
robot_id - id робота
log_type - Категории логов ('all' - все, 'self' - только свои действия, 'others' - наблюдения и слух).
format_style - Формат вывода ('json' - json, 'text' - только сообщения, 'brief' - краткий лог).
tick_start - Начать вывод логов с указанного тика симуляции.

**Результат для json**
```json
{
      "tick": 12,
      "robot_id": "robot_1",
      "category": "PERCEPTION",
      "action_type": "vision",
      "message": "Object human 'mem_human_304' is sighted at (69.42, -37.52).",
      "status": "NEW",
      "args": {
        "observed_id": "mem_human_304",
        "observed_type": "human",
        "observed_x": 69.42,
        "observed_y": -37.52
      }
    },
    {
      "tick": 12,
      "robot_id": "robot_1",
      "category": "OBSERVATION",
      "action_type": "move",
      "message": "Object human 'mem_human_304' started moving.",
      "status": "OBSERVATION",
      "args": {
        "observed_id": "mem_human_304",
        "observed_type": "human",
        "observed_x": 69.42,
        "observed_y": -37.52
      }
    },
    {
      "tick": 12,
      "robot_id": "robot_1",
      "category": "PERCEPTION",
      "action_type": "inventory",
      "message": "Object shovel 'mem_shovel_792' is now inside human 'mem_human_304'.",
      "status": "CHANGED",
      "args": {
        "observed_id": "mem_shovel_792",
        "observed_type": "shovel",
        "observed_x": null,
        "observed_y": null
      }
    },
    {
      "tick": 13,
      "robot_id": "robot_1",
      "category": "PERCEPTION",
      "action_type": "vision",
      "message": "Object human 'mem_human_304' moved from (69.42, -37.52) to (69.65, -38.5).",
      "status": "CHANGED",
      "args": {
        "observed_id": "mem_human_304",
        "observed_type": "human",
        "observed_x": 69.65,
        "observed_y": -38.5
      }
    },
    {
      "tick": 14,
      "robot_id": "robot_1",
      "category": "PERCEPTION",
      "action_type": "vision",
      "message": "Object human 'mem_human_304' moved from (69.65, -38.5) to (70.0, -40.0).",
      "status": "CHANGED",
      "args": {
        "observed_id": "mem_human_304",
        "observed_type": "human",
        "observed_x": 70,
        "observed_y": -40
      }
    },
    {
      "tick": 15,
      "robot_id": "robot_1",
      "category": "OBSERVATION",
      "action_type": "move",
      "message": "Object human 'mem_human_304' finished moving.",
      "status": "OBSERVATION",
      "args": {
        "observed_id": "mem_human_304",
        "observed_type": "human",
        "observed_x": 70,
        "observed_y": -40
      }
    },
    {
      "tick": 15,
      "robot_id": "robot_1",
      "category": "OBSERVATION",
      "action_type": "put",
      "message": "Object human 'mem_human_304' is placing object 'mem_shovel_792'.",
      "status": "OBSERVATION",
      "args": {
        "observed_id": "mem_human_304",
        "observed_type": "human",
        "observed_x": 70,
        "observed_y": -40
      }
    },
    {
      "tick": 16,
      "robot_id": "robot_1",
      "category": "OBSERVATION",
      "action_type": "put",
      "message": "Object human 'mem_human_304' dropped object 'mem_shovel_792'.",
      "status": "OBSERVATION",
      "args": {
        "observed_id": "mem_human_304",
        "observed_type": "human",
        "observed_x": 70,
        "observed_y": -40
      }
    }
  ]
}
```
## 6. API к памяти и карте
**Endpoint:** /api/v1/sessions/{session_id}/local-context
Генерирует контекст памяти для агента.

**Parameters**
token - токен
robot_id - id робота
format_style - формат вывода ('json', 'text-md', 'text-yaml', 'text-narrative').
sections - список секций через запятую ('all', 'current_state', 'visible_now', 'inventory'). По умолчанию 'all'.

**Результат для text-narrative**
```
[CURRENT STATE]
It is currently tick 931. You are located outside at coordinates (63.0, -38.0), facing 0.0 degrees.

[VISIBLE NOW]
You currently see 3 objects:
- A closet (ID: 10) - Массивный платяной шкаф на улице у забора. located outside at (63.5, -37.2). Distance: 0.94m. It contains:
  - A backpack (ID: mem_backpack_2f6) - Плотный брезентовый рюкзак. located inside 10 (parent coordinates: 63.5, -37.2).
- A human (ID: mem_human_304) - Человек в рабочей одежде, оглядывается. located outside at (70.0, -40.0). Distance: 7.28m.
- A shovel (ID: mem_shovel_792) - Обычная металлическая совковая лопата. located inside mem_human_304 at (71.11, -40.0). Distance: 8.35m.

[MEMORY]
You remember 12 objects from the past:
- A house (ID: 1) - Деревянный дом с зеленой крышей и небольшим крыльцом. located outside at (60.0, -35.0). Last seen at tick 0.
- A tree (ID: 2) - Дерево с дуплом на уровне глаз. located outside at (54.0, -31.0). Last seen at tick 0.
- A tree (ID: 3) - Высокий тополь. located outside at (60.0, -22.0). Last seen at tick 0.
- A car (ID: 4) - Красный легковой автомобиль, припаркован. located outside at (65.4, -33.8). Last seen at tick 0.
- A rock (ID: 5) - Острый гранитный камень, наполовину ушедший в землю. located outside at (56.0, -46.0). Last seen at tick 0.
- A broken_tree (ID: 6) - Поваленное дерево, ствол переломлен пополам. located outside at (58.0, -47.0). Last seen at tick 0.
- A tree (ID: 7) - Небольшое дерево с редкими ветвями. located outside at (69.0, -47.0). Last seen at tick 0.
- A tree (ID: 8) - Мощный ясень с толстым стволом. located outside at (46.0, -35.0). Last seen at tick 0.
- A tree (ID: 9) - Раскидистая ива. located outside at (39.0, -34.0). Last seen at tick 0.
- A closet (ID: 10) - Массивный платяной шкаф на улице у забора. located outside at (63.5, -37.2). Last seen at tick 931. It contains:
  - A backpack (ID: mem_backpack_2f6) - Плотный брезентовый рюкзак. located inside 10 (parent coordinates: 63.5, -37.2). Last seen at tick 931.
- A human (ID: mem_human_304) - Человек в рабочей одежде, оглядывается. located outside at (70.0, -40.0). Last seen at tick 931.
- A shovel (ID: mem_shovel_792) - Обычная металлическая совковая лопата. located inside mem_human_304 at (71.11, -40.0). Last seen at tick 931.
```