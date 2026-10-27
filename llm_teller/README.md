# llm_teller

`llm_teller` — небольшой пакет для:
1) генерации расширенного описания сцены (через LLM/`langchain`);
2) визуализации сцены в интерактивном HTML (через `plotly`).

## Установка

Вариант для разработки из исходников (editable):

```bash
pip install -e ./llm_teller
```

Зависимости можно также поставить вручную:

```bash
pip install -r llm_teller/requirements/all.txt
```

## Визуализация сцены (`llm_teller.render.plot_scene`)

Интерактивно (откроется браузер):

```bash
python -m llm_teller.render.plot_scene path/to/scene.json
```

Если путь к JSON не указан, будет использован `llm_teller/dummy-debug/objects.json`:

```bash
python -m llm_teller.render.plot_scene
```

С сохранением в HTML:

```bash
python -m llm_teller.render.plot_scene path/to/scene.json -o scene.html
```

Опции:
- `-o/--output FILE.html` — сохранить интерактивный HTML
- `--no-title` — не добавлять заголовок

## Формат JSON сцены

Визуализатор ожидает JSON в формате модели `ListObject`:

```json
{
  "objects": [
    {
      "object": "string",
      "shape": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
      "affordance": ["string"],
      "behavior": ["string"],
      "description": "string",
      "dynamic": false,
      "has_inventory": false,
      "inventory": ["string"],
      "pos": [0.0, 0.0]
    }
  ]
}
```

Поля объекта:
- `object`: имя объекта
- `shape`: плоский список координат вершин объекта в локальной системе (массив `[x0, y0, x1, y1, ...]`)
- `pos`: `[x, y]` в мировой системе, если объект размещён на сцене (для объектов “в инвентаре” может быть пустым)

## Генерация сцены (`llm_teller.core.pipelines.Morgan`)

Пример использования цепочек:

```python
from llm_teller.core.pipelines import Morgan

morgan = Morgan(
    model="gemini-2.5-flash-lite",
    provider="google_genai",
    secret="llm_teller/secrets/google.secret",
)

text = "Группа ученых расположилась в заснеженной тундре для исследований..."

scene = morgan.scene_chain.invoke({"short_scenario": text})
objects = morgan.description_chain.invoke({"objects": scene})
print(objects)
```

Ключ для LLM читается из переданного файла секрета (путь указывается в `secret`).

