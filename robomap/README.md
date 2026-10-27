# robomap

`robomap` — пакет для:
1) симуляции 2D-мира (через встроенный физический движок и планировщик пути A*);
2) визуализации состояния и генерации видео (через `plotly` и `imageio`);

## Структура проекта
```text
robot-scenario/
├── data/                      # Глобальные ресурсы и настройки
│   ├── assets/                # Шрифты (.ttf) и папка icons/ (.svg картинки)
│   ├── init_state/            # Базовые карты (initial_state.json)
│   ├── objects/               # База пресетов (blueprints) и параметров объектов
│   ├── scenarios/             # Динамические сценарии
│   └── config.json            # Цвета, настройки рендера, маппинг иконок и основные пути
├── renders/                   # Сгенерированные видео симуляций (.mp4)
├── src/
│   └── robomap/               # Исходный код движка (Ядро, Физика, API, Сервер)
└── pyproject.toml             # Файл сборки пакета
```
## Установка
```bash
pip install -e .
```

Зависимости можно также поставить вручную:
```bash
pip install -r requirements/all.txt
```

## Запуск сервера (`robomap.server`)

Интерактивно (запуск FastAPI сервера):
```bash
uvicorn robomap.server:app
```

Документация и панель тестирования (Swagger UI) будут доступны по адресу: **http://127.0.0.1:8000/docs**

Опции запуска сервера:
- `--reload` — автоматическая перезагрузка при изменении кода сценариев
- `--port 8000` — указать нестандартный порт

##  Доступные Команды

Управлять объектами можно, отправляя JSON на эндпоинт `/command`.
подробнее в [wiki](https://gitlab.sagteam.ru/sagteam/robot-scenario/-/wikis/Commands)

> **Важно:** Робот оперирует только теми ID, которые есть в их индивидуальной памяти (`robot.memory`). Если робот назовет ID воспоминания (например, `"mem_123"`), движок сам корректно переведет его в реальный физический ID объекта.

## API Эндпоинты

### 1. Управление мирами (Sessions)
*   `POST /api/v1/sessions` — Создать новый инстанс мира (`scenario` - название паки сценария, `auto_robot` - флаг для автоматически управляемого робота). Возвращает `session_id` мира.
*   `GET /api/v1/sessions` — Список активных миров на сервере.
*   `DELETE /api/v1/sessions/{id}` — Удалить мир и освободить память.
*   `POST /api/v1/sessions/{id}/reset` — Сбросить мир в стартовое состояние (Tick 0).
*   `GET /api/v1/sessions/{id}/state` — Скачать полный JSON слепок мира для текущего кадра.
*   `POST /api/v1/sessions/{session_id}/export` — Сохраняет историю сессии в exported_sessions/{session_id}.msgpack.gz.

### 2. Управление временем и физикой (Simulation)
*   `POST /api/v1/sessions/{id}/play` — Запустить мир в фоновом режиме (Real-time) с указанным fps.
*   `POST /api/v1/sessions/{id}/pause` — Остановить время.
*   `POST /api/v1/sessions/{id}/tick` — Сделать 1 физический шаг (пошаговый режим).
*   `POST /api/v1/sessions/{id}/run_auto` — Максимально быстро прокрутить мир до конца сценария (Movie Mode).
*   `POST /api/v1/sessions/{id}/command` — Отправить команду в очередь сущности.
*   `GET /api/v1/sessions/{id}/feedback` — Получить статус выполнения прошлых команд (ошибки) и текущую очередь (action_queue). Прочитанные ошибки удаляются.
*   `GET /api/v1/sessions/{session_id}/logs` — Получить отфильтрованные логи робота
*   `GET /api/v1/sessions/{session_id}/local-context` — Получить контекст локального мира агента

### 3. Авторизация (Players)
*   `POST /api/v1/sessions/{id}/players` — Привязать `token` игрока к списку `entity_id` (например, выдать игроку право управлять только `robot_1`).
*   `GET /api/v1/sessions/{id}/players` — Просмотр выданных прав.
*   `DELETE /api/v1/sessions/{id}/players/{token}` — Отозвать токен.

### 4. Рендер и Визуализация (Media & System)
*   `GET /api/v1/scenarios` — Получить список установленных в папке `data/scenarios/` сценариев.
*   `GET /api/v1/sessions/{id}/viewer` — Открыть в браузере Live-монитор (карта + память), обновляемый в реальном времени.
*   `GET /api/v1/sessions/{id}/plots` — Получить сырые данные Plotly (JSON) для кастомного фронтенда.
*   `POST /api/v1/sessions/{id}/render` — Запустить фоновую сборку `.mp4` видео (с поддержкой режимов `pip_real_main`, `side_by_side`, `real_only`).
*   `POST /api/v1/fast-render` — Формирует задачу на быстрый рендер MP4 из живой сессии или файла экспорта. Возвращает `job_id`.
*   `GET /api/v1/renders/{job_id}` — Статус фоновой задачи рендера (`queued` / `running` / `done` / `failed`).
*   `POST /api/v1/sessions/{session_id}/fast-render` — Alias для быстрого рендера из живой сессии (обратная совместимость).