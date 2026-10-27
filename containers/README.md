# Контейнеры

### Robomap

Конейнер для 2D мира, запускает сервер на FastAPI http://127.0.0.1:8000/docs

1. Сборка

```bash
cd containers/robomap;
docker compose build
```

2. Запуск

```bash
UID=$(id -u) GID=$(id -g) docker compose up -d
```

3. Посмотреть логи

```bash
docker compose logs -f robomap
```