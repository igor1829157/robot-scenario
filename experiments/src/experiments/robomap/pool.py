"""Пул контейнеров Robomap для параллельных прогонов."""

from __future__ import annotations

import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "robomap"
DEFAULT_BASE_PORT = 18000
DEFAULT_READY_TIMEOUT_SEC = 120.0
DEFAULT_READY_POLL_SEC = 1.0
DEFAULT_CPUS = "1"
DEFAULT_MEMORY = "4g"
CONTAINER_PORT = 8000
WORKSPACE_MOUNT = "/workspace/robot-scenario"
POOL_LABEL_KEY = "robomap.pool_id"


class RobomapPoolError(RuntimeError):
    """Ошибка жизненного цикла пула Robomap."""


class RobomapPool:
    """Поднимает N контейнеров Robomap и гасит их по завершении.

    Использует Docker CLI. Образ должен быть уже собран (`robomap`).
    """

    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
        image: str = DEFAULT_IMAGE,
        base_port: int = DEFAULT_BASE_PORT,
        cpus: str = DEFAULT_CPUS,
        memory: str = DEFAULT_MEMORY,
        ready_timeout_sec: float = DEFAULT_READY_TIMEOUT_SEC,
        ready_poll_sec: float = DEFAULT_READY_POLL_SEC,
    ) -> None:
        """
        Args:
            project_root: Корень репозитория для volume mount.
            image: Имя Docker-образа Robomap.
            base_port: Первый порт хоста (`base_port + i`).
            cpus: Лимит CPU на контейнер.
            memory: Лимит памяти на контейнер.
            ready_timeout_sec: Таймаут ожидания готовности API.
            ready_poll_sec: Интервал опроса readiness.
        """
        self.project_root = Path(
            project_root or Path.cwd()
        ).resolve()
        self.image = image
        self.base_port = base_port
        self.cpus = cpus
        self.memory = memory
        self.ready_timeout_sec = ready_timeout_sec
        self.ready_poll_sec = ready_poll_sec

        self.pool_id: str | None = None
        self.container_ids: list[str] = []
        self.base_urls: list[str] = []
        self._host_uid = os.getenv("HOST_UID") or str(os.getuid())
        self._host_gid = os.getenv("HOST_GID") or str(os.getgid())

    def __enter__(self) -> RobomapPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self, n: int) -> list[str]:
        """Поднимает ``n`` контейнеров и ждёт готовности API.

        Args:
            n: Число инстансов (>= 1).

        Returns:
            Список ``http://127.0.0.1:<port>``.

        Raises:
            RobomapPoolError: При ошибке Docker или таймауте ready.
            ValueError: Если ``n < 1``.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        if self.container_ids:
            raise RobomapPoolError("pool already started")

        self.pool_id = uuid.uuid4().hex[:8]
        urls: list[str] = []

        try:
            for index in range(n):
                host_port = self.base_port + index
                container_id = self._docker_run(index, host_port)
                self.container_ids.append(container_id)
                url = f"http://127.0.0.1:{host_port}"
                urls.append(url)
                logger.info(
                    "pool %s: started container %s on %s",
                    self.pool_id,
                    container_id[:12],
                    url,
                )

            for url in urls:
                self._wait_ready(url)

            self.base_urls = urls
            logger.info(
                "pool %s: ready (%s instance(s))",
                self.pool_id,
                n,
            )
            return list(urls)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        """Останавливает и удаляет контейнеры пула (идемпотентно)."""
        ids = list(self.container_ids)
        if self.pool_id:
            labeled = self._docker_ps_by_label(self.pool_id)
            for container_id in labeled:
                if container_id not in ids:
                    ids.append(container_id)

        if not ids:
            self.container_ids = []
            self.base_urls = []
            return

        logger.info(
            "pool %s: stopping %s container(s)",
            self.pool_id,
            len(ids),
        )
        try:
            self._run_docker(
                ["rm", "-f", *ids],
                check=False,
            )
        finally:
            self.container_ids = []
            self.base_urls = []

    def _docker_run(self, index: int, host_port: int) -> str:
        """Запускает один контейнер Robomap.

        Args:
            index: Индекс инстанса в пуле.
            host_port: Порт на хосте.

        Returns:
            ID контейнера.
        """
        assert self.pool_id is not None
        name = f"robomap-pool-{self.pool_id}-{index}"
        command = [
            "run",
            "-d",
            "--name",
            name,
            "--label",
            f"{POOL_LABEL_KEY}={self.pool_id}",
            "--user",
            f"{self._host_uid}:{self._host_gid}",
            "--cpus",
            self.cpus,
            "--memory",
            self.memory,
            "-v",
            f"{self.project_root}:{WORKSPACE_MOUNT}",
            "-w",
            WORKSPACE_MOUNT,
            "-p",
            f"{host_port}:{CONTAINER_PORT}",
            self.image,
            "uvicorn",
            "robomap.server:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(CONTAINER_PORT),
        ]
        result = self._run_docker(command, check=True)
        container_id = result.stdout.strip()
        if not container_id:
            raise RobomapPoolError(
                f"docker run returned empty id for {name}"
            )
        return container_id

    def _wait_ready(self, base_url: str) -> None:
        """Ждёт ответа HTTP GET ``{base_url}/docs``.

        Args:
            base_url: Базовый URL инстанса.

        Raises:
            RobomapPoolError: Если API не поднялся за таймаут.
        """
        deadline = time.monotonic() + self.ready_timeout_sec
        health_url = f"{base_url.rstrip('/')}/docs"
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            try:
                with urlopen(health_url, timeout=2) as response:
                    if 200 <= response.status < 500:
                        return
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                last_error = error
            time.sleep(self.ready_poll_sec)

        raise RobomapPoolError(
            f"Robomap at {base_url} not ready within "
            f"{self.ready_timeout_sec:.0f}s"
            + (f": {last_error}" if last_error else "")
        )

    def _docker_ps_by_label(self, pool_id: str) -> list[str]:
        """Возвращает ID контейнеров с label пула.

        Args:
            pool_id: Идентификатор пула.

        Returns:
            Список ID (может быть пустым).
        """
        result = self._run_docker(
            [
                "ps",
                "-aq",
                "--filter",
                f"label={POOL_LABEL_KEY}={pool_id}",
            ],
            check=False,
        )
        if result.returncode != 0:
            return []
        return [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        ]

    def _run_docker(
        self,
        args: list[str],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Выполняет ``docker <args>``.

        Args:
            args: Аргументы после ``docker``.
            check: Пробрасывать ненулевой код выхода.

        Returns:
            Результат subprocess.

        Raises:
            RobomapPoolError: Если команда завершилась с ошибкой и
                ``check=True``.
        """
        command = ["docker", *args]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as error:
            raise RobomapPoolError(
                "docker CLI not found on PATH"
            ) from error

        if check and result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RobomapPoolError(
                f"docker {' '.join(args[:3])} failed"
                + (f": {stderr}" if stderr else "")
            )
        return result
