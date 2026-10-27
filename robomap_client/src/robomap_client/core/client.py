from typing import Any, Literal
import logging
import time

import requests


FormatStyle = Literal["json", "text-md", "text-yaml", "text-narrative"]
Section = Literal["all", "current_state", "visible_now", "memory", "inventory", "virtual_map"]
LogType = Literal["all", "self", "others"]
LogFormatStyle = Literal["json", "text", "brief"]

logger = logging.getLogger(__name__)

# Опрос статуса рендера идёт в том же процессе, что и encode:
# короткий дефолтный timeout=15 рвёт прогон, пока воркер держит GIL.
RENDER_POLL_TIMEOUT_SEC = 120


class RobomapClient:

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        poll_interval: float = 0.5,
        timeout: int = 15,
    ):
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.session_id: str | None = None
        self.token: str | None = None
        self.robot_id: str | None = None
        self.last_read_tick = 0

        self.session = requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        timeout: int | None = None
    ) -> requests.Response:

        if timeout is None:
            timeout = self.timeout

        response = self.session.request(
            method=method,
            url=f"{self.base_url}{path}",
            params=params,
            data=data,
            json=json,
            timeout=timeout,
        )

        response.raise_for_status()
        return response

    def init_session(
        self,
        scenario: str = "empty",
        *,
        robot_id: str = "robot_1",
        token: str = "agent",
        fps: float = 2.0,
        random_robot_pose: bool = False,
        robot_pose_seed: int | None = None,
    ) -> str:
        self.robot_id = robot_id
        self.token = token

        session_data: dict[str, Any] = {
            "scenario": scenario,
            "auto_robot": False,
            "random_robot_pose": random_robot_pose,
            "robot_id": robot_id,
        }
        if robot_pose_seed is not None:
            session_data["robot_pose_seed"] = robot_pose_seed
        session_response = self._request("POST", "/api/v1/sessions", json=session_data)
        self.session_id = session_response.json()["session_id"]

        agent_data = {"token": self.token, "entity_ids": [self.robot_id]}
        self._request("POST", f"/api/v1/sessions/{self.session_id}/players", json=agent_data)

        play_data = {"fps": fps}
        self._request("POST", f"/api/v1/sessions/{self.session_id}/play", params=play_data)
        self.last_read_tick = 0

        return self.session_id

    def get_full_state(self) -> dict[str, Any]:
        response = self._request("GET", f"/api/v1/sessions/{self.session_id}/state")
        return response.json()

    def get_current_tick(self) -> int:
        """Возвращает текущий тик симуляции через GET /logs."""
        tick, _ = self._read_logs(self.last_read_tick)
        return tick

    def get_context(
        self,
        *,
        format_style: FormatStyle = "json",
        sections: Section = "all",
        include_geometry=False,
    ) -> Any:
        include_geometry = str(include_geometry).lower()

        params = {
            "token": self.token, "robot_id": self.robot_id, "format_style": format_style,
            "sections": sections, "include_geometry": include_geometry
        }
        response = self._request("GET", f"/api/v1/sessions/{self.session_id}/local-context", params=params)
        
        if format_style == "json":
            return response.json()
        
        return response.text
    
    def get_logs(
        self,
        tick_start: int = 0,
        *,
        log_type: LogType = "all",
        format_style: LogFormatStyle = "json",
    ) -> Any:
        params = {
            "token": self.token, "robot_id": self.robot_id, "log_type": log_type, 
            "format_style": format_style, "tick_start": tick_start
        }
        response = self._request("GET", f"/api/v1/sessions/{self.session_id}/logs", params=params)
        
        return response.json()

    def command(
        self,
        action: str,
        **kwargs: Any,
    ):
        json_data = {
            "token": self.token, "entity_id": self.robot_id, "action": action, "args": [], "kwargs": kwargs,
        }
        self._request("POST", f"/api/v1/sessions/{self.session_id}/command", json=json_data)
    
    def execute_and_wait(
        self,
        action: str,
        *,
        max_wait_ticks: int = 200,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Отправляет команду и ждёт терминального статуса в логах.

        Args:
            action: Имя команды API (``move``, ``rotate``, ...).
            max_wait_ticks: Лимит тиков симуляции.
            **kwargs: Аргументы команды.

        Returns:
            Сырой результат: action, status, final_message, elapsed_ticks, events.
        """
        terminal_statuses = {"SUCCESS", "FAIL", "ERROR"}

        self.command("clear")
        request_tick, _ = self._read_logs(self.last_read_tick)
        next_log_tick = request_tick + 1
        self.command(action, **kwargs)

        events: list[dict[str, Any]] = []
        while True:
            current_tick, new_logs = self._read_logs(next_log_tick)
            if new_logs:
                events.extend(new_logs)
                next_log_tick = max(int(log.get("tick", 0)) for log in new_logs) + 1

            for log in new_logs:
                status = str(log.get("status", "")).upper()
                if (
                    log.get("category") == "SELF_ACTION"
                    and log.get("action_type") == action
                    and status in terminal_statuses
                ):
                    self.last_read_tick = next_log_tick
                    return self._action_result(
                        action=action,
                        status=status,
                        final_message=str(log.get("message", "")),
                        elapsed_ticks=current_tick - request_tick,
                        events=events,
                    )

            if current_tick - request_tick >= max_wait_ticks:
                self.command("clear")
                self.last_read_tick = next_log_tick
                return self._action_result(
                    action=action,
                    status="TIMEOUT",
                    final_message=(
                        f"Action {action} aborted: exceeded max wait "
                        f"of {max_wait_ticks} ticks."
                    ),
                    elapsed_ticks=current_tick - request_tick,
                    events=events,
                )

            time.sleep(self.poll_interval)

    def _read_logs(self, tick_start: int) -> tuple[int, list[dict[str, Any]]]:
        """Читает логи и текущий тик симуляции."""
        payload = self.get_logs(
            tick_start=tick_start,
            log_type="all",
            format_style="json",
        )
        logs = sorted(
            payload.get("logs") or [],
            key=lambda item: int(item.get("tick", 0)),
        )
        return int(payload.get("tick", 0)), logs

    @staticmethod
    def _action_result(
        *,
        action: str,
        status: str,
        final_message: str,
        elapsed_ticks: int,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Собирает сырой результат команды без форматирования для LLM."""
        return {
            "action": action,
            "status": status,
            "final_message": final_message,
            "elapsed_ticks": elapsed_ticks,
            "events": events,
        }

    def pause_session(self) -> None:
        """Ставит сессию на паузу."""
        self._request("POST", f"/api/v1/sessions/{self.session_id}/pause")

    def export_session(self) -> dict[str, str]:
        """Экспортирует сессию на сервер и возвращает пути к файлу."""
        response = self._request(
            "POST",
            f"/api/v1/sessions/{self.session_id}/export",
            json={},
            timeout=600,
        )
        payload = response.json()
        return {
            "export_path": payload["export_path"],
            "export_filename": payload["filename"],
        }

    def delete_session(self) -> None:
        """Удаляет текущую сессию на сервере."""
        if not self.session_id:
            return
        self._request("DELETE", f"/api/v1/sessions/{self.session_id}")
        self.session_id = None

    def start_fast_render(
        self,
        export_file: str,
        *,
        filename: str = "output.mp4",
        layout_type: str = "side_by_side",
        workers: int | None = None,
    ) -> dict[str, str]:
        """Запускает fast-render из файла экспорта."""
        payload: dict[str, Any] = {
            "export_file": export_file,
            "filename": filename,
            "layout_type": layout_type,
        }
        if workers is not None:
            payload["workers"] = workers
        response = self._request(
            "POST",
            "/api/v1/fast-render",
            json=payload,
            timeout=600,
        )
        payload = response.json()
        return {
            "job_id": payload["job_id"],
            "expected_output_path": payload["expected_output_path"],
        }

    def get_render_status(
        self,
        job_id: str,
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Возвращает статус фоновой задачи рендера.

        Args:
            job_id: Идентификатор задачи, возвращённый ``start_fast_render``.
            timeout: Таймаут HTTP-запроса в секундах. По умолчанию
                ``RENDER_POLL_TIMEOUT_SEC``, а не короткий клиентский дефолт.
        """
        response = self._request(
            "GET",
            f"/api/v1/renders/{job_id}",
            timeout=RENDER_POLL_TIMEOUT_SEC if timeout is None else timeout,
        )
        return response.json()

    def wait_render(
        self,
        job_id: str,
        *,
        timeout_sec: float = 3600.0,
        poll_sec: float | None = None,
    ) -> dict[str, Any]:
        """Ждёт завершения задачи рендера (done/failed).

        ``ReadTimeout`` на одном опросе не завершает ожидание: сервер мог
        быть занят кодированием кадра. Повтор идёт до ``timeout_sec``.
        """
        if poll_sec is None:
            poll_sec = max(self.poll_interval, 1.0)

        deadline = time.time() + timeout_sec
        last: dict[str, Any] = {}
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            request_timeout = max(1, min(RENDER_POLL_TIMEOUT_SEC, int(remaining)))
            try:
                last = self.get_render_status(job_id, timeout=request_timeout)
            except requests.exceptions.ReadTimeout:
                logger.warning(
                    "Render job %s: status poll timed out after %ss, retrying",
                    job_id,
                    request_timeout,
                )
                time.sleep(poll_sec)
                continue

            status = last.get("status")
            if status == "done":
                return last
            if status == "failed":
                raise RuntimeError(
                    f"Render job {job_id} failed: {last.get('error') or 'unknown error'}"
                )
            time.sleep(poll_sec)

        raise TimeoutError(
            f"Render job {job_id} не завершился за {timeout_sec:.0f}s "
            f"(last status={last.get('status')})"
        )


if __name__ == "__main__":
    client = RobomapClient()
    client.init_session("exploring")
    print(client.get_logs(format_style="text"))
