"""Общие утилиты для baseline-агентов Robomap."""

from __future__ import annotations

import logging
import time

from robomap_client.core.client import RobomapClient

logger = logging.getLogger(__name__)


def wait_until_queue_empty(client: RobomapClient) -> None:
    """Ждёт завершения action_queue робота, логируя tick и размер очереди."""
    while True:
        state = client.get_full_state()
        queue = (
            state.get("robots_state", {})
            .get(client.robot_id, {})
            .get("action_queue", [])
        )
        if not queue:
            break

        logger.info("tick=%s queue=%s", state.get("tick"), len(queue))
        time.sleep(client.poll_interval)
