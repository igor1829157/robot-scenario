from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Возвращает текущее время в формате ISO 8601."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
