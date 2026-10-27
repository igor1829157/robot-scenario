"""Определение доступного числа CPU с учётом cgroup quota."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

DEFAULT_CPU_FALLBACK = 2
MAX_AUTO_WORKERS = 2


def _parse_cpu_max(text: str) -> Optional[int]:
    """Парсит cgroup v2 cpu.max (quota period)."""
    parts = text.strip().split()
    if len(parts) != 2 or parts[0] == 'max':
        return None
    quota, period = int(parts[0]), int(parts[1])
    if quota <= 0 or period <= 0:
        return None
    return max(1, math.ceil(quota / period))


def _read_cpu_max_file(path: Path) -> Optional[int]:
    """Читает cpu.max из указанного пути."""
    if not path.is_file():
        return None
    try:
        return _parse_cpu_max(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _cgroup_v2_cpu_count() -> Optional[int]:
    """Возвращает CPU limit из cgroup v2 через /proc/self/cgroup."""
    cgroup_path = Path('/proc/self/cgroup')
    if not cgroup_path.is_file():
        return None

    rel_path = ''
    for line in cgroup_path.read_text(encoding='utf-8').splitlines():
        if line.startswith('0::'):
            rel_path = line[3:].strip('/')
            break

    candidates = []
    if rel_path:
        candidates.append(Path('/sys/fs/cgroup') / rel_path / 'cpu.max')
    candidates.append(Path('/sys/fs/cgroup/cpu.max'))

    for candidate in candidates:
        count = _read_cpu_max_file(candidate)
        if count is not None:
            return count
    return None


def _cgroup_v1_cpu_count() -> Optional[int]:
    """Возвращает CPU limit из cgroup v1."""
    cgroup_path = Path('/proc/self/cgroup')
    if not cgroup_path.is_file():
        return None

    for line in cgroup_path.read_text(encoding='utf-8').splitlines():
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        controllers, rel_path = parts[1], parts[2].strip('/')
        if 'cpu' not in controllers:
            continue

        base = Path('/sys/fs/cgroup')
        for mount_name in controllers.split(','):
            quota_path = base / mount_name / rel_path / 'cpu.cfs_quota_us'
            period_path = base / mount_name / rel_path / 'cpu.cfs_period_us'
            if not quota_path.is_file() or not period_path.is_file():
                continue
            try:
                quota = int(quota_path.read_text(encoding='utf-8').strip())
                period = int(period_path.read_text(encoding='utf-8').strip())
            except (OSError, ValueError):
                continue
            if quota > 0 and period > 0:
                return max(1, math.ceil(quota / period))
    return None


def detect_cpu_limit() -> int:
    """Определяет доступное число CPU с учётом cgroup, affinity и fallback cap."""
    for detector in (_cgroup_v2_cpu_count, _cgroup_v1_cpu_count):
        count = detector()
        if count is not None:
            return count

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, min(os.cpu_count() or DEFAULT_CPU_FALLBACK, DEFAULT_CPU_FALLBACK))


def resolve_worker_count(workers: Optional[int]) -> int:
    """Определяет число worker-процессов; auto capped at MAX_AUTO_WORKERS."""
    if workers is not None:
        return max(1, workers)

    cpu_limit = detect_cpu_limit()
    auto_workers = max(1, cpu_limit - 1)
    return min(auto_workers, MAX_AUTO_WORKERS)
