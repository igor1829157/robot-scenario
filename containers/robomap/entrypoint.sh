#!/usr/bin/env bash
set -e

export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/robot-scenario}"
export PYTHONPATH="$PROJECT_ROOT/robomap/src"

git config --global --add safe.directory "$PROJECT_ROOT" 2>/dev/null || true

cd "$PROJECT_ROOT"

exec "$@"