#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

printf 'HOST_UID=%s\nHOST_GID=%s\nHOST_USER=%s\n' "$(id -u)" "$(id -g)" "$(id -un)" > .env

docker compose -f docker-compose.yml run --rm --service-ports robomap "$@"