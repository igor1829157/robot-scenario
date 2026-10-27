#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LLM_TELLER_KEY="${LLM_TELLER_KEY:-${PROJECT_DIR}/llm_teller/secrets/google.secret}"
export LLM_TELLER_KEY

MODEL="${MODEL:-gemini-2.5-flash-lite}"
if [[ -z "${TEXT:-}" ]]; then
  # Текст для первой цепочки `scene_chain` задается прямо здесь.
  # Если хотите переопределить без правки файла — задайте переменную окружения `TEXT`.
  TEXT="$(cat <<'EOF'
Группа ученых расположилась в заснеженной тундре для исследований.
В лесу за деревьями притаился медведь, готовый напасть в любой момент.
EOF
)"
fi

PYTHON="${PROJECT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Не найден python venv: ${PYTHON}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_DIR}/llm_teller/src:${PYTHONPATH:-}"

TEXT_ARGS=()
if [[ -n "${TEXT}" ]]; then
  TEXT_ARGS=(--text "${TEXT}")
fi

"${PYTHON}" -m llm_teller.core.pipelines \
  --model "${MODEL}" \
  "${TEXT_ARGS[@]}"

