#!/usr/bin/env bash
# Сборка русской заготовки статьи (XeLaTeX + biber через latexmk).
# Служебные файлы — в build/, PDF — рядом с исходниками.
set -euo pipefail
cd "$(dirname "$0")"

AUX_DIR=build
PDF_NAME=main_ru.pdf

if [[ "${1:-}" == "clean" ]]; then
  latexmk -C -outdir="${AUX_DIR}"
  rm -f "${PDF_NAME}"
  rm -rf "${AUX_DIR}"
  exit 0
fi

mkdir -p "${AUX_DIR}"
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir="${AUX_DIR}" \
  main_ru.tex
cp -f "${AUX_DIR}/${PDF_NAME}" "${PDF_NAME}"
# PDF оставляем рядом с исходниками; в build/ — только служебные файлы
rm -f "${AUX_DIR}/${PDF_NAME}"
