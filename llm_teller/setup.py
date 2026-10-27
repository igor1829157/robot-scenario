"""Сборочный скрипт `setup.py` для пакета `llm_teller`.

Проект использует структуру с каталогом `src/`:
`llm_teller/src/llm_teller/...`.
"""

from __future__ import annotations

from pathlib import Path

from setuptools import find_namespace_packages, setup


def _read_requirements(path: Path) -> list[str]:
    """Читает файл с зависимостями `pip requirements` в список строк."""
    if not path.is_file():
        return []

    requirements: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
SRC_REL = "src"

long_description = ""
readme_path = HERE / "README.md"
if readme_path.is_file():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="llm-teller",
    version="0.1.0",
    description="Утилиты для генерации и визуализации сцен на основе LLM.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="elnmax",
    python_requires=">=3.10",
    # Важно: для editable-сборок setuptools пути, передаваемые в `setup()`,
    # должны быть относительными относительно каталога `setup.py` (не абсолютными).
    package_dir={"": SRC_REL},
    packages=find_namespace_packages(where=SRC_REL),
    include_package_data=True,
    package_data={
        # Шаблоны промптов, используемые пакетом.
        "llm_teller.prompts": ["*.prompt"],
    },
    install_requires=_read_requirements(HERE / "requirements" / "all.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    zip_safe=False,
)

