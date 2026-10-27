# Статья: LLM-агент на северной станции

Русскоязычная заготовка статьи (`main_ru.tex`) и скрипт сборки.
Содержание опирается на план проекта `docs/plan_proekta_severnaya_stanciya.md`.

## Требования (Ubuntu 22.04 и новее)

Нужны XeLaTeX, Biber, `latexmk` и системные шрифты DejaVu (в преамбуле
подключены через `fontspec`).

```bash
sudo apt update
sudo apt install -y \
  texlive-xetex \
  texlive-latex-recommended \
  texlive-latex-extra \
  texlive-fonts-recommended \
  texlive-lang-cyrillic \
  texlive-science \
  latexmk \
  biber \
  fonts-dejavu-core
```

Кратко по пакетам:

| Пакет | Зачем |
| --- | --- |
| `texlive-xetex` | движок XeLaTeX |
| `texlive-latex-recommended` / `texlive-latex-extra` | базовые и дополнительные LaTeX-пакеты (`geometry`, `booktabs`, `hyperref`, …) |
| `texlive-fonts-recommended` | рекомендуемые шрифты TeX Live |
| `texlive-lang-cyrillic` | поддержка кириллицы / русского |
| `texlive-science` | научные пакеты (`amsmath` и окружение) |
| `latexmk` | автоматическая многопроходная сборка |
| `biber` | бэкенд для `biblatex` |
| `fonts-dejavu-core` | шрифты DejaVu для `fontspec` |

Проверка:

```bash
xelatex --version
biber --version
latexmk -v
fc-list : family | grep -i DejaVu
```

## Сборка

```bash
cd docs/paper
chmod +x build.sh
./build.sh
```

Результат: `main_ru.pdf` в этом же каталоге.
Служебные файлы сборки (`.aux`, `.bbl`, `.log`, …) пишутся в `build/`.

Очистка вспомогательных файлов и PDF:

```bash
./build.sh clean
```

Каталог `build/` и `*.pdf` в `docs/paper/` добавлены в `.gitignore`.

## Структура каталога

- `main_ru.tex` — текст статьи (заготовка с TODO)
- `refs.bib` — библиография
- `build.sh` — сборка через `latexmk -xelatex` (служебные файлы → `build/`)
- `build/` — служебные файлы сборки (не в git)
- `readme.md` — этот файл
