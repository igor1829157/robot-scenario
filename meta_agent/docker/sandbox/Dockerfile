FROM python:3.12-slim@sha256:e5c9fa26ffb76e11e0f054f30dc2523a2f9693f0c36c0cf1e39b27e152d899fc

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0

RUN groupadd --gid 10001 sandbox && \
    useradd --uid 10001 --gid sandbox --no-create-home --shell /usr/sbin/nologin sandbox

WORKDIR /sandbox

COPY docker/sandbox/worker.py /sandbox/worker.py
COPY src/meta_agent/runtime/ /sandbox/meta_agent/runtime/
COPY src/meta_agent/__init__.py /sandbox/meta_agent/__init__.py

USER 10001:10001

ENTRYPOINT ["python", "-I", "-u", "/sandbox/worker.py"]
