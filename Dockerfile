FROM python:3.12-slim AS base

WORKDIR /backend

ENV UV_COMPILE_BYTECODE=1 \
    UV_DIR=/root/.local \
    PATH="/root/.local/bin:$PATH"

# Install uv into the base image via pip for a clean, single-stage build.
RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY backend ./backend

RUN uv sync --no-cache --no-dev

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
