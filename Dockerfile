FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv

RUN apt-get update \
    && apt-get install --yes --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY .env.example ./

RUN uv sync --frozen --no-dev

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data /app/state \
    && chown -R appuser:appuser /app

USER appuser

CMD ["uv", "run", "pig-labeler-bot"]
