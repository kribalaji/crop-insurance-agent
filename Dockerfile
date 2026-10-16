# One image for all services (api, ui, mcp); docker-compose.yml picks the command.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    DATA_DIR=/app/data \
    CACHE_DIR=/app/cache \
    POLICIES_DIR=/app/policies

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY policies ./policies
RUN uv sync --frozen --no-dev

# Non-root user. Pre-create the cache dirs so named volumes inherit this ownership;
# /home/app/.cache holds Chroma's embedding model (downloaded once, ~80 MB).
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/cache /app/data /home/app/.cache \
    && chown -R app:app /app/cache /home/app/.cache
USER app

EXPOSE 8000 8001 8501
CMD ["crop-insurance-api", "--host", "0.0.0.0", "--port", "8000"]
