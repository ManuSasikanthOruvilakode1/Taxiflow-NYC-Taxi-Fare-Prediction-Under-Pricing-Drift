# Multi-stage production Dockerfile for Taxiflow.
# Requires a trained model: run `taxiflow train` before `docker build`.
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /bin/uv

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install exactly the locked runtime dependencies (no dev / geo / viz groups).
RUN UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --locked --no-dev --no-editable --no-cache

FROM python:3.11-slim AS runner

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TAXIFLOW_MODEL_PATH=/app/models/taxiflow_lgbm_serving.joblib

COPY --from=builder /opt/venv /opt/venv
COPY models/taxiflow_lgbm_serving.joblib /app/models/taxiflow_lgbm_serving.joblib

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["uvicorn", "taxiflow.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
