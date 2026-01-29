# syntax=docker/dockerfile:1
# Multi-stage build for a lean production image.

# ---------------------------------------------------------------------------
# Stage 1 – build wheels
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml requirements.txt ./
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt \
 && pip wheel --no-cache-dir --wheel-dir /wheels .

# ---------------------------------------------------------------------------
# Stage 2 – runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL maintainer="SurrogateFactory Contributors"
LABEL org.opencontainers.image.source="https://github.com/bismu-jet/SurrogateFactory"

# Non-root user for security (Canonical best-practice).
RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid app --create-home app

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels surrogate_factory \
 && pip install --no-cache-dir fastapi uvicorn[standard] \
 && rm -rf /wheels

# Copy only what the runtime needs.
COPY src/ src/

USER app

ENV SURROGATE_MODEL_PATH=/app/model.pkl \
    LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

ENTRYPOINT ["uvicorn", "surrogate_factory.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
