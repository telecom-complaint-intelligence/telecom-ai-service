# =============================================================================
# Telecom AI Service — Production Dockerfile
# FastAPI + uv | Python 3.11 | PyTorch CPU-only | Transformers | LangGraph
# =============================================================================
# torch is pinned to the CPU-only wheel via pyproject.toml [tool.uv.sources]
# (sys_platform == 'linux' → pytorch-cpu index). ~200MB vs ~2.5GB CUDA build.
# For GPU cloud (GKE T4/A100): change pytorch-cpu index to cu121 or cu124.
# =============================================================================

# Stage 1: Dependency resolution & build
FROM python:3.11-slim AS builder

# libpq-dev for psycopg2-binary wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy lock files — cached until deps change
COPY pyproject.toml uv.lock ./

# Install all production deps (CPU torch from pytorch-cpu index via uv.lock)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy source & trained models
COPY app ./app
COPY trained_models ./trained_models

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# =============================================================================
# Stage 2: Minimal production runtime
# =============================================================================
FROM python:3.11-slim AS runner

# libpq5 for psycopg2, libgomp1 for PyTorch OpenMP threading
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --no-create-home appuser

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app ./app
COPY --from=builder /app/trained_models ./trained_models

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Mount a persistent cloud volume here to cache HF model weights
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/hub

RUN mkdir -p /app/.cache/huggingface && chown -R appuser:appgroup /app/.cache

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=60s --timeout=15s --start-period=60s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:8001/health').raise_for_status()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]

