# syntax=docker/dockerfile:1

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    LOKY_MAX_CPU_COUNT=4 \
    DPLM_DB_BACKEND=sqlite \
    DPLM_SQLITE_PATH=/app/.runtime/dplm.sqlite \
    CI_ARTIFACT_DIR=/app/outputs/ci

WORKDIR /app

FROM base AS builder

ARG TORCH_INDEX_URL

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && grep -v -E '^torch([<>= ].*)?$' requirements.txt > /tmp/requirements-docker.txt \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" torch \
    && python -m pip install -r /tmp/requirements-docker.txt

FROM base AS runtime

LABEL org.opencontainers.image.title="GestureFlow Runtime"
LABEL org.opencontainers.image.description="Headless GestureFlow ML/runtime image for CI, smoke tests and release validation"
LABEL org.opencontainers.image.source="https://github.com/Rafaildavar/DPLM"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    make \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxi6 \
    libxrender1 \
    libxtst6 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY . .

RUN mkdir -p /app/outputs/ci /app/.runtime

CMD ["make", "ml-smoke"]
