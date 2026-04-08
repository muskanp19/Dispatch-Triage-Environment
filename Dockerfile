# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using the official OpenEnv base image.
# Works for both standalone HF Space deployment and local Docker builds.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# git is needed to resolve any VCS-pinned dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the project sources into the build context
COPY . /app/env

WORKDIR /app/env

# Ensure uv is available (base image ships it; this is a safety fallback)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv  /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install all project dependencies (respects uv.lock if present)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# ---------------------------------------------------------------------------
# Final runtime stage
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the virtualenv and project sources from the builder stage
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env       /app/env

# Activate the venv and expose the project root on PYTHONPATH so that
# both `from server.app import app` and `from models import ...` resolve.
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Liveness probe — required by the hackathon pre-submission checklist
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose the server port
EXPOSE 8000

# Start the FastAPI server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
