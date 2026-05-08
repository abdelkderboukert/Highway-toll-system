# ─── Stage 1 Dockerfile ───────────────────────────────────────────────────
# Edge Ingress — One container per physical camera.
# VPA-managed: Never scale horizontally for the same camera.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System dependencies: OpenCV runtime libs + ffmpeg for RTSP streams
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────
COPY stage1/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────
COPY stage1/ ./stage1/

# Models directory (weights mounted via Kubernetes PVC or baked in)
RUN mkdir -p models

# Non-root user for security
RUN addgroup --system detector && adduser --system --ingroup detector detector
USER detector

# Health check: verify Python can import the main module
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "from stage1.ledger import TrackingLedger; print('ok')"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "-m", "stage1.detector"]
