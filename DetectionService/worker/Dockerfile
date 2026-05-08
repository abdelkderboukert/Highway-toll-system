# ─── Worker Dockerfile ────────────────────────────────────────────────────
# Stage 2 + 3 combined worker — Stateless, HPA-managed via KEDA.
# Always keep minReplicas: 1 (never scale to 0).
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System dependencies: OpenCV + Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        tesseract-ocr \
        tesseract-ocr-eng \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────
COPY worker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────
COPY stage2/ ./stage2/
COPY stage3/ ./stage3/
COPY worker/ ./worker/

# Models directory (plate YOLO weights)
RUN mkdir -p models

# Non-root user
RUN addgroup --system worker && adduser --system --ingroup worker worker
USER worker

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "from stage2.preprocessor import Preprocessor; print('ok')"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "-m", "worker.main"]
