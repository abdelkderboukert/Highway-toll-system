# CLAUDE.md — Async CV Traffic Monitoring Pipeline

> This file gives Claude Code full context about the project architecture, conventions,
> design decisions, and operational rules. Read it entirely before touching any code.

---

## Project Overview

This is a **production-grade Automatic Number Plate Recognition (ANPR) system** for
multi-lane traffic monitoring. It is built as a three-stage microservice pipeline that
decouples real-time video ingestion from compute-heavy inference using an asynchronous
buffer. The result is a system that never drops a detection regardless of downstream
processing speed.

**Core design principle:** Input Speed must never be coupled to Processing Depth.
Stage 1 (the camera) always runs at 100% efficiency. Everything else is a consumer.

---

## Repository Structure

```
.
├── stage1/                  # Edge Ingress — Producer
│   ├── detector.py          # YOLOv26n inference loop
│   ├── tracker.py           # MOT wrapper (ByteTrack / StrongSORT)
│   ├── speed_estimator.py   # Vector displacement → km/h
│   ├── snapshot.py          # Snapshot Event builder and queue pusher
│   ├── ledger.py            # Redis-backed track_id deduplication
│   └── Dockerfile
│
├── buffer/                  # Async Broker — configuration only
│   ├── redis.conf           # Redis FIFO queue config
│   └── keda_scaledobject.yaml  # KEDA HPA trigger for Stage 2/3
│
├── stage2/                  # Targeted Inference
│   ├── plate_detector.py    # Custom YOLO — license plate bbox
│   ├── preprocessor.py      # DSP pipeline (grayscale → bilateral → threshold → deskew)
│   └── Dockerfile
│
├── stage3/                  # Digitization & Egress — Consumer
│   ├── ocr.py               # OCR engine wrapper (Tesseract / EasyOCR / CRNN)
│   ├── aggregator.py        # Merges OCR result with Stage 1 metadata
│   ├── egress.py            # Async HTTP POST / Kafka publisher
│   └── Dockerfile
│
├── worker/                  # Stage 2+3 combined worker process
│   ├── main.py              # Queue consumer loop — pops task, runs Stage 2 then Stage 3
│   └── Dockerfile
│
├── infra/                   # Kubernetes & DevOps
│   ├── k8s/
│   │   ├── stage1/
│   │   │   ├── statefulset.yaml     # One replica per camera, VPA-managed
│   │   │   └── vpa.yaml             # Vertical Pod Autoscaler config
│   │   ├── worker/
│   │   │   ├── deployment.yaml      # Stateless workers, HPA-managed
│   │   │   ├── hpa.yaml             # HPA — queue-depth metric via KEDA
│   │   │   └── pdb.yaml             # PodDisruptionBudget — min 1 worker always up
│   │   ├── redis/
│   │   │   └── statefulset.yaml     # Redis with persistent volume
│   │   └── network-policies/        # Strict pod-to-pod communication rules
│   ├── helm/                        # Helm chart for full deployment
│   └── terraform/                   # Infrastructure provisioning
│
├── scripts/
│   ├── calibrate_homography.py  # Camera calibration tool (pixel → meters)
│   └── test_pipeline.py         # End-to-end smoke test
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── .env.example             # Environment variable template — never commit .env
├── docker-compose.yml       # Local development stack
└── CLAUDE.md                # This file
```

---

## Architecture — The Three Stages

### Stage 1 — Edge Ingress (Producer)

**One process per physical IP camera. Never replicated horizontally.**

- Connects to the camera RTSP stream and runs YOLOv26n object detection at the
  camera's native frame rate (typically 30 FPS). Never slows down for anything downstream.
- Runs ByteTrack or StrongSORT MOT to assign a persistent `track_id` to each vehicle
  across consecutive frames.
- Computes speed via vector displacement of the bounding box centroid, converted to km/h
  using a pre-calibrated homography matrix.
- On first clean detection of a `track_id`, fires a **Snapshot Event**:
  - A JPEG crop of the vehicle (85% quality, padded bounding box)
  - Metadata: `{ track_id, speed_kmh, timestamp_utc, camera_id, frame_number, confidence }`
- Checks the **Tracking Ledger** (Redis SET) before enqueuing. If `track_id` already
  exists, the event is silently discarded.
- Pushes the task to Redis with `LPUSH`. Returns immediately. Never waits.

**Key file:** `stage1/snapshot.py` — the fire-and-forget pusher.  
**Key file:** `stage1/ledger.py` — the deduplication gate.

### Buffer Layer — Async Broker

**Redis FIFO queue + Tracking Ledger. The shock absorber.**

- Tasks are stored as Redis list entries (LPUSH / BRPOP = FIFO).
- Queue depth is the primary health signal for the entire system. Monitor it.
- KEDA reads queue depth and exposes it as a custom Kubernetes metric for HPA.
- Ledger TTL is configurable (default: 300 seconds). After TTL expires, the same
  vehicle can re-enter if it re-appears in the frame.

**Never modify the queue structure without updating the KEDA ScaledObject.**

### Stage 2 — Targeted Inference

**Stateless. Runs inside the combined `worker/` process.**

- **Stage 2a:** Custom YOLO (YOLOv8n or v5, plate-specific) detects the license plate
  bounding box within the vehicle crop.
- **Stage 2b:** OpenCV DSP preprocessing pipeline — always applied in this exact order:
  1. Grayscale conversion
  2. Bilateral filter (edge-preserving noise reduction)
  3. Adaptive thresholding (local binarization, corrects uneven lighting)
  4. Deskew via Hough Line Transform (optional, enabled by `DESKEW_ENABLED=true`)

**Do not reorder DSP steps.** Thresholding after grayscale is load-bearing.
Bilateral filter must run before thresholding or it operates on binary data uselessly.

### Stage 3 — Digitization & Egress (Consumer)

**Stateless. Runs inside the combined `worker/` process.**

- OCR engine (configurable via `OCR_ENGINE` env var: `tesseract`, `easyocr`, `crnn`).
- Character whitelist: `A-Z 0-9` only. Any read below `OCR_MIN_CONFIDENCE` (default: 0.7)
  is discarded and the task is sent to the dead-letter queue.
- Aggregates plate string with Stage 1 metadata to produce the final detection record.
- Dispatches via `aiohttp` async POST. **Never use synchronous requests here.**
  A slow external database must never propagate latency back into the worker loop.
- On HTTP failure, retries with exponential backoff (max 3 attempts), then pushes to
  the dead-letter queue. Does not crash the worker.

---

## Scaling Model — Hybrid VPA + HPA

This is the most important architectural decision in the project. Do not change it
without understanding the full rationale below.

### Stage 1 — Vertical Pod Autoscaler (VPA)

```
One pod per camera. VPA adjusts CPU/memory. HPA is NOT used here.
```

**Why VPA and not HPA:**  
Adding a second replica of a Stage 1 pod for the same camera creates two processes
consuming the same RTSP stream. Both would detect the same vehicles and generate
duplicate Snapshot Events, doubling queue load. There is no valid use case for multiple
Stage 1 replicas per camera.

VPA monitors historical CPU/memory usage and adjusts resource requests/limits over
time. During peak hours (many simultaneous tracks), the pod gets more CPU. During
quiet periods, it runs lean.

**Critical caveat — pod restarts:**  
In Kubernetes ≤ 1.26, VPA applies new resource values by restarting the pod
(5–15 second gap in camera coverage). This is acceptable because VPA scales slowly
(hours/days based on trends), not reactively to sudden spikes.

Watch **KEP-1287 (In-Place Pod Vertical Scaling)** — beta in K8s 1.29. When it reaches
GA, enable it to eliminate restarts. Track it in `infra/k8s/stage1/vpa.yaml`.

**Config location:** `infra/k8s/stage1/vpa.yaml`  
**VPA update mode:** `Auto` (recommended). Set `minAllowed` and `maxAllowed` to bound
the range and prevent runaway resource allocation.

### Stage 2/3 Workers — Horizontal Pod Autoscaler (HPA) via KEDA

```
Stateless workers. HPA scales replicas based on Redis queue depth.
```

**Why HPA and not VPA:**  
Worker load is bursty — it spikes when many vehicles are detected simultaneously and
drops when traffic is light. Vertical scaling hits the node's physical CPU ceiling.
Horizontal scaling can add workers across any available node in the cluster.

KEDA (Kubernetes Event-Driven Autoscaling) reads the Redis list length and publishes
it as a custom metric. HPA targets a configurable tasks-per-replica ratio.

**Config location:** `infra/k8s/worker/hpa.yaml` and `buffer/keda_scaledobject.yaml`

```yaml
# Target: 5 tasks per worker replica
# Queue depth 30 → HPA scales to 6 workers
# Queue depth returns to 0 → HPA scales back to minReplicas: 1
```

**Never scale Stage 2/3 workers to 0.** Always keep `minReplicas: 1` so there is
always a warm worker ready to consume. Cold start latency on the first task after
scaling from 0 creates a confusing UX and delays detection records.

---

## Environment Variables

All environment variables are defined in `.env.example`. Copy to `.env` for local dev.
**Never commit `.env` or any file containing credentials.**

| Variable | Stage | Description | Default |
|---|---|---|---|
| `RTSP_URL` | 1 | Camera stream URL | — |
| `CAMERA_ID` | 1 | Unique identifier for this camera | — |
| `REDIS_HOST` | 1, Worker | Redis hostname | `redis` |
| `REDIS_PORT` | 1, Worker | Redis port | `6379` |
| `REDIS_PASSWORD` | 1, Worker | Redis auth password — from Kubernetes Secret | — |
| `REDIS_QUEUE_KEY` | 1, Worker | List key used as the task queue | `pipeline:tasks` |
| `LEDGER_TTL_SECONDS` | 1 | How long a track_id stays in the deduplication ledger | `300` |
| `YOLO_MODEL_PATH` | 1 | Path to YOLOv26n weights | `models/yolov26n.pt` |
| `PLATE_MODEL_PATH` | 2 | Path to plate-specific YOLO weights | `models/plate.pt` |
| `DESKEW_ENABLED` | 2 | Whether to run Hough deskew step in DSP pipeline | `false` |
| `OCR_ENGINE` | 3 | OCR backend: `tesseract`, `easyocr`, or `crnn` | `tesseract` |
| `OCR_MIN_CONFIDENCE` | 3 | Minimum OCR confidence score to accept a read | `0.7` |
| `EGRESS_URL` | 3 | External microservice HTTP POST endpoint | — |
| `EGRESS_MAX_RETRIES` | 3 | Max retry attempts before dead-letter queue | `3` |
| `DEAD_LETTER_QUEUE_KEY` | 3 | Redis key for failed egress tasks | `pipeline:dlq` |

---

## Data Contracts

### Snapshot Event (Stage 1 → Redis Queue)

```python
{
    "track_id": int,             # Unique vehicle identity for this camera session
    "camera_id": str,            # e.g. "cam_01_north_entrance"
    "timestamp_utc": str,        # ISO 8601 — e.g. "2026-05-07T14:23:01.456Z"
    "frame_number": int,         # Frame index within the current stream session
    "speed_kmh": float,          # Estimated speed — 0.0 if track is < 3 frames old
    "confidence": float,         # YOLOv26n detection confidence 0.0–1.0
    "image_b64": str,            # Base64-encoded JPEG crop of the vehicle
}
```

### Final Detection Record (Stage 3 → External Egress)

```python
{
    "plate_number": str,         # e.g. "16ABK123" — empty string if OCR failed
    "ocr_confidence": float,     # 0.0–1.0 — below OCR_MIN_CONFIDENCE goes to DLQ
    "speed_kmh": float,          # Passed through from Stage 1
    "timestamp_utc": str,        # Passed through from Stage 1
    "camera_id": str,            # Passed through from Stage 1
    "track_id": int,             # Passed through from Stage 1
    "processing_latency_ms": int # Wall clock time from snapshot push to egress dispatch
}
```

---

## Critical Rules — Do Not Violate

1. **Stage 1 must never block on downstream state.** Any code that makes Stage 1 wait
   for a response from Redis, the worker, or any external service is a bug.

2. **Never use synchronous HTTP in Stage 3 egress.** Use `aiohttp` and `asyncio`.
   Synchronous requests will propagate external latency into the worker and stall the
   queue drain.

3. **Never replicate Stage 1 horizontally for the same camera.** One camera = one pod.
   This is enforced by the StatefulSet config but do not work around it.

4. **Always check the Tracking Ledger before enqueueing.** The ledger check is in
   `stage1/ledger.py::is_seen()`. If you bypass it for any reason you will flood
   the queue with duplicate tasks.

5. **DSP steps must run in order.** See Stage 2b above. The sequence is not arbitrary.

6. **No credentials in code or manifests.** All secrets via Kubernetes Secrets or
   Vault. The CI pipeline will reject any commit containing a hardcoded password,
   token, or RTSP URL with embedded credentials.

7. **Workers must be stateless.** No local caches, no in-memory state between tasks.
   All shared state lives in Redis. This is what makes HPA horizontal scaling safe.

---

## Local Development

```bash
# 1. Start the local stack (Redis + mock camera stream)
docker-compose up -d

# 2. Run Stage 1 against a local test video file
RTSP_URL=file://tests/fixtures/sample_traffic.mp4 \
CAMERA_ID=test_cam_01 \
python stage1/detector.py

# 3. Run a worker (Stage 2 + 3) against the local Redis queue
OCR_ENGINE=tesseract \
EGRESS_URL=http://localhost:8080/detections \
python worker/main.py

# 4. Run the full end-to-end smoke test
python scripts/test_pipeline.py
```

---

## Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests (requires Redis running)
pytest tests/integration/

# Test a specific DSP step in isolation
pytest tests/unit/test_preprocessor.py -v

# Confirm the ledger deduplication logic
pytest tests/unit/test_ledger.py -v
```

All tests must pass before merging to `main`. The CI pipeline runs `pytest`,
Trivy image scanning, and a KEDA metric dry-run check on every pull request.

---

## Camera Calibration

Before deploying a new camera, run the homography calibration tool:

```bash
python scripts/calibrate_homography.py \
  --rtsp-url <camera_rtsp_url> \
  --output infra/calibrations/cam_01.json
```

This generates a `homography_matrix` JSON file that maps pixel coordinates to real-world
meters. The matrix file path is passed to Stage 1 via `HOMOGRAPHY_PATH` env var.
Speed estimates will be meaningless (but won't crash) if this file is missing —
speed will default to `0.0`.

---

## Deployment

```bash
# Apply the full Kubernetes stack
kubectl apply -f infra/k8s/

# Or via Helm
helm upgrade --install traffic-pipeline infra/helm/ \
  --set stage1.cameraCount=4 \
  --set worker.maxReplicas=10
```

**Scaling a new camera site:**  
Add a new entry to the Stage 1 StatefulSet (or deploy a dedicated StatefulSet per site).
VPA will learn the resource profile of the new camera pod within 24–48 hours of traffic.

---

## Observability

Key metrics to monitor in Grafana:

| Metric | Warning Threshold | Critical Threshold |
|---|---|---|
| Redis queue depth | > 20 tasks | > 50 tasks for > 2 min |
| Stage 1 FPS per camera | < 25 FPS | < 15 FPS |
| Worker task throughput | < 2 tasks/sec | < 0.5 tasks/sec |
| OCR confidence p50 | < 0.80 | < 0.65 |
| Processing latency p95 | > 2000ms | > 5000ms |
| Dead-letter queue depth | > 5 | > 20 |

Dead-letter queue growth is the most important early warning signal for OCR quality
degradation (dirty lenses, changed lighting conditions, or a new plate standard
not covered by the current model).

---

## Known Limitations & Open Issues

- **VPA pod restart gap:** Stage 1 pods have a 5–15 second coverage gap when VPA
  applies new resource values. Mitigated by KEP-1287 (In-Place VPA) when it reaches
  GA in a future Kubernetes version. Track in `infra/k8s/stage1/vpa.yaml`.

- **Single-camera Re-ID only:** The Tracking Ledger is scoped per camera. A vehicle
  moving between two camera zones will generate two separate detection records.
  Cross-camera Re-ID is on the roadmap.

- **Speed accuracy at intersection entry:** Speed estimate requires a minimum of 3
  frames of stable tracking. Vehicles entering from off-screen will report `speed_kmh: 0.0`
  for their first detection. This is by design, not a bug.

- **Night mode not yet implemented:** The DSP pipeline uses daytime-optimized parameters.
  The `DESKEW_ENABLED` flag is the only current DSP toggle. Night mode (CLAHE, adjusted
  thresholding) is tracked as a future improvement.

---

## Glossary

| Term | Definition |
|---|---|
| `track_id` | Integer assigned by the MOT algorithm to a unique vehicle. Scoped per camera session. |
| Snapshot Event | The single task fired per `track_id` — vehicle JPEG crop + metadata. |
| Tracking Ledger | Redis SET storing seen `track_id` values. Prevents duplicate queue entries. |
| FIFO Queue | Redis list (LPUSH / BRPOP). Tasks are consumed in detection order. |
| DLQ | Dead-Letter Queue. Redis list for tasks that failed egress after all retries. |
| VPA | Vertical Pod Autoscaler. Adjusts CPU/memory of a pod. Used for Stage 1. |
| HPA | Horizontal Pod Autoscaler. Adjusts replica count. Used for Stage 2/3 workers. |
| KEDA | Kubernetes Event-Driven Autoscaling. Exposes Redis queue depth as an HPA metric. |
| Homography | Pixel-to-meter mapping matrix. Required for accurate speed estimation. |
| DSP | Digital Signal Processing. The image cleaning pipeline in Stage 2b. |
| MOT | Multi-Object Tracking. Maintains vehicle identity across frames. |