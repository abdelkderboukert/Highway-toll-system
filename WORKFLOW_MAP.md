# Tariq Sharq-Gharb — Highway Toll & Speed Management System
## Workflow Map for Claude Code

---

## Project Overview

A distributed microservices system for the Algerian East-West Highway that handles:
- Toll payment and pass validation at checkpoints
- Real-time license plate recognition (LPR) using YOLOv8
- Speed detection between camera pairs
- Automated alerts for unpaid tolls and speed violations
- Car owner lookup by plate number
- Automated model retraining pipeline
- Operator dashboard

---

## Monorepo Structure

```
highway-toll-system/
├── services/
│   ├── payment-service/          # FastAPI — toll payment & pass management
│   ├── lpr-service/              # FastAPI + YOLOv8 — plate detection
│   ├── speed-service/            # FastAPI + Redis — speed detection
│   ├── permission-service/       # FastAPI — pass validation at checkpoints
│   ├── alert-service/            # FastAPI + Kafka — alert broadcasting
│   ├── notification-service/     # FastAPI — SMS/push to drivers
│   ├── owner-lookup-service/     # FastAPI — owner info by plate
│   ├── model-training-service/   # Python + MLflow — retraining pipeline
│   ├── model-serving-service/    # Rust (Actix-web) — inference API
│   └── dashboard/                # Next.js — operator frontend
├── infrastructure/
│   ├── terraform/                # AWS infrastructure as code
│   ├── kubernetes/               # K8s manifests
│   ├── helm/                     # Helm charts per service
│   └── argocd/                   # GitOps deployment configs
├── ml/
│   ├── models/                   # YOLOv8 + OCR model weights
│   ├── training/                 # Training scripts
│   └── data/                     # Labeled plate datasets
├── docker/
│   └── docker-compose.yml        # Local dev environment
├── .github/
│   └── workflows/                # CI/CD pipelines
└── docs/
    └── api/                      # OpenAPI specs per service
```

---

## Phase 1 — Foundation (Start Here)

### Step 1.1 — Project Scaffold

**Task:** Create monorepo structure with shared configs.

```
Create the following at the root:
- .gitignore (Python, Node, Rust, Docker)
- README.md
- docker-compose.yml (postgres, redis, kafka, zookeeper)
- .env.example with all required env vars
- Makefile with commands: up, down, migrate, test, lint
```

**docker-compose.yml services to include:**
- `postgres` — image: postgres:15, port 5432, 3 databases: toll_db, alert_db, owner_db
- `redis` — image: redis:7, port 6379
- `zookeeper` — image: confluentinc/cp-zookeeper:7.4.0
- `kafka` — image: confluentinc/cp-kafka:7.4.0, port 9092
- `mlflow` — image: ghcr.io/mlflow/mlflow, port 5000
- `grafana` — image: grafana/grafana, port 3001
- `prometheus` — image: prom/prometheus, port 9090

---

### Step 1.2 — Shared Python Base

**Task:** Create shared library used by all Python microservices.

```
services/shared/
├── __init__.py
├── database.py        # SQLAlchemy async engine factory
├── kafka_client.py    # Kafka producer/consumer wrapper
├── redis_client.py    # Redis async client wrapper
├── models/
│   ├── plate.py       # Pydantic: PlateDetection, PlateEvent
│   ├── alert.py       # Pydantic: Alert, AlertStatus
│   ├── payment.py     # Pydantic: TollPass, PaymentRequest
│   └── violation.py   # Pydantic: SpeedViolation
└── config.py          # BaseSettings with env vars
```

**Requirements (requirements-shared.txt):**
```
fastapi==0.111.0
uvicorn[standard]==0.29.0
sqlalchemy[asyncio]==2.0.30
asyncpg==0.29.0
alembic==1.13.1
aiokafka==0.11.0
redis[asyncio]==5.0.4
pydantic-settings==2.2.1
httpx==0.27.0
prometheus-fastapi-instrumentator==7.0.0
```

---

## Phase 2 — Core Backend Services

### Step 2.1 — Payment Service

**Location:** `services/payment-service/`

**Database:** `toll_db` — PostgreSQL

**Tables to create (Alembic migrations):**
```sql
checkpoints (
  id UUID PRIMARY KEY,
  name VARCHAR NOT NULL,          -- e.g. "Blida Entry", "Setif Entry"
  location VARCHAR NOT NULL,
  km_marker FLOAT NOT NULL,
  created_at TIMESTAMP
)

toll_passes (
  id UUID PRIMARY KEY,
  plate_number VARCHAR(20) NOT NULL,
  entry_checkpoint_id UUID REFERENCES checkpoints(id),
  exit_checkpoint_id UUID REFERENCES checkpoints(id),
  amount_paid DECIMAL(10,2) NOT NULL,
  status VARCHAR(20) NOT NULL,    -- active | used | expired
  paid_at TIMESTAMP NOT NULL,
  expires_at TIMESTAMP NOT NULL,
  created_at TIMESTAMP
)

fare_matrix (
  id UUID PRIMARY KEY,
  from_checkpoint_id UUID REFERENCES checkpoints(id),
  to_checkpoint_id UUID REFERENCES checkpoints(id),
  price DECIMAL(10,2) NOT NULL,
  distance_km FLOAT NOT NULL
)
```

**API Endpoints (FastAPI):**
```
POST   /api/v1/passes/create
       Body: { plate_number, entry_checkpoint_id, exit_checkpoint_id }
       Returns: { pass_id, amount, qr_code_token }

GET    /api/v1/passes/{plate_number}/active
       Returns: active pass for this plate or 404

GET    /api/v1/fare?from={checkpoint_id}&to={checkpoint_id}
       Returns: { price, distance_km }

POST   /api/v1/checkpoints/seed
       Seeds initial checkpoint data (admin only)

GET    /api/v1/health
```

**Fare calculation logic:**
- Base fare per km: 2 DZD/km for cars, 4 DZD/km for trucks
- Minimum fare: 50 DZD
- Round trip discount: 10%

**File structure:**
```
payment-service/
├── main.py
├── router.py
├── models.py          # SQLAlchemy ORM models
├── schemas.py         # Pydantic request/response
├── service.py         # Business logic
├── database.py        # DB connection
├── migrations/
│   └── versions/
├── requirements.txt
└── Dockerfile
```

---

### Step 2.2 — Permission Service

**Location:** `services/permission-service/`

**Database:** Uses `toll_db` (read-only access to toll_passes)

**Purpose:** Called by checkpoints in real time when a plate is detected. Answers: does this car have a valid pass to pass through here?

**API Endpoints:**
```
POST   /api/v1/permission/check
       Body: { plate_number, current_checkpoint_id }
       Returns: { allowed: bool, pass_id, reason }
       Logic:
         1. Query toll_passes WHERE plate_number=X AND status='active'
         2. Check if current_checkpoint is between entry and exit
         3. Check pass not expired
         4. Return allowed=true or allowed=false with reason

GET    /api/v1/health
```

**Checkpoint range logic:**
```python
# A car is allowed if current checkpoint km_marker is between
# entry checkpoint km_marker and exit checkpoint km_marker
def is_checkpoint_in_range(entry_km, exit_km, current_km):
    return min(entry_km, exit_km) <= current_km <= max(entry_km, exit_km)
```

---

### Step 2.3 — Alert Service

**Location:** `services/alert-service/`

**Database:** `alert_db` — PostgreSQL

**Tables:**
```sql
alerts (
  id UUID PRIMARY KEY,
  plate_number VARCHAR(20) NOT NULL,
  alert_type VARCHAR(20) NOT NULL,   -- unpaid_toll | speed_violation
  status VARCHAR(20) NOT NULL,       -- active | resolved | expired
  triggered_at TIMESTAMP NOT NULL,
  expires_at TIMESTAMP,              -- NULL = permanent (violations)
  checkpoint_id UUID,
  metadata JSONB                     -- speed, amount owed, etc.
)

alert_broadcasts (
  id UUID PRIMARY KEY,
  alert_id UUID REFERENCES alerts(id),
  checkpoint_id UUID NOT NULL,
  broadcasted_at TIMESTAMP
)
```

**Kafka Topics (consumed):**
- `toll.unpaid` — from permission service when car has no pass
- `speed.violation` — from speed service when car exceeds 120 km/h

**Kafka Topics (produced):**
- `alert.broadcast` — sent to all checkpoints when alert created

**Business logic:**
```
On toll.unpaid event:
  1. Create alert (type=unpaid_toll, expires_at = now + 24h)
  2. Produce alert.broadcast to all checkpoints
  3. Trigger notification-service

On speed.violation event:
  1. Create alert (type=speed_violation, expires_at = NULL — permanent)
  2. Produce alert.broadcast
  3. Trigger notification-service

CRON job every 5 minutes:
  1. Find all active unpaid_toll alerts where expires_at < now
  2. Set status = expired (NOT deleted — stays in DB)
  3. Log: "Alert expired, car {plate} still in flagged database"
```

**API Endpoints:**
```
GET    /api/v1/alerts/active
       Returns: all currently active alerts

GET    /api/v1/alerts/plate/{plate_number}
       Returns: full alert history for this plate (active + expired)

POST   /api/v1/alerts/{alert_id}/resolve
       Marks alert as resolved (officer manually cleared it)

GET    /api/v1/health
```

---

### Step 2.4 — Notification Service

**Location:** `services/notification-service/`

**Purpose:** Sends messages to drivers and operators when alerts are triggered.

**Kafka Topics (consumed):**
- `alert.broadcast`

**Integrations:**
- SMS: Twilio API (mock with log output in dev)
- WebSocket: pushes live events to dashboard

**API Endpoints:**
```
POST   /api/v1/notify/driver
       Body: { plate_number, message, channel: sms|push }

POST   /api/v1/notify/operator
       Body: { checkpoint_id, alert_id, message }

WS     /ws/dashboard
       Streams live alert events to the Next.js dashboard
```

---

### Step 2.5 — Owner Lookup Service

**Location:** `services/owner-lookup-service/`

**Database:** `owner_db` — PostgreSQL

**Purpose:** Given a plate number, return the registered owner's info. In production this would call a government CNRC API. For dev/portfolio, uses a seeded mock database.

**Tables:**
```sql
vehicle_owners (
  id UUID PRIMARY KEY,
  plate_number VARCHAR(20) UNIQUE NOT NULL,
  owner_name VARCHAR(100) NOT NULL,
  phone_number VARCHAR(20),
  wilaya VARCHAR(50),
  vehicle_type VARCHAR(20),      -- car | truck | motorcycle
  vehicle_make VARCHAR(50),
  vehicle_model VARCHAR(50),
  vehicle_year INT,
  registered_at TIMESTAMP
)
```

**API Endpoints:**
```
GET    /api/v1/owner/{plate_number}
       Returns: owner info or 404 if not found

POST   /api/v1/owner/seed
       Seeds 500 fake owners using Faker (admin only, dev only)

GET    /api/v1/health
```

**Seeder script:** Use `faker` library to generate realistic Algerian names, wilayas, phone numbers (05x/06x/07x format), and plate numbers (format: XX-NNN-NN).

---

## Phase 3 — Detection Services

### Step 3.1 — LPR Service (License Plate Recognition)

**Location:** `services/lpr-service/`

**Purpose:** Receives camera frames, detects and reads the plate number, publishes result to Kafka.

**ML Stack:**
- YOLOv8n (nano) for plate detection bounding box
- EasyOCR for reading plate text from the cropped bounding box
- OpenCV for image preprocessing

**Install:**
```
ultralytics==8.2.0
easyocr==1.7.1
opencv-python-headless==4.9.0.80
Pillow==10.3.0
numpy==1.26.4
```

**API Endpoints:**
```
POST   /api/v1/detect
       Body: multipart/form-data with image file
       Returns: {
         plate_number: str,
         confidence: float,
         bounding_box: [x1, y1, x2, y2],
         processing_time_ms: int
       }

POST   /api/v1/detect/base64
       Body: { image_base64: str, checkpoint_id: str, camera_id: str }
       Returns: same as above + publishes to Kafka topic: plate.detected

GET    /api/v1/health
```

**Detection pipeline (service.py):**
```python
# Step 1: Preprocess image
# - Resize to 640x640
# - Normalize
# - Apply CLAHE for low light conditions

# Step 2: YOLOv8 inference
# - Load model from /models/yolov8_plates.pt
# - Run inference, get bounding boxes with confidence > 0.6

# Step 3: Crop plate region from image

# Step 4: EasyOCR on cropped region
# - Languages: ['ar', 'en']
# - Return raw text, clean to alphanumeric only

# Step 5: Format plate (XX-NNN-NN for Algerian format)

# Step 6: Publish to Kafka topic plate.detected
```

**Kafka event published:**
```json
{
  "plate_number": "16-234-47",
  "checkpoint_id": "uuid",
  "camera_id": "cam_1",
  "confidence": 0.94,
  "timestamp": "2025-01-01T10:00:00Z",
  "image_path": "s3://bucket/detections/uuid.jpg"
}
```

**Model loading:**
- In dev: use base YOLOv8n pretrained weights (ultralytics auto-downloads)
- In prod: load fine-tuned model from S3 path defined in env var MODEL_PATH

---

### Step 3.2 — Speed Detection Service

**Location:** `services/speed-service/`

**Database:** Redis (temporary state), PostgreSQL `alert_db` (permanent violations)

**Purpose:** Two cameras at each checkpoint, fixed distance apart (50 meters default). Camera 1 records entry timestamp, camera 2 records exit timestamp. Speed = distance / time.

**Redis key schema:**
```
speed:checkpoint:{checkpoint_id}:plate:{plate_number}
Value: { cam1_timestamp: ISO8601, cam1_image: s3_path }
TTL: 30 seconds (if cam2 not seen within 30s, discard)
```

**Kafka Topics (consumed):**
- `plate.detected` — listens for all plate events

**Kafka Topics (produced):**
- `speed.violation` — when speed > 120 km/h

**API Endpoints:**
```
POST   /api/v1/speed/event
       Body: { plate_number, checkpoint_id, camera_id: "cam1"|"cam2", timestamp }
       Logic:
         cam1: store in Redis with TTL 30s
         cam2: retrieve cam1 from Redis, calculate speed, check threshold

GET    /api/v1/violations
       Returns: all violations from PostgreSQL

GET    /api/v1/violations/{plate_number}
       Returns: violation history for plate

GET    /api/v1/health
```

**Speed calculation:**
```python
CAMERA_DISTANCE_METERS = 50  # configurable per checkpoint
SPEED_LIMIT_KMH = 120

def calculate_speed(cam1_ts: datetime, cam2_ts: datetime) -> float:
    elapsed_seconds = (cam2_ts - cam1_ts).total_seconds()
    if elapsed_seconds <= 0:
        return 0.0
    speed_ms = CAMERA_DISTANCE_METERS / elapsed_seconds
    speed_kmh = speed_ms * 3.6
    return round(speed_kmh, 2)

def is_violation(speed_kmh: float) -> bool:
    return speed_kmh > SPEED_LIMIT_KMH
```

**Violation record stored in PostgreSQL:**
```json
{
  "plate_number": "16-234-47",
  "checkpoint_id": "uuid",
  "speed_kmh": 147.3,
  "limit_kmh": 120,
  "cam1_timestamp": "...",
  "cam2_timestamp": "...",
  "evidence_image_url": "s3://..."
}
```

---

## Phase 4 — ML Pipeline

### Step 4.1 — Model Training Service

**Location:** `services/model-training-service/`

**Purpose:** Periodically retrains the YOLOv8 plate detection model on new labeled data. Tracks experiments with MLflow. Saves best model to S3.

**Trigger:** GitHub Actions CRON every Sunday at 2am, or manual POST request.

**Stack:**
```
ultralytics==8.2.0
mlflow==2.13.0
boto3==1.34.0
pandas==2.2.2
scikit-learn==1.5.0
```

**Training pipeline (train.py):**
```python
# Step 1: Download latest labeled dataset from S3
#   s3://highway-toll-system/datasets/plates/labeled/

# Step 2: Split dataset 80/10/10 train/val/test

# Step 3: Start MLflow run
with mlflow.start_run():
    # Step 4: Train YOLOv8n
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        project='plate_detection'
    )

    # Step 5: Log metrics to MLflow
    mlflow.log_metric("mAP50", results.results_dict['metrics/mAP50(B)'])
    mlflow.log_metric("precision", results.results_dict['metrics/precision(B)'])
    mlflow.log_metric("recall", results.results_dict['metrics/recall(B)'])

    # Step 6: If mAP50 > previous best, upload model to S3
    #   s3://highway-toll-system/models/yolov8_plates_v{version}.pt
    # Step 7: Update model-serving-service via API call to reload model
```

**API Endpoints:**
```
POST   /api/v1/train/trigger
       Starts a training run (async background task)
       Returns: { run_id, status: "started" }

GET    /api/v1/train/status/{run_id}
       Returns: { status, metrics, duration }

GET    /api/v1/models/list
       Returns: all model versions from MLflow

GET    /api/v1/health
```

---

### Step 4.2 — Model Serving Service (Rust)

**Location:** `services/model-serving-service/`

**Purpose:** High-throughput inference API. Wraps the trained ONNX-exported YOLOv8 model. Written in Rust with Actix-web for maximum performance at checkpoint scale.

**Stack:**
- Rust + Actix-web 4
- ort (ONNX Runtime bindings for Rust)
- image crate for preprocessing
- tokio for async

**Cargo.toml dependencies:**
```toml
[dependencies]
actix-web = "4"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
ort = "2.0.0-rc.2"
image = "0.25"
base64 = "0.22"
anyhow = "1"
tracing = "0.1"
tracing-actix-web = "0.7"
```

**API Endpoints:**
```
POST   /infer
       Body: { image_base64: str }
       Returns: {
         detections: [{ bbox, confidence, class }],
         inference_time_ms: int
       }

POST   /reload
       Reloads model weights from S3 (called by training service)

GET    /health
GET    /metrics   (Prometheus format)
```

**Model loading:**
```rust
// On startup: load ONNX model from MODEL_PATH env var
// Support hot-reload: store model in Arc<RwLock<Session>>
// On /reload: acquire write lock, load new model, release
```

---

## Phase 5 — Frontend Dashboard

### Step 5.1 — Operator Dashboard (Next.js)

**Location:** `services/dashboard/`

**Stack:**
- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- shadcn/ui components
- Recharts for charts
- Socket.io-client for live WebSocket events

**Pages:**
```
/                     → redirect to /dashboard
/dashboard            → live overview: active alerts, recent detections
/checkpoints          → list of all checkpoints with live status
/checkpoints/[id]     → single checkpoint: camera feeds, recent plates
/alerts               → all active + recent alerts table
/alerts/[id]          → single alert detail, resolve button
/violations           → speed violations table with filters
/vehicles/[plate]     → full history for one plate number
/analytics            → charts: traffic volume, revenue, violations over time
```

**Dashboard page components:**
```
<LiveAlertFeed />          — WebSocket-powered, auto-updates
<CheckpointStatusGrid />   — green/red status per checkpoint
<RecentDetections />       — last 20 plate detections
<RevenueChart />           — daily revenue bar chart (Recharts)
<ViolationCounter />       — today's violation count
```

**Environment variables:**
```
NEXT_PUBLIC_API_GATEWAY_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/dashboard
```

---

## Phase 6 — DevOps Layer

### Step 6.1 — Dockerfiles

**Each service needs a Dockerfile. Use multi-stage builds.**

**Python services (template):**
```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Rust service:**
```dockerfile
FROM rust:1.78-slim AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/model-serving-service /usr/local/bin/
EXPOSE 8080
CMD ["model-serving-service"]
```

**Next.js:**
```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
EXPOSE 3000
CMD ["node", "server.js"]
```

---

### Step 6.2 — GitHub Actions CI/CD

**Location:** `.github/workflows/`

**File: ci.yml** (runs on every PR)
```yaml
Triggers: pull_request to main

Jobs:
  lint-python:
    - ruff check services/*/
    - black --check services/*/

  lint-rust:
    - cargo clippy in model-serving-service/

  lint-frontend:
    - npm run lint in dashboard/

  test-python:
    - pytest services/payment-service/tests/
    - pytest services/permission-service/tests/
    - pytest services/alert-service/tests/
    - pytest services/speed-service/tests/

  security-scan:
    - trivy filesystem --severity HIGH,CRITICAL .
    - pip-audit in each Python service
    - trufflehog filesystem . (secret scanning)
    - semgrep --config auto services/

  docker-build:
    - Build all Docker images (no push on PR)
    - Confirm all images build successfully
```

**File: cd.yml** (runs on push to main)
```yaml
Triggers: push to main

Jobs:
  build-and-push:
    - Build multi-arch images (linux/amd64, linux/arm64)
    - Push to GHCR: ghcr.io/abdelkderboukert/highway-toll-{service}:sha

  deploy-staging:
    - Update Helm values with new image tags
    - ArgoCD auto-syncs to staging namespace

  deploy-prod:
    - Manual approval gate
    - ArgoCD sync to prod namespace
```

**File: model-retrain.yml** (CRON)
```yaml
Triggers: schedule: cron '0 2 * * 0'  (every Sunday 2am)

Jobs:
  retrain:
    - Call POST /api/v1/train/trigger on model-training-service
    - Wait for completion (poll /status)
    - If mAP50 improved: tag new model, notify via Slack
    - If failed: create GitHub issue automatically
```

---

### Step 6.3 — Kubernetes + Helm

**Location:** `infrastructure/helm/`

**Create one Helm chart per service. Each chart contains:**
```
charts/{service-name}/
├── Chart.yaml
├── values.yaml          # default values
├── values-staging.yaml  # staging overrides
├── values-prod.yaml     # prod overrides
└── templates/
    ├── deployment.yaml
    ├── service.yaml
    ├── configmap.yaml
    ├── hpa.yaml          # HorizontalPodAutoscaler
    └── ingress.yaml
```

**values.yaml template:**
```yaml
replicaCount: 2

image:
  repository: ghcr.io/abdelkderboukert/highway-toll-{service}
  tag: latest
  pullPolicy: Always

service:
  type: ClusterIP
  port: 8000

resources:
  requests:
    memory: "128Mi"
    cpu: "100m"
  limits:
    memory: "512Mi"
    cpu: "500m"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

env:
  DATABASE_URL: ""
  KAFKA_BOOTSTRAP_SERVERS: ""
  REDIS_URL: ""
```

---

### Step 6.4 — Terraform (AWS)

**Location:** `infrastructure/terraform/`

**Resources to provision:**
```hcl
# modules/networking/
- VPC with 3 public + 3 private subnets
- Internet Gateway
- NAT Gateway
- Security Groups

# modules/eks/
- EKS Cluster (v1.29)
- Node group: t3.medium x3 (min), t3.large x10 (max)

# modules/database/
- RDS PostgreSQL 15 (Multi-AZ in prod)
  - toll_db
  - alert_db
  - owner_db

# modules/elasticache/
- ElastiCache Redis 7 cluster

# modules/storage/
- S3 bucket: highway-toll-system
  - /datasets/
  - /models/
  - /detections/

# modules/monitoring/
- CloudWatch log groups per service

# main.tf — wires all modules
# variables.tf — env, region, cluster_name
# outputs.tf — eks endpoint, rds endpoints, s3 bucket name
# backend.tf — S3 remote state + DynamoDB locking
```

---

### Step 6.5 — ArgoCD GitOps

**Location:** `infrastructure/argocd/`

```yaml
# apps/payment-service.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: payment-service
  namespace: argocd
spec:
  project: highway-toll
  source:
    repoURL: https://github.com/abdelkderboukert/highway-toll-system
    targetRevision: HEAD
    path: infrastructure/helm/payment-service
  destination:
    server: https://kubernetes.default.svc
    namespace: highway-toll-prod
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

**Create one Application manifest per service.**

---

## Phase 7 — Monitoring

### Step 7.1 — Prometheus + Grafana

**Each FastAPI service exposes /metrics via prometheus-fastapi-instrumentator.**

**Key metrics to track:**
```
- http_requests_total (per service, per endpoint, per status code)
- plate_detections_total (LPR service)
- speed_violations_total (speed service)
- alerts_created_total (alert service)
- toll_revenue_total (payment service)
- model_inference_time_seconds (model serving)
- kafka_consumer_lag (all services)
```

**Grafana dashboards to create:**
```
1. System Overview — all services health, error rates
2. Checkpoint Activity — detections per checkpoint per hour
3. Revenue Dashboard — daily/weekly/monthly toll revenue
4. Violations Dashboard — speed violations by checkpoint, time of day
5. ML Performance — model inference time, accuracy drift over time
```

---

## Environment Variables Reference

```env
# Shared
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/toll_db
REDIS_URL=redis://localhost:6379/0
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=eu-west-3
S3_BUCKET=highway-toll-system

# Payment Service
PAYMENT_DB_URL=postgresql+asyncpg://user:pass@localhost:5432/toll_db
BASE_FARE_PER_KM=2.0
MIN_FARE=50.0

# Speed Service
CAMERA_DISTANCE_METERS=50
SPEED_LIMIT_KMH=120
ALERT_EXPIRY_HOURS=24

# LPR Service
MODEL_PATH=s3://highway-toll-system/models/yolov8_plates_latest.pt
DETECTION_CONFIDENCE_THRESHOLD=0.6

# Model Serving (Rust)
ONNX_MODEL_PATH=/models/yolov8_plates.onnx
MODEL_S3_PATH=s3://highway-toll-system/models/

# Notification Service
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Dashboard
NEXT_PUBLIC_API_GATEWAY_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/dashboard
```

---

## Build Order for Claude Code

**Give Claude Code these phases in order. Do not skip ahead.**

```
Phase 1 → Project scaffold + docker-compose
Phase 2.1 → Payment service (includes DB schema + migrations)
Phase 2.2 → Permission service
Phase 2.3 → Alert service + Kafka integration
Phase 2.4 → Notification service + WebSocket
Phase 2.5 → Owner lookup service + data seeder
Phase 3.1 → LPR service (YOLOv8 + EasyOCR)
Phase 3.2 → Speed detection service (Redis state machine)
Phase 4.1 → Model training service (MLflow)
Phase 4.2 → Model serving service (Rust)
Phase 5.1 → Next.js dashboard
Phase 6.1 → All Dockerfiles
Phase 6.2 → GitHub Actions CI/CD
Phase 6.3 → Helm charts
Phase 6.4 → Terraform
Phase 6.5 → ArgoCD manifests
Phase 7.1 → Prometheus + Grafana
```

---

## Notes for Claude Code

- All Python services use **async/await** throughout — no sync SQLAlchemy or blocking I/O
- All Kafka consumers use **consumer groups** — group_id = service name
- All database migrations use **Alembic** — never raw SQL in application code
- All secrets come from **environment variables** — never hardcoded
- All services expose **/health** and **/metrics** endpoints
- Use **UUID** primary keys everywhere — never auto-increment integers
- All timestamps stored as **UTC**
- Plate number format: **XX-NNN-NN** (Algerian standard) — validate on input
- For the YOLOv8 model in dev: use `yolov8n.pt` base weights (no fine-tuning needed to get started)
- For the Rust model serving: export YOLOv8 to ONNX first: `model.export(format='onnx')`
```
