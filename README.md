# Smart Nutrition Tracker

A multi-agent AI-powered nutrition tracking system built with FastAPI and Gemini 2.0-Flash. The platform uses specialized agents to help users log meals, plan nutrition, and receive personalized health advice — all through natural language conversation.

The system supports dual deployment modes (monolith and microservices), three interchangeable communication protocols, full Kubernetes orchestration with horizontal auto-scaling, and **ML-based food image classification** using MobileNetV2 trained on Food-101.

---

## Table of Contents

- [Architecture](#architecture)
- [Agents](#agents)
- [Communication Protocols](#communication-protocols)
- [ML Food Classifier](#ml-food-classifier)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Database Schema](#database-schema)
- [Monitoring & Observability](#monitoring--observability)
- [Benchmarking](#benchmarking)
- [Backup Strategy](#backup-strategy)
- [Project Structure](#project-structure)
- [Configuration](#configuration)

---

## Architecture

The system follows a hub-and-spoke multi-agent architecture. An **Orchestrator** receives user messages, classifies intent via LLM, and routes requests to the appropriate specialized agent. Agents communicate through pluggable protocols (A2A, PNP, or TOON) and share a common PostgreSQL database via a dedicated DB Writer agent.

```
                        +--------------+
                        |    Nginx     |
                        |  (Port 80)   |
                        +------+-------+
                               |
                    +----------+----------+
                    |    Orchestrator      |
                    |     (Port 8000)      |
                    |  Intent classifier  |
                    +--+---+---+---+------+
                       |   |   |   |
          +------------+   |   |   +------------+
          v                v   v                v
   +-------------+ +--------------+ +---------------+
   | Food Logger  | | Meal Planner | | Health Advisor |
   |  (8001)      | |   (8002)     | |    (8003)      |
   | + ML Model   | |              | |                |
   +------+-------+ +------+------+ +-------+--------+
          |                |                 |
          +--------+-------+--------+--------+
                   v                |
            +-------------+        |
            |  DB Writer   |<------+
            |   (8004)     |
            +------+-------+
                   |
            +------+-------+
            |  PostgreSQL   |
            |  (Port 5432)  |
            +--------------+
```

**Dual deployment modes** from a single codebase:

| Mode | Use Case | How It Works |
|------|----------|-------------|
| **Monolith** | Local development | All agents run in one FastAPI process. Inter-agent calls are direct function calls. |
| **Microservices** | Production / K8s | Each agent runs in its own container. Communication happens over HTTP. |

Switch between modes with a single environment variable: `DEPLOY_MODE=monolith|microservice`.

---

## Agents

### Orchestrator
The central router. Receives every user message, uses Gemini to classify intent (food logging, meal planning, health advice, or general conversation), and dispatches to the right agent. Manages conversation state and trace ID propagation.

### Food Logger
Parses natural language food descriptions ("I had a chicken sandwich and a coffee") into structured nutrition data — calories, protein, carbs, fat, fiber, sugar, sodium. Infers meal type and sends parsed data to the DB Writer for storage. Also supports **ML-based food image classification** via a trained MobileNetV2 model, accessible through all three protocols.

### Meal Planner
Generates personalized daily or weekly meal plans. Considers user goals (weight loss, muscle gain, maintenance), dietary preferences, allergies, and activity level. Provides shopping lists for weekly plans.

### Health Advisor
Analyzes the user's nutrition history and provides evidence-based advice. Identifies nutrient imbalances, tracks progress toward goals, and generates actionable improvement suggestions.

### DB Writer
Handles all database operations — no LLM dependency. Exposes actions: `create_user`, `get_user`, `log_meal`, `get_meals`, `get_daily_summary`, `cache_food`, `lookup_food`, `log_message`, `get_conversation`. Acts as the single source of truth for all writes.

---

## Communication Protocols

The system implements three interchangeable protocols, selectable at runtime from the frontend or API:

| Protocol | Message Format | ID Format | Transport | Serialization |
|----------|---------------|-----------|-----------|---------------|
| **A2A** (Agent-to-Agent) | 7 fields (full metadata) | 36-char UUID | HTTP | JSON |
| **PNP** (Paolo Nutrition Protocol) | 5 fields (lean) | 8-char hex | HTTP or Direct | JSON or MessagePack |
| **TOON** | 6 fields | Short IDs | HTTP | JSON |

**PNP** is a custom-designed protocol optimized for this system. It achieves ~2x lower latency than A2A through shorter field names, compact IDs, and optional binary serialization via MessagePack. In monolith mode, PNP can bypass HTTP entirely with direct in-process function calls.

### Protocol Benchmark Results

**Text payloads — Monolith (50 iterations):**

| Protocol | Avg Latency | Throughput | vs A2A |
|----------|-------------|-----------|--------|
| A2A (HTTP+JSON) | 2,564 ms | 0.39 msg/s | baseline |
| PNP (HTTP+JSON) | 1,271 ms | 0.79 msg/s | 2.0x faster |
| PNP (HTTP+MsgPack) | 1,879 ms | 0.53 msg/s | 1.4x faster |

**Text payloads — Kubernetes containers (50 iterations):**

| Protocol | Avg Latency | Throughput | vs Monolith |
|----------|-------------|-----------|-------------|
| A2A (HTTP+JSON) | 249 ms | 4.01 msg/s | 10x faster |
| PNP (HTTP+JSON) | 78 ms | 12.76 msg/s | 16x faster |
| PNP (HTTP+MsgPack) | 79 ms | 12.66 msg/s | 24x faster |
| TOON (HTTP+JSON) | 72 ms | 13.82 msg/s | new baseline |

**Key finding:** TOON wins on containers (72ms, tightest P99 at 96ms). Containerization improves all protocols by 10-24x by eliminating monolith process contention.

---

## ML Food Classifier

The Food Logger agent includes an ML-powered image classification pipeline for identifying food from photos.

### Model Details

| Property | Value |
|----------|-------|
| **Architecture** | MobileNetV2 (transfer learning) |
| **Dataset** | Food-101 (101 categories, 101K images) |
| **Training strategy** | 2-phase: classifier head (5 epochs) + full fine-tune (10 epochs) |
| **Input size** | 224x224 RGB |
| **Model size** | ~9 MB (.pth) |
| **Framework** | PyTorch |
| **Training environment** | Google Colab (T4 GPU) |

### Training

A Google Colab notebook is provided for training:

```
1. Upload  food_classifier_training.ipynb  to Google Colab
2. Set runtime to T4 GPU
3. Run all cells (~50 min total)
4. Download output files:
   - food_classifier.pth         (model weights)
   - food_classifier_traced.pt   (TorchScript, optimized for CPU)
   - food_classes.json           (class names + config)
5. Copy files to  models/  directory
```

### Integration

The classifier is integrated into the Food Logger agent and accessible via all three protocols:

```python
# Direct Python usage
from ml_food_classifier import get_classifier

classifier = get_classifier()
result = classifier.classify(image_bytes)      # raw bytes
result = classifier.classify_base64(b64_str)   # base64 string

# Result:
# {
#     "predicted_class": "Pizza",
#     "confidence": 0.92,
#     "top_k": [...],
#     "inference_time_ms": 25.3,
#     "image_size_bytes": 45230
# }
```

### Image Protocol Benchmarking

A dedicated benchmark compares how each protocol handles image payloads (50-500KB, much larger than text):

```bash
python benchmark_ml.py --iterations 30 --export
```

This measures serialization overhead, transport latency, and total round-trip for image classification through A2A, PNP (JSON + MsgPack), and TOON. Results are automatically compared against the text-only benchmarks.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.12 |
| **Web Framework** | FastAPI 0.115 + Uvicorn |
| **LLM** | Gemini 2.0-Flash (OpenAI-compatible API) |
| **ML Framework** | PyTorch + torchvision |
| **ML Model** | MobileNetV2 (Food-101, transfer learning) |
| **Database** | PostgreSQL 16 |
| **Inter-Agent HTTP** | HTTPX (async) |
| **Binary Serialization** | MessagePack |
| **Validation** | Pydantic 2.9 |
| **Image Processing** | Pillow |
| **Reverse Proxy** | Nginx 1.27 |
| **Containerization** | Docker (Python 3.12-slim) |
| **Orchestration** | Kubernetes |
| **Frontend** | Vanilla JS + CSS (single-page app) |

---

## Getting Started

### Prerequisites

- Python 3.12+
- PostgreSQL 16+
- A Gemini API key
- (Optional) Trained ML model files in `models/`

### Local Development (without Docker)

```bash
# Clone the repository
git clone <repository-url>
cd my_product

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set GEMINI_API_KEY

# Start PostgreSQL (must be running on localhost:5432)

# Run the application
python main.py
```

The app starts at `http://localhost:8000`. The frontend is served at `http://localhost:8000/static/`.

### Docker Compose (Monolith)

```bash
docker-compose up --build
```

Starts PostgreSQL, the monolith app, and Nginx. Access the app at `http://localhost`.

### Docker Compose (Microservices)

```bash
docker-compose -f docker-compose.microservices.yml up --build
```

Starts PostgreSQL, all 5 agent containers, and Nginx. Access the app at `http://localhost`.

### ML Model Setup

```bash
# 1. Train on Google Colab (upload food_classifier_training.ipynb)
# 2. Download the trained model files
# 3. Place them in the models/ directory:

models/
  food_classifier.pth           # Model weights
  food_classifier_traced.pt     # TorchScript (faster CPU inference)
  food_classes.json             # Class names + model config

# 4. Place test food photos in test_images/ for benchmarking:

test_images/
  pizza.jpg
  sushi.png
  ...
```

---

## Deployment

### Kubernetes

```bash
# Create namespace and apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/postgres-pvc.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/orchestrator.yaml
kubectl apply -f k8s/food-logger.yaml
kubectl apply -f k8s/meal-planner.yaml
kubectl apply -f k8s/health-advisor.yaml
kubectl apply -f k8s/db-writer.yaml
kubectl apply -f k8s/nginx.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/backup-cronjob.yaml
```

**Resource allocation per agent:**

| Agent | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-------|-----------|---------|--------------|------------|
| Orchestrator | 200m | 500m | 256Mi | 512Mi |
| Food Logger | 100m | 500m | 128Mi | 512Mi |
| Meal Planner | 100m | 500m | 128Mi | 512Mi |
| Health Advisor | 100m | 500m | 128Mi | 512Mi |
| DB Writer | 100m | 500m | 128Mi | 512Mi |

**Auto-scaling (HPA):**

| Agent | Min Replicas | Max Replicas | Scale-up Threshold |
|-------|------------|------------|-------------------|
| Food Logger | 1 | 3 | 70% CPU |
| Meal Planner | 1 | 2 | 70% CPU |
| Health Advisor | 1 | 2 | 70% CPU |
| Orchestrator | 1 | 2 | 70% CPU |

**Health checks** are configured on all deployments — readiness probes at 5s intervals and liveness probes at 10s intervals, both hitting each agent's `/health` endpoint.

**Service discovery** uses Kubernetes DNS: `<service>.nutrition-tracker.svc.cluster.local`.

---

## API Reference

### Orchestrator (Port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a natural language message. The orchestrator classifies intent and routes to the appropriate agent. |
| `GET` | `/health` | Health check |
| `GET` | `/agents` | List all registered agents and their capabilities |
| `GET` | `/metrics` | System performance metrics |
| `GET` | `/traces` | Recent execution traces |
| `GET` | `/traces/{trace_id}` | Detailed trace for a specific request |

### Food Logger (Port 8001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/parse` | Parse a food description into structured nutrition data |
| `POST` | `/classify-image` | Classify a food image (multipart file upload) |
| `POST` | `/a2a/messages` | A2A protocol endpoint (text) |
| `POST` | `/a2a/classify` | A2A protocol endpoint (image classification) |
| `POST` | `/pnp` | PNP protocol endpoint (text) |
| `POST` | `/pnp/classify` | PNP protocol endpoint (image classification) |
| `POST` | `/toon/classify` | TOON protocol endpoint (image classification) |
| `GET` | `/health` | Health check (includes `ml_model_loaded` status) |

### Meal Planner (Port 8002)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate` | Generate a personalized meal plan |
| `POST` | `/a2a/messages` | A2A protocol endpoint |
| `POST` | `/pnp/messages` | PNP protocol endpoint |

### Health Advisor (Port 8003)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/advise` | Get health analysis and advice |
| `POST` | `/a2a/messages` | A2A protocol endpoint |
| `POST` | `/pnp/messages` | PNP protocol endpoint |

### DB Writer (Port 8004)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/action` | Execute a database action (`create_user`, `log_meal`, `get_meals`, etc.) |
| `POST` | `/a2a/messages` | A2A protocol endpoint |
| `POST` | `/pnp` | PNP protocol endpoint |
| `POST` | `/toon` | TOON protocol endpoint |

---

## Database Schema

PostgreSQL with 6 tables:

```
users
+-- id (serial, PK)
+-- username, email
+-- age, gender, weight, height, activity_level
+-- dietary_goals (JSON)
+-- allergies (JSON)
+-- preferences (JSON)
+-- created_at, updated_at

meals
+-- id (serial, PK)
+-- user_id (FK -> users)
+-- food_name, meal_type (breakfast/lunch/dinner/snack)
+-- quantity, unit
+-- calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg
+-- source (gemini_estimate / cache / ml_model)
+-- meal_date, created_at

nutrition_log
+-- id (serial, PK)
+-- user_id (FK -> users)
+-- log_date
+-- total_calories, total_protein_g, total_carbs_g, total_fat_g, total_fiber_g, total_sugar_g
+-- created_at

food_cache
+-- id (serial, PK)
+-- food_name, quantity, unit
+-- calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg
+-- source (open_food_facts / user_input / llm_estimate)
+-- cached_at

conversations
+-- id (serial, PK)
+-- user_id, message_id, conversation_id
+-- sender_id, message_type, payload (JSON)
+-- timestamp

agents
+-- agent_id (PK), name, url, description
+-- capabilities (JSON)
+-- status, version
+-- registered_at, last_seen
```

---

## Monitoring & Observability

### Execution Tracing

Every request receives a **trace ID** that propagates across all agent hops. Traces are written to `logs/execution_log.jsonl` in structured JSONL format:

```json
{
  "trace_id": "a1b2c3d4e5f67890",
  "timestamp": "2026-02-24T10:30:00.123Z",
  "agent": "food-logger",
  "action": "llm_parse_food",
  "protocol": "pnp",
  "duration_ms": 150.5,
  "tokens_in": 200,
  "tokens_out": 50,
  "llm_call": true,
  "seed": 42
}
```

Traces are queryable via the `/traces` and `/traces/{trace_id}` endpoints.

### Per-Request Performance Metrics

Each response includes performance data:
- Total request time (ms)
- LLM call count and token usage (input + output)
- Memory usage (MB, via psutil)
- Serialization time (microseconds)
- Step-by-step breakdown of each operation

### Structured Logging

JSON-formatted logs compatible with Docker log drivers, Loki, and CloudWatch. Enabled via `JSON_LOGS=true`.

---

## Benchmarking

The project includes three benchmark suites for measuring protocol performance across different workloads and deployment modes.

### 1. In-Process Benchmark (Monolith)

Compares all protocol/transport combinations within a single process:

```bash
python benchmark.py --iterations 50 --export
```

### 2. Container Benchmark (Kubernetes)

Measures protocol overhead across Docker containers:

```bash
docker compose -f docker-compose.microservices.yml up --build -d
python benchmark_containers.py --iterations 50 --export
```

### 3. ML Image Benchmark

Compares protocols with **image payloads** (50-500KB) for food classification:

```bash
# Place test food images in test_images/
python benchmark_ml.py --iterations 30 --export
```

If no test images are present, the benchmark generates synthetic 224x224 JPEG images automatically.

**Benchmark configurations tested:**

| # | Protocol | Transport | Serialization | Benchmark |
|---|----------|-----------|---------------|-----------|
| 1 | A2A | HTTP | JSON | text, containers, ML |
| 2 | PNP | HTTP | JSON | text, containers, ML |
| 3 | PNP | HTTP | MessagePack | text, containers, ML |
| 4 | TOON | HTTP | JSON | text, containers, ML |
| 5 | PNP | Direct | JSON | text only |
| 6 | PNP | Direct | MessagePack | text only |

**Metrics collected:** avg/p50/p95/p99 latency, throughput (msgs/sec), message size (bytes/KB), serialization overhead, error rate.

**Overall protocol rankings (containers, text payloads):**

| Rank | Protocol | Avg Latency | P99 | Throughput |
|------|----------|-------------|-----|-----------|
| 1 | TOON (HTTP+JSON) | 72 ms | 96 ms | 13.82 msg/s |
| 2 | PNP (HTTP+JSON) | 78 ms | 183 ms | 12.76 msg/s |
| 3 | PNP (HTTP+MsgPack) | 79 ms | 142 ms | 12.66 msg/s |
| 4 | A2A (HTTP+JSON) | 249 ms | 915 ms | 4.01 msg/s |

The ML image benchmark additionally compares text vs image overhead per protocol, showing how larger payloads affect each protocol differently.

---

## Backup Strategy

Automated daily PostgreSQL backups:

- **Schedule:** 3:00 AM daily (CronJob in K8s, cron in Docker)
- **Format:** Gzipped SQL dumps (`nutrition_tracker_YYYY-MM-DD_HHMMSS.sql.gz`)
- **Retention:** 7 days (automatic cleanup of older backups)
- **Location:** `/backups/` volume mount
- **Script:** `backup.sh` (pg_dump + gzip)

---

## Project Structure

```
my_product/
+-- main.py                          # FastAPI entry point
+-- config.py                        # Central configuration & agent registry
+-- database.py                      # PostgreSQL data layer
+-- llm_client.py                    # Gemini LLM client (OpenAI-compatible)
|
+-- Agents
|   +-- agent_orchestrator.py        # Intent classifier & router
|   +-- agent_food_logger.py         # Food parsing, nutrition extraction, image classification
|   +-- agent_meal_planner.py        # Meal plan generation
|   +-- agent_health_advisor.py      # Health analysis & advice
|   +-- agent_db_writer.py           # Database operations
|
+-- Protocols
|   +-- a2a_protocol.py              # Agent-to-Agent Protocol v1.0
|   +-- pnp_protocol.py              # Paolo Nutrition Protocol v1.0
|   +-- toon_protocol.py             # TOON Protocol v1.0
|
+-- ML
|   +-- ml_food_classifier.py        # MobileNetV2 inference module
|   +-- food_classifier_training.ipynb  # Google Colab training notebook
|   +-- models/                      # Trained model files (.pth, .pt, .json)
|   +-- test_images/                 # Test food images for benchmarking
|
+-- Observability
|   +-- performance.py               # Per-request performance tracking
|   +-- execution_log.py             # Structured JSONL tracing
|   +-- logging_config.py            # JSON log formatter
|
+-- Frontend
|   +-- static/
|       +-- index.html               # Single-page application
|       +-- app.js                   # UI logic & protocol switching
|       +-- style.css                # Responsive styling
|
+-- Infrastructure
|   +-- Dockerfile
|   +-- docker-compose.yml              # Monolith deployment
|   +-- docker-compose.microservices.yml # Microservice deployment
|   +-- nginx.conf                       # Nginx (monolith)
|   +-- nginx.microservices.conf         # Nginx (microservices)
|   +-- entrypoint.sh                    # Container startup
|   +-- backup.sh                        # Database backup script
|
+-- k8s/                             # Kubernetes manifests
|   +-- namespace.yaml
|   +-- secrets.yaml
|   +-- configmap.yaml
|   +-- postgres.yaml & postgres-pvc.yaml
|   +-- orchestrator.yaml
|   +-- food-logger.yaml
|   +-- meal-planner.yaml
|   +-- health-advisor.yaml
|   +-- db-writer.yaml
|   +-- nginx.yaml
|   +-- hpa.yaml
|   +-- backup-cronjob.yaml
|
+-- Testing & Benchmarks
|   +-- test_system.py               # End-to-end system tests
|   +-- test_toon.py                 # TOON protocol tests
|   +-- benchmark.py                 # Protocol benchmark (monolith)
|   +-- benchmark_containers.py      # Protocol benchmark (containers)
|   +-- benchmark_ml.py              # Protocol benchmark (image payloads)
|
+-- requirements.txt
+-- .env
+-- .gitignore
```

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | -- | Gemini API key |
| `DEPLOY_MODE` | No | `monolith` | `monolith` or `microservice` |
| `PNP_DEFAULT_MODE` | No | `direct` | `direct` (in-process) or `http` |
| `DATABASE_URL` | No | `postgresql://nutrition_user:nutrition_pass@localhost:5432/nutrition_tracker` | PostgreSQL connection string |
| `DB_HOST` | No | `localhost` | Database host |
| `DB_PORT` | No | `5432` | Database port |
| `DB_NAME` | No | `nutrition_tracker` | Database name |
| `DB_USER` | No | `nutrition_user` | Database user |
| `DB_PASSWORD` | No | `nutrition_pass` | Database password |
| `APP_HOST` | No | `0.0.0.0` | Server bind address |
| `APP_PORT` | No | `8000` | Server bind port |
| `JSON_LOGS` | No | `false` | Enable structured JSON logging |
| `ML_MODEL_PATH` | No | `models/food_classifier.pth` | Path to PyTorch model weights |
| `ML_TRACED_MODEL_PATH` | No | `models/food_classifier_traced.pt` | Path to TorchScript model |
| `ML_CLASSES_PATH` | No | `models/food_classes.json` | Path to class names JSON |
| `ORCHESTRATOR_URL` | No | `http://orchestrator:8000` | Orchestrator URL (microservice mode) |
| `FOOD_LOGGER_URL` | No | `http://food-logger:8001` | Food Logger URL (microservice mode) |
| `MEAL_PLANNER_URL` | No | `http://meal-planner:8002` | Meal Planner URL (microservice mode) |
| `HEALTH_ADVISOR_URL` | No | `http://health-advisor:8003` | Health Advisor URL (microservice mode) |
| `DB_WRITER_URL` | No | `http://db-writer:8004` | DB Writer URL (microservice mode) |

### Nginx Configuration

- **Rate limiting:** 10 requests/second per IP
- **Max upload size:** 10 MB
- **Gzip compression:** Enabled
- **Static file caching:** 7 days
- **LLM proxy timeout:** 120 seconds

---

## License

This project is proprietary. All rights reserved.
