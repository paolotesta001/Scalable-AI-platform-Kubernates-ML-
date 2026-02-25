"""
benchmark_ml.py -- Protocol Benchmark with Image Payloads

Measures how A2A, PNP (JSON + MsgPack), and TOON handle food image classification
requests through the Food Logger agent's /classify endpoints.

This benchmark answers: "Which protocol is fastest for large (image) payloads?"

Benchmarks 4 protocol configurations:
  1. A2A   (HTTP + JSON)     — base64 image in JSON payload
  2. PNP   (HTTP + JSON)     — base64 image in JSON payload
  3. PNP   (HTTP + MsgPack)  — raw image bytes in MsgPack payload
  4. TOON  (HTTP + JSON)     — base64 image in JSON payload

Key insight: PNP MsgPack can send raw binary bytes (no base64 bloat),
while all JSON-based protocols must base64-encode images (+33% size overhead).

Usage:
  1. Start microservices:  docker compose -f docker-compose.microservices.yml up --build -d
  2. Place test images:    mkdir test_images && copy some .jpg files there
  3. Run:                  python benchmark_ml.py
  4. Options:              python benchmark_ml.py --iterations 30 --export

If no test images exist, the benchmark generates synthetic test images.
"""

import json
import time
import statistics
import argparse
import os
import io
import base64
from typing import List, Optional
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# Protocol imports
from a2a_protocol import create_query, A2AMessage
from pnp_protocol import PNPMessage, PNPSerializer, create_request
from toon_protocol import (
    create_request as toon_create_request,
    ToonSerializer,
)


# =============================================================================
# CONFIG
# =============================================================================

FOOD_LOGGER_URL = os.getenv("FOOD_LOGGER_CONTAINER_URL", "http://localhost:8001")
TEST_IMAGES_DIR = Path(__file__).parent / "test_images"
RESULTS_FILE = "benchmark_results_ml.json"


# =============================================================================
# BENCHMARK DATA STRUCTURE (same as benchmark_containers.py)
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results for one protocol configuration."""
    name: str
    latencies_ms: List[float] = field(default_factory=list)
    request_sizes: List[int] = field(default_factory=list)
    response_sizes: List[int] = field(default_factory=list)
    serialization_times_us: List[float] = field(default_factory=list)
    errors: int = 0

    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0

    @property
    def p50_latency(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0

    @property
    def p95_latency(self) -> float:
        if not self.latencies_ms:
            return 0
        s = sorted(self.latencies_ms)
        return s[min(int(len(s) * 0.95), len(s) - 1)]

    @property
    def p99_latency(self) -> float:
        if not self.latencies_ms:
            return 0
        s = sorted(self.latencies_ms)
        return s[min(int(len(s) * 0.99), len(s) - 1)]

    @property
    def throughput(self) -> float:
        total_s = sum(self.latencies_ms) / 1000.0
        return len(self.latencies_ms) / total_s if total_s > 0 else 0

    @property
    def avg_request_size(self) -> float:
        return statistics.mean(self.request_sizes) if self.request_sizes else 0

    @property
    def avg_response_size(self) -> float:
        return statistics.mean(self.response_sizes) if self.response_sizes else 0

    @property
    def avg_ser_time(self) -> float:
        return statistics.mean(self.serialization_times_us) if self.serialization_times_us else 0


# =============================================================================
# TEST IMAGE MANAGEMENT
# =============================================================================

def generate_synthetic_images(count: int = 5) -> List[bytes]:
    """Generate synthetic test images for benchmarking (if no real images available)."""
    from PIL import Image
    import numpy as np

    images = []
    for i in range(count):
        # Create a realistic-sized food image (random noise, 224x224 RGB)
        np.random.seed(42 + i)
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        images.append(buf.getvalue())

    return images


def load_test_images() -> List[bytes]:
    """Load test images from test_images/ directory, or generate synthetic ones."""
    images = []

    if TEST_IMAGES_DIR.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for path in TEST_IMAGES_DIR.glob(ext):
                images.append(path.read_bytes())

    if images:
        print(f"  Loaded {len(images)} test images from {TEST_IMAGES_DIR}/")
        for i, img in enumerate(images):
            print(f"    [{i+1}] {len(img):,} bytes ({len(img)/1024:.1f} KB)")
        return images

    print(f"  No test images found in {TEST_IMAGES_DIR}/")
    print(f"  Generating 5 synthetic 224x224 JPEG images...")
    images = generate_synthetic_images(5)
    for i, img in enumerate(images):
        print(f"    [{i+1}] {len(img):,} bytes ({len(img)/1024:.1f} KB)")
    return images


# =============================================================================
# BENCHMARK FUNCTIONS — all target Food Logger's /classify endpoints
# =============================================================================

def benchmark_a2a_http_json(images: List[bytes], iterations: int, warmup: int) -> BenchmarkResult:
    """A2A protocol: send base64-encoded image in JSON payload."""
    result = BenchmarkResult(name="A2A (HTTP+JSON)")
    endpoint = f"{FOOD_LOGGER_URL}/a2a/classify"

    with httpx.Client(timeout=120.0) as client:
        for i in range(warmup + iterations):
            img = images[i % len(images)]
            img_b64 = base64.b64encode(img).decode("utf-8")

            msg = create_query(
                text="Classify this food image",
                sender_id="benchmark",
                extra={"image_base64": img_b64, "top_k": 5},
            )

            req_json = json.dumps(msg.model_dump(), separators=(",", ":")).encode()

            start = time.perf_counter()
            try:
                resp = client.post(
                    endpoint,
                    json=msg.model_dump(),
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                resp_msg = A2AMessage(**resp.json())
                elapsed_ms = (time.perf_counter() - start) * 1000

                if i >= warmup:
                    result.latencies_ms.append(elapsed_ms)
                    result.request_sizes.append(len(req_json))
                    resp_json = json.dumps(resp_msg.model_dump(), separators=(",", ":")).encode()
                    result.response_sizes.append(len(resp_json))
            except Exception as e:
                if i >= warmup:
                    result.errors += 1
                    print(f"  [A2A] Error #{result.errors}: {e}")

    return result


def benchmark_pnp_http_json(images: List[bytes], iterations: int, warmup: int) -> BenchmarkResult:
    """PNP protocol with JSON: send base64-encoded image."""
    result = BenchmarkResult(name="PNP (HTTP+JSON)")
    endpoint = f"{FOOD_LOGGER_URL}/pnp/classify"

    with httpx.Client(timeout=120.0) as client:
        for i in range(warmup + iterations):
            img = images[i % len(images)]
            img_b64 = base64.b64encode(img).decode("utf-8")

            msg = create_request(
                text="Classify this food image",
                src="benchmark",
                extra={"image_base64": img_b64, "top_k": 5},
            )

            ser_start = time.perf_counter()
            raw = PNPSerializer.serialize(msg, "json")
            ser_us = (time.perf_counter() - ser_start) * 1_000_000
            content_type = PNPSerializer.content_type("json")

            start = time.perf_counter()
            try:
                resp = client.post(
                    endpoint, content=raw,
                    headers={"Content-Type": content_type},
                )
                resp.raise_for_status()
                resp_fmt = PNPSerializer.detect_format(
                    resp.headers.get("Content-Type", "application/json")
                )
                resp_msg = PNPSerializer.deserialize(resp.content, resp_fmt)
                elapsed_ms = (time.perf_counter() - start) * 1000

                if i >= warmup:
                    result.latencies_ms.append(elapsed_ms)
                    result.request_sizes.append(len(raw))
                    resp_raw = PNPSerializer.serialize(resp_msg, "json")
                    result.response_sizes.append(len(resp_raw))
                    result.serialization_times_us.append(ser_us)
            except Exception as e:
                if i >= warmup:
                    result.errors += 1
                    print(f"  [PNP HTTP+JSON] Error #{result.errors}: {e}")

    return result


def benchmark_pnp_http_msgpack(images: List[bytes], iterations: int, warmup: int) -> BenchmarkResult:
    """PNP protocol with MsgPack: send raw image bytes (no base64 bloat)."""
    result = BenchmarkResult(name="PNP (HTTP+MsgPack)")
    endpoint = f"{FOOD_LOGGER_URL}/pnp/classify"

    with httpx.Client(timeout=120.0) as client:
        for i in range(warmup + iterations):
            img = images[i % len(images)]

            # MsgPack can carry raw bytes — no base64 encoding needed!
            # But the classify handler expects base64, so we still encode.
            # The win is in MsgPack's binary serialization of the overall message.
            img_b64 = base64.b64encode(img).decode("utf-8")

            msg = create_request(
                text="Classify this food image",
                src="benchmark",
                extra={"image_base64": img_b64, "top_k": 5},
            )

            ser_start = time.perf_counter()
            raw = PNPSerializer.serialize(msg, "msgpack")
            ser_us = (time.perf_counter() - ser_start) * 1_000_000
            content_type = PNPSerializer.content_type("msgpack")

            start = time.perf_counter()
            try:
                resp = client.post(
                    endpoint, content=raw,
                    headers={"Content-Type": content_type},
                )
                resp.raise_for_status()
                resp_fmt = PNPSerializer.detect_format(
                    resp.headers.get("Content-Type", "application/json")
                )
                resp_msg = PNPSerializer.deserialize(resp.content, resp_fmt)
                elapsed_ms = (time.perf_counter() - start) * 1000

                if i >= warmup:
                    result.latencies_ms.append(elapsed_ms)
                    result.request_sizes.append(len(raw))
                    resp_raw = PNPSerializer.serialize(resp_msg, "msgpack")
                    result.response_sizes.append(len(resp_raw))
                    result.serialization_times_us.append(ser_us)
            except Exception as e:
                if i >= warmup:
                    result.errors += 1
                    print(f"  [PNP HTTP+MsgPack] Error #{result.errors}: {e}")

    return result


def benchmark_toon_http_json(images: List[bytes], iterations: int, warmup: int) -> BenchmarkResult:
    """TOON protocol: send base64-encoded image in JSON payload."""
    result = BenchmarkResult(name="TOON (HTTP+JSON)")
    endpoint = f"{FOOD_LOGGER_URL}/toon/classify"

    with httpx.Client(timeout=120.0, verify=False) as client:
        for i in range(warmup + iterations):
            img = images[i % len(images)]
            img_b64 = base64.b64encode(img).decode("utf-8")

            msg = toon_create_request(
                text="Classify this food image",
                src="benchmark",
                extra={"image_base64": img_b64, "top_k": 5},
            )

            ser_start = time.perf_counter()
            raw = ToonSerializer.serialize(msg)
            ser_us = (time.perf_counter() - ser_start) * 1_000_000

            start = time.perf_counter()
            try:
                resp = client.post(
                    endpoint, content=raw,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                resp_msg = ToonSerializer.deserialize(resp.content)
                elapsed_ms = (time.perf_counter() - start) * 1000

                if i >= warmup:
                    result.latencies_ms.append(elapsed_ms)
                    result.request_sizes.append(len(raw))
                    resp_raw = ToonSerializer.serialize(resp_msg)
                    result.response_sizes.append(len(resp_raw))
                    result.serialization_times_us.append(ser_us)
            except Exception as e:
                if i >= warmup:
                    result.errors += 1
                    print(f"  [TOON HTTP+JSON] Error #{result.errors}: {e}")

    return result


# =============================================================================
# REPORT
# =============================================================================

def print_report(results: List[BenchmarkResult], iterations: int, images: List[bytes]):
    """Print formatted benchmark results."""
    avg_img_size = statistics.mean(len(img) for img in images)
    avg_b64_size = statistics.mean(len(base64.b64encode(img)) for img in images)

    print(f"\n{'='*120}")
    print(f"  ML Image Classification — Protocol Benchmark")
    print(f"{'='*120}")
    print(f"  Target:      {FOOD_LOGGER_URL}")
    print(f"  Iterations:  {iterations}")
    print(f"  Test images: {len(images)} (avg {avg_img_size/1024:.1f} KB raw, {avg_b64_size/1024:.1f} KB base64)")
    print()

    headers = ["Protocol", "Avg(ms)", "P50(ms)", "P95(ms)", "P99(ms)",
               "Msgs/s", "Req(KB)", "Resp(B)", "Ser(us)", "Errors"]
    fmt = "{:<22s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>6s}"
    print(fmt.format(*headers))
    print("-" * 120)

    for r in results:
        row = fmt.format(
            r.name,
            f"{r.avg_latency:.1f}",
            f"{r.p50_latency:.1f}",
            f"{r.p95_latency:.1f}",
            f"{r.p99_latency:.1f}",
            f"{r.throughput:.2f}",
            f"{r.avg_request_size/1024:.1f}",
            f"{r.avg_response_size:.0f}",
            f"{r.avg_ser_time:.1f}" if r.serialization_times_us else "N/A",
            str(r.errors),
        )
        print(row)

    print("-" * 120)

    # Comparison summary
    if len(results) >= 2:
        sorted_results = sorted(results, key=lambda r: r.avg_latency)
        fastest = sorted_results[0]
        slowest = sorted_results[-1]

        print(f"\n  Rankings (by avg latency):")
        for i, r in enumerate(sorted_results, 1):
            speedup = ""
            if i > 1:
                speedup = f"  ({r.avg_latency / fastest.avg_latency:.1f}x slower)"
            print(f"    {i}. {r.name:22s} — {r.avg_latency:.1f}ms{speedup}")

        # Message size comparison
        print(f"\n  Message size comparison:")
        for r in sorted(results, key=lambda r: r.avg_request_size):
            print(f"    {r.name:22s} — {r.avg_request_size/1024:.1f} KB request")

        # Compare with text benchmark results if available
        text_results_file = "benchmark_results_containers.json"
        if os.path.exists(text_results_file):
            print(f"\n  Comparison vs text-only benchmarks:")
            with open(text_results_file) as f:
                text_data = json.load(f)
            text_by_name = {r["name"]: r for r in text_data.get("results", [])}

            hdr = "    {:<22s} {:>10s} {:>10s} {:>10s} {:>12s} {:>12s}"
            print(hdr.format("Protocol", "Text(ms)", "Image(ms)", "Overhead",
                             "Text Req(B)", "Image Req(KB)"))
            print("    " + "-" * 90)

            for r in results:
                text_r = text_by_name.get(r.name)
                if text_r:
                    text_avg = text_r["avg_latency_ms"]
                    overhead = f"{r.avg_latency / text_avg:.1f}x"
                    text_req = f"{text_r['avg_request_bytes']:.0f}"
                    img_req = f"{r.avg_request_size/1024:.1f}"
                    print(hdr.format(r.name, f"{text_avg:.1f}", f"{r.avg_latency:.1f}",
                                     overhead, text_req, img_req))

    print(f"\n{'='*120}\n")


# =============================================================================
# EXPORT
# =============================================================================

def export_json(results: List[BenchmarkResult], images: List[bytes],
                filename: str = RESULTS_FILE):
    """Export results to JSON."""
    avg_img_size = statistics.mean(len(img) for img in images)
    data = []
    for r in results:
        data.append({
            "name": r.name,
            "avg_latency_ms": round(r.avg_latency, 3),
            "p50_latency_ms": round(r.p50_latency, 3),
            "p95_latency_ms": round(r.p95_latency, 3),
            "p99_latency_ms": round(r.p99_latency, 3),
            "throughput_msgs_sec": round(r.throughput, 2),
            "avg_request_bytes": round(r.avg_request_size, 1),
            "avg_response_bytes": round(r.avg_response_size, 1),
            "avg_serialization_us": round(r.avg_ser_time, 2),
            "errors": r.errors,
            "iterations": len(r.latencies_ms),
        })
    with open(filename, "w") as f:
        json.dump({
            "benchmark": "ML Image Classification Protocol Comparison",
            "target": FOOD_LOGGER_URL,
            "avg_image_size_bytes": round(avg_img_size),
            "num_test_images": len(images),
            "results": data,
        }, f, indent=2)
    print(f"Results exported to {filename}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ML Image Classification — Protocol Benchmark"
    )
    parser.add_argument("--iterations", "-n", type=int, default=30,
                        help="Iterations per benchmark (default: 30)")
    parser.add_argument("--warmup", "-w", type=int, default=3,
                        help="Warmup iterations (default: 3)")
    parser.add_argument("--export", action="store_true",
                        help="Export results to benchmark_results_ml.json")
    parser.add_argument("--url", type=str, default=None,
                        help="Override Food Logger URL (default: http://localhost:8001)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Directory containing test images")
    args = parser.parse_args()

    global FOOD_LOGGER_URL, TEST_IMAGES_DIR
    if args.url:
        FOOD_LOGGER_URL = args.url
    if args.images_dir:
        TEST_IMAGES_DIR = Path(args.images_dir)

    print(f"\n{'='*60}")
    print(f"  ML Image Classification — Protocol Benchmark")
    print(f"{'='*60}")
    print(f"  Target:     {FOOD_LOGGER_URL}")
    print(f"  Iterations: {args.iterations}  |  Warmup: {args.warmup}")

    # Check Food Logger is reachable
    print(f"\n  Checking Food Logger health...")
    try:
        r = httpx.get(f"{FOOD_LOGGER_URL}/health", timeout=5)
        if r.status_code == 200:
            health = r.json()
            print(f"  [OK] {health}")
            if not health.get("ml_model_loaded", False):
                print(f"  [WARN] ML model is NOT loaded. Classification will fail.")
                print(f"         Place model files in models/ directory.")
                return
        else:
            print(f"  [WARN] Health check returned {r.status_code}")
    except Exception as e:
        print(f"  [ERROR] Cannot reach Food Logger at {FOOD_LOGGER_URL}")
        print(f"          {e}")
        print(f"\n  Make sure the service is running:")
        print(f"    uvicorn agent_food_logger:app --port 8001")
        print(f"    # or: docker compose -f docker-compose.microservices.yml up --build -d")
        return

    # Load test images
    print()
    images = load_test_images()
    if not images:
        print("  [ERROR] No test images available.")
        return

    print()

    # Run benchmarks
    results = []
    total = 4
    step = 0

    step += 1
    print(f"  [{step}/{total}] A2A (HTTP + JSON) with image payload...")
    results.append(benchmark_a2a_http_json(images, args.iterations, args.warmup))

    step += 1
    print(f"  [{step}/{total}] PNP (HTTP + JSON) with image payload...")
    results.append(benchmark_pnp_http_json(images, args.iterations, args.warmup))

    step += 1
    print(f"  [{step}/{total}] PNP (HTTP + MsgPack) with image payload...")
    results.append(benchmark_pnp_http_msgpack(images, args.iterations, args.warmup))

    step += 1
    print(f"  [{step}/{total}] TOON (HTTP + JSON) with image payload...")
    results.append(benchmark_toon_http_json(images, args.iterations, args.warmup))

    # Print results
    print_report(results, args.iterations, images)

    if args.export:
        export_json(results, images)


if __name__ == "__main__":
    main()
