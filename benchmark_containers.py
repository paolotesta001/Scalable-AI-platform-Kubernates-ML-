"""
benchmark_containers.py -- Cross-Container Protocol Benchmark

Measures protocol overhead when agents run in separate Docker containers
communicating over the Docker bridge network.

Benchmarks 4 HTTP protocol configurations against the DB Writer container:
  1. A2A   (HTTP + JSON)
  2. PNP   (HTTP + JSON)
  3. PNP   (HTTP + MessagePack)
  4. TOON  (HTTP + JSON)

No direct-mode benchmarks — impossible across containers (that's the point).

Loads existing monolith results (benchmark_results.json) for a unified
3-dimension comparison: Direct vs Monolith-HTTP vs Container-HTTP.

Usage:
  1. Start microservices:  docker compose -f docker-compose.microservices.yml up --build -d
  2. Run:                  python benchmark_containers.py
  3. Options:              python benchmark_containers.py --iterations 50 --warmup 5 --export
"""

import json
import time
import statistics
import argparse
import os
from typing import List, Optional
from dataclasses import dataclass, field

import httpx

# A2A imports
from a2a_protocol import create_query, A2AMessage

# PNP imports
from pnp_protocol import PNPMessage, PNPSerializer, create_request

# TOON imports
from toon_protocol import (
    create_request as toon_create_request,
    ToonSerializer,
)

# =============================================================================
# CONFIG
# =============================================================================

DB_WRITER_URL = os.getenv("DB_WRITER_CONTAINER_URL", "http://localhost:8004")
MONOLITH_RESULTS_FILE = "benchmark_results.json"
CONTAINER_RESULTS_FILE = "benchmark_results_containers.json"


# =============================================================================
# BENCHMARK DATA STRUCTURE (same as benchmark.py)
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
# BENCHMARK FUNCTIONS — all target DB Writer container at DB_WRITER_URL
# =============================================================================

def benchmark_a2a_http_json(iterations: int, warmup: int) -> BenchmarkResult:
    """A2A protocol over HTTP with JSON — targeting container."""
    result = BenchmarkResult(name="A2A (HTTP+JSON)")
    endpoint = f"{DB_WRITER_URL}/a2a/messages"

    with httpx.Client(timeout=120.0) as client:
        for i in range(warmup + iterations):
            msg = create_query(
                text=f"Benchmark message {i}",
                sender_id="benchmark",
                extra={"action": "get_user", "data": {"user_id": 1}},
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


def benchmark_pnp_http_json(iterations: int, warmup: int) -> BenchmarkResult:
    """PNP protocol over HTTP with JSON — targeting container."""
    result = BenchmarkResult(name="PNP (HTTP+JSON)")
    endpoint = f"{DB_WRITER_URL}/pnp"

    with httpx.Client(timeout=120.0) as client:
        for i in range(warmup + iterations):
            msg = create_request(
                text=f"Benchmark message {i}",
                src="benchmark",
                extra={"action": "get_user", "data": {"user_id": 1}},
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


def benchmark_pnp_http_msgpack(iterations: int, warmup: int) -> BenchmarkResult:
    """PNP protocol over HTTP with MessagePack — targeting container."""
    result = BenchmarkResult(name="PNP (HTTP+MsgPack)")
    endpoint = f"{DB_WRITER_URL}/pnp"

    with httpx.Client(timeout=120.0) as client:
        for i in range(warmup + iterations):
            msg = create_request(
                text=f"Benchmark message {i}",
                src="benchmark",
                extra={"action": "get_user", "data": {"user_id": 1}},
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


def benchmark_toon_http_json(iterations: int, warmup: int) -> BenchmarkResult:
    """TOON protocol over HTTP with JSON — targeting container."""
    result = BenchmarkResult(name="TOON (HTTP+JSON)")
    endpoint = f"{DB_WRITER_URL}/toon"

    with httpx.Client(timeout=120.0, verify=False) as client:
        for i in range(warmup + iterations):
            msg = toon_create_request(
                text=f"Benchmark message {i}",
                src="benchmark",
                extra={"action": "get_user", "data": {"user_id": 1}},
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
# REPORT — Container results + cross-dimension comparison
# =============================================================================

def print_report(results: List[BenchmarkResult], iterations: int):
    """Print formatted benchmark table for container results."""
    print(f"\n{'='*110}")
    print(f"  Container Benchmark — DB Writer via Docker Bridge Network")
    print(f"{'='*110}")
    print(f"  Target: {DB_WRITER_URL}  |  Iterations: {iterations}")
    print()

    headers = ["Protocol", "Avg(ms)", "P50(ms)", "P95(ms)", "P99(ms)",
               "Msgs/s", "Req(B)", "Resp(B)", "Ser(us)", "Errors"]
    fmt = "{:<22s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>6s}"
    print(fmt.format(*headers))
    print("-" * 110)

    for r in results:
        row = fmt.format(
            r.name,
            f"{r.avg_latency:.2f}",
            f"{r.p50_latency:.2f}",
            f"{r.p95_latency:.2f}",
            f"{r.p99_latency:.2f}",
            f"{r.throughput:.1f}",
            f"{r.avg_request_size:.0f}",
            f"{r.avg_response_size:.0f}",
            f"{r.avg_ser_time:.1f}" if r.serialization_times_us else "N/A",
            str(r.errors),
        )
        print(row)

    print("-" * 110)


def load_monolith_results(filepath: str) -> Optional[dict]:
    """Load monolith benchmark results from JSON file."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return None


def print_comparison(container_results: List[BenchmarkResult]):
    """Print unified 3-dimension comparison: Direct vs Monolith-HTTP vs Container-HTTP."""
    monolith_data = load_monolith_results(MONOLITH_RESULTS_FILE)

    print(f"\n{'='*130}")
    print(f"  Cross-Dimension Comparison: Direct vs Monolith HTTP vs Container HTTP")
    print(f"{'='*130}")

    if not monolith_data:
        print(f"\n  [!] Monolith results not found ({MONOLITH_RESULTS_FILE}).")
        print(f"      Run 'python benchmark.py --export' first to generate them.")
        print(f"      Showing container results only.\n")
        return

    # Build lookup from monolith results
    monolith_by_name = {}
    for r in monolith_data.get("results", []):
        monolith_by_name[r["name"]] = r

    # Build lookup from container results
    container_by_name = {}
    for r in container_results:
        container_by_name[r.name] = r

    # Protocol configs to compare across dimensions
    protocols = [
        ("A2A (HTTP+JSON)",     "A2A (HTTP+JSON)",     "A2A (HTTP+JSON)"),
        ("PNP (HTTP+JSON)",     "PNP (HTTP+JSON)",     "PNP (HTTP+JSON)"),
        ("PNP (HTTP+MsgPack)",  "PNP (HTTP+MsgPack)",  "PNP (HTTP+MsgPack)"),
        ("TOON (HTTP+JSON)",    "TOON (HTTP+JSON)",    "TOON (HTTP+JSON)"),
    ]

    # Direct-mode results (from monolith file)
    direct_protocols = [
        ("PNP (Direct+JSON)",    None),
        ("PNP (Direct+MsgPack)", None),
    ]

    print()
    hdr = "{:<22s} {:>14s} {:>14s} {:>14s} {:>12s}"
    print(hdr.format("Protocol", "Direct(ms)", "Monolith(ms)", "Container(ms)", "Slowdown"))
    print("-" * 130)

    # Show direct-mode results first
    for mono_name, _ in direct_protocols:
        mono = monolith_by_name.get(mono_name)
        if mono:
            print(hdr.format(
                mono_name,
                f"{mono['avg_latency_ms']:.2f}",
                "N/A",
                "N/A",
                "baseline",
            ))

    # Show HTTP results across all 3 dimensions
    for mono_name, _, cont_name in protocols:
        mono = monolith_by_name.get(mono_name)
        cont = container_by_name.get(cont_name)

        mono_avg = mono["avg_latency_ms"] if mono else None
        cont_avg = cont.avg_latency if cont else None

        # Slowdown = container / monolith
        if mono_avg and cont_avg and mono_avg > 0:
            slowdown = f"{cont_avg / mono_avg:.2f}x"
        else:
            slowdown = "N/A"

        print(hdr.format(
            cont_name if cont else mono_name,
            "N/A",
            f"{mono_avg:.2f}" if mono_avg else "N/A",
            f"{cont_avg:.2f}" if cont_avg else "N/A",
            slowdown,
        ))

    print("-" * 130)

    # Summary insight
    mono_a2a = monolith_by_name.get("A2A (HTTP+JSON)")
    cont_a2a = container_by_name.get("A2A (HTTP+JSON)")
    if mono_a2a and cont_a2a and cont_a2a.avg_latency > 0 and mono_a2a["avg_latency_ms"] > 0:
        overhead = cont_a2a.avg_latency - mono_a2a["avg_latency_ms"]
        print(f"\n  Docker bridge overhead (A2A baseline): ~{overhead:.1f}ms per request")

    print(f"\n{'='*130}\n")


# =============================================================================
# EXPORT
# =============================================================================

def export_json(results: List[BenchmarkResult], filename: str = CONTAINER_RESULTS_FILE):
    """Export container benchmark results as JSON."""
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
            "benchmark": "Container Protocol Comparison",
            "target": DB_WRITER_URL,
            "protocol": "PNP/1.0",
            "results": data,
        }, f, indent=2)
    print(f"Results exported to {filename}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Container Protocol Benchmark (Docker bridge network)"
    )
    parser.add_argument("--iterations", "-n", type=int, default=50,
                        help="Iterations per benchmark (default: 50)")
    parser.add_argument("--warmup", "-w", type=int, default=5,
                        help="Warmup iterations (default: 5)")
    parser.add_argument("--export", action="store_true",
                        help="Export results to benchmark_results_containers.json")
    parser.add_argument("--url", type=str, default=None,
                        help="Override DB Writer URL (default: http://localhost:8004)")
    args = parser.parse_args()

    global DB_WRITER_URL
    if args.url:
        DB_WRITER_URL = args.url

    print(f"\n{'='*60}")
    print(f"  Cross-Container Protocol Benchmark")
    print(f"{'='*60}")
    print(f"  Target:     {DB_WRITER_URL}")
    print(f"  Iterations: {args.iterations}  |  Warmup: {args.warmup}")

    # Check DB Writer container is reachable
    print(f"\n  Checking DB Writer health...")
    try:
        r = httpx.get(f"{DB_WRITER_URL}/health", timeout=5)
        if r.status_code == 200:
            print(f"  [OK] {r.json()}")
        else:
            print(f"  [WARN] Health check returned {r.status_code}")
    except Exception as e:
        print(f"  [ERROR] Cannot reach DB Writer at {DB_WRITER_URL}")
        print(f"          {e}")
        print(f"\n  Make sure microservices are running:")
        print(f"    docker compose -f docker-compose.microservices.yml up --build -d")
        return

    print()

    results = []
    total = 4
    step = 0

    step += 1
    print(f"  [{step}/{total}] A2A (HTTP + JSON)...")
    results.append(benchmark_a2a_http_json(args.iterations, args.warmup))

    step += 1
    print(f"  [{step}/{total}] PNP (HTTP + JSON)...")
    results.append(benchmark_pnp_http_json(args.iterations, args.warmup))

    step += 1
    print(f"  [{step}/{total}] PNP (HTTP + MessagePack)...")
    results.append(benchmark_pnp_http_msgpack(args.iterations, args.warmup))

    step += 1
    print(f"  [{step}/{total}] TOON (HTTP + JSON)...")
    results.append(benchmark_toon_http_json(args.iterations, args.warmup))

    # Print container-only results table
    print_report(results, args.iterations)

    # Print cross-dimension comparison (loads monolith results)
    print_comparison(results)

    if args.export:
        export_json(results)


if __name__ == "__main__":
    main()
