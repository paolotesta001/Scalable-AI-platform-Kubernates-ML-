"""
benchmark.py -- Protocol Benchmark: A2A vs PNP vs TOON

Compares 6 protocol configurations side-by-side:
  1. A2A   (HTTP + JSON)           -- baseline
  2. PNP   (HTTP + JSON)           -- PNP over HTTP, minimal JSON
  3. PNP   (HTTP + MessagePack)    -- PNP over HTTP, binary format
  4. TOON  (HTTP + JSON)           -- TOON protocol over HTTP
  5. PNP   (Direct + JSON)         -- PNP direct call, JSON serialization
  6. PNP   (Direct + MessagePack)  -- PNP direct call, binary

Metrics measured:
  - Latency: avg, p50, p95, p99 (milliseconds)
  - Throughput: messages per second
  - Message size: bytes (request + response)
  - Serialization time: encode + decode (microseconds)
  - Error count

Usage:
  1. Start app:   uvicorn main:app --reload --port 8000
  2. Run:         python benchmark.py
  3. Options:     python benchmark.py --iterations 100 --warmup 5 --export

Note: HTTP benchmarks (configs 1-4) require the server running.
      Direct benchmarks (configs 5-6) work standalone (no server needed).
"""

import json
import time
import statistics
import argparse
from typing import List
from dataclasses import dataclass, field
from toon_protocol import create_request as toon_create_request, ToonSerializer, toon_transport
import httpx

# A2A imports
from a2a_protocol import create_query, send_message, A2AMessage

# PNP imports
from pnp_protocol import (
    PNPMessage, PNPSerializer, PNPTransport,
    pnp_registry, pnp_transport,
    create_request, create_reply, send_pnp,
)
from config import APP_BASE, get_agent_url


# =============================================================================
# BENCHMARK DATA STRUCTURE
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
# BENCHMARK FUNCTIONS
# =============================================================================

def benchmark_a2a_http_json(iterations: int, warmup: int) -> BenchmarkResult:
    """Benchmark 1: A2A protocol over HTTP with JSON."""
    result = BenchmarkResult(name="A2A (HTTP+JSON)")

    for i in range(warmup + iterations):
        msg = create_query(
            text=f"Benchmark message {i}",
            sender_id="benchmark",
            extra={"action": "get_user", "data": {"user_id": 1}},
        )

        req_json = json.dumps(msg.model_dump(), separators=(",", ":")).encode()

        start = time.perf_counter()
        try:
            resp = send_message(get_agent_url("db_writer"), msg)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if i >= warmup:
                result.latencies_ms.append(elapsed_ms)
                result.request_sizes.append(len(req_json))
                resp_json = json.dumps(resp.model_dump(), separators=(",", ":")).encode()
                result.response_sizes.append(len(resp_json))
        except Exception as e:
            if i >= warmup:
                result.errors += 1
                print(f"  [A2A] Error #{result.errors}: {e}")

    return result


def benchmark_pnp_http_json(iterations: int, warmup: int) -> BenchmarkResult:
    """Benchmark 2: PNP protocol over HTTP with JSON."""
    result = BenchmarkResult(name="PNP (HTTP+JSON)")

    for i in range(warmup + iterations):
        msg = create_request(
            text=f"Benchmark message {i}",
            src="benchmark",
            extra={"action": "get_user", "data": {"user_id": 1}},
        )

        ser_start = time.perf_counter()
        raw = PNPSerializer.serialize(msg, "json")
        ser_us = (time.perf_counter() - ser_start) * 1_000_000

        start = time.perf_counter()
        try:
            resp = pnp_transport.send(get_agent_url("db_writer"), msg, mode="http", fmt="json")
            elapsed_ms = (time.perf_counter() - start) * 1000

            if i >= warmup:
                result.latencies_ms.append(elapsed_ms)
                result.request_sizes.append(len(raw))
                resp_raw = PNPSerializer.serialize(resp, "json")
                result.response_sizes.append(len(resp_raw))
                result.serialization_times_us.append(ser_us)
        except Exception as e:
            if i >= warmup:
                result.errors += 1
                print(f"  [PNP HTTP+JSON] Error #{result.errors}: {e}")

    return result


def benchmark_pnp_http_msgpack(iterations: int, warmup: int) -> BenchmarkResult:
    """Benchmark 3: PNP protocol over HTTP with MessagePack."""
    result = BenchmarkResult(name="PNP (HTTP+MsgPack)")

    for i in range(warmup + iterations):
        msg = create_request(
            text=f"Benchmark message {i}",
            src="benchmark",
            extra={"action": "get_user", "data": {"user_id": 1}},
        )

        ser_start = time.perf_counter()
        raw = PNPSerializer.serialize(msg, "msgpack")
        ser_us = (time.perf_counter() - ser_start) * 1_000_000

        start = time.perf_counter()
        try:
            resp = pnp_transport.send(get_agent_url("db_writer"), msg, mode="http", fmt="msgpack")
            elapsed_ms = (time.perf_counter() - start) * 1000

            if i >= warmup:
                result.latencies_ms.append(elapsed_ms)
                result.request_sizes.append(len(raw))
                resp_raw = PNPSerializer.serialize(resp, "msgpack")
                result.response_sizes.append(len(resp_raw))
                result.serialization_times_us.append(ser_us)
        except Exception as e:
            if i >= warmup:
                result.errors += 1
                print(f"  [PNP HTTP+MsgPack] Error #{result.errors}: {e}")

    return result


def benchmark_pnp_direct_json(iterations: int, warmup: int) -> BenchmarkResult:
    """Benchmark 4: PNP protocol direct call with JSON serialization."""
    result = BenchmarkResult(name="PNP (Direct+JSON)")

    for i in range(warmup + iterations):
        msg = create_request(
            text=f"Benchmark message {i}",
            src="benchmark",
            extra={"action": "get_user", "data": {"user_id": 1}},
        )

        ser_start = time.perf_counter()
        raw = PNPSerializer.serialize(msg, "json")
        ser_us = (time.perf_counter() - ser_start) * 1_000_000

        start = time.perf_counter()
        try:
            resp = pnp_transport.send("db_writer", msg, mode="direct", fmt="json")
            elapsed_ms = (time.perf_counter() - start) * 1000

            if i >= warmup:
                result.latencies_ms.append(elapsed_ms)
                result.request_sizes.append(len(raw))
                resp_raw = PNPSerializer.serialize(resp, "json")
                result.response_sizes.append(len(resp_raw))
                result.serialization_times_us.append(ser_us)
        except Exception as e:
            if i >= warmup:
                result.errors += 1
                print(f"  [PNP Direct+JSON] Error #{result.errors}: {e}")

    return result


def benchmark_pnp_direct_msgpack(iterations: int, warmup: int) -> BenchmarkResult:
    """Benchmark 5: PNP protocol direct call with MessagePack."""
    result = BenchmarkResult(name="PNP (Direct+MsgPack)")

    for i in range(warmup + iterations):
        msg = create_request(
            text=f"Benchmark message {i}",
            src="benchmark",
            extra={"action": "get_user", "data": {"user_id": 1}},
        )

        ser_start = time.perf_counter()
        raw = PNPSerializer.serialize(msg, "msgpack")
        ser_us = (time.perf_counter() - ser_start) * 1_000_000

        start = time.perf_counter()
        try:
            resp = pnp_transport.send("db_writer", msg, mode="direct", fmt="msgpack")
            elapsed_ms = (time.perf_counter() - start) * 1000

            if i >= warmup:
                result.latencies_ms.append(elapsed_ms)
                result.request_sizes.append(len(raw))
                resp_raw = PNPSerializer.serialize(resp, "msgpack")
                result.response_sizes.append(len(resp_raw))
                result.serialization_times_us.append(ser_us)
        except Exception as e:
            if i >= warmup:
                result.errors += 1
                print(f"  [PNP Direct+MsgPack] Error #{result.errors}: {e}")

    return result

def benchmark_toon_http_json(iterations: int, warmup: int) -> BenchmarkResult:
    """Benchmark: TOON protocol over HTTP with JSON."""
    result = BenchmarkResult(name="TOON (HTTP+JSON)")

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
            resp = toon_transport.send_http(get_agent_url("db_writer"), msg)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if i >= warmup:
                result.latencies_ms.append(elapsed_ms)
                result.request_sizes.append(len(raw))
                resp_raw = ToonSerializer.serialize(resp)
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

def print_report(results: List[BenchmarkResult], iterations: int):
    """Print formatted benchmark comparison table."""
    print(f"\n{'='*110}")
    print(f"  A2A vs PNP vs TOON - Protocol Benchmark Report")
    print(f"{'='*110}")
    print(f"  Iterations per config: {iterations}")
    print()

    # Header
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

    # Speedup comparison vs baseline (A2A is always first if present)
    baseline = None
    for r in results:
        if "A2A" in r.name:
            baseline = r
            break

    if baseline and baseline.avg_latency > 0 and len(results) > 1:
        print(f"\n  Comparison vs {baseline.name} baseline:")
        print(f"  {'-'*70}")
        for r in results:
            if r is baseline:
                continue
            if r.avg_latency > 0:
                speedup = baseline.avg_latency / r.avg_latency
                size_pct = (1 - r.avg_request_size / baseline.avg_request_size) * 100 if baseline.avg_request_size > 0 else 0
                print(f"    {r.name:22s}  {speedup:>6.2f}x faster  |  {size_pct:>5.1f}% smaller messages")
        print(f"  {'-'*70}")
    elif len(results) > 1:
        # No A2A baseline — compare against first result
        first = results[0]
        print(f"\n  Comparison vs {first.name}:")
        print(f"  {'-'*70}")
        for r in results[1:]:
            if r.avg_latency > 0 and first.avg_latency > 0:
                speedup = first.avg_latency / r.avg_latency
                size_pct = (1 - r.avg_request_size / first.avg_request_size) * 100 if first.avg_request_size > 0 else 0
                print(f"    {r.name:22s}  {speedup:>6.2f}x faster  |  {size_pct:>5.1f}% smaller messages")
        print(f"  {'-'*70}")

    print(f"\n{'='*110}\n")


def export_json(results: List[BenchmarkResult], filename: str = "benchmark_results.json"):
    """Export results as JSON for further analysis."""
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
        json.dump({"benchmark": "A2A vs PNP vs TOON", "protocol": "PNP/1.0", "results": data}, f, indent=2)
    print(f"Results exported to {filename}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="A2A vs PNP vs TOON Protocol Benchmark")
    parser.add_argument("--iterations", "-n", type=int, default=50,
                        help="Iterations per benchmark (default: 50)")
    parser.add_argument("--warmup", "-w", type=int, default=5,
                        help="Warmup iterations (default: 5)")
    parser.add_argument("--export", action="store_true",
                        help="Export results to benchmark_results.json")
    parser.add_argument("--direct-only", action="store_true",
                        help="Run only direct-mode benchmarks (no server needed)")
    parser.add_argument("--http-only", action="store_true",
                        help="Run only HTTP-mode benchmarks (server must be running)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  A2A vs PNP vs TOON - Protocol Benchmark")
    print(f"{'='*60}")
    print(f"  Iterations: {args.iterations}  |  Warmup: {args.warmup}")
    print(f"  Server: {APP_BASE}")

    # Ensure PNP agents are registered (import triggers registration)
    from agent_db_writer import handle_pnp as _db_pnp  # noqa: F401

    print(f"  PNP registry: {pnp_registry.list_agents()}")
    print()

    results = []

    run_http = not args.direct_only
    run_direct = not args.http_only

    # Check server is reachable for HTTP benchmarks
    if run_http:
        try:
            r = httpx.get(f"{APP_BASE}/health", timeout=3)
            if r.status_code != 200:
                print("  Server not healthy. Skipping HTTP benchmarks.")
                run_http = False
        except Exception:
            print(f"  Server not reachable at {APP_BASE}. Skipping HTTP benchmarks.")
            print(f"  Start with: uvicorn main:app --reload --port 8000")
            run_http = False
        print()

    total = (4 if run_http else 0) + (2 if run_direct else 0)
    step = 0

    if run_http:
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

    if run_direct:
        step += 1
        print(f"  [{step}/{total}] PNP (Direct + JSON)...")
        results.append(benchmark_pnp_direct_json(args.iterations, args.warmup))

        step += 1
        print(f"  [{step}/{total}] PNP (Direct + MessagePack)...")
        results.append(benchmark_pnp_direct_msgpack(args.iterations, args.warmup))

    if results:
        print_report(results, args.iterations)
        if args.export:
            export_json(results)
    else:
        print("\n  No benchmarks were run. Use --direct-only or start the server.")


if __name__ == "__main__":
    main()
