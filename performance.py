"""
performance.py — Performance Tracking for Smart Nutrition Tracker
Measures time, LLM calls, tokens, cost, and memory for every operation.

Every agent uses this to track how efficient it is.
Results are included in every A2A response for the frontend to display.

Usage in an agent:
    from performance import PerformanceTracker

    tracker = PerformanceTracker()
    # ... do work ...
    tracker.log_step("classify_food", input_text, output_text, duration)
    # ... more work ...
    metrics = tracker.get_summary()  # → dict with all metrics
"""

import time
import psutil
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from fastapi import FastAPI


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PERFORMANCE TRACKER — Tracks metrics for a single request
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Step:
    """A single tracked step within a request."""
    name: str
    input_chars: int
    output_chars: int
    duration_seconds: float
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0

    def __post_init__(self):
        # Rough estimate: 1 token ≈ 4 characters
        self.estimated_input_tokens = self.input_chars // 4
        self.estimated_output_tokens = self.output_chars // 4


class PerformanceTracker:
    """
    Tracks performance metrics for a single request lifecycle.

    Create one at the start of handling a request, log each step,
    then get the summary at the end.

    Example:
        tracker = PerformanceTracker()

        # Step 1: LLM classification
        start = time.time()
        result = llm.classify(text)
        tracker.log_step("classify", text, result, time.time() - start)

        # Step 2: Database lookup
        start = time.time()
        data = db.query(food_name)
        tracker.log_step("db_lookup", food_name, str(data), time.time() - start)

        # Get final metrics
        summary = tracker.get_summary()
        # → {"total_time": 1.5, "llm_calls": 1, "total_tokens": 450, ...}
    """

    def __init__(self):
        self.steps: list[Step] = []
        self.start_time: float = time.time()
        self.start_memory: float = self._get_memory_mb()

    @staticmethod
    def _get_memory_mb() -> float:
        """Get current process memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def log_step(
        self,
        name: str,
        input_text: str,
        output_text: str,
        duration: float,
    ) -> None:
        """
        Record a completed step.

        Args:
            name:        Step name (e.g., "llm_classify", "db_write", "ml_predict")
            input_text:  What went in (for token estimation)
            output_text: What came out
            duration:    How long it took in seconds
        """
        self.steps.append(Step(
            name=name,
            input_chars=len(input_text) if input_text else 0,
            output_chars=len(output_text) if output_text else 0,
            duration_seconds=duration,
        ))

    def get_summary(self) -> Dict[str, Any]:
        """
        Get complete performance summary.

        Returns dict with:
            total_time:    Total seconds from start to now
            llm_calls:     Number of LLM-related steps
            total_tokens:  Estimated total tokens (in + out)
            input_tokens:  Estimated input tokens
            output_tokens: Estimated output tokens
            cost_usd:      Estimated cost (Gemini Flash = free, so $0.00)
            memory_mb:     Memory usage delta in MB
            steps:         List of individual step summaries
        """
        total_time = time.time() - self.start_time
        current_memory = self._get_memory_mb()

        total_input_tokens = sum(s.estimated_input_tokens for s in self.steps)
        total_output_tokens = sum(s.estimated_output_tokens for s in self.steps)

        # Count LLM calls (steps with "llm" or "classify" or "generate" in name)
        llm_keywords = ["llm", "classify", "generate", "gemini", "predict"]
        llm_calls = sum(
            1 for s in self.steps
            if any(kw in s.name.lower() for kw in llm_keywords)
        )

        # Cost estimation (Gemini Flash is free tier)
        cost_per_input = 0.0  # Free!
        cost_per_output = 0.0
        cost = (total_input_tokens * cost_per_input +
                total_output_tokens * cost_per_output)

        return {
            "total_time": round(total_time, 3),
            "llm_calls": llm_calls,
            "total_tokens": total_input_tokens + total_output_tokens,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cost_usd": round(cost, 6),
            "memory_mb": round(current_memory - self.start_memory, 2),
            "memory_current_mb": round(current_memory, 2),
            "steps_count": len(self.steps),
            "steps": [
                {
                    "name": s.name,
                    "duration": round(s.duration_seconds, 3),
                    "tokens": s.estimated_input_tokens + s.estimated_output_tokens,
                }
                for s in self.steps
            ],
        }

    def print_report(self, agent_name: str = "Agent") -> None:
        """Print a formatted performance report to the terminal."""
        s = self.get_summary()
        print(f"\n{'-'*50}")
        print(f"[PERF] {agent_name} Performance")
        print(f"{'-'*50}")
        print(f"  Time:    {s['total_time']:.3f}s")
        print(f"  LLM:     {s['llm_calls']} calls")
        print(f"  Tokens:  {s['total_tokens']} ({s['input_tokens']} in, {s['output_tokens']} out)")
        print(f"  Cost:    ${s['cost_usd']:.6f}")
        print(f"  Memory:  {s['memory_mb']:+.2f} MB")
        if s['steps']:
            print(f"  Steps:")
            for step in s['steps']:
                print(f"     -{step['name']:30s} {step['duration']:.3f}s  ({step['tokens']} tok)")
        print(f"{'-'*50}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PERFORMANCE MIDDLEWARE — Automatically tracks every POST request
# ═══════════════════════════════════════════════════════════════════════════════
# Matches ExVenture multi-agent PerformanceMiddleware pattern.

import os
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


def get_memory_usage() -> Dict:
    """Get current process memory usage."""
    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        return {
            "rss_mb": round(mem.rss / 1024 / 1024, 2),
            "vms_mb": round(mem.vms / 1024 / 1024, 2),
        }
    except Exception:
        return {"rss_mb": 0, "vms_mb": 0}


class PerformanceMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that automatically adds timing to every response.

    Add to any agent with:
        app.add_middleware(PerformanceMiddleware, agent_name="food-logger")
    """

    def __init__(self, app, agent_name: str = "agent"):
        super().__init__(app)
        self.agent_name = agent_name

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        start_memory = get_memory_usage()

        response = await call_next(request)

        if request.method == "POST":
            duration = time.time() - start_time
            end_memory = get_memory_usage()

            response.headers["X-Processing-Time"] = f"{duration:.3f}s"
            response.headers["X-Agent"] = self.agent_name
            response.headers["X-Memory-Delta-MB"] = f"{end_memory['rss_mb'] - start_memory['rss_mb']:.2f}"

            print(f"[{self.agent_name}] Request processed in {duration:.3f}s "
                  f"(Memory: {start_memory['rss_mb']}->{end_memory['rss_mb']} MB)")

        return response


def add_performance_to_app(app: FastAPI, agent_name: str) -> None:
    """Helper to add performance middleware to any FastAPI app.

    Usage in any agent:
        from performance import add_performance_to_app
        app = FastAPI(...)
        add_performance_to_app(app, "food-logger")
    """
    app.add_middleware(PerformanceMiddleware, agent_name=agent_name)


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("[PERF] Testing Performance Tracker...\n")

    tracker = PerformanceTracker()

    # Simulate LLM classification
    import time as t
    t.sleep(0.1)
    tracker.log_step("llm_classify_food", "What food is this? pasta with sauce", "spaghetti_carbonara", 0.8)

    # Simulate DB lookup
    t.sleep(0.05)
    tracker.log_step("db_lookup_nutrition", "spaghetti_carbonara", '{"calories": 420, "protein": 15}', 0.02)

    # Simulate DB write
    t.sleep(0.02)
    tracker.log_step("db_write_meal", "INSERT INTO meals...", "OK", 0.01)

    # Print report
    tracker.print_report("Food Logger Agent")

    # Get summary (this is what goes in A2A response)
    summary = tracker.get_summary()
    print(f"Summary dict keys: {list(summary.keys())}")
    print(f"\n[OK] Performance tracking works!")
