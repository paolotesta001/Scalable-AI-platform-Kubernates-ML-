"""
execution_log.py — Structured Execution Logging + Trace ID for Smart Nutrition Tracker

Provides:
    1. generate_trace_id()  — 16-char hex ID that follows a request across all agent hops
    2. log_execution()      — Appends one JSONL line per action (LLM call, agent call, DB call)
    3. get_trace_log()      — Read back all entries for a given trace_id (for debugging/replay)

Every agent imports this. Every significant action gets logged.

Output: logs/execution_log.jsonl
Format per line:
    {
        "trace_id": "a1b2c3d4e5f6g7h8",
        "timestamp": "2026-02-20T10:30:00.123456",
        "agent": "food-logger",
        "action": "llm_parse_food",
        "protocol": "a2a",
        "input": {"text": "I ate pasta"},
        "output": {"food_name": "pasta"},
        "duration_ms": 150.5,
        "tokens_in": 200,
        "tokens_out": 50,
        "llm_call": true,
        "seed": 42,
        "error": null
    }
"""

import json
import os
import uuid
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "execution_log.jsonl")
_write_lock = threading.Lock()


def _ensure_log_dir():
    """Create logs/ directory if it doesn't exist."""
    os.makedirs(_LOG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Trace ID generation
# ---------------------------------------------------------------------------

def generate_trace_id() -> str:
    """
    Generate a 16-character hex trace ID.

    This ID is created once at the entry point (orchestrator)
    and propagated through every agent hop in the request chain.
    """
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# 2. JSONL structured logging
# ---------------------------------------------------------------------------

def log_execution(
    trace_id: str,
    agent: str,
    action: str,
    *,
    protocol: str = "a2a",
    input_data: Any = None,
    output_data: Any = None,
    duration_ms: float = 0.0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    llm_call: bool = False,
    seed: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """
    Append one structured log entry to execution_log.jsonl.

    Args:
        trace_id:    The request trace ID (from generate_trace_id)
        agent:       Which agent is logging (e.g., "orchestrator", "food-logger")
        action:      What happened (e.g., "classify_task", "llm_parse_food", "db_log_meal")
        protocol:    "a2a" or "pnp"
        input_data:  What went in (truncated for safety)
        output_data: What came out (truncated for safety)
        duration_ms: How long it took in milliseconds
        tokens_in:   Estimated input tokens
        tokens_out:  Estimated output tokens
        llm_call:    Whether this was an LLM call
        seed:        LLM seed used (if llm_call)
        error:       Error message if something failed
    """
    _ensure_log_dir()

    entry = {
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "action": action,
        "protocol": protocol,
        "input": _truncate(input_data),
        "output": _truncate(output_data),
        "duration_ms": round(duration_ms, 3),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "llm_call": llm_call,
        "seed": seed,
        "error": error,
    }

    line = json.dumps(entry, default=str, ensure_ascii=False) + "\n"

    with _write_lock:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)


def _truncate(data: Any, max_len: int = 2000) -> Any:
    """Truncate large strings/dicts to keep log entries manageable."""
    if data is None:
        return None
    if isinstance(data, str):
        return data[:max_len] + ("..." if len(data) > max_len else "")
    if isinstance(data, dict):
        s = json.dumps(data, default=str, ensure_ascii=False)
        if len(s) > max_len:
            return json.loads(s[:max_len] + "}")  # best-effort truncation
        return data
    # For lists and other types, convert to string
    s = str(data)
    return s[:max_len] + ("..." if len(s) > max_len else "")


# ---------------------------------------------------------------------------
# 3. Read back logs for a trace_id
# ---------------------------------------------------------------------------

def get_trace_log(trace_id: str) -> List[Dict]:
    """
    Read all execution log entries for a given trace_id.

    Returns list of dicts, ordered by timestamp.
    Useful for debugging: "show me everything that happened for this request."
    """
    if not os.path.exists(_LOG_FILE):
        return []

    entries = []
    with open(_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("trace_id") == trace_id:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue

    return sorted(entries, key=lambda e: e.get("timestamp", ""))


def get_recent_traces(limit: int = 20) -> List[Dict]:
    """
    Get the most recent unique trace_ids with summary info.

    Returns list of {"trace_id", "timestamp", "agent", "actions_count"}.
    """
    if not os.path.exists(_LOG_FILE):
        return []

    traces: Dict[str, Dict] = {}
    with open(_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                tid = entry.get("trace_id", "")
                if tid not in traces:
                    traces[tid] = {
                        "trace_id": tid,
                        "timestamp": entry.get("timestamp", ""),
                        "first_agent": entry.get("agent", ""),
                        "first_action": entry.get("action", ""),
                        "actions_count": 0,
                        "has_error": False,
                    }
                traces[tid]["actions_count"] += 1
                if entry.get("error"):
                    traces[tid]["has_error"] = True
            except json.JSONDecodeError:
                continue

    result = sorted(traces.values(), key=lambda t: t["timestamp"], reverse=True)
    return result[:limit]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing execution_log...\n")

    tid = generate_trace_id()
    print(f"Generated trace_id: {tid} ({len(tid)} chars)")

    # Log a few entries
    log_execution(tid, "orchestrator", "classify_task",
                  protocol="a2a",
                  input_data={"text": "I ate pasta for lunch"},
                  output_data={"plan": "log_food"},
                  duration_ms=150.5, tokens_in=200, tokens_out=50,
                  llm_call=True, seed=42)

    log_execution(tid, "food-logger", "llm_parse_food",
                  protocol="a2a",
                  input_data={"text": "I ate pasta for lunch"},
                  output_data={"food_name": "pasta", "calories": 420},
                  duration_ms=800.2, tokens_in=500, tokens_out=150,
                  llm_call=True, seed=42)

    log_execution(tid, "food-logger", "db_log_meal",
                  protocol="a2a",
                  input_data={"action": "log_meal", "food_name": "pasta"},
                  output_data={"status": "ok"},
                  duration_ms=12.3)

    # Read back
    entries = get_trace_log(tid)
    print(f"\nEntries for trace {tid}: {len(entries)}")
    for e in entries:
        print(f"  [{e['agent']}] {e['action']} - {e['duration_ms']}ms"
              f" {'(LLM)' if e['llm_call'] else ''}")

    # Recent traces
    recent = get_recent_traces()
    print(f"\nRecent traces: {len(recent)}")
    for t in recent:
        print(f"  {t['trace_id']} - {t['actions_count']} actions "
              f"[{t['first_agent']}/{t['first_action']}]")

    print(f"\nLog file: {_LOG_FILE}")
    print("Execution log works!")
