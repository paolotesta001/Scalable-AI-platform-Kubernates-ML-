"""
DB Writer Agent — reads/writes nutrition data to PostgreSQL.
No LLM — pure database operations. Wraps database.py functions.

Supported actions (sent in payload["action"]):
    - "create_user"       -> database.create_user(**payload["data"])
    - "get_user"          -> database.get_user(**payload["data"])
    - "log_meal"          -> database.log_meal(**payload["data"])
    - "get_meals"         -> database.get_meals(**payload["data"])
    - "get_daily_summary" -> database.get_daily_summary(**payload["data"])
    - "cache_food"        -> database.cache_food(**payload["data"])
    - "lookup_food"       -> database.lookup_food(**payload["data"])
    - "log_message"       -> database.log_message(**payload["data"])
    - "get_conversation"  -> database.get_conversation(**payload["data"])

Start with (standalone):
    uvicorn agent_db_writer:app --reload --port 8004
Or via main.py (mounted at /db-writer on port 8000).
"""
import os
print("[DB_WRITER LOADED FROM]", os.path.abspath(__file__))

import json
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Dict

from fastapi import FastAPI, HTTPException

from a2a_protocol import A2AMessage, Sender, fix_default_ids
from performance import PerformanceTracker
from config import AGENTS, agent_registry
from database import log_message as db_log_message
import database
from pnp_protocol import (
    PNPMessage, PNPType, PNPSerializer,
    pnp_registry, create_reply,
)
from fastapi import Request, Response
from toon_protocol import (
    ToonMessage, ToonType, ToonSerializer,
    create_reply as toon_create_reply,
)
app = FastAPI(
    title="DB Writer Agent",
    version="1.0.0",
    description="A2A DB Writer Agent — reads/writes nutrition data to PostgreSQL.",
)


# Map action names to database.py functions
ACTIONS = {
    "create_user": database.create_user,
    "get_user": database.get_user,
    "get_all_users": database.get_all_users,
    "update_user": database.update_user,
    "delete_user": database.delete_user,
    "log_meal": database.log_meal,
    "update_meal": database.update_meal,
    "delete_meal": database.delete_meal,
    "get_meals": database.get_meals,
    "get_daily_summary": database.get_daily_summary,
    "cache_food": database.cache_food,
    "lookup_food": database.lookup_food,
    "log_message": database.log_message,
    "get_conversation": database.get_conversation,
}


def serialize(obj):
    """Convert database results to JSON-safe format."""
    if obj is None:
        return None
    if isinstance(obj, list):
        return [serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    return obj


@app.post("/a2a/messages", response_model=A2AMessage)
def handle_message(msg: A2AMessage) -> A2AMessage:
    msg = fix_default_ids(msg)

    # Log incoming message
    try:
        db_log_message(msg.model_dump(), agent="db-writer")
    except Exception:
        pass  # Don't fail if logging fails

    if msg.type != "query":
        return A2AMessage(
            sender=Sender(id="db-writer", role="assistant"),
            conversation_id=msg.conversation_id,
            type="event",
            payload={"status": "received", "received_type": msg.type},
        )

    action = msg.payload.get("action")
    data = msg.payload.get("data", {})

    if not action:
        return A2AMessage(
            sender=Sender(id="db-writer", role="assistant"),
            conversation_id=msg.conversation_id,
            type="response",
            payload={
                "text": "Missing 'action' in payload. Supported: " + ", ".join(ACTIONS.keys()),
                "error": True,
            },
        )

    if action not in ACTIONS:
        return A2AMessage(
            sender=Sender(id="db-writer", role="assistant"),
            conversation_id=msg.conversation_id,
            type="response",
            payload={
                "text": f"Unknown action '{action}'. Supported: " + ", ".join(ACTIONS.keys()),
                "error": True,
            },
        )

    # Execute the database function
    tracker = PerformanceTracker()
    start = time.time()

    try:
        result = ACTIONS[action](**data)
        duration = time.time() - start
        tracker.log_step(f"db_{action}", json.dumps(data, default=str), str(result), duration)

        serialized = serialize(result)

        perf = tracker.get_summary()
        tracker.print_report("DB Writer Agent")

        response = A2AMessage(
            sender=Sender(id="db-writer", role="assistant"),
            conversation_id=msg.conversation_id,
            type="response",
            payload={
                "text": f"Action '{action}' completed successfully.",
                "action": action,
                "result": serialized,
                "error": False,
                "performance": perf,
            },
        )
        try:
            db_log_message(response.model_dump(), agent="db-writer")
        except Exception:
            pass
        return response

    except Exception as e:
        duration = time.time() - start
        tracker.log_step(f"db_{action}_error", json.dumps(data, default=str), str(e), duration)
        perf = tracker.get_summary()
        tracker.print_report("DB Writer Agent")

        return A2AMessage(
            sender=Sender(id="db-writer", role="assistant"),
            conversation_id=msg.conversation_id,
            type="response",
            payload={
                "text": f"Database error in '{action}': {str(e)}",
                "action": action,
                "error": True,
                "detail": str(e),
                "performance": perf,
            },
        )


# ---------------------------------------------------------------------------
# PNP Protocol Handler
# ---------------------------------------------------------------------------

def handle_pnp(msg: PNPMessage) -> PNPMessage:
    """PNP handler for DB Writer. Same action-dispatch logic as A2A."""
    if msg.t != PNPType.QUERY:
        return create_reply("received", "db-writer", msg.cid)

    action = msg.p.get("action")
    data = msg.p.get("data", {})

    if not action:
        return create_reply(
            "Missing 'action' in payload. Supported: " + ", ".join(ACTIONS.keys()),
            "db-writer", msg.cid, extra={"error": True},
        )

    if action not in ACTIONS:
        return create_reply(
            f"Unknown action '{action}'. Supported: " + ", ".join(ACTIONS.keys()),
            "db-writer", msg.cid, extra={"error": True},
        )

    tracker = PerformanceTracker()
    start = time.time()

    try:
        result = ACTIONS[action](**data)
        duration = time.time() - start
        tracker.log_step(f"db_{action}", json.dumps(data, default=str), str(result), duration)
        serialized = serialize(result)
        perf = tracker.get_summary()

        return create_reply(
            f"Action '{action}' completed successfully.", "db-writer", msg.cid,
            extra={"action": action, "result": serialized, "error": False, "performance": perf},
        )
    except Exception as e:
        duration = time.time() - start
        tracker.log_step(f"db_{action}_error", json.dumps(data, default=str), str(e), duration)
        perf = tracker.get_summary()
        return create_reply(
            f"Database error in '{action}': {str(e)}", "db-writer", msg.cid,
            extra={"action": action, "error": True, "detail": str(e), "performance": perf},
        )


def handle_toon(msg: ToonMessage) -> ToonMessage:
    """TOON handler for DB Writer. Same action-dispatch logic as A2A/PNP."""
    if msg.kind != ToonType.QUERY:
        return toon_create_reply("received", "db-writer", msg.cid)

    action = msg.body.get("action")
    data = msg.body.get("data", {})

    if not action:
        return toon_create_reply(
            "Missing 'action' in payload. Supported: " + ", ".join(ACTIONS.keys()),
            "db-writer",
            msg.cid,
            extra={"error": True},
        )

    if action not in ACTIONS:
        return toon_create_reply(
            f"Unknown action '{action}'. Supported: " + ", ".join(ACTIONS.keys()),
            "db-writer",
            msg.cid,
            extra={"error": True},
        )

    tracker = PerformanceTracker()
    start = time.time()

    try:
        result = ACTIONS[action](**data)
        duration = time.time() - start
        tracker.log_step(f"db_{action}", json.dumps(data, default=str), str(result), duration)

        serialized = serialize(result)
        perf = tracker.get_summary()

        return toon_create_reply(
            f"Action '{action}' completed successfully.",
            "db-writer",
            msg.cid,
            extra={"action": action, "result": serialized, "error": False, "performance": perf},
        )

    except Exception as e:
        duration = time.time() - start
        tracker.log_step(f"db_{action}_error", json.dumps(data, default=str), str(e), duration)
        perf = tracker.get_summary()

        return toon_create_reply(
            f"Database error in '{action}': {str(e)}",
            "db-writer",
            msg.cid,
            extra={"action": action, "error": True, "detail": str(e), "performance": perf},
        )


@app.post("/pnp")
async def pnp_endpoint(request: Request) -> Response:
    """PNP HTTP endpoint. Accepts JSON or MessagePack."""
    content_type = request.headers.get("Content-Type", "application/json")
    fmt = PNPSerializer.detect_format(content_type)
    raw = await request.body()
    msg = PNPSerializer.deserialize(raw, fmt)
    response = handle_pnp(msg)
    resp_bytes = PNPSerializer.serialize(response, fmt)
    return Response(content=resp_bytes, media_type=PNPSerializer.content_type(fmt))


# Register in PNP registry at module load
pnp_registry.register("db_writer", handle_pnp)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "agent": "db-writer"}

@app.post("/toon")
async def toon_endpoint(request: Request) -> Response:
    """TOON HTTP endpoint. Accepts JSON."""
    raw = await request.body()
    msg = ToonSerializer.deserialize(raw)
    response = handle_toon(msg)
    resp_bytes = ToonSerializer.serialize(response)
    return Response(content=resp_bytes, media_type="application/json")

# Self-register in the dynamic agent registry
agent_registry.register(
    agent_id="db_writer",
    name="DB Writer",
    url=AGENTS["db_writer"]["url"],
    path="/db-writer",
    description="Reads/writes nutrition data to PostgreSQL",
    capabilities=["db_read", "db_write", "user_management", "meal_management"],
    protocols=["a2a", "pnp", "toon"],    
    version="1.0.0",
)

