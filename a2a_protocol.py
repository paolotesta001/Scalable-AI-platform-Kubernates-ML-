"""
a2a_protocol.py — Agent-to-Agent Communication Protocol
Built from scratch for Smart Nutrition Tracker.

This defines HOW agents talk to each other:
- Message format (what data is sent)
- Message types (query, response, event)
- Validation (reject bad messages)
- Sending (HTTP POST between agents)

Every agent imports this file. It's the universal language.

Protocol: A2A/1.0
Transport: HTTP POST (JSON body)
Endpoint: /a2a/messages (every agent listens here)
"""

import uuid
import httpx
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MESSAGE MODELS — The data structure agents exchange
# ═══════════════════════════════════════════════════════════════════════════════

class Sender(BaseModel):
    """
    Who sent this message?

    Fields:
        id:   Unique agent identifier (e.g., "food-logger", "orchestrator")
        role: Either "user" (human) or "assistant" (agent)
    """
    id: str = "unknown"
    role: str = "assistant"  # "user" or "assistant"


class A2AMessage(BaseModel):
    """
    The core message format for all agent communication.

    This is what gets sent as JSON between agents via HTTP POST.
    Every message has the same structure, regardless of which agent
    sends or receives it.

    Fields:
        message_id:      Unique ID for this specific message (auto-generated)
        conversation_id: Groups messages into a conversation thread
        timestamp:       When the message was created (ISO format)
        protocol:        Always "A2A/1.0" — version tracking
        sender:          Who sent it (agent ID + role)
        type:            What kind of message: "query", "response", or "event"
        payload:         The actual data (flexible dict — can contain anything)
    """
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    protocol: str = "A2A/1.0"
    sender: Sender = Field(default_factory=Sender)
    type: str = "query"  # "query" | "response" | "event"
    payload: Dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MESSAGE TYPES — What kinds of messages exist
# ═══════════════════════════════════════════════════════════════════════════════

class MessageType:
    """
    Three types of messages in A2A protocol:

    QUERY:    "Hey agent, I need something from you"
              - User asks a question
              - Orchestrator sends task to an agent
              - Agent asks another agent for data

    RESPONSE: "Here's your answer"
              - Agent returns results
              - ML model returns prediction
              - DB writer confirms data was saved

    EVENT:    "FYI, something happened"
              - Agent started processing
              - Progress update (50% done)
              - Error occurred
              - Status change
    """
    QUERY = "query"
    RESPONSE = "response"
    EVENT = "event"

    @staticmethod
    def is_valid(msg_type: str) -> bool:
        return msg_type in ("query", "response", "event")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MESSAGE BUILDERS — Helper functions to create messages easily
# ═══════════════════════════════════════════════════════════════════════════════

def create_query(
    text: str,
    sender_id: str,
    conversation_id: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> A2AMessage:
    """
    Create a QUERY message (asking an agent to do something).

    Args:
        text:            The question or task description
        sender_id:       Who is sending (e.g., "orchestrator", "food-logger")
        conversation_id: Thread ID (auto-generated if not provided)
        extra:           Additional data to include in payload

    Returns:
        A2AMessage ready to send

    Example:
        msg = create_query(
            text="What food is in this image?",
            sender_id="orchestrator",
            extra={"image_base64": "..."}
        )
    """
    payload = {"text": text}
    if extra:
        payload.update(extra)

    return A2AMessage(
        conversation_id=conversation_id or str(uuid.uuid4()),
        sender=Sender(id=sender_id, role="user"),
        type=MessageType.QUERY,
        payload=payload,
    )


def create_response(
    text: str,
    sender_id: str,
    conversation_id: str,
    extra: Optional[Dict] = None,
) -> A2AMessage:
    """
    Create a RESPONSE message (returning results).

    Args:
        text:            The answer or result
        sender_id:       Who is responding
        conversation_id: Must match the original query's conversation_id
        extra:           Additional data (e.g., nutrition facts, predictions)

    Returns:
        A2AMessage with the response

    Example:
        msg = create_response(
            text="Detected: Spaghetti Carbonara (92% confidence)",
            sender_id="ml-model",
            conversation_id=original_msg.conversation_id,
            extra={"food": "spaghetti_carbonara", "confidence": 0.92}
        )
    """
    payload = {"text": text}
    if extra:
        payload.update(extra)

    return A2AMessage(
        conversation_id=conversation_id,
        sender=Sender(id=sender_id, role="assistant"),
        type=MessageType.RESPONSE,
        payload=payload,
    )


def create_event(
    event_type: str,
    sender_id: str,
    conversation_id: str,
    data: Optional[Dict] = None,
) -> A2AMessage:
    """
    Create an EVENT message (status updates, errors, progress).

    Args:
        event_type: What happened — "started", "progress", "error", "completed"
        sender_id:  Who is reporting
        conversation_id: Thread this event belongs to
        data:       Event-specific data

    Returns:
        A2AMessage with the event

    Example:
        msg = create_event(
            event_type="progress",
            sender_id="ml-model",
            conversation_id=conv_id,
            data={"percent": 50, "status": "Classifying image..."}
        )
    """
    payload = {"event": event_type}
    if data:
        payload.update(data)

    return A2AMessage(
        conversation_id=conversation_id,
        sender=Sender(id=sender_id, role="assistant"),
        type=MessageType.EVENT,
        payload=payload,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MESSAGE SENDER — How to send messages between agents
# ═══════════════════════════════════════════════════════════════════════════════

def send_message(
    target_url: str,
    message: A2AMessage,
    timeout: float = 120.0,
) -> A2AMessage:
    """
    Send an A2A message to another agent via HTTP POST.

    This is the core communication function. When the orchestrator
    wants to ask the Food Logger something, it calls:

        response = send_message(
            "http://127.0.0.1:8001",
            my_query_message
        )

    Args:
        target_url: Base URL of the target agent (e.g., "http://127.0.0.1:8001")
        message:    The A2AMessage to send
        timeout:    Max seconds to wait for response

    Returns:
        A2AMessage — the agent's response

    Raises:
        ConnectionError: If agent is not running
        TimeoutError:    If agent takes too long
    """
    endpoint = f"{target_url}/a2a/messages"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                endpoint,
                json=message.model_dump(),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return A2AMessage(**response.json())

    except httpx.ConnectError:
        raise ConnectionError(
            f"Cannot reach agent at {target_url}. Is it running?"
        )
    except httpx.TimeoutException:
        raise TimeoutError(
            f"Agent at {target_url} took too long to respond (>{timeout}s)"
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Agent at {target_url} returned error {e.response.status_code}: "
            f"{e.response.text[:200]}"
        )


def send_query(
    target_url: str,
    text: str,
    sender_id: str,
    conversation_id: Optional[str] = None,
    extra: Optional[Dict] = None,
    timeout: float = 120.0,
) -> A2AMessage:
    """
    Shortcut: create a query and send it in one step.

    Example:
        response = send_query(
            target_url="http://127.0.0.1:8001",
            text="Log this meal: 2 eggs and toast",
            sender_id="orchestrator",
        )
        print(response.payload["text"])
    """
    message = create_query(text, sender_id, conversation_id, extra)
    return send_message(target_url, message, timeout)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VALIDATION — Ensure messages are well-formed
# ═══════════════════════════════════════════════════════════════════════════════

def validate_message(msg: A2AMessage) -> List[str]:
    """
    Check if a message is valid. Returns list of errors (empty = valid).

    Example:
        errors = validate_message(incoming_msg)
        if errors:
            return {"error": errors}
    """
    errors = []

    if not msg.message_id:
        errors.append("Missing message_id")

    if not msg.conversation_id:
        errors.append("Missing conversation_id")

    if not MessageType.is_valid(msg.type):
        errors.append(f"Invalid type '{msg.type}'. Must be: query, response, event")

    if msg.type == MessageType.QUERY and "text" not in msg.payload:
        errors.append("Query messages must have 'text' in payload")

    if not msg.sender.id or msg.sender.id == "unknown":
        errors.append("Sender ID must be set")

    if msg.sender.role not in ("user", "assistant"):
        errors.append(f"Invalid sender role '{msg.sender.role}'. Must be: user, assistant")

    return errors


def fix_defaults(msg: A2AMessage) -> A2AMessage:
    """
    Fill in missing fields with sensible defaults.
    Useful when receiving messages that might be incomplete.
    """
    if not msg.message_id:
        msg.message_id = str(uuid.uuid4())
    if not msg.conversation_id:
        msg.conversation_id = str(uuid.uuid4())
    if not msg.timestamp:
        msg.timestamp = datetime.now(timezone.utc).isoformat()
    return msg


def fix_default_ids(msg: A2AMessage) -> A2AMessage:
    """
    Replace placeholder IDs from Swagger UI with real UUIDs.
    Matches ExVenture multi-agent pattern.
    """
    if msg.message_id == "string" or len(msg.message_id) < 10:
        msg.message_id = str(uuid.uuid4())
    if msg.conversation_id == "string" or len(msg.conversation_id) < 10:
        msg.conversation_id = str(uuid.uuid4())
    return msg


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DISPLAY — Pretty-print messages for debugging
# ═══════════════════════════════════════════════════════════════════════════════

def format_message(msg: A2AMessage) -> str:
    """
    Human-readable display of a message (for terminal output).

    Example output:
        ┌─ QUERY from orchestrator ─────────────────
        │ Conv:    abc-123
        │ Text:    What food is in this image?
        │ Payload: {"text": "...", "image": "..."}
        └───────────────────────────────────────────
    """
    text = msg.payload.get("text", str(msg.payload)[:100])
    border = "─" * 50

    type_labels = {
        "query": "[OUT]",
        "response": "[IN]",
        "event": "[EVT]",
    }
    label = type_labels.get(msg.type, "[MSG]")

    return (
        f"+- {label} {msg.type.upper()} from {msg.sender.id} {border[:30]}\n"
        f"| Conv:    {msg.conversation_id[:8]}...\n"
        f"| Text:    {text[:80]}\n"
        f"| Keys:    {list(msg.payload.keys())}\n"
        f"+{border}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing A2A Protocol...\n")

    # Create a query
    q = create_query(
        text="I just ate a bowl of pasta with tomato sauce",
        sender_id="orchestrator",
        extra={"user_id": "user_001"},
    )
    print(format_message(q))

    # Validate it
    errors = validate_message(q)
    print(f"[OK] Validation: {'PASSED' if not errors else 'FAILED: ' + str(errors)}\n")

    # Create a response
    r = create_response(
        text="Logged: Pasta with tomato sauce — 420 kcal, 15g protein, 62g carbs, 8g fat",
        sender_id="food-logger",
        conversation_id=q.conversation_id,
        extra={
            "food": "pasta_tomato",
            "calories": 420,
            "protein_g": 15,
            "carbs_g": 62,
            "fat_g": 8,
        },
    )
    print(format_message(r))

    # Create an event
    e = create_event(
        event_type="meal_logged",
        sender_id="db-writer",
        conversation_id=q.conversation_id,
        data={"table": "meals", "rows_inserted": 1},
    )
    print(format_message(e))

    # Test validation on a bad message
    bad = A2AMessage(type="invalid_type", sender=Sender(id="", role="alien"))
    bad_errors = validate_message(bad)
    print(f"\n[FAIL] Bad message errors: {bad_errors}")

    print("\n[OK] A2A Protocol tests complete!")
