"""
pnp_protocol.py -- Paolo Nutrition Protocol v1.0

Custom inter-agent communication protocol for Smart Nutrition Tracker.
Designed from scratch for benchmarking against A2A and MCP protocols.

Key differences from A2A:
    - 5 lean fields vs 7 (no timestamp, no protocol version, flat sender)
    - Short field names: id, cid, src, t, p
    - 8-char hex IDs vs 36-char UUIDs
    - Single-char message types: "q", "r", "e"
    - Dual transport: direct function calls (in-process) + HTTP fallback
    - Dual serialization: minimal JSON + MessagePack (binary)

Usage:
    from pnp_protocol import (
        PNPMessage, create_request, create_reply, send_pnp,
        pnp_registry, PNPSerializer,
    )

    # Direct mode (in-process, no HTTP)
    resp = send_pnp("food_logger", msg, mode="direct", fmt="json")

    # HTTP mode (network)
    resp = send_pnp("http://127.0.0.1:8000/food-logger", msg, mode="http", fmt="msgpack")
"""

import json
import uuid
import time
from typing import Optional, Any, Dict, List, Callable

import httpx
import msgpack
from pydantic import BaseModel, Field


# =============================================================================
# CONSTANTS
# =============================================================================

PNP_VERSION = "PNP/1.0"
PNP_CONTENT_TYPE_JSON = "application/json"
PNP_CONTENT_TYPE_MSGPACK = "application/x-msgpack"


# =============================================================================
# 1. MESSAGE MODEL -- Lean 5-field structure
# =============================================================================

def _short_id() -> str:
    """Generate 8-character hex ID (vs A2A's 36-char UUID4)."""
    return uuid.uuid4().hex[:8]


def _short_cid() -> str:
    """Generate 12-character hex conversation ID."""
    return uuid.uuid4().hex[:12]


class PNPMessage(BaseModel):
    """
    PNP message format -- 5 fields, minimal overhead.

    Fields:
        id:  Short message identifier (8 hex chars)
        cid: Conversation ID (12 hex chars)
        src: Source agent ID (flat string, no nested object)
        t:   Message type: "q" (query), "r" (response), "e" (event)
        p:   Payload dict -- the actual data
    """
    id: str = Field(default_factory=_short_id)
    cid: str = Field(default_factory=_short_cid)
    src: str = "unknown"
    t: str = "q"
    p: Dict[str, Any] = Field(default_factory=dict)


class PNPType:
    """Message type constants -- single character for minimal overhead."""
    QUERY = "q"
    RESPONSE = "r"
    EVENT = "e"

    _VALID = {"q", "r", "e"}

    @staticmethod
    def is_valid(t: str) -> bool:
        return t in PNPType._VALID


# =============================================================================
# 2. AGENT REGISTRY -- In-memory registry for direct (no-HTTP) calls
# =============================================================================

AgentHandler = Callable[[PNPMessage], PNPMessage]


class AgentRegistry:
    """
    In-memory registry of agent handler functions.

    When all agents run in the same process (via main.py),
    each agent registers its handler here at import time.
    Direct-mode send() looks up the handler and calls it
    as a Python function -- zero HTTP overhead.

    Usage:
        registry = AgentRegistry()
        registry.register("food_logger", food_logger_handle_pnp)
        handler = registry.get("food_logger")
        response = handler(my_message)
    """

    def __init__(self):
        self._agents: Dict[str, AgentHandler] = {}

    def register(self, agent_id: str, handler: AgentHandler) -> None:
        self._agents[agent_id] = handler
        print(f"[PNP] Registered agent: {agent_id}")

    def unregister(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    def get(self, agent_id: str) -> Optional[AgentHandler]:
        return self._agents.get(agent_id)

    def has(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

    def clear(self) -> None:
        self._agents.clear()


# Global singleton registry
pnp_registry = AgentRegistry()


# =============================================================================
# 3. SERIALIZER -- Dual format: JSON + MessagePack
# =============================================================================

class PNPSerializer:
    """
    Dual-format serializer: minimal JSON and MessagePack (binary).

    MessagePack is ~30-50% smaller than JSON and faster to parse.
    JSON is human-readable and better for debugging.
    """

    @staticmethod
    def serialize(msg: PNPMessage, fmt: str = "json") -> bytes:
        """Serialize a PNPMessage to bytes."""
        data = msg.model_dump()
        if fmt == "msgpack":
            return msgpack.packb(data, use_bin_type=True)
        # Minimal JSON: no extra whitespace
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def deserialize(raw: bytes, fmt: str = "json") -> PNPMessage:
        """Deserialize bytes to a PNPMessage."""
        if fmt == "msgpack":
            data = msgpack.unpackb(raw, raw=False)
        else:
            data = json.loads(raw.decode("utf-8"))
        return PNPMessage(**data)

    @staticmethod
    def content_type(fmt: str) -> str:
        """Get HTTP Content-Type header for a format."""
        if fmt == "msgpack":
            return PNP_CONTENT_TYPE_MSGPACK
        return PNP_CONTENT_TYPE_JSON

    @staticmethod
    def detect_format(content_type: str) -> str:
        """Detect serialization format from Content-Type header."""
        if "msgpack" in content_type:
            return "msgpack"
        return "json"


# =============================================================================
# 4. TRANSPORT -- Direct (in-process) + HTTP
# =============================================================================

class PNPTransport:
    """
    Unified transport: send via direct function call or HTTP.

    Direct mode:
        Looks up agent handler in pnp_registry.
        Calls it as a Python function (zero network overhead).
        Serializes/deserializes for message isolation and fair benchmarking.

    HTTP mode:
        POST to target_url/pnp endpoint.
        Supports both JSON and MessagePack Content-Type.
    """

    def __init__(self, timeout: float = 120.0):
        self.timeout = timeout

    def send(
        self,
        target: str,
        message: PNPMessage,
        mode: str = "direct",
        fmt: str = "json",
    ) -> PNPMessage:
        """
        Send a PNP message to a target agent.

        Args:
            target: Agent ID (direct mode) or base URL (HTTP mode)
            message: The PNPMessage to send
            mode:   "direct" (in-process) or "http" (network)
            fmt:    "json" or "msgpack"

        Returns:
            PNPMessage response

        Raises:
            ConnectionError: Agent not found or unreachable
            TimeoutError:    HTTP timeout
            RuntimeError:    HTTP error from agent
        """
        if mode == "direct":
            return self._send_direct(target, message, fmt)
        return self._send_http(target, message, fmt)

    def _send_direct(self, agent_id: str, message: PNPMessage, fmt: str) -> PNPMessage:
        """Direct in-process call via registry lookup."""
        handler = pnp_registry.get(agent_id)
        if handler is None:
            raise ConnectionError(
                f"[PNP] Agent '{agent_id}' not in registry. "
                f"Registered: {pnp_registry.list_agents()}"
            )

        # Serialize/deserialize for message isolation and fair benchmarking
        raw = PNPSerializer.serialize(message, fmt)
        msg_copy = PNPSerializer.deserialize(raw, fmt)

        # Direct function call -- no HTTP
        response = handler(msg_copy)

        # Serialize response too (fair benchmark)
        resp_raw = PNPSerializer.serialize(response, fmt)
        return PNPSerializer.deserialize(resp_raw, fmt)

    def _send_http(self, target_url: str, message: PNPMessage, fmt: str) -> PNPMessage:
        """HTTP POST to target_url/pnp endpoint."""
        endpoint = f"{target_url}/pnp"
        raw = PNPSerializer.serialize(message, fmt)
        content_type = PNPSerializer.content_type(fmt)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    endpoint,
                    content=raw,
                    headers={"Content-Type": content_type},
                )
                response.raise_for_status()

                resp_fmt = PNPSerializer.detect_format(
                    response.headers.get("Content-Type", "application/json")
                )
                return PNPSerializer.deserialize(response.content, resp_fmt)

        except httpx.ConnectError:
            raise ConnectionError(
                f"[PNP] Cannot reach agent at {endpoint}. Is it running?"
            )
        except httpx.TimeoutException:
            raise TimeoutError(
                f"[PNP] Agent at {endpoint} timed out (>{self.timeout}s)"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"[PNP] HTTP {e.response.status_code} from {endpoint}: "
                f"{e.response.text[:200]}"
            )


# Global transport instance
pnp_transport = PNPTransport()


# =============================================================================
# 5. HELPER FUNCTIONS -- Message builders + convenience senders
# =============================================================================

def create_request(
    text: str,
    src: str,
    cid: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> PNPMessage:
    """Create a query/request message (PNP equivalent of A2A's create_query)."""
    p = {"text": text}
    if extra:
        p.update(extra)
    return PNPMessage(
        cid=cid or _short_cid(),
        src=src,
        t=PNPType.QUERY,
        p=p,
    )


def create_reply(
    text: str,
    src: str,
    cid: str,
    extra: Optional[Dict] = None,
) -> PNPMessage:
    """Create a response/reply message (PNP equivalent of A2A's create_response)."""
    p = {"text": text}
    if extra:
        p.update(extra)
    return PNPMessage(
        cid=cid,
        src=src,
        t=PNPType.RESPONSE,
        p=p,
    )


def create_event(
    event: str,
    src: str,
    cid: str,
    data: Optional[Dict] = None,
) -> PNPMessage:
    """Create an event notification message."""
    p = {"event": event}
    if data:
        p.update(data)
    return PNPMessage(
        cid=cid,
        src=src,
        t=PNPType.EVENT,
        p=p,
    )


def resolve_pnp_target(agent_id: str, mode: str) -> str:
    """
    Resolve PNP target based on mode.

    Direct mode: returns agent_id (for in-process registry lookup).
    HTTP mode: returns the agent's base URL (for network call).
    """
    if mode == "direct":
        return agent_id
    from config import get_agent_url
    return get_agent_url(agent_id)


def send_pnp(
    target: str,
    message: PNPMessage,
    mode: str = "direct",
    fmt: str = "json",
) -> PNPMessage:
    """Send a PNP message using the global transport."""
    return pnp_transport.send(target, message, mode, fmt)


def send_request(
    target: str,
    text: str,
    src: str,
    cid: Optional[str] = None,
    extra: Optional[Dict] = None,
    mode: str = "direct",
    fmt: str = "json",
) -> PNPMessage:
    """Shortcut: create a request and send it in one call."""
    msg = create_request(text, src, cid, extra)
    return send_pnp(target, msg, mode, fmt)


# =============================================================================
# 6. VALIDATION & DISPLAY
# =============================================================================

def validate_pnp(msg: PNPMessage) -> List[str]:
    """Validate a PNP message. Returns list of errors (empty = valid)."""
    errors = []
    if not msg.id:
        errors.append("Missing id")
    if not msg.cid:
        errors.append("Missing cid")
    if not PNPType.is_valid(msg.t):
        errors.append(f"Invalid type '{msg.t}'. Must be: q, r, e")
    if msg.t == PNPType.QUERY and "text" not in msg.p:
        errors.append("Query messages must have 'text' in payload")
    if not msg.src or msg.src == "unknown":
        errors.append("Source (src) must be set")
    return errors


def format_pnp(msg: PNPMessage) -> str:
    """Human-readable display of a PNP message."""
    text = msg.p.get("text", str(msg.p)[:80])
    type_map = {"q": "QUERY", "r": "REPLY", "e": "EVENT"}
    return (
        f"[PNP {type_map.get(msg.t, '?')}] {msg.src} -> "
        f"cid={msg.cid[:8]}.. | {text[:80]}"
    )


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  PNP Protocol v1.0 -- Self-Test")
    print(f"{'='*60}\n")

    # 1. Message creation
    print("1. Message creation:")
    req = create_request("I ate pasta for lunch", "orchestrator", extra={"user_id": 1})
    print(f"   {format_pnp(req)}")
    print(f"   Fields: id={req.id}, cid={req.cid}, src={req.src}, t={req.t}")

    # 2. Validation
    print("\n2. Validation:")
    errors = validate_pnp(req)
    print(f"   Valid message:   {'PASS' if not errors else errors}")

    bad = PNPMessage(t="x", src="unknown")
    errors = validate_pnp(bad)
    print(f"   Invalid message: {errors}")

    # 3. Serialization comparison
    print("\n3. Serialization size comparison:")
    json_bytes = PNPSerializer.serialize(req, "json")
    msgpack_bytes = PNPSerializer.serialize(req, "msgpack")
    print(f"   JSON size:     {len(json_bytes)} bytes")
    print(f"   MsgPack size:  {len(msgpack_bytes)} bytes")
    savings = 100 - (len(msgpack_bytes) * 100 // len(json_bytes))
    print(f"   Savings:       {savings}%")

    # 4. Serialization round-trip
    print("\n4. Serialization round-trip:")
    json_rt = PNPSerializer.deserialize(json_bytes, "json")
    mp_rt = PNPSerializer.deserialize(msgpack_bytes, "msgpack")
    print(f"   JSON round-trip:    {'PASS' if json_rt.id == req.id else 'FAIL'}")
    print(f"   MsgPack round-trip: {'PASS' if mp_rt.id == req.id else 'FAIL'}")

    # 5. Serialization speed
    print("\n5. Serialization speed (1000 iterations):")
    start = time.perf_counter()
    for _ in range(1000):
        PNPSerializer.serialize(req, "json")
    json_time = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    for _ in range(1000):
        PNPSerializer.serialize(req, "msgpack")
    mp_time = (time.perf_counter() - start) * 1000
    print(f"   JSON:    {json_time:.2f} ms (1000 serializations)")
    print(f"   MsgPack: {mp_time:.2f} ms (1000 serializations)")

    # 6. Registry and direct call
    print("\n6. Agent registry + direct call:")

    def echo_handler(msg: PNPMessage) -> PNPMessage:
        return create_reply(f"Echo: {msg.p.get('text', '')}", "echo", msg.cid)

    pnp_registry.register("echo", echo_handler)
    print(f"   Registered agents: {pnp_registry.list_agents()}")

    resp = send_pnp("echo", req, mode="direct", fmt="json")
    print(f"   Direct JSON:    {format_pnp(resp)}")

    resp_mp = send_pnp("echo", req, mode="direct", fmt="msgpack")
    print(f"   Direct MsgPack: {format_pnp(resp_mp)}")

    # 7. Direct call speed
    print("\n7. Direct call speed (1000 round-trips):")
    start = time.perf_counter()
    for _ in range(1000):
        send_pnp("echo", req, mode="direct", fmt="json")
    direct_json_time = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    for _ in range(1000):
        send_pnp("echo", req, mode="direct", fmt="msgpack")
    direct_mp_time = (time.perf_counter() - start) * 1000
    print(f"   Direct+JSON:    {direct_json_time:.2f} ms (1000 round-trips)")
    print(f"   Direct+MsgPack: {direct_mp_time:.2f} ms (1000 round-trips)")
    print(f"   Avg latency:    {direct_json_time/1000:.3f} ms/call (JSON), "
          f"{direct_mp_time/1000:.3f} ms/call (MsgPack)")

    # 8. Compare message size with A2A
    print("\n8. Size comparison vs A2A:")
    a2a_equivalent = {
        "message_id": str(uuid.uuid4()),
        "conversation_id": str(uuid.uuid4()),
        "timestamp": "2025-02-20T12:34:56.789000",
        "protocol": "A2A/1.0",
        "sender": {"id": "orchestrator", "role": "user"},
        "type": "query",
        "payload": {"text": "I ate pasta for lunch", "user_id": 1},
    }
    a2a_json = json.dumps(a2a_equivalent, separators=(",", ":")).encode()
    print(f"   A2A JSON:       {len(a2a_json)} bytes")
    print(f"   PNP JSON:       {len(json_bytes)} bytes  ({100 - len(json_bytes)*100//len(a2a_json)}% smaller)")
    print(f"   PNP MsgPack:    {len(msgpack_bytes)} bytes  ({100 - len(msgpack_bytes)*100//len(a2a_json)}% smaller)")

    pnp_registry.clear()
    print(f"\n{'='*60}")
    print(f"  All PNP self-tests passed!")
    print(f"{'='*60}\n")
