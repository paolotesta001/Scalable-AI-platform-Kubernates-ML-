# toon_protocol.py
import json
import uuid
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import httpx

TOON_VERSION = "TOON/1.0"
TOON_CONTENT_TYPE_JSON = "application/json"

def _sid(n=10) -> str:
    return uuid.uuid4().hex[:n]

class ToonMessage(BaseModel):
    # lean-ish: 6 fields
    mid: str = Field(default_factory=lambda: _sid(10))     # message id
    cid: str = Field(default_factory=lambda: _sid(12))     # conversation id
    src: str = "unknown"                                   # sender id
    kind: str = "q"                                        # q/r/e
    body: Dict[str, Any] = Field(default_factory=dict)     # payload
    v: str = Field(default=TOON_VERSION)                   # protocol version (optional)

class ToonType:
    QUERY = "q"
    RESPONSE = "r"
    EVENT = "e"
    VALID = {QUERY, RESPONSE, EVENT}

class ToonSerializer:
    @staticmethod
    def serialize(msg: ToonMessage) -> bytes:
        return json.dumps(msg.model_dump(), separators=(",", ":")).encode("utf-8")

    @staticmethod
    def deserialize(raw: bytes) -> ToonMessage:
        return ToonMessage(**json.loads(raw.decode("utf-8")))

def create_request(text: str, src: str, cid: Optional[str] = None, extra: Optional[Dict] = None) -> ToonMessage:
    """Create a TOON query message."""
    body = {"text": text}
    if extra:
        body.update(extra)
    return ToonMessage(
        cid=cid or _sid(12),
        src=src,
        kind=ToonType.QUERY,
        body=body
    )

def create_reply(text: str, src: str, cid: str, extra: Optional[Dict] = None) -> ToonMessage:
    """Create a TOON response message."""
    body = {"text": text}
    if extra:
        body.update(extra)
    return ToonMessage(
        cid=cid,
        src=src,
        kind=ToonType.RESPONSE,
        body=body
    )


class ToonTransport:
    """HTTP transport for TOON protocol."""
    
    def __init__(self, timeout: float = 120.0):
        self.timeout = timeout

    def send_http(self, target_url: str, message: ToonMessage) -> ToonMessage:
        """Send a TOON message via HTTP and get response."""
        endpoint = f"{target_url}/toon"
        raw = ToonSerializer.serialize(message)

        with httpx.Client(timeout=self.timeout, verify=False) as client:
            resp = client.post(
                endpoint,
                content=raw,
                headers={"Content-Type": TOON_CONTENT_TYPE_JSON}
            )
            resp.raise_for_status()
            return ToonSerializer.deserialize(resp.content)

toon_transport = ToonTransport()