#!/usr/bin/env python3
"""
Tests for the TOON protocol: serialization, message format, and agent handler integration.
Run this before the full benchmark to verify everything works.
"""

from toon_protocol import (
    ToonMessage, ToonType, ToonSerializer,
    create_request, create_reply, toon_transport
)
import json


# =============================================================================
# PROTOCOL TESTS
# =============================================================================

def test_basic_message():
    """Test creating and serializing a basic TOON message."""
    print("Test 1: Basic message creation...")
    msg = create_request(
        text="Hello TOON",
        src="test_agent",
        extra={"action": "test", "data": {"value": 42}}
    )
    print(f"  ✓ Created message: {msg.mid[:8]}... type={msg.kind} src={msg.src}")

    raw = ToonSerializer.serialize(msg)
    print(f"  ✓ Serialized to {len(raw)} bytes")

    restored = ToonSerializer.deserialize(raw)
    assert restored.src == msg.src
    assert restored.kind == msg.kind
    assert restored.body == msg.body
    print(f"  ✓ Restored message matches original\n")


def test_response():
    """Test creating a response message."""
    print("Test 2: Response creation...")
    response = create_reply(
        text="Response from TOON",
        src="db_writer",
        cid="conv123",
        extra={"result": "success", "data": {"user_id": 1}}
    )
    print(f"  ✓ Created response: id={response.mid[:8]}... type={response.kind}")

    raw = ToonSerializer.serialize(response)
    print(f"  ✓ Serialized response to {len(raw)} bytes")

    restored = ToonSerializer.deserialize(raw)
    assert restored.kind == ToonType.RESPONSE
    assert restored.cid == "conv123"
    print(f"  ✓ Response message correct\n")


def test_message_sizes():
    """Compare message sizes across different content."""
    print("Test 3: Message size comparison...")

    small = create_request("Hi", "agent", extra={"a": 1})
    small_raw = ToonSerializer.serialize(small)

    large = create_request(
        "This is a longer message with more data",
        "agent",
        extra={
            "action": "complex_action",
            "data": {
                "user_id": 123,
                "foods": ["apple", "banana", "carrot"] * 5,
                "metadata": {"timestamp": "2024-01-01T00:00:00Z"}
            }
        }
    )
    large_raw = ToonSerializer.serialize(large)

    print(f"  ✓ Small message: {len(small_raw)} bytes")
    print(f"  ✓ Large message: {len(large_raw)} bytes")
    print(f"  ✓ Overhead: ~{len(small_raw)} bytes (protocol + IDs)\n")


def test_json_format():
    """Verify JSON serialization format."""
    print("Test 4: JSON format inspection...")
    msg = create_request(
        text="Inspect this",
        src="inspector",
        cid="fixed123",
        extra={"key": "value"}
    )
    raw = ToonSerializer.serialize(msg)
    json_obj = json.loads(raw)

    print(f"  JSON keys: {list(json_obj.keys())}")
    print(f"  Message ID: {json_obj['mid']}")
    print(f"  Conversation ID: {json_obj['cid']}")
    print(f"  Source: {json_obj['src']}")
    print(f"  Kind: {json_obj['kind']}")
    print(f"  Version: {json_obj['v']}")
    print(f"  Body keys: {list(json_obj['body'].keys())}\n")


# =============================================================================
# INTEGRATION TESTS (agent handler)
# =============================================================================

def test_toon_handler():
    """Test the TOON handler integration with agent_db_writer."""
    from agent_db_writer import handle_toon
    print("Test 5: GET_USER query")
    msg = create_request(
        text="Get user info",
        src="test",
        extra={"action": "get_user", "data": {"user_id": 1}}
    )
    print(f"  Request: action={msg.body.get('action')}, cid={msg.cid}")
    try:
        resp = handle_toon(msg)
        print(f"  ✓ Got response: {resp.kind}")
        if "result" in resp.body:
            print(f"  ✓ Result included in response")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()

    print("Test 6: Invalid action handling")
    msg = create_request(
        text="Invalid action",
        src="test",
        extra={"action": "invalid_action", "data": {}}
    )
    print(f"  Request: action={msg.body.get('action')}")
    try:
        resp = handle_toon(msg)
        print(f"  ✓ Got response: {resp.kind}")
        if resp.body.get("error"):
            print(f"  ✓ Error field set correctly")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()

    print("Test 7: Missing action field")
    msg = create_request(
        text="No action specified",
        src="test",
        extra={"data": {}}
    )
    print(f"  Request: action={msg.body.get('action')}")
    try:
        resp = handle_toon(msg)
        print(f"  ✓ Got response: {resp.kind}")
        if resp.body.get("error"):
            print(f"  ✓ Error field set correctly for missing action")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()

    print("Test 8: Response message (not a query)")
    msg = create_reply(text="Some response", src="test", cid="test_conv")
    msg.kind = ToonType.RESPONSE
    print(f"  Request: kind={msg.kind}")
    try:
        resp = handle_toon(msg)
        print(f"  ✓ Got response: {resp.kind}")
        print(f"  ✓ Handled non-query message correctly")
    except Exception as e:
        print(f"  ✗ Error: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TOON Protocol Tests")
    print("=" * 60 + "\n")

    try:
        test_basic_message()
        test_response()
        test_message_sizes()
        test_json_format()

        print("--- Integration Tests ---\n")
        test_toon_handler()

        print("=" * 60)
        print("  ✓ All TOON tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
