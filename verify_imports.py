#!/usr/bin/env python3
"""
Verify all imports work correctly after the fixes.
"""

print("Checking imports...")

try:
    print("  1. Importing toon_protocol...", end="")
    from toon_protocol import (
        ToonMessage, ToonType, ToonSerializer,
        create_request, create_reply, toon_transport
    )
    print(" ✓")
except Exception as e:
    print(f" ✗ Failed: {e}")
    exit(1)

try:
    print("  2. Importing benchmark...", end="")
    import benchmark
    print(" ✓")
except Exception as e:
    print(f" ✗ Failed: {e}")
    exit(1)

try:
    print("  3. Importing agent_db_writer...", end="")
    from agent_db_writer import handle_toon
    print(" ✓")
except Exception as e:
    print(f" ✗ Failed: {e}")
    exit(1)

try:
    print("  4. Importing pnp_protocol...", end="")
    from pnp_protocol import PNPMessage, PNPSerializer, pnp_transport
    print(" ✓")
except Exception as e:
    print(f" ✗ Failed: {e}")
    exit(1)

try:
    print("  5. Importing a2a_protocol...", end="")
    from a2a_protocol import A2AMessage, send_message
    print(" ✓")
except Exception as e:
    print(f" ✗ Failed: {e}")
    exit(1)

print("\n✓ All imports successful! Ready for benchmarking.")
print("\nYou can now run:")
print("  1. Start server:  python -m uvicorn main:app --port 8000")
print("  2. Run benchmark: python benchmark.py --http-only --iterations 50")
