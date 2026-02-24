"""
test_system.py — End-to-end tests for Smart Nutrition Tracker.
Assumes the app is running: uvicorn main:app --reload --port 8000

Tests:
    1. Health check all agents
    2. Create a test user via DB Writer
    3. Log a meal via Orchestrator -> Food Logger -> DB Writer
    4. Request a meal plan via Orchestrator -> Meal Planner -> DB Writer
    5. Request health advice via Orchestrator -> Health Advisor -> DB Writer
    6. Direct DB Writer query

Usage:
    1. Start app:   uvicorn main:app --reload --port 8000
    2. Run tests:   python test_system.py
"""

import sys
import time
import httpx
from a2a_protocol import send_query
from config import AGENTS, get_agent_url, APP_BASE


def divider(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_health_checks():
    """Test 1: Verify all agents are reachable."""
    divider("TEST 1: Health Checks")
    all_ok = True
    for agent_id, info in AGENTS.items():
        if agent_id == "ml_model":
            continue
        url = f"{info['url']}/health"
        try:
            resp = httpx.get(url, timeout=5)
            status = "OK" if resp.status_code == 200 else f"FAIL ({resp.status_code})"
            if resp.status_code != 200:
                all_ok = False
        except Exception:
            status = "UNREACHABLE"
            all_ok = False
        print(f"  {info['emoji']} {info['name']:20s} {status}")
    return all_ok


def test_create_user():
    """Test 2: Create a test user via DB Writer."""
    divider("TEST 2: Create Test User (via DB Writer)")
    try:
        response = send_query(
            target_url=get_agent_url("db_writer"),
            text="Create test user",
            sender_id="test_script",
            extra={
                "action": "create_user",
                "data": {
                    "username": f"test_user_{int(time.time())}",
                    "age": 30,
                    "weight_kg": 75.0,
                    "height_cm": 175.0,
                    "gender": "male",
                    "activity_level": "moderate",
                    "goal": "maintain",
                    "daily_cal_target": 2200,
                    "dietary_prefs": ["balanced"],
                    "allergies": [],
                },
            },
        )
        print(f"  Status: {response.payload.get('text', 'No text')}")
        user = response.payload.get("result", {})
        user_id = user.get("id") if user else None
        print(f"  User ID: {user_id}")
        print(f"  Error: {response.payload.get('error', 'unknown')}")
        return user_id
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def test_log_meal(user_id: int):
    """Test 3: Log a meal via Orchestrator -> Food Logger -> DB Writer."""
    divider("TEST 3: Log Food (Orchestrator -> Food Logger -> DB Writer)")
    try:
        response = send_query(
            target_url=APP_BASE,  # Orchestrator is at root
            text="I just had a big bowl of chicken alfredo pasta and a side salad for lunch",
            sender_id="test_script",
            extra={"user_id": user_id},
        )
        print(f"  Workflow: {response.payload.get('workflow', '?')}")
        print(f"  Agents used: {response.payload.get('agents_used', '?')}")
        print(f"  Response: {response.payload.get('text', 'No text')[:300]}")
    except Exception as e:
        print(f"  FAILED: {e}")


def test_meal_plan(user_id: int):
    """Test 4: Request a meal plan via Orchestrator -> Meal Planner."""
    divider("TEST 4: Meal Plan (Orchestrator -> Meal Planner -> DB Writer)")
    try:
        response = send_query(
            target_url=APP_BASE,
            text="Plan my meals for tomorrow. I want something healthy and high in protein.",
            sender_id="test_script",
            extra={"user_id": user_id},
        )
        print(f"  Workflow: {response.payload.get('workflow', '?')}")
        print(f"  Agents used: {response.payload.get('agents_used', '?')}")
        text = response.payload.get("text", "No text")
        print(f"  Response preview: {text[:400]}...")
    except Exception as e:
        print(f"  FAILED: {e}")


def test_health_advice(user_id: int):
    """Test 5: Request health advice via Orchestrator -> Health Advisor."""
    divider("TEST 5: Health Advice (Orchestrator -> Health Advisor -> DB Writer)")
    try:
        response = send_query(
            target_url=APP_BASE,
            text="How am I doing with my nutrition today? Any suggestions?",
            sender_id="test_script",
            extra={"user_id": user_id},
        )
        print(f"  Workflow: {response.payload.get('workflow', '?')}")
        print(f"  Agents used: {response.payload.get('agents_used', '?')}")
        text = response.payload.get("text", "No text")
        print(f"  Response preview: {text[:400]}...")
    except Exception as e:
        print(f"  FAILED: {e}")


def test_get_meals(user_id: int):
    """Test 6: Query meals directly from DB Writer."""
    divider("TEST 6: Get Meals (direct DB Writer query)")
    try:
        response = send_query(
            target_url=get_agent_url("db_writer"),
            text="Get meals",
            sender_id="test_script",
            extra={
                "action": "get_meals",
                "data": {"user_id": user_id, "limit": 5},
            },
        )
        meals = response.payload.get("result", [])
        print(f"  Meals found: {len(meals) if meals else 0}")
        if meals:
            for meal in meals[:3]:
                print(
                    f"    - {meal.get('food_name', '?')} "
                    f"({meal.get('meal_type', '?')}): "
                    f"{meal.get('calories', '?')} kcal"
                )
    except Exception as e:
        print(f"  FAILED: {e}")


def main():
    print("\n" + "=" * 60)
    print("  Smart Nutrition Tracker - End-to-End Tests")
    print(f"  Server: {APP_BASE}")
    print("=" * 60)

    # Test 1: Health checks
    if not test_health_checks():
        print("\n  Some agents are not running!")
        print("  Start with: uvicorn main:app --reload --port 8000")
        sys.exit(1)

    # Test 2: Create test user
    user_id = test_create_user()
    if not user_id:
        print("\n  Failed to create test user.")
        print("  Is PostgreSQL running? Did you run: python database.py")
        sys.exit(1)

    # Test 3: Log a meal
    test_log_meal(user_id)

    # Test 4: Meal plan
    test_meal_plan(user_id)

    # Test 5: Health advice
    test_health_advice(user_id)

    # Test 6: Verify meals in DB
    test_get_meals(user_id)

    # Summary
    divider("ALL TESTS COMPLETED")
    print(f"  Test user ID: {user_id}")
    print(f"  All flows exercised through the orchestrator.")
    print()


if __name__ == "__main__":
    main()
