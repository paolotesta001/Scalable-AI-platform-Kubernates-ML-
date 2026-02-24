"""
Meal Planner Agent — generates personalized meal plans.
Fetches user profile from DB Writer, uses Gemini to generate plans.

Start with (standalone):
    uvicorn agent_meal_planner:app --reload --port 8002
Or via main.py (mounted at /meal-planner on port 8000).
"""

import json
import time
from typing import Dict

from fastapi import FastAPI, HTTPException

from a2a_protocol import (
    A2AMessage, Sender, fix_default_ids,
    create_query, send_message,
)
from performance import PerformanceTracker
from llm_client import get_llm_client
from config import get_agent_url, LLM_SEED, agent_registry, AGENTS
from database import log_message as db_log_message
from execution_log import log_execution

from pnp_protocol import (
    PNPMessage, PNPType, PNPSerializer,
    pnp_registry, create_request, create_reply, send_pnp,
)
from fastapi import Request, Response


app = FastAPI(
    title="Meal Planner Agent",
    version="1.0.0",
    description="A2A Meal Planner Agent — creates personalized meal plans based on goals.",
)


SYSTEM_PROMPT = """You are a professional nutritionist AI. Generate personalized meal plans.

Consider the user's:
- Daily calorie target
- Dietary preferences (vegetarian, vegan, keto, etc.)
- Allergies
- Health goals (lose weight, maintain, gain muscle)
- Activity level

Provide a structured meal plan with:
- Meals for the requested period (1 day or 1 week)
- Each meal: name, approximate calories, protein/carbs/fat
- Shopping list (if weekly plan)
- Brief nutritional notes

Be practical and realistic. Use common, accessible foods."""


def generate_plan(query: str, profile_context: str, tracker: PerformanceTracker,
                  provider: str = "gemini", model_name: str = None) -> str:
    """Use LLM to generate a personalized meal plan."""
    client, model = get_llm_client(provider, model_name)

    prompt = f"{profile_context}\nUser request: {query}"
    input_text = SYSTEM_PROMPT + prompt

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
        # seed=LLM_SEED,  # Gemini API does not support seed param
    )
    duration = time.time() - start

    result = response.choices[0].message.content.strip()
    tracker.log_step("llm_generate_meal_plan", input_text, result, duration)
    return result


@app.post("/a2a/messages", response_model=A2AMessage)
def handle_message(msg: A2AMessage) -> A2AMessage:
    msg = fix_default_ids(msg)

    try:
        db_log_message(msg.model_dump(), agent="meal-planner")
    except Exception:
        pass

    if msg.type != "query":
        return A2AMessage(
            sender=Sender(id="meal-planner", role="assistant"),
            conversation_id=msg.conversation_id,
            type="event",
            payload={"status": "received", "received_type": msg.type},
        )

    user_text = msg.payload.get("text")
    user_id = msg.payload.get("user_id", 1)
    provider = msg.payload.get("model_provider", "gemini")
    model_name = msg.payload.get("model_name", None)
    trace_id = msg.payload.get("trace_id", "")

    if not user_text:
        raise HTTPException(status_code=400, detail='Payload must include {"text": "..."}')

    tracker = PerformanceTracker()

    # Step 1: Fetch user profile from DB Writer
    db_url = get_agent_url("db_writer")
    profile_query = create_query(
        text="Get user profile",
        sender_id="meal-planner",
        conversation_id=msg.conversation_id,
        extra={"action": "get_user", "data": {"user_id": user_id}},
    )

    start = time.time()
    profile_response = send_message(db_url, profile_query)
    db_duration = time.time() - start
    tracker.log_step("db_get_user", str(user_id), str(profile_response.payload), db_duration)

    if trace_id:
        log_execution(trace_id, "meal-planner", "db_get_user",
                      protocol="a2a",
                      input_data={"user_id": user_id},
                      output_data=profile_response.payload.get("result"),
                      duration_ms=db_duration * 1000)

    user_profile = profile_response.payload.get("result", {})

    # Step 2: Build context
    profile_context = "User profile:\n"
    if user_profile:
        profile_context += (
            f"- Age: {user_profile.get('age', 'unknown')}\n"
            f"- Weight: {user_profile.get('weight_kg', 'unknown')} kg\n"
            f"- Height: {user_profile.get('height_cm', 'unknown')} cm\n"
            f"- Gender: {user_profile.get('gender', 'unknown')}\n"
            f"- Activity level: {user_profile.get('activity_level', 'moderate')}\n"
            f"- Goal: {user_profile.get('goal', 'maintain')}\n"
            f"- Daily calorie target: {user_profile.get('daily_cal_target', 2000)} kcal\n"
            f"- Dietary preferences: {user_profile.get('dietary_prefs', [])}\n"
            f"- Allergies: {user_profile.get('allergies', [])}\n"
        )
    else:
        profile_context += "No profile found. Use defaults: 2000 kcal, no restrictions.\n"

    # Step 3: Generate meal plan via LLM
    start = time.time()
    plan_text = generate_plan(user_text, profile_context, tracker, provider, model_name)
    llm_dur = (time.time() - start) * 1000

    if trace_id:
        log_execution(trace_id, "meal-planner", "llm_generate_meal_plan",
                      protocol="a2a",
                      input_data={"text": user_text[:200]},
                      output_data=plan_text[:500],
                      duration_ms=llm_dur,
                      tokens_in=len(user_text) // 4,
                      tokens_out=len(plan_text) // 4,
                      llm_call=True, seed=LLM_SEED)

    perf = tracker.get_summary()
    tracker.print_report("Meal Planner Agent")

    response = A2AMessage(
        sender=Sender(id="meal-planner", role="assistant"),
        conversation_id=msg.conversation_id,
        type="response",
        payload={
            "text": plan_text,
            "agent": "meal-planner",
            "user_id": user_id,
            "error": False,
            "performance": perf,
        },
    )
    try:
        db_log_message(response.model_dump(), agent="meal-planner")
    except Exception:
        pass
    return response


# ---------------------------------------------------------------------------
# PNP Protocol Handler
# ---------------------------------------------------------------------------

def handle_pnp(msg: PNPMessage) -> PNPMessage:
    """PNP handler for Meal Planner."""
    if msg.t != PNPType.QUERY:
        return create_reply("received", "meal-planner", msg.cid)

    user_text = msg.p.get("text")
    if not user_text:
        return create_reply("Missing 'text' in payload", "meal-planner", msg.cid,
                            extra={"error": True})

    user_id = msg.p.get("user_id", 1)
    provider = msg.p.get("model_provider", "gemini")
    model_name = msg.p.get("model_name")
    pnp_mode = msg.p.get("_pnp_mode", "direct")
    pnp_fmt = msg.p.get("_pnp_fmt", "json")
    trace_id = msg.p.get("trace_id", "")

    tracker = PerformanceTracker()

    # Step 1: Fetch user profile from DB Writer via PNP
    profile_req = create_request(
        "Get user profile", "meal-planner", msg.cid,
        extra={"action": "get_user", "data": {"user_id": user_id}},
    )
    start = time.time()
    profile_resp = send_pnp("db_writer", profile_req, mode=pnp_mode, fmt=pnp_fmt)
    db_dur = (time.time() - start) * 1000
    tracker.log_step("db_get_user", str(user_id), str(profile_resp.p), db_dur / 1000)
    user_profile = profile_resp.p.get("result", {})

    if trace_id:
        log_execution(trace_id, "meal-planner", "db_get_user",
                      protocol="pnp",
                      input_data={"user_id": user_id},
                      output_data=user_profile,
                      duration_ms=db_dur)

    # Step 2: Build context
    profile_context = "User profile:\n"
    if user_profile:
        profile_context += (
            f"- Age: {user_profile.get('age', 'unknown')}\n"
            f"- Weight: {user_profile.get('weight_kg', 'unknown')} kg\n"
            f"- Height: {user_profile.get('height_cm', 'unknown')} cm\n"
            f"- Gender: {user_profile.get('gender', 'unknown')}\n"
            f"- Activity level: {user_profile.get('activity_level', 'moderate')}\n"
            f"- Goal: {user_profile.get('goal', 'maintain')}\n"
            f"- Daily calorie target: {user_profile.get('daily_cal_target', 2000)} kcal\n"
            f"- Dietary preferences: {user_profile.get('dietary_prefs', [])}\n"
            f"- Allergies: {user_profile.get('allergies', [])}\n"
        )
    else:
        profile_context += "No profile found. Use defaults: 2000 kcal, no restrictions.\n"

    # Step 3: Generate meal plan via LLM
    start = time.time()
    plan_text = generate_plan(user_text, profile_context, tracker, provider, model_name)
    llm_dur = (time.time() - start) * 1000

    if trace_id:
        log_execution(trace_id, "meal-planner", "llm_generate_meal_plan",
                      protocol="pnp",
                      input_data={"text": user_text[:200]},
                      output_data=plan_text[:500],
                      duration_ms=llm_dur,
                      llm_call=True, seed=LLM_SEED)

    perf = tracker.get_summary()
    tracker.print_report("Meal Planner Agent")

    return create_reply(
        plan_text, "meal-planner", msg.cid,
        extra={"agent": "meal-planner", "user_id": user_id, "error": False, "performance": perf},
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
pnp_registry.register("meal_planner", handle_pnp)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "agent": "meal-planner"}


# Self-register in the dynamic agent registry
agent_registry.register(
    agent_id="meal_planner",
    name="Meal Planner",
    url=AGENTS["meal_planner"]["url"],
    path="/meal-planner",
    description="Creates personalized meal plans based on goals",
    capabilities=["meal_plan", "recipe_suggestion"],
    protocols=["a2a", "pnp"],
    version="1.0.0",
)
