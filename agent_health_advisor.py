"""
Health Advisor Agent — provides health tips and nutrition analysis.
Fetches nutrition history from DB Writer, uses Gemini to generate advice.

Start with (standalone):
    uvicorn agent_health_advisor:app --reload --port 8003
Or via main.py (mounted at /health-advisor on port 8000).
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
    pnp_registry, create_request, create_reply, send_pnp, resolve_pnp_target,
)
from config import PNP_DEFAULT_MODE
from fastapi import Request, Response
from toon_protocol import (
    ToonMessage, ToonType, ToonSerializer,
    create_request as toon_create_request,
    create_reply as toon_create_reply,
    toon_transport,
)


app = FastAPI(
    title="Health Advisor Agent",
    version="1.0.0",
    description="A2A Health Advisor Agent — health tips, progress tracking, nutrient alerts.",
)


SYSTEM_PROMPT = """You are a supportive health and nutrition advisor AI.

Based on the user's nutrition history and profile, provide:
- Analysis of their eating patterns
- Nutrient balance assessment (are they getting enough protein, fiber, etc.?)
- Specific, actionable health tips
- Encouragement and positive reinforcement
- Warnings if calorie intake is too low or too high
- Suggestions for missing nutrients

Be friendly, evidence-based, and non-judgmental.
Disclaimer: Always note you are an AI and not a medical professional."""


def generate_advice(query: str, context: str, tracker: PerformanceTracker,
                    provider: str = "gemini", model_name: str = None) -> str:
    """Use LLM to generate personalized health advice."""
    client, model = get_llm_client(provider, model_name)

    prompt = f"{context}\nUser question: {query}"
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
        max_tokens=1536,
        # seed=LLM_SEED,  # Gemini API does not support seed param
    )
    duration = time.time() - start

    result = response.choices[0].message.content.strip()
    tracker.log_step("llm_generate_health_advice", input_text, result, duration)
    return result


@app.post("/a2a/messages", response_model=A2AMessage)
def handle_message(msg: A2AMessage) -> A2AMessage:
    msg = fix_default_ids(msg)

    try:
        db_log_message(msg.model_dump(), agent="health-advisor")
    except Exception:
        pass

    if msg.type != "query":
        return A2AMessage(
            sender=Sender(id="health-advisor", role="assistant"),
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
    db_url = get_agent_url("db_writer")

    # Step 1: Fetch user profile
    profile_query = create_query(
        text="Get user profile",
        sender_id="health-advisor",
        conversation_id=msg.conversation_id,
        extra={"action": "get_user", "data": {"user_id": user_id}},
    )
    start = time.time()
    profile_resp = send_message(db_url, profile_query)
    db_dur = (time.time() - start) * 1000
    tracker.log_step("db_get_user", str(user_id), str(profile_resp.payload), db_dur / 1000)
    user_profile = profile_resp.payload.get("result", {})

    if trace_id:
        log_execution(trace_id, "health-advisor", "db_get_user",
                      protocol="a2a",
                      input_data={"user_id": user_id},
                      output_data=user_profile,
                      duration_ms=db_dur)

    # Step 2: Fetch recent meals
    meals_query = create_query(
        text="Get recent meals",
        sender_id="health-advisor",
        conversation_id=msg.conversation_id,
        extra={"action": "get_meals", "data": {"user_id": user_id, "limit": 20}},
    )
    start = time.time()
    meals_resp = send_message(db_url, meals_query)
    db_dur = (time.time() - start) * 1000
    tracker.log_step("db_get_meals", str(user_id), str(meals_resp.payload), db_dur / 1000)
    recent_meals = meals_resp.payload.get("result", [])

    if trace_id:
        log_execution(trace_id, "health-advisor", "db_get_meals",
                      protocol="a2a",
                      input_data={"user_id": user_id, "limit": 20},
                      output_data={"meals_count": len(recent_meals) if recent_meals else 0},
                      duration_ms=db_dur)

    # Step 3: Fetch today's daily summary
    summary_query = create_query(
        text="Get daily summary",
        sender_id="health-advisor",
        conversation_id=msg.conversation_id,
        extra={"action": "get_daily_summary", "data": {"user_id": user_id}},
    )
    start = time.time()
    summary_resp = send_message(db_url, summary_query)
    db_dur = (time.time() - start) * 1000
    tracker.log_step("db_get_daily_summary", str(user_id), str(summary_resp.payload), db_dur / 1000)
    daily_summary = summary_resp.payload.get("result", {})

    if trace_id:
        log_execution(trace_id, "health-advisor", "db_get_daily_summary",
                      protocol="a2a",
                      input_data={"user_id": user_id},
                      output_data=daily_summary,
                      duration_ms=db_dur)

    # Step 4: Build context
    context = f"User profile: {json.dumps(user_profile, default=str)}\n\n"
    context += f"Today's nutrition summary: {json.dumps(daily_summary, default=str)}\n\n"
    context += f"Recent meals ({len(recent_meals) if recent_meals else 0} meals):\n"
    if recent_meals:
        for meal in recent_meals[:10]:
            context += (
                f"  - {meal.get('food_name', '?')} ({meal.get('meal_type', '?')}): "
                f"{meal.get('calories', '?')} kcal\n"
            )
    else:
        context += "  No meals logged yet.\n"

    # Step 5: Generate health advice via LLM
    start = time.time()
    advice_text = generate_advice(user_text, context, tracker, provider, model_name)
    llm_dur = (time.time() - start) * 1000

    if trace_id:
        log_execution(trace_id, "health-advisor", "llm_generate_health_advice",
                      protocol="a2a",
                      input_data={"text": user_text[:200]},
                      output_data=advice_text[:500],
                      duration_ms=llm_dur,
                      tokens_in=len(user_text) // 4,
                      tokens_out=len(advice_text) // 4,
                      llm_call=True, seed=LLM_SEED)

    perf = tracker.get_summary()
    tracker.print_report("Health Advisor Agent")

    response = A2AMessage(
        sender=Sender(id="health-advisor", role="assistant"),
        conversation_id=msg.conversation_id,
        type="response",
        payload={
            "text": advice_text,
            "agent": "health-advisor",
            "user_id": user_id,
            "error": False,
            "performance": perf,
        },
    )
    try:
        db_log_message(response.model_dump(), agent="health-advisor")
    except Exception:
        pass
    return response


# ---------------------------------------------------------------------------
# PNP Protocol Handler
# ---------------------------------------------------------------------------

def handle_pnp(msg: PNPMessage) -> PNPMessage:
    """PNP handler for Health Advisor."""
    if msg.t != PNPType.QUERY:
        return create_reply("received", "health-advisor", msg.cid)

    user_text = msg.p.get("text")
    if not user_text:
        return create_reply("Missing 'text' in payload", "health-advisor", msg.cid,
                            extra={"error": True})

    user_id = msg.p.get("user_id", 1)
    provider = msg.p.get("model_provider", "gemini")
    model_name = msg.p.get("model_name")
    pnp_mode = msg.p.get("_pnp_mode", PNP_DEFAULT_MODE)
    pnp_fmt = msg.p.get("_pnp_fmt", "json")
    trace_id = msg.p.get("trace_id", "")

    tracker = PerformanceTracker()

    # Step 1: Fetch user profile via PNP
    db_target = resolve_pnp_target("db_writer", pnp_mode)
    profile_req = create_request(
        "Get user profile", "health-advisor", msg.cid,
        extra={"action": "get_user", "data": {"user_id": user_id}},
    )
    start = time.time()
    profile_resp = send_pnp(db_target, profile_req, mode=pnp_mode, fmt=pnp_fmt)
    db_dur = (time.time() - start) * 1000
    tracker.log_step("db_get_user", str(user_id), str(profile_resp.p), db_dur / 1000)
    user_profile = profile_resp.p.get("result", {})

    if trace_id:
        log_execution(trace_id, "health-advisor", "db_get_user",
                      protocol="pnp",
                      input_data={"user_id": user_id},
                      output_data=user_profile,
                      duration_ms=db_dur)

    # Step 2: Fetch recent meals via PNP
    meals_req = create_request(
        "Get recent meals", "health-advisor", msg.cid,
        extra={"action": "get_meals", "data": {"user_id": user_id, "limit": 20}},
    )
    start = time.time()
    meals_resp = send_pnp(db_target, meals_req, mode=pnp_mode, fmt=pnp_fmt)
    db_dur = (time.time() - start) * 1000
    tracker.log_step("db_get_meals", str(user_id), str(meals_resp.p), db_dur / 1000)
    recent_meals = meals_resp.p.get("result", [])

    if trace_id:
        log_execution(trace_id, "health-advisor", "db_get_meals",
                      protocol="pnp",
                      input_data={"user_id": user_id, "limit": 20},
                      output_data={"meals_count": len(recent_meals) if recent_meals else 0},
                      duration_ms=db_dur)

    # Step 3: Fetch daily summary via PNP
    summary_req = create_request(
        "Get daily summary", "health-advisor", msg.cid,
        extra={"action": "get_daily_summary", "data": {"user_id": user_id}},
    )
    start = time.time()
    summary_resp = send_pnp(db_target, summary_req, mode=pnp_mode, fmt=pnp_fmt)
    db_dur = (time.time() - start) * 1000
    tracker.log_step("db_get_daily_summary", str(user_id), str(summary_resp.p), db_dur / 1000)
    daily_summary = summary_resp.p.get("result", {})

    if trace_id:
        log_execution(trace_id, "health-advisor", "db_get_daily_summary",
                      protocol="pnp",
                      input_data={"user_id": user_id},
                      output_data=daily_summary,
                      duration_ms=db_dur)

    # Step 4: Build context
    context = f"User profile: {json.dumps(user_profile, default=str)}\n\n"
    context += f"Today's nutrition summary: {json.dumps(daily_summary, default=str)}\n\n"
    context += f"Recent meals ({len(recent_meals) if recent_meals else 0} meals):\n"
    if recent_meals:
        for meal in recent_meals[:10]:
            context += (
                f"  - {meal.get('food_name', '?')} ({meal.get('meal_type', '?')}): "
                f"{meal.get('calories', '?')} kcal\n"
            )
    else:
        context += "  No meals logged yet.\n"

    # Step 5: Generate health advice via LLM
    start = time.time()
    advice_text = generate_advice(user_text, context, tracker, provider, model_name)
    llm_dur = (time.time() - start) * 1000

    if trace_id:
        log_execution(trace_id, "health-advisor", "llm_generate_health_advice",
                      protocol="pnp",
                      input_data={"text": user_text[:200]},
                      output_data=advice_text[:500],
                      duration_ms=llm_dur,
                      llm_call=True, seed=LLM_SEED)

    perf = tracker.get_summary()
    tracker.print_report("Health Advisor Agent")

    return create_reply(
        advice_text, "health-advisor", msg.cid,
        extra={"agent": "health-advisor", "user_id": user_id, "error": False, "performance": perf},
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
pnp_registry.register("health_advisor", handle_pnp)


# ---------------------------------------------------------------------------
# TOON Protocol Handler
# ---------------------------------------------------------------------------

def handle_toon(msg: ToonMessage) -> ToonMessage:
    """TOON handler for Health Advisor."""
    if msg.kind != ToonType.QUERY:
        return toon_create_reply("received", "health-advisor", msg.cid)

    user_text = msg.body.get("text")
    if not user_text:
        return toon_create_reply("Missing 'text' in payload", "health-advisor", msg.cid,
                                  extra={"error": True})

    user_id = msg.body.get("user_id", 1)
    provider = msg.body.get("model_provider", "gemini")
    model_name = msg.body.get("model_name")
    trace_id = msg.body.get("trace_id", "")

    tracker = PerformanceTracker()
    db_url = get_agent_url("db_writer")

    # Step 1: Fetch user profile via TOON
    profile_req = toon_create_request(
        "Get user profile", "health-advisor", msg.cid,
        extra={"action": "get_user", "data": {"user_id": user_id}},
    )
    start = time.time()
    profile_resp = toon_transport.send_http(db_url, profile_req)
    db_dur = (time.time() - start) * 1000
    tracker.log_step("db_get_user", str(user_id), str(profile_resp.body), db_dur / 1000)
    user_profile = profile_resp.body.get("result", {})

    if trace_id:
        log_execution(trace_id, "health-advisor", "db_get_user",
                      protocol="toon",
                      input_data={"user_id": user_id},
                      output_data=user_profile,
                      duration_ms=db_dur)

    # Step 2: Fetch recent meals via TOON
    meals_req = toon_create_request(
        "Get recent meals", "health-advisor", msg.cid,
        extra={"action": "get_meals", "data": {"user_id": user_id, "limit": 20}},
    )
    start = time.time()
    meals_resp = toon_transport.send_http(db_url, meals_req)
    db_dur = (time.time() - start) * 1000
    tracker.log_step("db_get_meals", str(user_id), str(meals_resp.body), db_dur / 1000)
    recent_meals = meals_resp.body.get("result", [])

    if trace_id:
        log_execution(trace_id, "health-advisor", "db_get_meals",
                      protocol="toon",
                      input_data={"user_id": user_id, "limit": 20},
                      output_data={"meals_count": len(recent_meals) if recent_meals else 0},
                      duration_ms=db_dur)

    # Step 3: Fetch daily summary via TOON
    summary_req = toon_create_request(
        "Get daily summary", "health-advisor", msg.cid,
        extra={"action": "get_daily_summary", "data": {"user_id": user_id}},
    )
    start = time.time()
    summary_resp = toon_transport.send_http(db_url, summary_req)
    db_dur = (time.time() - start) * 1000
    tracker.log_step("db_get_daily_summary", str(user_id), str(summary_resp.body), db_dur / 1000)
    daily_summary = summary_resp.body.get("result", {})

    if trace_id:
        log_execution(trace_id, "health-advisor", "db_get_daily_summary",
                      protocol="toon",
                      input_data={"user_id": user_id},
                      output_data=daily_summary,
                      duration_ms=db_dur)

    # Step 4: Build context
    context = f"User profile: {json.dumps(user_profile, default=str)}\n\n"
    context += f"Today's nutrition summary: {json.dumps(daily_summary, default=str)}\n\n"
    context += f"Recent meals ({len(recent_meals) if recent_meals else 0} meals):\n"
    if recent_meals:
        for meal in recent_meals[:10]:
            context += (
                f"  - {meal.get('food_name', '?')} ({meal.get('meal_type', '?')}): "
                f"{meal.get('calories', '?')} kcal\n"
            )
    else:
        context += "  No meals logged yet.\n"

    # Step 5: Generate health advice via LLM
    start = time.time()
    advice_text = generate_advice(user_text, context, tracker, provider, model_name)
    llm_dur = (time.time() - start) * 1000

    if trace_id:
        log_execution(trace_id, "health-advisor", "llm_generate_health_advice",
                      protocol="toon",
                      input_data={"text": user_text[:200]},
                      output_data=advice_text[:500],
                      duration_ms=llm_dur,
                      llm_call=True, seed=LLM_SEED)

    perf = tracker.get_summary()
    tracker.print_report("Health Advisor Agent")

    return toon_create_reply(
        advice_text, "health-advisor", msg.cid,
        extra={"agent": "health-advisor", "user_id": user_id, "error": False, "performance": perf},
    )


@app.post("/toon")
def toon_endpoint(msg: ToonMessage):
    """TOON HTTP endpoint for health advice."""
    response = handle_toon(msg)
    resp_bytes = ToonSerializer.serialize(response)
    return Response(content=resp_bytes, media_type="application/json")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "agent": "health-advisor"}


# Self-register in the dynamic agent registry
agent_registry.register(
    agent_id="health_advisor",
    name="Health Advisor",
    url=AGENTS["health_advisor"]["url"],
    path="/health-advisor",
    description="Health tips, progress tracking, nutrient alerts",
    capabilities=["health_advice", "nutrition_analysis", "progress_review"],
    protocols=["a2a", "pnp", "toon"],
    version="1.0.0",
)
