"""
Food Logger Agent — parses food descriptions, estimates calories/macros.
Uses Gemini LLM to parse text, sends structured data to DB Writer.

Start with (standalone):
    uvicorn agent_food_logger:app --reload --port 8001
Or via main.py (mounted at /food-logger on port 8000).
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
from database import log_message as db_log_message, lookup_food as db_lookup_food, cache_food as db_cache_food
from execution_log import log_execution

from pnp_protocol import (
    PNPMessage, PNPType, PNPSerializer,
    pnp_registry, create_request, create_reply, send_pnp,
)
from fastapi import Request, Response


app = FastAPI(
    title="Food Logger Agent",
    version="1.0.0",
    description="A2A Food Logger Agent — identifies food, logs meals with calories/macros.",
)


SYSTEM_PROMPT = """You are a nutrition expert AI. Given a food description,
extract structured nutrition information.

Respond with a JSON object containing:
{
    "food_name": "name of the food item",
    "meal_type": "breakfast" | "lunch" | "dinner" | "snack",
    "quantity": 1.0,
    "unit": "serving" | "gram" | "cup" | "piece",
    "calories": estimated total calories (number),
    "protein_g": grams of protein (number),
    "carbs_g": grams of carbohydrates (number),
    "fat_g": grams of fat (number),
    "fiber_g": grams of fiber (number),
    "sugar_g": grams of sugar (number),
    "sodium_mg": milligrams of sodium (number),
    "confidence_note": "brief note about estimation confidence"
}

If the description mentions multiple items, return a JSON array of objects.
Be realistic with calorie and macro estimates. Use standard USDA-like values.
If the user does not specify the meal type, infer it from context or default to "snack".

You MUST respond with valid JSON only. No markdown, no explanation, no code fences."""


def parse_food(query: str, tracker: PerformanceTracker, provider: str = "gemini", model_name: str = None) -> str:
    """Use LLM to parse food description into structured nutrition data."""
    client, model = get_llm_client(provider, model_name)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Parse this food description and estimate nutrition:\n\n{query}"},
    ]

    input_text = SYSTEM_PROMPT + query
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
        # seed=LLM_SEED,  # Gemini API does not support seed param
    )
    duration = time.time() - start

    result = response.choices[0].message.content.strip()
    tracker.log_step("llm_parse_food", input_text, result, duration)
    return result


def _infer_meal_type(text: str) -> str:
    """Infer meal type from user text keywords."""
    t = text.lower()
    if "breakfast" in t: return "breakfast"
    if "lunch" in t: return "lunch"
    if "dinner" in t or "supper" in t: return "dinner"
    return "snack"


def _try_cache_lookup(text: str, tracker: PerformanceTracker):
    """Try to find food in cache before calling LLM. Returns cached item dict or None."""
    cleaned = text.lower().strip()
    for prefix in ["i ate ", "i had ", "i just ate ", "i just had ",
                    "log ", "had ", "ate ", "just had ", "just ate "]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    for suffix in [" for breakfast", " for lunch", " for dinner",
                   " as a snack", " for snack", " today", " yesterday"]:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:len(cleaned) - len(suffix)]
            break
    for article in ["a ", "an ", "some ", "the "]:
        if cleaned.startswith(article):
            cleaned = cleaned[len(article):]
            break
    cleaned = cleaned.strip()
    if not cleaned or len(cleaned) < 2:
        return None
    try:
        start = time.time()
        cached = db_lookup_food(cleaned)
        duration = time.time() - start
        tracker.log_step("cache_lookup", cleaned, str(bool(cached)), duration)
        if cached and cached.get("raw_data"):
            return cached["raw_data"]
    except Exception:
        pass
    return None


def _cache_food_items(items: list, tracker: PerformanceTracker):
    """Cache parsed food items for future lookups."""
    for item in items:
        food_name = item.get("food_name", "")
        if not food_name:
            continue
        try:
            start = time.time()
            db_cache_food(
                food_name=food_name,
                calories_per_100g=item.get("calories"),
                protein_per_100g=item.get("protein_g"),
                carbs_per_100g=item.get("carbs_g"),
                fat_per_100g=item.get("fat_g"),
                fiber_per_100g=item.get("fiber_g"),
                sugar_per_100g=item.get("sugar_g"),
                source="gemini_cache",
                raw_data=item,
            )
            duration = time.time() - start
            tracker.log_step("cache_food", food_name, "cached", duration)
        except Exception:
            pass


@app.post("/a2a/messages", response_model=A2AMessage)
def handle_message(msg: A2AMessage) -> A2AMessage:
    msg = fix_default_ids(msg)

    try:
        db_log_message(msg.model_dump(), agent="food-logger")
    except Exception:
        pass

    if msg.type != "query":
        return A2AMessage(
            sender=Sender(id="food-logger", role="assistant"),
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

    # Step 0: Try food cache
    cached = _try_cache_lookup(user_text, tracker)

    if cached:
        # Cache hit - skip LLM call
        items = [cached] if not isinstance(cached, list) else cached
        inferred_type = _infer_meal_type(user_text)
        for item in items:
            if inferred_type != "snack":
                item["meal_type"] = inferred_type
        food_source = "cache"
        if trace_id:
            log_execution(trace_id, "food-logger", "cache_hit",
                          protocol="a2a",
                          input_data={"text": user_text},
                          output_data={"food_name": items[0].get("food_name", ""), "source": "cache"},
                          duration_ms=0)
    else:
        # Step 1: Parse food via LLM
        start_llm = time.time()
        llm_response = parse_food(user_text, tracker, provider, model_name)
        llm_dur = (time.time() - start_llm) * 1000

        if trace_id:
            log_execution(trace_id, "food-logger", "llm_parse_food",
                          protocol="a2a",
                          input_data={"text": user_text},
                          output_data=llm_response[:500],
                          duration_ms=llm_dur,
                          tokens_in=len(user_text) // 4,
                          tokens_out=len(llm_response) // 4,
                          llm_call=True, seed=LLM_SEED)

        # Step 2: Parse JSON
        try:
            raw = llm_response
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            parsed = json.loads(raw.strip())
        except json.JSONDecodeError:
            perf = tracker.get_summary()
            return A2AMessage(
                sender=Sender(id="food-logger", role="assistant"),
                conversation_id=msg.conversation_id,
                type="response",
                payload={
                    "text": f"Failed to parse LLM response as JSON: {llm_response[:200]}",
                    "error": True,
                    "raw_llm_response": llm_response,
                    "performance": perf,
                },
            )

        items = parsed if isinstance(parsed, list) else [parsed]
        food_source = "gemini_estimate"

        # Cache the results for future lookups
        _cache_food_items(items, tracker)

    # Step 3: Send each item to DB Writer
    logged_meals = []
    db_url = get_agent_url("db_writer")

    for item in items:
        meal_data = {
            "user_id": user_id,
            "food_name": item.get("food_name", "Unknown food"),
            "meal_type": item.get("meal_type", "snack"),
            "calories": item.get("calories"),
            "protein_g": item.get("protein_g"),
            "carbs_g": item.get("carbs_g"),
            "fat_g": item.get("fat_g"),
            "fiber_g": item.get("fiber_g"),
            "sugar_g": item.get("sugar_g"),
            "sodium_mg": item.get("sodium_mg"),
            "quantity": item.get("quantity", 1.0),
            "unit": item.get("unit", "serving"),
            "source": food_source,
        }

        db_query = create_query(
            text="Log a meal",
            sender_id="food-logger",
            conversation_id=msg.conversation_id,
            extra={"action": "log_meal", "data": meal_data},
        )

        start = time.time()
        db_response = send_message(db_url, db_query)
        db_duration = time.time() - start
        tracker.log_step("db_log_meal", str(meal_data), str(db_response.payload), db_duration)

        logged_meals.append(item)

    # Build summary
    summary_lines = []
    for item in logged_meals:
        summary_lines.append(
            f"- {item.get('food_name', '?')}: "
            f"{item.get('calories', '?')} kcal, "
            f"{item.get('protein_g', '?')}g protein, "
            f"{item.get('carbs_g', '?')}g carbs, "
            f"{item.get('fat_g', '?')}g fat"
        )
    summary = "Logged meals:\n" + "\n".join(summary_lines)

    perf = tracker.get_summary()
    tracker.print_report("Food Logger Agent")

    response = A2AMessage(
        sender=Sender(id="food-logger", role="assistant"),
        conversation_id=msg.conversation_id,
        type="response",
        payload={
            "text": summary,
            "agent": "food-logger",
            "meals_logged": len(logged_meals),
            "items": logged_meals,
            "error": False,
            "performance": perf,
        },
    )
    try:
        db_log_message(response.model_dump(), agent="food-logger")
    except Exception:
        pass
    return response


# ---------------------------------------------------------------------------
# PNP Protocol Handler
# ---------------------------------------------------------------------------

def _process_food_log(text: str, user_id: int, conversation_id: str,
                      provider: str = "gemini", model_name: str = None,
                      pnp_mode: str = None, pnp_fmt: str = None,
                      trace_id: str = "") -> dict:
    """Core food logging logic, protocol-agnostic. Used by both A2A and PNP."""
    tracker = PerformanceTracker()
    protocol = "pnp" if pnp_mode else "a2a"

    # Step 0: Try food cache
    cached = _try_cache_lookup(text, tracker)

    if cached:
        items = [cached] if not isinstance(cached, list) else cached
        inferred_type = _infer_meal_type(text)
        for item in items:
            if inferred_type != "snack":
                item["meal_type"] = inferred_type
        food_source = "cache"
        if trace_id:
            log_execution(trace_id, "food-logger", "cache_hit",
                          protocol=protocol,
                          input_data={"text": text},
                          output_data={"food_name": items[0].get("food_name", ""), "source": "cache"},
                          duration_ms=0)
    else:
        # Step 1: Parse food via LLM
        start = time.time()
        llm_response = parse_food(text, tracker, provider, model_name)
        llm_dur = (time.time() - start) * 1000

        if trace_id:
            log_execution(trace_id, "food-logger", "llm_parse_food",
                          protocol=protocol,
                          input_data={"text": text},
                          output_data=llm_response[:500],
                          duration_ms=llm_dur,
                          tokens_in=len(text) // 4,
                          tokens_out=len(llm_response) // 4,
                          llm_call=True, seed=LLM_SEED)

        # Step 2: Parse JSON
        try:
            raw = llm_response
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            parsed = json.loads(raw.strip())
        except json.JSONDecodeError:
            perf = tracker.get_summary()
            return {
                "text": f"Failed to parse LLM response: {llm_response[:200]}",
                "error": True,
                "raw_llm_response": llm_response,
                "performance": perf,
            }

        items = parsed if isinstance(parsed, list) else [parsed]
        food_source = "gemini_estimate"

        # Cache the results for future lookups
        _cache_food_items(items, tracker)

    # Step 3: Log each item to DB Writer
    logged_meals = []
    db_url = get_agent_url("db_writer")

    for item in items:
        meal_data = {
            "user_id": user_id,
            "food_name": item.get("food_name", "Unknown food"),
            "meal_type": item.get("meal_type", "snack"),
            "calories": item.get("calories"),
            "protein_g": item.get("protein_g"),
            "carbs_g": item.get("carbs_g"),
            "fat_g": item.get("fat_g"),
            "fiber_g": item.get("fiber_g"),
            "sugar_g": item.get("sugar_g"),
            "sodium_mg": item.get("sodium_mg"),
            "quantity": item.get("quantity", 1.0),
            "unit": item.get("unit", "serving"),
            "source": food_source,
        }

        if pnp_mode:
            db_req = create_request(
                "Log a meal", "food-logger", conversation_id,
                extra={"action": "log_meal", "data": meal_data},
            )
            start = time.time()
            db_resp = send_pnp("db_writer", db_req, mode=pnp_mode, fmt=pnp_fmt or "json")
            tracker.log_step("db_log_meal", str(meal_data), str(db_resp.p), time.time() - start)
        else:
            db_query = create_query(
                "Log a meal", "food-logger", conversation_id,
                extra={"action": "log_meal", "data": meal_data},
            )
            start = time.time()
            db_response = send_message(db_url, db_query)
            tracker.log_step("db_log_meal", str(meal_data), str(db_response.payload), time.time() - start)

        logged_meals.append(item)

    # Build summary
    summary_lines = []
    for item in logged_meals:
        summary_lines.append(
            f"- {item.get('food_name', '?')}: "
            f"{item.get('calories', '?')} kcal, "
            f"{item.get('protein_g', '?')}g protein, "
            f"{item.get('carbs_g', '?')}g carbs, "
            f"{item.get('fat_g', '?')}g fat"
        )
    summary = "Logged meals:\n" + "\n".join(summary_lines)

    perf = tracker.get_summary()
    tracker.print_report("Food Logger Agent")

    return {
        "text": summary,
        "agent": "food-logger",
        "meals_logged": len(logged_meals),
        "items": logged_meals,
        "error": False,
        "performance": perf,
    }


def handle_pnp(msg: PNPMessage) -> PNPMessage:
    """PNP handler for Food Logger."""
    if msg.t != PNPType.QUERY:
        return create_reply("received", "food-logger", msg.cid)

    text = msg.p.get("text")
    if not text:
        return create_reply("Missing 'text' in payload", "food-logger", msg.cid,
                            extra={"error": True})

    user_id = msg.p.get("user_id", 1)
    provider = msg.p.get("model_provider", "gemini")
    model_name = msg.p.get("model_name")
    pnp_mode = msg.p.get("_pnp_mode", "direct")
    pnp_fmt = msg.p.get("_pnp_fmt", "json")
    trace_id = msg.p.get("trace_id", "")

    result = _process_food_log(text, user_id, msg.cid, provider, model_name, pnp_mode, pnp_fmt, trace_id)

    return create_reply(
        result.get("text", ""), "food-logger", msg.cid,
        extra={k: v for k, v in result.items() if k != "text"},
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
pnp_registry.register("food_logger", handle_pnp)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "agent": "food-logger"}


# Self-register in the dynamic agent registry
agent_registry.register(
    agent_id="food_logger",
    name="Food Logger",
    url=AGENTS["food_logger"]["url"],
    path="/food-logger",
    description="Identifies food, logs meals with calories/macros",
    capabilities=["log_food", "parse_food", "food_cache"],
    protocols=["a2a", "pnp"],
    version="1.0.0",
)
