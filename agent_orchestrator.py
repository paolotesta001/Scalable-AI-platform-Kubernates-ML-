"""
Orchestrator — routes to 4 agents: Food Logger, Meal Planner, Health Advisor, DB Writer.
Includes performance monitoring. Runs on port 8000 (root app).

Start with:
    uvicorn main:app --reload --port 8000
"""

import json
import os
import time
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from a2a_protocol import (
    A2AMessage, Sender, fix_default_ids,
    send_message, create_query,
)
from performance import PerformanceTracker, add_performance_to_app, get_memory_usage
from llm_client import get_llm_client
from config import AGENTS, get_agent_url, LLM_SEED, agent_registry
from database import log_message as db_log_message

from pnp_protocol import (
    PNPMessage, PNPType, PNPSerializer,
    pnp_registry, create_request, create_reply, send_pnp, resolve_pnp_target,
)
from config import PNP_DEFAULT_MODE
from execution_log import generate_trace_id, log_execution
from fastapi import Request, Response


# ---------------------------------------------------------------------------
# Agent URLs (internal routing via mount paths)
# ---------------------------------------------------------------------------

FOOD_LOGGER_URL    = get_agent_url("food_logger")
MEAL_PLANNER_URL   = get_agent_url("meal_planner")
HEALTH_ADVISOR_URL = get_agent_url("health_advisor")
DB_WRITER_URL      = get_agent_url("db_writer")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Smart Nutrition Tracker — Orchestrator",
    version="1.0.0",
    description="Multi-agent orchestrator with 4 agents: Food Logger, Meal Planner, Health Advisor, DB Writer.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

add_performance_to_app(app, "orchestrator")


# ---------------------------------------------------------------------------
# Intent Classification
# ---------------------------------------------------------------------------

def classify_task(user_text: str, tracker: PerformanceTracker,
                  provider: str = "gemini", model_name: str = None) -> Dict:
    """Classify user intent and determine which agent to route to."""

    classification_prompt = """You are a task router for a nutrition tracking app.
Analyze the user's request and decide the best workflow.

You have 4 agents:
- FOOD LOGGER: Logs food/meals, identifies what was eaten, estimates calories/macros.
  Use when: "I had pizza", "Log 2 eggs", "Just ate a banana", "I drank a smoothie"
- MEAL PLANNER: Creates personalized meal plans and recipe suggestions.
  Use when: "Plan my meals", "What should I eat?", "Give me a high-protein lunch idea"
- HEALTH ADVISOR: Health tips, nutrition analysis, progress review, nutrient alerts.
  Use when: "How am I doing?", "Am I eating enough protein?", "Give me health tips"
- DB WRITER: Direct database queries (create user, get meals, get daily summary).
  Use when: "Show my meals today", "Create a user profile", "What did I eat yesterday"

Respond ONLY with a valid JSON object (no markdown, no backticks):
{
    "plan": "log_food" | "meal_plan" | "health_advice" | "db_query" | "direct_answer",
    "food_logger_query": "what to ask the food logger (or empty string)",
    "meal_planner_query": "what to ask the meal planner (or empty string)",
    "health_advisor_query": "what to ask the health advisor (or empty string)",
    "db_writer_query": "what to ask the db writer (or empty string)",
    "direct_answer": "your answer if plan is direct_answer (or empty string)"
}

User request: """

    try:
        client, model = get_llm_client(provider, model_name)
        input_text = classification_prompt + user_text

        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": classification_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.1,
            max_tokens=500,
            # seed=LLM_SEED,  # Gemini API does not support seed param
        )
        raw = response.choices[0].message.content.strip()
        duration = time.time() - start
        tracker.log_step("classify_task", input_text, raw, duration)
        print(f"[DEBUG] LLM raw response: {raw[:300]}")

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        return json.loads(raw)

    except Exception as exc:
        print(f"[WARN] classify_task failed: {type(exc).__name__}: {exc}")
        return {
            "plan": "direct_answer",
            "food_logger_query": "", "meal_planner_query": "",
            "health_advisor_query": "", "db_writer_query": "",
            "direct_answer": "I had trouble understanding that. Could you rephrase?",
        }


# ---------------------------------------------------------------------------
# Retry & Fallback Logic
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
RETRY_BACKOFF = [0.5, 1.0, 2.0]  # seconds between retries (exponential backoff)

FALLBACK_MESSAGES = {
    "food_logger": "I wasn't able to reach the Food Logger agent right now. "
                   "Please try logging your meal again in a moment.",
    "meal_planner": "The Meal Planner agent is temporarily unavailable. "
                    "Please try requesting a meal plan again shortly.",
    "health_advisor": "The Health Advisor agent is temporarily unavailable. "
                      "Please try again in a moment for health advice.",
    "db_writer": "The Database agent is temporarily unavailable. "
                 "Your data may not have been saved. Please retry.",
}


def call_agent_with_retry(agent_name, call_fn, tracker, trace_id=None):
    """
    Call an agent function with retry logic and exponential backoff.

    Args:
        agent_name: Agent identifier (for logging and fallback)
        call_fn: Callable that performs the actual agent call, returns text
        tracker: PerformanceTracker instance
        trace_id: Optional trace ID for execution logging

    Returns:
        tuple: (text, success_bool, retries_used)
    """
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            text = call_fn()
            if attempt > 0:
                print(f"[RETRY] {agent_name} succeeded on attempt {attempt + 1}")
                if trace_id:
                    log_execution(trace_id, "orchestrator", "retry_success",
                                  input_data={"agent": agent_name, "attempt": attempt + 1},
                                  output_data={"status": "recovered"})
            return text, True, attempt

        except Exception as exc:
            last_error = exc
            print(f"[RETRY] {agent_name} attempt {attempt + 1}/{MAX_RETRIES} failed: "
                  f"{type(exc).__name__}: {exc}")

            # Update registry status
            agent_registry.set_status(agent_name, "error")

            if trace_id:
                log_execution(trace_id, "orchestrator", "retry_attempt",
                              input_data={"agent": agent_name, "attempt": attempt + 1},
                              output_data={"error": str(exc)},
                              error=str(exc))

            # Wait before next retry (unless last attempt)
            if attempt < MAX_RETRIES - 1:
                backoff = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                print(f"[RETRY] Waiting {backoff}s before retry...")
                time.sleep(backoff)

    # All retries exhausted — return fallback
    fallback = FALLBACK_MESSAGES.get(agent_name, "Agent temporarily unavailable. Please try again.")
    print(f"[FALLBACK] {agent_name} failed after {MAX_RETRIES} attempts. Using fallback.")

    if trace_id:
        log_execution(trace_id, "orchestrator", "fallback",
                      input_data={"agent": agent_name, "max_retries": MAX_RETRIES},
                      output_data={"fallback_message": fallback},
                      error=str(last_error))

    tracker.log_step(f"{agent_name}_fallback", "all retries failed", fallback, 0)
    return fallback, False, MAX_RETRIES


# ---------------------------------------------------------------------------
# Agent call helpers (with retry)
# ---------------------------------------------------------------------------

def call_food_logger(query: str, conv_id: str, tracker: PerformanceTracker,
                     user_id: int = 1, model_extra: dict = None,
                     trace_id: str = None) -> str:
    print(f">Food Logger: {query[:80]}...")

    def _call():
        start = time.time()
        extra = {"user_id": user_id}
        if model_extra:
            extra.update(model_extra)
        resp = send_message(
            FOOD_LOGGER_URL,
            create_query(query, "orchestrator", conv_id, extra),
        )
        duration = time.time() - start
        text = resp.payload.get("text", "No food logged.")
        tracker.log_step("food_logger_agent", query, text, duration)
        print(f"[OK]Food Logger: {len(text)} chars in {duration:.1f}s")
        return text

    text, success, retries = call_agent_with_retry("food_logger", _call, tracker, trace_id)
    return text


def call_meal_planner(query: str, conv_id: str, tracker: PerformanceTracker,
                      user_id: int = 1, model_extra: dict = None,
                      trace_id: str = None) -> str:
    print(f">Meal Planner: {query[:80]}...")

    def _call():
        start = time.time()
        extra = {"user_id": user_id}
        if model_extra:
            extra.update(model_extra)
        resp = send_message(
            MEAL_PLANNER_URL,
            create_query(query, "orchestrator", conv_id, extra),
        )
        duration = time.time() - start
        text = resp.payload.get("text", "No meal plan generated.")
        tracker.log_step("meal_planner_agent", query, text, duration)
        print(f"[OK]Meal Planner: {len(text)} chars in {duration:.1f}s")
        return text

    text, success, retries = call_agent_with_retry("meal_planner", _call, tracker, trace_id)
    return text


def call_health_advisor(query: str, conv_id: str, tracker: PerformanceTracker,
                        user_id: int = 1, model_extra: dict = None,
                        trace_id: str = None) -> str:
    print(f">Health Advisor: {query[:80]}...")

    def _call():
        start = time.time()
        extra = {"user_id": user_id}
        if model_extra:
            extra.update(model_extra)
        resp = send_message(
            HEALTH_ADVISOR_URL,
            create_query(query, "orchestrator", conv_id, extra),
        )
        duration = time.time() - start
        text = resp.payload.get("text", "No health advice generated.")
        tracker.log_step("health_advisor_agent", query, text, duration)
        print(f"[OK]Health Advisor: {len(text)} chars in {duration:.1f}s")
        return text

    text, success, retries = call_agent_with_retry("health_advisor", _call, tracker, trace_id)
    return text


# ---------------------------------------------------------------------------
# Plan execution
# ---------------------------------------------------------------------------

def execute_plan(plan: Dict, conversation_id: str, user_text: str,
                 tracker: PerformanceTracker, user_id: int = 1,
                 model_extra: dict = None, trace_id: str = None) -> str:
    workflow = plan.get("plan", "direct_answer")

    if workflow == "direct_answer":
        answer = plan.get("direct_answer", "")
        if not answer:
            answer = (
                "I'm a Smart Nutrition Tracker with 4 specialized agents:\n"
                "**Food Logger** -- log what you eat, get calories/macros\n"
                "**Meal Planner** -- personalized meal plans for your goals\n"
                "**Health Advisor** -- nutrition tips and progress tracking\n"
                "**DB Writer** -- direct database queries\n\n"
                "Try asking me something like:\n"
                "- 'I had pasta and salad for lunch'\n"
                "- 'Plan my meals for tomorrow, high protein'\n"
                "- 'How am I doing with my nutrition?'"
            )
        return f"**Orchestrator**\n\n{answer}"

    m = model_extra

    if workflow == "log_food":
        logged = call_food_logger(
            plan.get("food_logger_query", user_text),
            conversation_id, tracker, user_id, m, trace_id,
        )
        return f"**Food Logged**\n\n{logged}"

    if workflow == "meal_plan":
        plan_text = call_meal_planner(
            plan.get("meal_planner_query", user_text),
            conversation_id, tracker, user_id, m, trace_id,
        )
        return f"**Meal Plan**\n\n{plan_text}"

    if workflow == "health_advice":
        advice = call_health_advisor(
            plan.get("health_advisor_query", user_text),
            conversation_id, tracker, user_id, m, trace_id,
        )
        return f"**Health Advice**\n\n{advice}"

    if workflow == "db_query":
        return f"Use the DB Writer directly at {DB_WRITER_URL}/a2a/messages"

    return "Could not determine workflow. Please try rephrasing your request."


# ---------------------------------------------------------------------------
# A2A endpoint
# ---------------------------------------------------------------------------

@app.post("/a2a/messages", response_model=A2AMessage)
def handle_message(msg: A2AMessage) -> A2AMessage:
    msg = fix_default_ids(msg)

    # Generate trace_id for this request chain
    trace_id = msg.payload.get("trace_id") or generate_trace_id()

    try:
        db_log_message(msg.model_dump(), agent="orchestrator")
    except Exception:
        pass

    if msg.type != "query":
        return A2AMessage(
            sender=Sender(id="orchestrator", role="assistant"),
            conversation_id=msg.conversation_id,
            type="event",
            payload={"status": "received", "received_type": msg.type, "trace_id": trace_id},
        )

    user_text = msg.payload.get("text")
    user_id = msg.payload.get("user_id", 1)
    provider = msg.payload.get("model_provider", "gemini")
    model_name = msg.payload.get("model_name", None)

    if not user_text:
        raise HTTPException(status_code=400, detail='Payload must include {"text": "..."}')

    tracker = PerformanceTracker()
    model_extra = {"model_provider": provider, "model_name": model_name, "trace_id": trace_id} if model_name else {"model_provider": provider, "trace_id": trace_id}

    print(f"\n{'='*60}")
    print(f"[REQ] New request [{trace_id}]: {user_text[:100]}...")

    # Classify intent (LLM call — logged)
    start = time.time()
    plan = classify_task(user_text, tracker, provider, model_name)
    classify_dur = (time.time() - start) * 1000
    print(f"[PLAN] Plan: {plan.get('plan')}")

    log_execution(trace_id, "orchestrator", "classify_task",
                  protocol="a2a",
                  input_data={"text": user_text},
                  output_data=plan,
                  duration_ms=classify_dur,
                  tokens_in=len(user_text) // 4,
                  tokens_out=len(str(plan)) // 4,
                  llm_call=True, seed=LLM_SEED)

    # Execute plan (sub-agent calls)
    start = time.time()
    final_answer = execute_plan(plan, msg.conversation_id, user_text, tracker, user_id, model_extra, trace_id)
    exec_dur = (time.time() - start) * 1000
    perf = tracker.get_summary()

    log_execution(trace_id, "orchestrator", "execute_plan",
                  protocol="a2a",
                  input_data={"workflow": plan.get("plan"), "user_text": user_text[:200]},
                  output_data={"answer_length": len(final_answer)},
                  duration_ms=exec_dur)

    # Print performance report
    print(f"\n[PERF] PERFORMANCE REPORT [{trace_id}]")
    print(f"   Total time:    {perf['total_time']}s")
    print(f"   LLM calls:     {perf['llm_calls']}")
    print(f"   Total tokens:  {perf['total_tokens']}")
    print(f"   Est. cost:     ${perf['cost_usd']:.6f}")
    print(f"   Memory delta:  {perf['memory_mb']} MB")
    for step in perf["steps"]:
        print(f"   - {step['name']}: {step['duration']}s, "
              f"{step['tokens']} tokens")
    print(f"{'='*60}\n")

    # Identify which agents were used
    agents_used = []
    for step in perf["steps"]:
        for name in ["food_logger", "meal_planner", "health_advisor", "db_writer"]:
            if name in step["name"] and name not in agents_used:
                agents_used.append(name)

    response = A2AMessage(
        sender=Sender(id="orchestrator", role="assistant"),
        conversation_id=msg.conversation_id,
        type="response",
        payload={
            "text": final_answer,
            "workflow": plan.get("plan"),
            "agents_used": agents_used,
            "performance": perf,
            "trace_id": trace_id,
        },
    )
    try:
        db_log_message(response.model_dump(), agent="orchestrator")
    except Exception:
        pass
    return response


# ---------------------------------------------------------------------------
# PNP Protocol — Call helpers
# ---------------------------------------------------------------------------

def call_food_logger_pnp(query: str, cid: str, tracker: PerformanceTracker,
                         user_id: int = 1, mode: str = None, fmt: str = "json",
                         trace_id: str = None) -> str:
    """Call Food Logger via PNP protocol with retry."""
    mode = mode or PNP_DEFAULT_MODE
    target = resolve_pnp_target("food_logger", mode)
    print(f"[PNP] -> Food Logger ({mode}): {query[:80]}...")

    def _call():
        msg = create_request(query, "orchestrator", cid,
                             extra={"user_id": user_id, "_pnp_mode": mode, "_pnp_fmt": fmt})
        start = time.time()
        resp = send_pnp(target, msg, mode=mode, fmt=fmt)
        duration = time.time() - start
        text = resp.p.get("text", "No food logged.")
        tracker.log_step("food_logger_agent", query, text, duration)
        print(f"[PNP] Food Logger: {len(text)} chars in {duration:.1f}s")
        return text

    text, success, retries = call_agent_with_retry("food_logger", _call, tracker, trace_id)
    return text


def call_meal_planner_pnp(query: str, cid: str, tracker: PerformanceTracker,
                          user_id: int = 1, mode: str = None, fmt: str = "json",
                          trace_id: str = None) -> str:
    """Call Meal Planner via PNP protocol with retry."""
    mode = mode or PNP_DEFAULT_MODE
    target = resolve_pnp_target("meal_planner", mode)
    print(f"[PNP] -> Meal Planner ({mode}): {query[:80]}...")

    def _call():
        msg = create_request(query, "orchestrator", cid,
                             extra={"user_id": user_id, "_pnp_mode": mode, "_pnp_fmt": fmt})
        start = time.time()
        resp = send_pnp(target, msg, mode=mode, fmt=fmt)
        duration = time.time() - start
        text = resp.p.get("text", "No meal plan generated.")
        tracker.log_step("meal_planner_agent", query, text, duration)
        print(f"[PNP] Meal Planner: {len(text)} chars in {duration:.1f}s")
        return text

    text, success, retries = call_agent_with_retry("meal_planner", _call, tracker, trace_id)
    return text


def call_health_advisor_pnp(query: str, cid: str, tracker: PerformanceTracker,
                            user_id: int = 1, mode: str = None, fmt: str = "json",
                            trace_id: str = None) -> str:
    """Call Health Advisor via PNP protocol with retry."""
    mode = mode or PNP_DEFAULT_MODE
    target = resolve_pnp_target("health_advisor", mode)
    print(f"[PNP] -> Health Advisor ({mode}): {query[:80]}...")

    def _call():
        msg = create_request(query, "orchestrator", cid,
                             extra={"user_id": user_id, "_pnp_mode": mode, "_pnp_fmt": fmt})
        start = time.time()
        resp = send_pnp(target, msg, mode=mode, fmt=fmt)
        duration = time.time() - start
        text = resp.p.get("text", "No health advice generated.")
        tracker.log_step("health_advisor_agent", query, text, duration)
        print(f"[PNP] Health Advisor: {len(text)} chars in {duration:.1f}s")
        return text

    text, success, retries = call_agent_with_retry("health_advisor", _call, tracker, trace_id)
    return text


# ---------------------------------------------------------------------------
# PNP Protocol — Orchestrator handler
# ---------------------------------------------------------------------------

def handle_pnp(msg: PNPMessage) -> PNPMessage:
    """PNP handler for Orchestrator. Classifies intent, routes via PNP transport."""
    if msg.t != PNPType.QUERY:
        return create_reply("received", "orchestrator", msg.cid)

    user_text = msg.p.get("text")
    if not user_text:
        return create_reply("Missing 'text'", "orchestrator", msg.cid, extra={"error": True})

    # Generate or propagate trace_id
    trace_id = msg.p.get("trace_id") or generate_trace_id()

    user_id = msg.p.get("user_id", 1)
    provider = msg.p.get("model_provider", "gemini")
    model_name = msg.p.get("model_name")
    pnp_mode = msg.p.get("_pnp_mode", PNP_DEFAULT_MODE)
    pnp_fmt = msg.p.get("_pnp_fmt", "json")

    tracker = PerformanceTracker()

    # Classify intent (uses LLM)
    start = time.time()
    plan = classify_task(user_text, tracker, provider, model_name)
    classify_dur = (time.time() - start) * 1000
    workflow = plan.get("plan", "direct_answer")

    log_execution(trace_id, "orchestrator", "classify_task",
                  protocol="pnp",
                  input_data={"text": user_text},
                  output_data=plan,
                  duration_ms=classify_dur,
                  llm_call=True, seed=LLM_SEED)

    if workflow == "direct_answer":
        answer = plan.get("direct_answer", "Could not determine workflow.")
        answer = f"Orchestrator\n\n{answer}"
    elif workflow == "log_food":
        result = call_food_logger_pnp(
            plan.get("food_logger_query", user_text), msg.cid, tracker, user_id, pnp_mode, pnp_fmt, trace_id)
        answer = f"Food Logged\n\n{result}"
    elif workflow == "meal_plan":
        result = call_meal_planner_pnp(
            plan.get("meal_planner_query", user_text), msg.cid, tracker, user_id, pnp_mode, pnp_fmt, trace_id)
        answer = f"Meal Plan\n\n{result}"
    elif workflow == "health_advice":
        result = call_health_advisor_pnp(
            plan.get("health_advisor_query", user_text), msg.cid, tracker, user_id, pnp_mode, pnp_fmt, trace_id)
        answer = f"Health Advice\n\n{result}"
    else:
        answer = "Could not determine workflow. Please try rephrasing."

    perf = tracker.get_summary()
    return create_reply(
        answer, "orchestrator", msg.cid,
        extra={"workflow": workflow, "performance": perf, "trace_id": trace_id},
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
pnp_registry.register("orchestrator", handle_pnp)


# ---------------------------------------------------------------------------
# Monitoring endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict:
    """Health check that verifies DB connectivity."""
    from database import test_connection
    db_ok = False
    try:
        db_ok = test_connection()
    except Exception:
        pass
    status = "ok" if db_ok else "degraded"
    return {
        "status": status,
        "agent": "orchestrator",
        "database": "connected" if db_ok else "unreachable",
    }


@app.get("/agents")
def list_agents() -> Dict:
    """List all agents and their health status, updating registry."""
    import httpx
    agents = {}
    for name, info in AGENTS.items():
        if name in ("orchestrator", "ml_model"):
            continue
        try:
            r = httpx.get(f"{info['url']}/health", timeout=5.0)
            agents[name] = r.json()
            agent_registry.set_status(name, "online")
            agent_registry.heartbeat(name)
        except Exception:
            agents[name] = {"status": "offline"}
            agent_registry.set_status(name, "offline")
    agent_registry.set_status("orchestrator", "online")
    agent_registry.heartbeat("orchestrator")
    return {"orchestrator": "online", "agents": agents}


@app.get("/metrics")
def get_metrics() -> Dict:
    return {
        "memory": get_memory_usage(),
        "agents": list(AGENTS.keys()),
    }


# ---------------------------------------------------------------------------
# Agent Discovery / Registry — A2A Agent Registration
# ---------------------------------------------------------------------------

@app.post("/api/registry/register")
def api_register_agent(body: dict):
    """Agent self-registration endpoint. Agents call this on startup."""
    agent_id = body.get("agent_id")
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id required")
    info = agent_registry.register(
        agent_id=agent_id,
        name=body.get("name", agent_id),
        url=body.get("url", ""),
        path=body.get("path", ""),
        description=body.get("description", ""),
        capabilities=body.get("capabilities", []),
        protocols=body.get("protocols", ["a2a"]),
        version=body.get("version", "1.0.0"),
    )
    print(f"[REGISTRY] Agent registered: {agent_id} -> {info['url']}")
    return {"status": "registered", "agent": info}


@app.post("/api/registry/heartbeat")
def api_heartbeat(body: dict):
    """Agent heartbeat — agents call periodically to stay registered."""
    agent_id = body.get("agent_id")
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id required")
    ok = agent_registry.heartbeat(agent_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    return {"status": "ok", "agent_id": agent_id}


@app.delete("/api/registry/{agent_id}")
def api_unregister_agent(agent_id: str):
    """Remove an agent from the registry."""
    removed = agent_registry.unregister(agent_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    print(f"[REGISTRY] Agent unregistered: {agent_id}")
    return {"status": "unregistered", "agent_id": agent_id}


@app.get("/api/registry")
def api_list_registry():
    """List all registered agents with status."""
    return {"agents": agent_registry.list_all()}


@app.get("/api/registry/discover")
def api_discover_agents(capability: Optional[str] = None, protocol: Optional[str] = None):
    """Discover agents by capability or protocol."""
    results = agent_registry.discover(capability=capability, protocol=protocol)
    return {"agents": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Web Interface — Static files + REST API wrappers
# ---------------------------------------------------------------------------

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/")
def serve_index():
    """Serve the web interface."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/user/{user_id}")
def api_get_user(user_id: int):
    """Get user profile — wrapper around DB Writer."""
    msg = create_query("Get user", "web-ui", extra={"action": "get_user", "data": {"user_id": user_id}})
    resp = send_message(DB_WRITER_URL, msg)
    return resp.payload


@app.get("/api/meals/{user_id}")
def api_get_meals(user_id: int, limit: int = 50, meal_type: Optional[str] = None):
    """Get meals for a user."""
    data = {"user_id": user_id, "limit": limit}
    if meal_type:
        data["meal_type"] = meal_type
    msg = create_query("Get meals", "web-ui", extra={"action": "get_meals", "data": data})
    resp = send_message(DB_WRITER_URL, msg)
    return resp.payload


@app.get("/api/daily/{user_id}")
def api_get_daily(user_id: int):
    """Get today's daily nutrition summary."""
    msg = create_query("Get daily summary", "web-ui", extra={"action": "get_daily_summary", "data": {"user_id": user_id}})
    resp = send_message(DB_WRITER_URL, msg)
    return resp.payload


@app.get("/api/users")
def api_get_all_users():
    """Get all users."""
    msg = create_query("Get all users", "web-ui", extra={"action": "get_all_users", "data": {}})
    resp = send_message(DB_WRITER_URL, msg)
    return resp.payload


@app.post("/api/user")
def api_create_user(body: dict):
    """Create a new user."""
    msg = create_query("Create user", "web-ui", extra={"action": "create_user", "data": body})
    resp = send_message(DB_WRITER_URL, msg)
    return resp.payload


@app.put("/api/user/{user_id}")
def api_update_user(user_id: int, body: dict):
    """Update a user's fields."""
    data = {"user_id": user_id}
    data.update(body)
    msg = create_query("Update user", "web-ui", extra={"action": "update_user", "data": data})
    resp = send_message(DB_WRITER_URL, msg)
    return resp.payload


@app.delete("/api/user/{user_id}")
def api_delete_user(user_id: int):
    """Delete a user."""
    msg = create_query("Delete user", "web-ui", extra={"action": "delete_user", "data": {"user_id": user_id}})
    resp = send_message(DB_WRITER_URL, msg)
    return resp.payload


@app.put("/api/meal/{meal_id}")
def api_update_meal(meal_id: int, body: dict):
    """Update a meal's fields."""
    data = {"meal_id": meal_id}
    data.update(body)
    msg = create_query("Update meal", "web-ui", extra={"action": "update_meal", "data": data})
    resp = send_message(DB_WRITER_URL, msg)
    return resp.payload


@app.delete("/api/meal/{meal_id}")
def api_delete_meal(meal_id: int):
    """Delete a meal."""
    msg = create_query("Delete meal", "web-ui", extra={"action": "delete_meal", "data": {"meal_id": meal_id}})
    resp = send_message(DB_WRITER_URL, msg)
    return resp.payload


@app.get("/api/app-config")
def api_app_config():
    """Return frontend configuration (public mode hides technical sections)."""
    from config import PUBLIC_MODE
    return {"public_mode": PUBLIC_MODE}


@app.get("/api/benchmark-results")
def api_benchmark_results():
    """Return saved benchmark results."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "No benchmark results found. Run: python benchmark.py --export"}


@app.get("/api/traces")
def api_get_traces(limit: int = 20):
    """Get recent execution traces."""
    from execution_log import get_recent_traces
    return {"traces": get_recent_traces(limit)}


@app.get("/api/trace/{trace_id}")
def api_get_trace(trace_id: str):
    """Get all execution log entries for a trace."""
    from execution_log import get_trace_log
    entries = get_trace_log(trace_id)
    return {"trace_id": trace_id, "entries": entries, "count": len(entries)}


# Mount static files (CSS, JS)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
