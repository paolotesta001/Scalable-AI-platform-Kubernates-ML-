"""
Smart Nutrition Tracker — Entry point (monolith + microservice modes).

Monolith mode (default):
    All 4 agents + orchestrator run on port 8000 via FastAPI mount.
    Start with: uvicorn main:app --reload --port 8000

Microservice mode (DEPLOY_MODE=microservice):
    Only the orchestrator app is loaded. Each agent runs as its own service.
    Agents communicate via HTTP (PNP/A2A) using per-agent URLs from env vars.

Agent paths (monolith routing):
    /food-logger/     — Food Logger Agent
    /meal-planner/    — Meal Planner Agent
    /health-advisor/  — Health Advisor Agent
    /db-writer/       — DB Writer Agent
    /                 — Orchestrator (chat, health, metrics, agents, etc.)
"""

import os
if os.getenv("JSON_LOGS", "").lower() in ("1", "true", "yes"):
    from logging_config import setup_logging
    setup_logging()

from config import DEPLOY_MODE

# Orchestrator becomes the root app
from agent_orchestrator import app

if DEPLOY_MODE == "monolith":
    # Import all agent sub-apps and mount them under path prefixes
    from agent_food_logger    import app as food_logger_app
    from agent_meal_planner   import app as meal_planner_app
    from agent_health_advisor import app as health_advisor_app
    from agent_db_writer      import app as db_writer_app

    app.mount("/food-logger",    food_logger_app)
    app.mount("/meal-planner",   meal_planner_app)
    app.mount("/health-advisor", health_advisor_app)
    app.mount("/db-writer",      db_writer_app)
else:
    # Microservice mode: orchestrator only — agents are separate services.
    # Agent URLs are resolved from environment variables in config.py.
    print(f"[MAIN] Running in microservice mode (DEPLOY_MODE={DEPLOY_MODE})")
