"""
Smart Nutrition Tracker — Single-port entry point.

All 4 agents + orchestrator run on port 8000.
No separate terminals needed.

Agent paths (internal routing):
    /food-logger/     — Food Logger Agent
    /meal-planner/    — Meal Planner Agent
    /health-advisor/  — Health Advisor Agent
    /db-writer/       — DB Writer Agent
    /                 — Orchestrator (chat, health, metrics, agents, etc.)

Start with:
    uvicorn main:app --reload --port 8000
"""

import os
if os.getenv("JSON_LOGS", "").lower() in ("1", "true", "yes"):
    from logging_config import setup_logging
    setup_logging()

# Orchestrator becomes the root app
from agent_orchestrator import app

# Import all agent sub-apps
from agent_food_logger    import app as food_logger_app
from agent_meal_planner   import app as meal_planner_app
from agent_health_advisor import app as health_advisor_app
from agent_db_writer      import app as db_writer_app

# Mount each agent under its own path prefix.
# Requests to /food-logger/a2a/messages are forwarded to food_logger_app as /a2a/messages.
app.mount("/food-logger",    food_logger_app)
app.mount("/meal-planner",   meal_planner_app)
app.mount("/health-advisor", health_advisor_app)
app.mount("/db-writer",      db_writer_app)
