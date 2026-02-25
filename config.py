"""
config.py — Central Configuration for Smart Nutrition Tracker
All ports, database settings, API keys, and agent definitions.

Change settings HERE, not in individual agent files.
"""

import os

# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT INFO
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_NAME = "Smart Nutrition Tracker"
PROJECT_VERSION = "1.0.0"
PROTOCOL_VERSION = "A2A/1.0"
PNP_PROTOCOL_VERSION = "PNP/1.0"
PNP_DEFAULT_MODE = os.getenv("PNP_DEFAULT_MODE", "direct")    # "direct" or "http"
PNP_DEFAULT_FORMAT = "json"    # "json" or "msgpack"

# ═══════════════════════════════════════════════════════════════════════════════
# LLM CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-2.0-flash"
LLM_SEED = int(os.getenv("LLM_SEED", "42"))  # Deterministic seed for reproducibility

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE — PostgreSQL
# ═══════════════════════════════════════════════════════════════════════════════

# Connection string format: postgresql://user:password@host:port/database
# Override with environment variable DATABASE_URL for production
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://nutrition_user:nutrition_pass@localhost:5433/nutrition_tracker"
)

# Individual components (used if you need them separately)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "nutrition_tracker")
DB_USER = os.getenv("DB_USER", "nutrition_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "nutrition_pass")

# ═══════════════════════════════════════════════════════════════════════════════
# DEPLOYMENT MODE
# ═══════════════════════════════════════════════════════════════════════════════
# "monolith" = all agents on single port via FastAPI mount (local dev)
# "microservice" = each agent is its own process/container (K8s / docker-compose)

DEPLOY_MODE = os.getenv("DEPLOY_MODE", "monolith")

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT REGISTRY — Configurable URLs per agent
# ═══════════════════════════════════════════════════════════════════════════════
# Monolith: all agents on single port (8000) via FastAPI mount
# Microservice: each agent on its own port, resolved by env var / K8s DNS

APP_PORT = int(os.getenv("APP_PORT", "8000"))
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_BASE = f"http://{APP_HOST}:{APP_PORT}"

# Per-agent URLs — override with env vars for microservice / K8s mode
ORCHESTRATOR_URL  = os.getenv("ORCHESTRATOR_URL",  APP_BASE)
FOOD_LOGGER_URL   = os.getenv("FOOD_LOGGER_URL",   f"{APP_BASE}/food-logger")
MEAL_PLANNER_URL  = os.getenv("MEAL_PLANNER_URL",  f"{APP_BASE}/meal-planner")
HEALTH_ADVISOR_URL = os.getenv("HEALTH_ADVISOR_URL", f"{APP_BASE}/health-advisor")
DB_WRITER_URL     = os.getenv("DB_WRITER_URL",     f"{APP_BASE}/db-writer")
ML_MODEL_URL      = os.getenv("ML_MODEL_URL",      f"{APP_BASE}/ml-model")

AGENTS = {
    "orchestrator": {
        "name": "Orchestrator",
        "emoji": "🤖",
        "path": "",
        "url": ORCHESTRATOR_URL,
        "description": "Routes queries to the right agent",
    },
    "food_logger": {
        "name": "Food Logger",
        "emoji": "📸",
        "path": "/food-logger",
        "url": FOOD_LOGGER_URL,
        "description": "Identifies food, logs meals with calories/macros",
    },
    "meal_planner": {
        "name": "Meal Planner",
        "emoji": "🍽️",
        "path": "/meal-planner",
        "url": MEAL_PLANNER_URL,
        "description": "Creates personalized meal plans based on goals",
    },
    "health_advisor": {
        "name": "Health Advisor",
        "emoji": "💪",
        "path": "/health-advisor",
        "url": HEALTH_ADVISOR_URL,
        "description": "Health tips, progress tracking, nutrient alerts",
    },
    "db_writer": {
        "name": "DB Writer",
        "emoji": "🗄️",
        "path": "/db-writer",
        "url": DB_WRITER_URL,
        "description": "Reads/writes nutrition data to PostgreSQL",
    },
    "ml_model": {
        "name": "ML Model",
        "emoji": "🧠",
        "path": "/ml-model",
        "url": ML_MODEL_URL,
        "description": "Food image classifier → calories/macros estimation",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC AGENT REGISTRY — A2A Agent Discovery
# ═══════════════════════════════════════════════════════════════════════════════

import threading
from datetime import datetime, timezone


class AgentRegistry:
    """
    Dynamic agent registry implementing A2A Agent Discovery.

    Agents self-register on startup with their capabilities.
    The orchestrator uses this registry to discover and route to agents
    instead of relying solely on hardcoded URLs.

    Features:
        - Runtime registration/deregistration
        - Heartbeat tracking (last_seen timestamp)
        - Capability-based discovery
        - Backward compatible with static AGENTS dict
    """

    def __init__(self):
        self._agents = {}
        self._lock = threading.Lock()
        # Pre-populate from static AGENTS config
        for agent_id, info in AGENTS.items():
            self._agents[agent_id] = {
                "agent_id": agent_id,
                "name": info["name"],
                "url": info["url"],
                "path": info.get("path", ""),
                "description": info.get("description", ""),
                "capabilities": [],
                "protocols": ["a2a"],
                "status": "unknown",
                "registered_at": None,
                "last_seen": None,
                "version": "1.0.0",
                "source": "static",  # "static" = from config, "dynamic" = self-registered
            }

    def register(self, agent_id, name, url, description="",
                 capabilities=None, protocols=None, version="1.0.0", path=""):
        """Register or update an agent in the registry."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._agents[agent_id] = {
                "agent_id": agent_id,
                "name": name,
                "url": url,
                "path": path,
                "description": description,
                "capabilities": capabilities or [],
                "protocols": protocols or ["a2a"],
                "status": "online",
                "registered_at": now,
                "last_seen": now,
                "version": version,
                "source": "dynamic",
            }
        return self._agents[agent_id]

    def unregister(self, agent_id):
        """Remove an agent from the registry."""
        with self._lock:
            return self._agents.pop(agent_id, None)

    def heartbeat(self, agent_id):
        """Update last_seen for an agent (heartbeat)."""
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id]["last_seen"] = datetime.now(timezone.utc).isoformat()
                self._agents[agent_id]["status"] = "online"
                return True
        return False

    def set_status(self, agent_id, status):
        """Update agent status (online/offline/error)."""
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id]["status"] = status

    def get(self, agent_id):
        """Get a single agent's info."""
        return self._agents.get(agent_id)

    def list_all(self):
        """List all registered agents."""
        return list(self._agents.values())

    def discover(self, capability=None, protocol=None):
        """Find agents by capability or protocol."""
        results = []
        for info in self._agents.values():
            if capability and capability not in info.get("capabilities", []):
                continue
            if protocol and protocol not in info.get("protocols", []):
                continue
            results.append(info)
        return results

    def get_url(self, agent_id):
        """Get agent URL from registry (falls back to static config)."""
        agent = self._agents.get(agent_id)
        if agent:
            return agent["url"]
        # Fallback to static AGENTS
        static = AGENTS.get(agent_id)
        if static:
            return static["url"]
        raise ValueError(f"Unknown agent: {agent_id}")


# Global registry instance
agent_registry = AgentRegistry()


# ═══════════════════════════════════════════════════════════════════════════════
# EXTERNAL APIs
# ═══════════════════════════════════════════════════════════════════════════════

# Open Food Facts — free, no API key needed
OPEN_FOOD_FACTS_URL = "https://world.openfoodfacts.org/api/v2"

# ═══════════════════════════════════════════════════════════════════════════════
# ML MODEL SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Path to the trained model file (exported from Google Colab)
# PyTorch: food_classifier.pth (weights) or food_classifier_traced.pt (TorchScript)
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "models/food_classifier.pth")
ML_TRACED_MODEL_PATH = os.getenv("ML_TRACED_MODEL_PATH", "models/food_classifier_traced.pt")
ML_CLASSES_PATH = os.getenv("ML_CLASSES_PATH", "models/food_classes.json")
ML_MODEL_CLASSES = 101  # Food-101 dataset has 101 categories
ML_IMAGE_SIZE = (224, 224)  # Input size for MobileNetV2
ML_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to accept prediction

# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

ENABLE_PERFORMANCE_TRACKING = True
COST_PER_INPUT_TOKEN = 0.0  # Gemini Flash is free
COST_PER_OUTPUT_TOKEN = 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_agent_url(agent_id: str) -> str:
    """Get the base URL for an agent by its ID."""
    agent = AGENTS.get(agent_id)
    if not agent:
        raise ValueError(f"Unknown agent: {agent_id}")
    return agent["url"]


def get_all_agent_urls() -> dict:
    """Get a mapping of agent_id → URL for all agents."""
    return {aid: a["url"] for aid, a in AGENTS.items()}


def print_config():
    """Print current configuration (for debugging)."""
    print(f"\n{'='*60}")
    print(f"  {PROJECT_NAME} v{PROJECT_VERSION}")
    print(f"  Protocol: {PROTOCOL_VERSION}")
    print(f"{'='*60}")
    print(f"\n  Database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"  Gemini:   {'[OK] Key set' if GEMINI_API_KEY else '[MISSING] GEMINI_API_KEY'}")
    print(f"  ML Model: {ML_MODEL_PATH}")
    print(f"\n  Agents:")
    for aid, a in AGENTS.items():
        print(f"    [{aid}] {a['name']:20s} -> {a['url']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print_config()
