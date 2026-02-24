"""
database.py — PostgreSQL Database Layer for Smart Nutrition Tracker
Handles all database operations: connection, tables, queries.

Uses asyncpg for async PostgreSQL access (fast, production-grade).
Falls back to psycopg2 for sync operations.

Tables:
    - users:           User profiles and goals
    - meals:           Logged meals (what was eaten, when)
    - nutrition_log:   Daily nutrition totals
    - food_cache:      Cached food lookups (avoid re-querying APIs)
    - conversations:   A2A message history (for debugging)

Setup PostgreSQL first:
    1. Install: https://www.postgresql.org/download/
    2. Create database:
        psql -U postgres
        CREATE USER nutrition_user WITH PASSWORD 'nutrition_pass';
        CREATE DATABASE nutrition_tracker OWNER nutrition_user;
        \\q
    3. Run this file to create tables:
        python database.py
"""

import json
from datetime import datetime, date, timezone
from typing import Optional, Dict, List, Any

from config import DATABASE_URL, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONNECTION — Connect to PostgreSQL
# ═══════════════════════════════════════════════════════════════════════════════

# We use psycopg2 (sync) for simplicity. Can upgrade to asyncpg later.
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False
    print("[WARN] psycopg2 not installed. Run: pip install psycopg2-binary")


def get_connection():
    """
    Get a new PostgreSQL connection.

    Returns a connection object. Always close it when done:
        conn = get_connection()
        try:
            # ... do stuff
        finally:
            conn.close()

    Or use the helper functions below which handle this for you.
    """
    if not PG_AVAILABLE:
        raise RuntimeError("psycopg2 not installed. Run: pip install psycopg2-binary")

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def test_connection() -> bool:
    """Test if PostgreSQL is reachable."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[FAIL] Database connection failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TABLE CREATION — Define the database schema
# ═══════════════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
-- ╔═══════════════════════════════════════════╗
-- ║  Smart Nutrition Tracker — Database Schema ║
-- ╚═══════════════════════════════════════════╝

-- Users table: who is using the system
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    username        VARCHAR(100) UNIQUE NOT NULL,
    email           VARCHAR(255),
    age             INTEGER,
    weight_kg       DECIMAL(5,2),
    height_cm       DECIMAL(5,2),
    gender          VARCHAR(20),
    activity_level  VARCHAR(50),       -- sedentary, light, moderate, active, very_active
    goal            VARCHAR(50),       -- lose_weight, maintain, gain_muscle
    daily_cal_target INTEGER,          -- calculated from profile
    dietary_prefs   JSONB DEFAULT '[]', -- vegetarian, vegan, keto, gluten_free, etc.
    allergies       JSONB DEFAULT '[]', -- nuts, dairy, shellfish, etc.
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

-- Meals table: every logged meal
CREATE TABLE IF NOT EXISTS meals (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id) ON DELETE CASCADE,
    meal_type       VARCHAR(20) NOT NULL,  -- breakfast, lunch, dinner, snack
    food_name       VARCHAR(255) NOT NULL,
    food_category   VARCHAR(100),          -- from Food-101 classification
    quantity        DECIMAL(8,2) DEFAULT 1.0,
    unit            VARCHAR(50) DEFAULT 'serving', -- serving, gram, cup, piece
    calories        DECIMAL(8,2),
    protein_g       DECIMAL(8,2),
    carbs_g         DECIMAL(8,2),
    fat_g           DECIMAL(8,2),
    fiber_g         DECIMAL(8,2),
    sugar_g         DECIMAL(8,2),
    sodium_mg       DECIMAL(8,2),
    confidence      DECIMAL(5,4),          -- ML model confidence (0-1)
    source          VARCHAR(50),           -- manual, ml_model, open_food_facts, barcode
    image_path      TEXT,                  -- path to food image (if uploaded)
    notes           TEXT,
    logged_at       TIMESTAMP DEFAULT NOW(),
    meal_date       DATE DEFAULT CURRENT_DATE
);

-- Daily nutrition summary: aggregated per day per user
CREATE TABLE IF NOT EXISTS daily_nutrition (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id) ON DELETE CASCADE,
    log_date        DATE NOT NULL,
    total_calories  DECIMAL(8,2) DEFAULT 0,
    total_protein   DECIMAL(8,2) DEFAULT 0,
    total_carbs     DECIMAL(8,2) DEFAULT 0,
    total_fat       DECIMAL(8,2) DEFAULT 0,
    total_fiber     DECIMAL(8,2) DEFAULT 0,
    total_sugar     DECIMAL(8,2) DEFAULT 0,
    total_sodium    DECIMAL(8,2) DEFAULT 0,
    meals_count     INTEGER DEFAULT 0,
    water_ml        INTEGER DEFAULT 0,
    notes           TEXT,
    UNIQUE(user_id, log_date)
);

-- Food cache: avoid re-querying Open Food Facts for the same food
CREATE TABLE IF NOT EXISTS food_cache (
    id              SERIAL PRIMARY KEY,
    food_name       VARCHAR(255) NOT NULL,
    food_name_lower VARCHAR(255) NOT NULL,  -- lowercase for searching
    calories_per_100g DECIMAL(8,2),
    protein_per_100g  DECIMAL(8,2),
    carbs_per_100g    DECIMAL(8,2),
    fat_per_100g      DECIMAL(8,2),
    fiber_per_100g    DECIMAL(8,2),
    sugar_per_100g    DECIMAL(8,2),
    source          VARCHAR(50),           -- open_food_facts, manual, usda
    raw_data        JSONB,                 -- full API response for reference
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE(food_name_lower, source)
);

-- Conversation log: A2A message history for debugging
CREATE TABLE IF NOT EXISTS conversations (
    id              SERIAL PRIMARY KEY,
    message_id      VARCHAR(100) NOT NULL,
    conversation_id VARCHAR(100) NOT NULL,
    sender_id       VARCHAR(100),
    sender_role     VARCHAR(20),
    message_type    VARCHAR(20),           -- query, response, event
    payload         JSONB,
    agent           VARCHAR(100),          -- which agent handled this
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_meals_user_date ON meals(user_id, meal_date);
CREATE INDEX IF NOT EXISTS idx_meals_date ON meals(meal_date);
CREATE INDEX IF NOT EXISTS idx_daily_user_date ON daily_nutrition(user_id, log_date);
CREATE INDEX IF NOT EXISTS idx_food_cache_name ON food_cache(food_name_lower);
CREATE INDEX IF NOT EXISTS idx_conversations_conv ON conversations(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at);
"""


def create_tables():
    """
    Create all tables in the database.
    Safe to run multiple times (uses IF NOT EXISTS).
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(SCHEMA_SQL)
        conn.commit()
        cur.close()
        print("[OK] All tables created successfully!")
    except Exception as e:
        conn.rollback()
        print(f"[FAIL] Table creation failed: {e}")
        raise
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. USER OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_user(
    username: str,
    email: str = None,
    age: int = None,
    weight_kg: float = None,
    height_cm: float = None,
    gender: str = None,
    activity_level: str = "moderate",
    goal: str = "maintain",
    daily_cal_target: int = 2000,
    dietary_prefs: list = None,
    allergies: list = None,
) -> Dict:
    """
    Create a new user.

    Returns the created user as a dict.
    """
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            INSERT INTO users (username, email, age, weight_kg, height_cm, gender,
                              activity_level, goal, daily_cal_target, dietary_prefs, allergies)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (username, email, age, weight_kg, height_cm, gender,
             activity_level, goal, daily_cal_target,
             Json(dietary_prefs or []), Json(allergies or [])),
        )
        user = dict(cur.fetchone())
        conn.commit()
        return user
    finally:
        conn.close()


def get_user(user_id: int = None, username: str = None) -> Optional[Dict]:
    """Get a user by ID or username."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if user_id:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        elif username:
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        else:
            return None
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_users() -> List[Dict]:
    """Get all users."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM users ORDER BY id")
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def update_user(user_id: int, **fields) -> Optional[Dict]:
    """Update user fields. Only non-None fields are updated."""
    if not fields:
        return get_user(user_id=user_id)

    # Handle list fields that need JSON wrapping
    for key in ('dietary_prefs', 'allergies'):
        if key in fields and isinstance(fields[key], list):
            fields[key] = Json(fields[key])

    set_parts = []
    values = []
    for k, v in fields.items():
        set_parts.append(f"{k} = %s")
        values.append(v)
    values.append(user_id)

    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            f"UPDATE users SET {', '.join(set_parts)}, updated_at = NOW() WHERE id = %s RETURNING *",
            values,
        )
        row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_user(user_id: int) -> bool:
    """Delete a user and all their data (CASCADE). Returns True if deleted."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MEAL OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def log_meal(
    user_id: int,
    food_name: str,
    meal_type: str = "snack",
    calories: float = None,
    protein_g: float = None,
    carbs_g: float = None,
    fat_g: float = None,
    fiber_g: float = None,
    sugar_g: float = None,
    sodium_mg: float = None,
    quantity: float = 1.0,
    unit: str = "serving",
    food_category: str = None,
    confidence: float = None,
    source: str = "manual",
    image_path: str = None,
    notes: str = None,
    meal_date: date = None,
) -> Dict:
    """
    Log a meal for a user.

    Args:
        user_id:     User's ID
        food_name:   What was eaten (e.g., "Spaghetti Carbonara")
        meal_type:   breakfast, lunch, dinner, or snack
        calories:    Total calories
        protein_g:   Protein in grams
        carbs_g:     Carbohydrates in grams
        fat_g:       Fat in grams
        confidence:  ML model confidence (0.0 to 1.0)
        source:      Where the data came from (manual, ml_model, open_food_facts)

    Returns:
        The created meal record as a dict
    """
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            INSERT INTO meals (user_id, food_name, meal_type, calories, protein_g,
                              carbs_g, fat_g, fiber_g, sugar_g, sodium_mg,
                              quantity, unit, food_category, confidence, source,
                              image_path, notes, meal_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (user_id, food_name, meal_type, calories, protein_g,
             carbs_g, fat_g, fiber_g, sugar_g, sodium_mg,
             quantity, unit, food_category, confidence, source,
             image_path, notes, meal_date or date.today()),
        )
        meal = dict(cur.fetchone())
        conn.commit()

        # Update daily nutrition summary
        _update_daily_nutrition(conn, user_id, meal_date or date.today())

        return meal
    finally:
        conn.close()


def get_meals(
    user_id: int,
    meal_date: date = None,
    meal_type: str = None,
    limit: int = 20,
) -> List[Dict]:
    """
    Get meals for a user, optionally filtered by date and type.
    """
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        query = "SELECT * FROM meals WHERE user_id = %s"
        params = [user_id]

        if meal_date:
            query += " AND meal_date = %s"
            params.append(meal_date)
        if meal_type:
            query += " AND meal_type = %s"
            params.append(meal_type)

        query += " ORDER BY logged_at DESC LIMIT %s"
        params.append(limit)

        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def update_meal(meal_id: int, **fields) -> Optional[Dict]:
    """Update meal fields. Returns updated meal or None."""
    if not fields:
        return None
    set_parts = []
    values = []
    for k, v in fields.items():
        set_parts.append(f"{k} = %s")
        values.append(v)
    values.append(meal_id)
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            f"UPDATE meals SET {', '.join(set_parts)} WHERE id = %s RETURNING *",
            values,
        )
        row = cur.fetchone()
        conn.commit()
        if row:
            meal = dict(row)
            _update_daily_nutrition(conn, meal['user_id'], meal['meal_date'])
            return meal
        return None
    finally:
        conn.close()


def delete_meal(meal_id: int) -> bool:
    """Delete a meal and recalculate daily totals. Returns True if deleted."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT user_id, meal_date FROM meals WHERE id = %s", (meal_id,))
        info = cur.fetchone()
        if not info:
            return False
        user_id, meal_date = info['user_id'], info['meal_date']
        cur.execute("DELETE FROM meals WHERE id = %s", (meal_id,))
        conn.commit()
        # Check if any meals remain for this day (use plain cursor for COUNT)
        cur2 = conn.cursor()
        cur2.execute("SELECT COUNT(*) FROM meals WHERE user_id = %s AND meal_date = %s",
                     (user_id, meal_date))
        remaining = cur2.fetchone()[0]
        if remaining > 0:
            _update_daily_nutrition(conn, user_id, meal_date)
        else:
            cur2.execute("DELETE FROM daily_nutrition WHERE user_id = %s AND log_date = %s",
                         (user_id, meal_date))
            conn.commit()
        return True
    finally:
        conn.close()


def _update_daily_nutrition(conn, user_id: int, log_date: date):
    """Recalculate daily nutrition totals from meals."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO daily_nutrition (user_id, log_date, total_calories, total_protein,
                                     total_carbs, total_fat, total_fiber, total_sugar,
                                     total_sodium, meals_count)
        SELECT user_id, meal_date,
               COALESCE(SUM(calories), 0),
               COALESCE(SUM(protein_g), 0),
               COALESCE(SUM(carbs_g), 0),
               COALESCE(SUM(fat_g), 0),
               COALESCE(SUM(fiber_g), 0),
               COALESCE(SUM(sugar_g), 0),
               COALESCE(SUM(sodium_mg), 0),
               COUNT(*)
        FROM meals
        WHERE user_id = %s AND meal_date = %s
        GROUP BY user_id, meal_date
        ON CONFLICT (user_id, log_date)
        DO UPDATE SET
            total_calories = EXCLUDED.total_calories,
            total_protein = EXCLUDED.total_protein,
            total_carbs = EXCLUDED.total_carbs,
            total_fat = EXCLUDED.total_fat,
            total_fiber = EXCLUDED.total_fiber,
            total_sugar = EXCLUDED.total_sugar,
            total_sodium = EXCLUDED.total_sodium,
            meals_count = EXCLUDED.meals_count
        """,
        (user_id, log_date),
    )
    conn.commit()


def get_daily_summary(user_id: int, log_date: date = None) -> Optional[Dict]:
    """Get daily nutrition summary for a user."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM daily_nutrition WHERE user_id = %s AND log_date = %s",
            (user_id, log_date or date.today()),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FOOD CACHE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def cache_food(
    food_name: str,
    calories_per_100g: float = None,
    protein_per_100g: float = None,
    carbs_per_100g: float = None,
    fat_per_100g: float = None,
    fiber_per_100g: float = None,
    sugar_per_100g: float = None,
    source: str = "manual",
    raw_data: dict = None,
) -> Dict:
    """Cache nutrition data for a food item to avoid re-querying APIs."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            INSERT INTO food_cache (food_name, food_name_lower, calories_per_100g,
                                    protein_per_100g, carbs_per_100g, fat_per_100g,
                                    fiber_per_100g, sugar_per_100g, source, raw_data)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (food_name_lower, source) DO UPDATE SET
                calories_per_100g = EXCLUDED.calories_per_100g,
                protein_per_100g = EXCLUDED.protein_per_100g,
                carbs_per_100g = EXCLUDED.carbs_per_100g,
                fat_per_100g = EXCLUDED.fat_per_100g,
                raw_data = EXCLUDED.raw_data
            RETURNING *
            """,
            (food_name, food_name.lower(), calories_per_100g, protein_per_100g,
             carbs_per_100g, fat_per_100g, fiber_per_100g, sugar_per_100g,
             source, Json(raw_data) if raw_data else None),
        )
        result = dict(cur.fetchone())
        conn.commit()
        return result
    finally:
        conn.close()


def lookup_food(food_name: str) -> Optional[Dict]:
    """Look up cached nutrition data for a food item."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM food_cache WHERE food_name_lower = %s ORDER BY created_at DESC LIMIT 1",
            (food_name.lower(),),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CONVERSATION LOG — Store A2A messages for debugging
# ═══════════════════════════════════════════════════════════════════════════════

def log_message(msg_dict: Dict, agent: str = "unknown"):
    """Save an A2A message to the conversations table."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO conversations (message_id, conversation_id, sender_id,
                                       sender_role, message_type, payload, agent)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                msg_dict.get("message_id", ""),
                msg_dict.get("conversation_id", ""),
                msg_dict.get("sender", {}).get("id", ""),
                msg_dict.get("sender", {}).get("role", ""),
                msg_dict.get("type", ""),
                Json(msg_dict.get("payload", {})),
                agent,
            ),
        )
        conn.commit()
    except Exception as e:
        print(f"[WARN] Failed to log message: {e}")
    finally:
        conn.close()


def get_conversation(conversation_id: str) -> List[Dict]:
    """Get all messages in a conversation thread."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM conversations WHERE conversation_id = %s ORDER BY created_at",
            (conversation_id,),
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP & TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Smart Nutrition Tracker - Database Setup\n")

    # Test connection
    print("1. Testing connection...")
    if test_connection():
        print("   [OK] Connected to PostgreSQL!\n")
    else:
        print("   [FAIL] Cannot connect. Make sure PostgreSQL is running and the")
        print("      database exists. Run these commands in psql:\n")
        print("      CREATE USER nutrition_user WITH PASSWORD 'nutrition_pass';")
        print("      CREATE DATABASE nutrition_tracker OWNER nutrition_user;\n")
        exit(1)

    # Create tables
    print("2. Creating tables...")
    create_tables()

    # Verify tables
    print("\n3. Verifying tables...")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' ORDER BY table_name
    """)
    tables = [row[0] for row in cur.fetchall()]
    conn.close()

    expected = ["conversations", "daily_nutrition", "food_cache", "meals", "users"]
    for t in expected:
        status = "[OK]" if t in tables else "[MISSING]"
        print(f"   {status} {t}")

    print(f"\nDatabase ready! {len(tables)} tables created.")
