#!/bin/bash
set -e

echo "=== Smart Nutrition Tracker — Starting ==="

# Wait for PostgreSQL and initialize tables
echo "[INIT] Running database setup..."
python database.py

echo "[INIT] Database ready. Starting server..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
