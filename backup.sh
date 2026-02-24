#!/bin/sh
# PostgreSQL backup script for Smart Nutrition Tracker
# Runs via cron inside the backup container (see docker-compose.yml)
#
# Keeps the last 7 daily backups. Older files are deleted automatically.

set -e

BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
FILENAME="nutrition_tracker_${TIMESTAMP}.sql.gz"

echo "[BACKUP] Starting backup at $(date -Iseconds)"

pg_dump -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" "$PGDATABASE" \
  | gzip > "${BACKUP_DIR}/${FILENAME}"

SIZE=$(du -h "${BACKUP_DIR}/${FILENAME}" | cut -f1)
echo "[BACKUP] Created ${FILENAME} (${SIZE})"

# Delete backups older than 7 days
find "$BACKUP_DIR" -name "nutrition_tracker_*.sql.gz" -mtime +7 -delete
REMAINING=$(find "$BACKUP_DIR" -name "nutrition_tracker_*.sql.gz" | wc -l)
echo "[BACKUP] Done. ${REMAINING} backups on disk."
