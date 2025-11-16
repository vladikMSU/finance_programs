#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
MIGRATIONS_DIR="$SCRIPT_DIR/migrations"

if [[ ! -d "$MIGRATIONS_DIR" ]]; then
  echo "[initdb] No migrations directory mounted at $MIGRATIONS_DIR; skipping."
  exit 0
fi

shopt -s nullglob
declare -a migrations=("$MIGRATIONS_DIR"/*.sql)
if [[ ${#migrations[@]} -eq 0 ]]; then
  echo "[initdb] No SQL files found in $MIGRATIONS_DIR; skipping."
  exit 0
fi

for file in "${migrations[@]}"; do
  echo "[initdb] Applying migration: $(basename "$file")"
  psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    --file "$file"
  echo "[initdb] Completed: $(basename "$file")"
  echo
done
