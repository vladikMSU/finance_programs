# Database Migrations

This folder keeps raw SQL migrations that set up the normalized market-data schema. Files are timestamp/sequence prefixed so Docker and developers can replay them deterministically:

- `001_create_core_tables.sql` creates the `countries`, `assets`, and `asset_daily_prices` tables, declares the `asset_type_enum`, enforces FK/unique constraints, and adds the `(asset_id, price_date)` index required by downstream analytics.
- `002_seed_reference_data.sql` seeds ISO country codes and a handful of representative assets so notebooks have immediately queryable reference data.

## Running migrations locally

The `postgres` service automatically runs the SQL files during its first boot via the mounted `docker/init/01_apply_migrations.sh` helper. If you need to re-run them manually against an existing database, either:

1. Drop the Docker volumes (`docker compose down --volumes`) and start the stack again, or
2. Run them explicitly with `psql` once the container is up:
   ```bash
   source .env
   for file in db/migrations/*.sql; do
       PGPASSWORD="$POSTGRES_PASSWORD" psql \
         -h localhost -p 5432 \
         -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
         -v ON_ERROR_STOP=1 -f "$file"
   done
   ```

Keep new migrations additive: never rewrite the older SQL files so teammates can rely on deterministic bootstrap behavior.
