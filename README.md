# Containerized PostgreSQL Stack

This project now ships with a lightweight Docker Compose stack so analysts can spin up a local PostgreSQL instance that mirrors production schemas without polluting their host.

## 1. Configure environment variables
1. Copy the sample variables and fill in secrets:
   ```bash
   cp .env.example .env
   ```
2. Adjust the required values in `.env` (`POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `PGDATA`, `TZ`). Optional overrides such as `LANG`, `LC_ALL`, or the pgAdmin login values can also live there.

> The Compose file relies on named volumes (`postgres_data`, `pgadmin_data`) so Docker Desktop can create consistent Linux paths whether you run it from Windows or WSL. Avoid binding host folders under `C:\Users` directly to Postgres unless you explicitly enable file-sharing inside Docker Desktop.

## 2. Launch the services
Run the stack from the repository root:
```bash
docker compose up -d
```
This command starts the `postgres` container and, if desired, the pgAdmin UI. Follow up with `docker compose logs -f postgres` until you see `database system is ready to accept connections`.

To stop containers while preserving volumes:
```bash
docker compose down
```
To remove the data volumes entirely (drops all databases):
```bash
docker compose down --volumes
```

## 3. Verify connectivity
- **From WSL** (requires `postgresql-client`):
  ```bash
  source .env && \
  PGPASSWORD="$POSTGRES_PASSWORD" psql -h localhost -p 5432 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\\conninfo'
  ```
  If you installed Docker Desktop + WSL integration, `localhost` resolves to the Linux VM and this command should return the connected database name.

- **From Windows PowerShell** (assuming `psql.exe` is on `PATH`):
  ```powershell
  $env:POSTGRES_USER = "finance_admin"
  $env:POSTGRES_DB = "finance"
  $env:PGPASSWORD = "please-change-me"
  psql.exe -h localhost -p 5432 -U $env:POSTGRES_USER -d $env:POSTGRES_DB -c '\conninfo'
  Remove-Item Env:PGPASSWORD
  ```
  Replace the placeholder values with those stored in your `.env`. Docker Desktop exposes the port directly to Windows, so `localhost:5432` works from PowerShell, pgAdmin, DBeaver, etc.

## 4. pgAdmin helper (optional)
Access pgAdmin at [http://localhost:5050](http://localhost:5050) (the port is configurable via `PGADMIN_PORT`). Use the credentials from `.env`. Once logged in, register a new server pointing at `host=postgres`, `port=5432`, and the Postgres user/password you configured.

## 5. Database migrations & seeds
Normalized schema files live under `db/migrations/` (e.g., `001_create_core_tables.sql`, `002_seed_reference_data.sql`). The Compose file now mounts that folder into the container and `docker/init/01_apply_migrations.sh` replays every `.sql` file automatically during the initial boot so you get `countries`, `assets`, and `asset_daily_prices` tables plus starter data for ISO codes and representative assets.

To reapply migrations after changing the SQL:
- Drop the data volume (`docker compose down --volumes`) before running `docker compose up -d` again, **or**
- Run them manually: `source .env && for f in db/migrations/*.sql; do PGPASSWORD=\"$POSTGRES_PASSWORD\" psql -h localhost -p 5432 -U \"$POSTGRES_USER\" -d \"$POSTGRES_DB\" -v ON_ERROR_STOP=1 -f \"$f\"; done`

## 6. Troubleshooting tips
- Confirm Docker Desktop file sharing for your cloned path if volumes fail to mount on Windows.
- Ensure `PGDATA` in `.env` stays under `/var/lib/postgresql/data` so the named volume captures the full data directory; only change the trailing folder name if needed.
- If port `5432` already runs on the host, change the left side of the port mapping inside `docker-compose.yml` (e.g., `6543:5432`).
