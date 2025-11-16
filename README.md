# Finance Programs Infrastructure

## Running Postgres with Docker Compose
This repository ships a lightweight Docker Compose stack that starts a Postgres database with optional tooling. The stack is defined in [`docker-compose.yml`](docker-compose.yml) and uses environment variables from your local `.env` file. A sample configuration is provided in [`.env.example`](.env.example).

### 1. Configure environment variables
1. Copy the example file and edit the secrets:
   ```bash
   cp .env.example .env
   # update POSTGRES_PASSWORD, PGADMIN credentials, etc.
   ```
2. The required variables are `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `PGDATA`, and `TZ`. Optional entries configure pgAdmin and locale (`LANG`, `LC_ALL`).
3. Any SQL or shell files placed under `docker/postgres-init/` will be executed automatically by the official Postgres entrypoint. Use this folder for repeatable seed data or schema changes.

### 2. Start the containers
```bash
docker compose up -d              # launches postgres
docker compose --profile pgadmin up -d  # adds pgAdmin when you need it
```
The stack uses named Docker volumes (`pg_data`, `pgadmin_data`) so it works out of the box on Linux, Windows (Docker Desktop), and WSL without path conversion issues.

### 3. Check status and logs
```bash
docker compose ps
docker compose logs -f postgres
```

### 4. Connect to the database
* **From WSL/Linux**
  ```bash
  psql -h localhost -p 5432 -U "$POSTGRES_USER" -d "$POSTGRES_DB"
  # or run inside the container
  docker exec -it finance-postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"
  ```
* **From Windows host**
  1. Use `psql` from PgAdmin/psql installed via PostgreSQL installer.
  2. Connect to `Host=localhost`, `Port=5432`, `Database=$POSTGRES_DB`, `User=$POSTGRES_USER`, `Password` set in `.env`.
  3. Docker Desktop shares the same networking namespace, so `localhost` works both from Windows CMD/PowerShell and WSL.

### Windows & WSL considerations
* Keep this repository inside your WSL filesystem (`/home/<user>/...`) or another Linux path for best performance. If you need to work from Windows Explorer, use the `\\wsl$\distro-name\workspace\finance_programs` network path.
* Named volumes (`pg_data`, `pgadmin_data`) are handled entirely by Docker Desktop, so there is no need to bind mount Windows paths. This avoids path translation bugs between NTFS and ext4 filesystems.
* When you *do* want to mount a Windows folder (for SQL scripts, backups, etc.), reference it using Docker Desktop's `/run/desktop/mnt/host/c/...` path format so it resolves in both Windows and WSL shells. Document those mounts alongside the service definition.
* Verify connectivity from both environments:
  ```powershell
  # Windows PowerShell
  docker exec -it finance-postgres pg_isready -U $env:POSTGRES_USER
  psql "host=localhost port=5432 dbname=$env:POSTGRES_DB user=$env:POSTGRES_USER"
  ```

## Shutting down
```bash
docker compose down                # stops containers, preserves volumes
# remove everything including volumes if you need a clean slate
docker compose down -v
```
