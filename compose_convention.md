---
title: Docker Compose
---

# Docker Compose Standards

These are our conventions for writing Docker Compose files. Following them makes it easier to jump between projects and onboard new team members.

---

## File Naming

!!! note "`.yml` vs `.yaml`"
    Both `.yml` and `.yaml` extensions work everywhere. But, prefer using `.yml` - it's shorter to type.

For local overrides, use `compose.override.yml` (Docker auto-loads it). Add it to `.gitignore` so each dev can customize without affecting others.

### Override Templates

We keep environment-specific overrides in a `templates/` folder:

```
project/
├── compose.yml              # Base config (committed)
├── compose.override.yml     # Your local tweaks (gitignored)
└── templates/
    └── compose/
        ├── compose.override.dev.yml
        ├── compose.override.staging.yml
        └── compose.override.prod.yml
```

Copy the template you need, then run `docker compose up` - Docker auto-loads `compose.override.yml`:

```bash
cp templates/compose/compose.override.dev.yml compose.override.yml
docker compose up
```

Use `compose.override.yml` instead of editing the templates directly:

- **Templates are shared** - editing them affects everyone who copies that template and every environment using it
- **Override is gitignored** - your local tweaks (ports, volumes, debug flags) stay off the repo and don't pollute others' environments
- **Each dev can customize freely** - different machines, different paths, no conflicts

If a change is intentional and should apply across the whole project for everyone, edit the template directly instead of your personal override.

### Avoid `-f` when possible

!!! warning ""
    Prefer `compose.yml` + `compose.override.yml`. Avoid chaining multiple `-f` flags.

Using `-f` (e.g. `docker compose -f compose.yml -f override.yml up`) becomes hard to maintain as the number of compose files grows - it's easy to lose track of which files are being merged and in what order. It quickly turns into this:

```bash
docker compose -f compose.yml -f compose.fixed.yml -f compose.final-fixed.yml -f compose.please-work.yml up
```

You should only rely on `compose.yml` (base) and `compose.override.yml` - focus on those two files. Prefer copying the template to `compose.override.yml` and letting Docker auto-load it.

**Exception:** `-f` is reasonable for isolated 3rd party services that have their own separate compose file (e.g. `docker compose -f compose.yml -f compose.service-3rd.yml up`).

---

## Naming Conventions

**Project names** - lowercase, alphanumeric and hyphens allowed (not underscores). By default it uses your folder name, which usually works fine.

**Service names** - lowercase, alphanumeric and hyphens allowed (not underscores), keep them short. Order services by dependency - what needs to start first goes first:

```yaml
services:
  db:           # starts first
  redis:        # depends on nothing
  api:          # depends on db, redis
  worker:       # depends on api
  frontend:     # depends on api
```

Don't set `container_name` unless necessary - let Docker Compose handle it. It auto-generates names as **`{project-name-or-slug}-{service}-{index}`** (e.g. `myproject-db-1`, `myproject-api-2`). Explicit container names break scaling (`docker compose up --scale`) and cause conflicts when running multiple instances. If you do set one, avoid generic names like `db`, `redis`, or `api` - they collide with other projects running on the same host. Use a project-scoped name like `myproject-db` instead.

!!! danger "Silent conflict with shared container names"
    If two projects share the same folder name **and** the same `container_name`, Docker silently reuses the existing container - the second project never actually starts its own. No error, no warning. If the folder names differ, Docker will throw a conflict error instead.

Prefix the folder name based on environment to avoid conflicts:

| Environment | Folder Name |
|-------------|-------------|
| Production | `prod-myproject` |
| Staging | `staging-myproject` |
| Shared dev server | `john-myproject`, `jane-myproject` |

**Production image naming** - always use a registry path with an explicit version tag:

```yaml
services:
  api:
    image: registry.humblebee.ai/owner/myproject-api:1.2.3   # private registry
    # image: myproject-api:1.2.3                             # Docker Hub
    # image: myproject-api:latest                           # dev only
```

!!! warning "Never use `latest` in production"
    It makes rollbacks and debugging impossible.

---

## Service Property Order

When writing a service, use this order. It's not required, but consistency helps when scanning files:

1. `image`
2. `build`
3. `restart`
4. `depends_on` (if applicable)
5. `environment`
6. `env_file` (optional - only for custom filenames)
7. `volumes`
8. `networks`
9. `ports` / `expose`
10. `entrypoint` (if necessary)
11. `command`
12. `healthcheck`
13. `deploy`

---

## Ports

Use `ports` to map to the host, `expose` to make a port available only to other containers (not the host):

| Property | Accessible From |
|----------|----------------|
| `ports` | Host and other containers |
| `expose` | Other containers only |

Always define ports dynamically using environment variables - never hardcode them:

```yaml
services:
  api:
    ports:
      - "${MP_API_PORT:-8080}:8080"
```

!!! danger "Never hardcode ports"
    Hardcoded ports cause conflicts when running multiple instances or environments on the same host. Dynamic ports are required.

---

## Environment Variables

Keep them in `.env` files, not hardcoded in `compose.yml`. Group them by concern (e.g. app config, database, logging) - it's easier to find and manage related variables together.

Variable substitution syntax:

| Syntax | Behavior |
|--------|----------|
| `${MP_VAR}` | Use value - allows empty, shows warning if unset |
| `${MP_VAR:-default}` | Use default if unset or empty |
| `${MP_VAR:?error message}` | Throw error and stop if unset or empty |

Use `:?` for variables that must be explicitly set - place these in the production override, not in the base `compose.yml`. The base should always have defaults so it works locally:

```yaml
# compose.yml - base, has defaults
environment:
  - MP_SECRET_KEY=${MP_SECRET_KEY:-dev-secret}

# compose.override.prod.yml - production, enforces required values
environment:
  - MP_SECRET_KEY=${MP_SECRET_KEY:?MP_SECRET_KEY is required}
```

Use `${MP_VAR:-default}` so the service works even without a `.env` file:

```yaml
services:
  api:
    environment:
      # App config
      - MP_API_PORT=${MP_API_PORT:-8080}
      - MP_DEBUG=${MP_DEBUG:-false}
      - MP_LOG_LEVEL=${MP_LOG_LEVEL:-info}
      # Database
      - MP_DB_PASSWORD=${MP_DB_PASSWORD:-secret}
      # Security
      - MP_SECRET_KEY=${MP_SECRET_KEY:?MP_SECRET_KEY is required}
```

Without `env_file`, variables from `.env` are only used for substitution inside `compose.yml` (e.g. `${MP_API_PORT}`) - they are **not** passed into the container. To actually inject them into the container's environment, you need `env_file`.

Use `required: false` so the service still starts even without the file, falling back to default values defined in `environment:`:

```yaml
services:
  api:
    env_file:
      - path: .env
        required: false    # service starts even if .env doesn't exist
```

Prefix variables with the service or project name to avoid conflicts with other services or system variables. Avoid generic names like `PORT`, `HOST`, or `PASSWORD`. In the examples below, `MP_` is short for **my-project** - replace it with your own project prefix:

```yaml
# Bad - conflicts with other services
- PORT=8080
- PASSWORD=secret

# Good - scoped to the service
- MP_API_PORT=8080
- MP_DB_PASSWORD=secret
```

!!! danger "Never commit secrets"
    Add `.env*` to `.gitignore`. Passwords in git history are painful to remove.

---

## Secrets

The `secrets` field is an optional alternative to environment variables for sensitive values like passwords and API keys. Environment variables can be exposed in logs and `docker inspect`, so secrets provide an extra layer of isolation - they are mounted as files at `/run/secrets/` instead:

```yaml
secrets:
  db_password:
    file: ./secrets/db_password.txt

services:
  db:
    secrets:
      - db_password
```

For production, use a proper secret manager ([Vault](https://www.vaultproject.io/), [AWS SSM](https://docs.aws.amazon.com/systems-manager/latest/userguide/systems-manager-parameter-store.html), [Infisical](https://infisical.com/), etc.). For local dev, `.env` files are fine - just don't commit them.

---

## Volumes

**Named volumes** - Docker manages the storage location. You only provide a name. The default path is `/var/lib/docker/volumes/` on Linux - it can differ on Mac and Windows, and can be configured in Docker settings. Recommended for stateful services like databases and test environments - Docker optimizes I/O and it works consistently across all platforms (Linux, Mac, Windows). Mandatory when deploying to Kubernetes:

```yaml

services:
  db:
    volumes:
      - postgres-data:/var/lib/postgresql/data   # reference by name

volumes:
  postgres-data:
```

!!! warning "Risk - no volume on stateful services"
    Skipping volumes on databases or stateful services means all data is permanently lost on container removal or system shutdown.

Name them descriptively: `postgres-data`, `db-data`, `postgres-db`, `redis-cache` - not `vol1`, `data`.

---

**Bind mounts** - you control the path, directly linking a host directory into the container. Preferred for config files, host paths, and development hot reload:

```yaml
services:
  api:
    volumes:
      - ./src:/app/src                      # Dev only - hot reload
      - ./config.yaml:/app/config.yaml:ro
```

Bind mounts are not limited to the project folder - you can mount any path on the host. You can also use symbolic links to point to paths outside the project:

```bash
ln -s /data/models ./models
```

```yaml
services:
  api:
    volumes:
      - /data/shared:/app/data              # Absolute path outside project
      - ./models:/app/models                # Symlink resolves to /data/models
```

!!! danger "Never bind mount `src` in production"
    Mounting source code in production breaks image immutability and reproducibility. In production, always rebuild and redeploy the image.

---

## Networks

By default Docker Compose creates a bridge network for all services - this is sufficient for most cases. Custom networks are recommended for applications that require stricter network isolation (e.g. separating public-facing services from internal ones).

| Driver | When to Use |
|--------|-------------|
| `bridge` | Default - isolated network between containers on the same host |
| `host` | Shares host network stack - use only in specific cases (e.g. high-performance networking, monitoring agents) |

Use custom networks when you need to control which services can reach each other:

```yaml




services:
  db:
    networks: [backend]      # Can't be reached from frontend
  api:
    networks: [backend, frontend]
  nginx:
    networks: [frontend]

networks:
  backend:
  frontend:
```

Default naming: `backend`, `frontend`, or `{project}-internal`.

!!! warning "Don't use `host` network in production"
    `host` removes network isolation between the container and the host. Avoid it in production services - only use it in specific cases where performance or low-level network access is required.

---

## Health Checks

Health checks are recommended. They let Docker know if your service is actually working and enable proper `depends_on` readiness checks.

Simple `depends_on` only waits for the container to **start**, not to be **ready**. Use `condition: service_healthy` to wait for the health check to pass:

```yaml
services:
  api:
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

The `start_period` should cover your app's startup time, otherwise it'll restart during boot.

The `test` command depends on what's available in your image - not all images have `curl`:

| Command | Use Case |
|---------|----------|
| `curl -f http://localhost/health` | HTTP services with curl |
| `wget -qO- http://localhost/health` | Lightweight images (alpine) |
| `pg_isready -U postgres` | PostgreSQL |
| `redis-cli ping` | Redis |
| `ls /tmp/healthy` | Simple file-based check |
| `nc -z localhost 8080` | TCP port check |

!!! note "Adjust timeouts per environment"
    Services start slower and respond under more load in production. Increase `timeout`, `interval`, and `start_period` in production overrides - values that work in dev may cause false failures in prod.

---

## Resource Limits

Required in production, recommended in development - a runaway process or memory leak shouldn't consume all host resources:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
```

- **`limits`** - hard cap. The service cannot exceed this. Other services can use whatever is left. Set this **above your highest expected spike** - if your service hits the limit during a traffic spike, it will be throttled or killed.
- **`reservations`** - guaranteed allocation. Docker reserves these resources exclusively for this service - other services cannot use them even if idle.

---

## Logging Driver

By default Docker uses `json-file`. Configure it to prevent logs from filling disk:

```yaml
services:
  api:
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "90"
```

For centralized logging, switch to `fluentd`, `loki` or your aggregation tool. See [Logging Conventions](../codebase/logs.md) for more details.

---

## Restart Policies

- **`unless-stopped`** *(highly recommended)* - for most services. Restarts on crash, survives reboots, but respects `docker stop`.
- `on-failure` - for jobs/workers that should retry but eventually give up.

```yaml
services:
  api:
    restart: unless-stopped
  worker:
    restart: on-failure
```

---

## Quick Checks

Before committing, verify:

- [ ] Using `compose.yml` (not docker-compose.yml)
- [ ] Services ordered by dependency (dependencies first)
- [ ] Environment variables grouped by concern
- [ ] Ports use environment variables (not hardcoded)
- [ ] No secrets in the file (use `.env` or secrets)
- [ ] Image versions pinned (`nginx:1.25` not `nginx:latest`)
- [ ] Health checks configured (mandatory in production, recommended in dev)
- [ ] `depends_on` uses `condition: service_healthy` where needed
- [ ] Logging driver configured with size limits
- [ ] Resource limits set (required in production, recommended in dev)
- [ ] `.env` files in `.gitignore`

---

## References

- [Compose File Reference](https://docs.docker.com/reference/compose-file/)
- [Services Reference](https://docs.docker.com/reference/compose-file/services/)
- [Project Naming](https://docs.docker.com/compose/how-tos/project-name/)
- [Environment Variables](https://docs.docker.com/compose/how-tos/environment-variables/best-practices/)
- [Secrets](https://docs.docker.com/compose/how-tos/use-secrets/)
- [Deploy Specification](https://docs.docker.com/reference/compose-file/deploy/)
- [Startup Order & Health Checks](https://docs.docker.com/compose/how-tos/startup-order/)
- [Image Tagging](https://www.docker.com/blog/docker-best-practices-using-tags-and-labels-to-manage-docker-image-sprawl/)
- [Coding Convention](https://gist.github.com/ccdle12/6095c492af084f67a0b730dd01d75865)