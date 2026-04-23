---
title: Logging Conventions
---

# Logging Conventions

> **You are blind without logs.**

![You are blind without logs](../../assets/images/memes/meme-loggers-blind.jpeg)
<p style="text-align: right; font-size: 0.8em;"><a href="https://x.com/reactjpg/status/1353257878933520384" target="_blank" rel="noopener noreferrer">source</a></p>

Logs are your eyes in production. When something breaks at 3 AM, logs are the first thing you'll check. Make them count.

---

## Why Not Just Use Print?

**Don't use `print()` for debugging. Use a logger.**

`print()` has no log levels, no timestamps, no file/line info, and no way to filter or route output. A logger gives you all of that out of the box.

---

## Log Levels

Use the right level for the right situation:

| Level | When to Use | Example |
| ------- | ------------- | --------- |
| **DEBUG** | Detailed info (dev only) | Variable values, function entry/exit |
| **INFO** | Normal operations | "Server started", "User logged in" |
| **WARNING** | Something odd, but not broken | Retry attempt, deprecated API used |
| **ERROR** | Something failed | API call failed, database error |
| **CRITICAL** | System is down | Database unreachable, out of memory |
| **CUSTOM** | Your own levels if needed | AUDIT, TRACE, METRIC, SUCCESS |

Always include the file name and line number in your logs - it makes debugging so much easier.

---

## What to Log

=== ":material-check: Do Log"

    - Application start/stop events
    - User authentication (login, logout, failed attempts)
    - API requests (method, endpoint, response time, status)
    - Database operations (errors, slow queries)
    - External service calls (request, response, latency)
    - Errors with stack traces
    - Business-critical events (payments, orders)

=== ":material-close: Don't Log"

    - Passwords, tokens, API keys
    - Personal data (PII) - names, emails, phone numbers
    - Credit card numbers, SSN
    - Full request/response bodies with sensitive data

!!! danger "Never Log Secrets"
    Logging sensitive data is a security violation. Implement controls to exclude sensitive fields.

!!! tip "Pydantic SecretStr"
    Use Pydantic's `SecretStr` type for sensitive fields - it won't expose values in logs. Use `.get_secret_value()` only when you actually need the value. See [Security Conventions](../../policies/security/index.md) for more details.

---

## Log Format

JSON format is mandatory - machine-friendly, parseable, and queryable. Plain text is optional - human-friendly and easy to read, but hard to aggregate and search at scale.

=== "Plain text"

    ```text
    [2026-02-26 14:40:09.766 +09:00 | DEBUG | module_name._base:101]: Item '0.1' is below the threshold '0.7', removing it...
    ```

=== "JSON format"

    ```json
    {
      "timestamp": "2026-02-26 14:40:09.766 +09:00",
      "level": "DEBUG",
      "module_name": "module_name._base",
      "line": 101,
      "message": "Item '0.1' is below the threshold '0.7', removing it..."
    }
    ```

### Log Fields

!!! info
    Field names may vary between frameworks and libraries (e.g. `timestamp` vs `datetime`, `module_name` vs `logger`). What matters is the value - make sure the same information is present regardless of the key name.

#### Basic

Mandatory for all log entries:

| Field | Description |
| ----------- | ------------------------------------ |
| `timestamp` | ISO 8601 format with timezone |
| `level` | Log level (DEBUG, INFO, etc.) |
| `message` | Human-readable description |
| `filename` | Source file that triggered the log |
| `line` | Line number that triggered the log |

**Example:**

```json
{
  "timestamp": "2026-02-26T14:37:00.092+09:00",
  "level": "ERROR",
  "message": "division by zero",
  "filename": "__main__.py",
  "line": 23
}
```

#### Advanced

Mandatory for JSON logs in production/distributed systems, optional for plain text:

| Field              | Description                              |
| ------------------ | ---------------------------------------- |
| `service`          | Service/application name                 |
| `trace_id`         | Request trace ID for distributed tracing |
| `request_id`       | Unique identifier per request            |
| `user_id`          | User identifier (if applicable)          |
| `duration_ms`      | Operation duration in milliseconds       |
| `error`            | Error type/message                       |
| `instance_id`      | Identifies which prod server/pod failed  |
| `process_id`       | Process ID - useful for multiprocessing  |
| `thread_id`        | Thread ID - useful for multithreading    |

> **`request_id`** is generated per web request and stays within a single service - use it to track everything that happens during one HTTP call.
> **`trace_id`** spans across multiple services or packages - use it when a single operation triggers calls to other services, queues, or background workers.

**Example:**

```json
{
  "text": "2026-02-26 15:12:39.401 | INFO     | __main__:main:38 - Done!\n\n",
  "record": {
    "elapsed": {
      "repr": "0:00:00.094393",
      "seconds": 0.094393
    },
    "exception": null,
    "extra": {
      "level_short": "INFO"
    },
    "file": {
      "name": "main.py",
      "path": "/home/user/workspaces/projects/module-python-template/examples/simple/./main.py"
    },
    "function": "main",
    "level": {
      "icon": "ℹ️",
      "name": "INFO",
      "no": 20
    },
    "line": 38,
    "message": "Done!\n",
    "module": "main",
    "name": "__main__",
    "process": {
      "id": 2518837,
      "name": "MainProcess"
    },
    "thread": {
      "id": 133456131309696,
      "name": "MainThread"
    },
    "time": {
      "repr": "2026-02-26 15:12:39.401476+09:00",
      "timestamp": 1772086359.401476
    }
  }
}
```

#### HTTP Access Logs

| Field              | Description                             |
| ------------------ | --------------------------------------- |
| `http_version`     | HTTP protocol version (1.1, 2.0)        |
| `method`           | HTTP method (GET, POST, etc.)           |
| `user_id`          | Authenticated user identifier           |
| `request_id`       | Unique identifier for the request       |
| `path`             | Request URL path                        |
| `status_code`      | HTTP response status code               |
| `bytes_sent`       | Response size in bytes                  |
| `response_time_ms` | Response time for SLA monitoring        |
| `client_ip`        | Client IP address                       |
| `user_agent`       | Client browser/tool identifier          |
| `referer`          | Referring URL                           |

##### Plain Text — Apache Combined Log Format

```text
127.0.0.1 - frank [2026-03-15T14:36:20+09:00] "GET /api/v1/ping HTTP/1.1" 200 2326 "http://example.com/start" "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
```

Format: `%h %l %u %t "%r" %>s %b "%{Referer}i" "%{User-agent}i"`

| Field            | Value                                             |
| ---------------- | ------------------------------------------------- |
| `%h`             | Client IP (`127.0.0.1`)                           |
| `%l`             | Identity - usually `-`                            |
| `%u`             | Authenticated user (`frank`, or `-` if anonymous) |
| `%t`             | Timestamp                                         |
| `%r`             | Request line - method, path, HTTP version         |
| `%>s`            | Status code                                       |
| `%b`             | Bytes sent                                        |
| `%{Referer}i`    | Referring URL                                     |
| `%{User-agent}i` | User agent                                        |

##### JSON Format

```json
{
    "timestamp": "2026-03-15T14:36:20.048+09:00",
    "http_version": "1.1",
    "method": "GET",
    "path": "/api/v1/ping",
    "status_code": 200,
    "bytes_sent": 2326,
    "response_time_ms": 200,
    "client_ip": "127.0.0.1",
    "user_id": "frank",
    "request_id": "3d18f2fa574248099a2b03fa49a43ff6",
    "referer": "http://example.com/start",
    "user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
}
```

---

## Track Requests Across Services

Generate a `request_id` when a request comes in. Pass it through all your logs. If you have multiple services, use a `trace_id` to follow requests across them.

When something breaks, you can find all related logs instantly.

---

## Exception Logging

- Always log the full stack trace
- Use `logger.exception()` in Python
- **Log at the point where you have context, then re-raise** - don't let exceptions bubble up silently. When an exception propagates to a higher-level handler, local context like `trace_id`, `request_id`, and `user_id` is lost, making it impossible to properly analyze the issue.
- Each function that holds relevant context should log and re-raise, not delegate logging to a top-level handler.

!!! warning "Python Thread-Safe Logging"
    In multi-threaded Python applications, be careful with logging. Use [Loguru](https://loguru.readthedocs.io/en/stable/api/logger.html) or [beans-logging](https://pypi.org/project/beans-logging/) - they handle thread-safety, multiprocessing, formatting, and rotation out of the box.

---

## Log Levels by Environment

| Environment | Level | Why |
| ------------- | ------------ | ----------------------------------------------------------------------------------------------------------------- |
| Development | DEBUG | See everything |
| Staging | INFO | Catch issues early |
| Production | INFO/WARNING | Use WARNING to reduce noise when system is stable. Switch to INFO when debugging or investigating issues |

Allow overriding the log level via an environment variable when debugging. The name varies by language and framework (e.g. `LOG_LEVEL`, `RUST_LOG`, `LOGGING_LEVEL`) - use whatever fits your stack.

---

## Log Handlers

Handlers define **where** logs are sent:

| Handler | Status |
| -------------------- | ---------- |
| **Console (stdout)** | Mandatory |
| **Log files** | Recommended |
| **Remote** | Optional - send to aggregation system (ELK, Loki, etc.) |
| **Custom** | Optional - database, 3rd party services (e.g. Sentry, PagerDuty), or any handler specific to your service needs |

Use **separate log files per level** to make filtering and alerting easier:

| File | Contents |
| ------ | ---------- |
| `error.log` | ERROR, CRITICAL, and WARNING |
| `app.log` | Anything else (INFO, DEBUG, etc.) |
| `custom.log` | Any custom levels (e.g. AUDIT, TRACE, METRIC) |

---

## Log Rotation

![Log rotation meme](../../assets/images/memes/logger-rotation-meme.jpg)

Don't let log files grow forever. Both rotation strategies are **mandatory**:

### Based on Size

Rotate when a log file reaches a maximum size to prevent slow I/O and disk exhaustion:

- **Max file size:** 10-100MB
- **Compress** old files

### Based on Time

Rotate on a schedule regardless of file size to make log management and retention predictable:

- **Daily rotation** is recommended for most services

!!! danger "Large Files = Slow I/O"
    Large log files slow down read/write operations and can impact application performance. Always configure rotation.

### How Long to Keep Logs

| Log Type | Keep For | Storage |
| ---------- | ---------- | --------- |
| Security/Audit | 1 year | Database - queryable, tamper-evident |
| Errors | 3 months | Log files |
| Debug | 7 days | Log files |

### Containers

- **Kubernetes/Cloud Run:** Log to stdout, the platform handles the rest
- **Docker Volume:** Mount a volume and configure rotation yourself

---

## Writing Good Log Messages

- Be specific: "Order created" not "Done"
- Use past tense: "User logged in" not "User logging in"
- Keep the message static, put variables in fields

---

## Canonical Log Lines

Generate **one comprehensive log entry per request** instead of multiple entries. Include: method, path, status, duration, user, and trace ID.

This makes searching and debugging much simpler.

---

## High Traffic? Use Sampling

**Everything should be logged based on your application requirements.** Sampling applies to what is displayed or queried - not what is stored.

- **Always log 100% of errors** - never sample these
- Sample what you display (e.g. show 10% of successful requests in dashboards)
- Decrease sampling during incidents for full visibility

**Use batch log writing to reduce I/O overhead.** Instead of writing every log entry immediately, buffer logs and flush in batches. Sampling (what to log) and batching (how to write) are separate concerns.

---

## Best Practices

=== ":material-check: Dos"

    | Practice                       | Why                                              |
    | ------------------------------ | ------------------------------------------------ |
    | **Use structured logging**     | JSON format for easy parsing and querying        |
    | **Include trace/request IDs**  | Essential for distributed systems                |
    | **Use canonical log lines**    | One entry per request with all context           |
    | **Centralize logs**            | Aggregate into unified system (ELK, Loki, etc.)  |
    | **Set up alerts**              | Alert on ERROR/CRITICAL levels                   |
    | **Configure retention policy** | Balance accessibility, compliance, and costs     |
    | **Standardize field names**    | Consistent naming across all services            |
    | **Use metrics for resource monitoring** | CPU, memory, and throughput belong in metrics (Prometheus, Grafana) - not logs |

=== ":material-close: Don'ts"

    | Mistake                        | Why It's Bad                                     |
    | ------------------------------ | ------------------------------------------------ |
    | **Log sensitive data**         | Security violation, compliance risk              |
    | **Ignore performance costs**   | Poor logging can reduce throughput 20%           |
    | **Log inside tight loops without DEBUG** | Use DEBUG level only - INFO or above floods logs and slows performance |
    | **Skip log rotation**          | Disk fills up, service crashes                   |
    | **Use only plain text logs**        | Hard to parse, search, and analyze               |

---

## Where to Send Logs

Pick a log aggregation tool:

| Tool | Type | Collector |
| ---------------------------------------------------- | ----------------------------------------- | --------------------- |
| [**ELK Stack**](https://www.elastic.co/elastic-stack) | Self-hosted, full-featured | Logstash / Filebeat |
| [**Grafana Loki**](https://grafana.com/oss/loki/) | Lightweight, works great with Grafana | Promtail / Alloy |
| [**CloudWatch**](https://aws.amazon.com/cloudwatch/) | AWS-native | CloudWatch Agent |
| [**Datadog**](https://www.datadoghq.com/) | SaaS, easy setup | Datadog Agent |
| [**Sentry**](https://sentry.io/welcome/) | Error tracking and performance monitoring | Sentry SDK |

---

## Quick Checklist

Before shipping, make sure you have:

- [ ] Structured logging (JSON)
- [ ] Correct log levels
- [ ] No sensitive data in logs
- [ ] Request/trace IDs
- [ ] Stack traces for exceptions
- [ ] Log rotation configured
- [ ] Logs going to a central place
- [ ] Alerts on errors

---

## References

- [Apache HTTP Server Log Files](https://httpd.apache.org/docs/2.4/logs.html)
- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
- [Node.js Console Documentation](https://nodejs.org/api/console.html)
- [Rust log crate Documentation](https://docs.rs/log/latest/log/)
- [Loguru Documentation](https://loguru.readthedocs.io/en/stable/api/logger.html)
- [beans-logging](https://pypi.org/project/beans-logging/)
- [beans-logging-fastapi](https://pypi.org/project/beans-logging-fastapi/)
- [Engineer's Checklist of Logging Best Practices](https://www.honeycomb.io/blog/engineers-checklist-logging-best-practices)
- [Logging Best Practices](https://betterstack.com/community/guides/logging/logging-best-practices/)
- [Leveling Up Your Python Logs with Structlog](https://www.dash0.com/guides/python-logging-with-structlog)
- [Logging and Monitoring in Node.js - Best Practices](https://dev.to/imsushant12/logging-and-monitoring-in-nodejs-best-practices-2j1k)







