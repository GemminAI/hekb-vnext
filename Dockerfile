# python:3.12-slim, not this repo's own 3.14+ dev interpreter: the
# `zstandard` package already declared in requirements.txt exists
# precisely as a fallback for interpreters without the stdlib
# `compression.zstd` module (PEP 784, Python 3.14+) — see
# storage/trajectory_blob.py and requirements.txt's own comment. This
# image relies on that fallback rather than requiring a 3.14 base image.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HEKB_HOST=0.0.0.0 \
    HEKB_PORT=8300 \
    HEKB_DATA_DIR=/app/experience \
    HEKB_APP_SUPPORT_DIR=/app/app_support

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# auth/ and index/ are real runtime dependencies of store_service.py
# (auth.local_auth.BearerTokenGuard, index.recalibrate) added since this
# Dockerfile was first written -- without them the container fails at
# import time with ModuleNotFoundError. storage/ is index/recalibrate.py's
# own dependency (storage.trajectory_blob).
COPY store_service.py .
COPY vendor ./vendor
COPY auth ./auth
COPY index ./index
COPY storage ./storage

RUN mkdir -p /app/experience/objects /app/experience/relations /app/experience/lineages /app/experience/meta \
    && mkdir -p /app/app_support

EXPOSE 8300

# The local transport-auth Bearer guard (auth/local_auth.py) applies to
# EVERY route including /health, not just POST /experience -- an
# unauthenticated healthcheck gets a real 401, not a pass. main() writes
# the token to $HEKB_APP_SUPPORT_DIR/hekb.token at startup; read it fresh
# on each check rather than baking a value in, since it's regenerated
# every container start.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD curl -sf -H "Authorization: Bearer $(cat /app/app_support/hekb.token)" http://localhost:8300/health || exit 1

# --no-uds: the Unix Domain Socket / LOCAL_PEERCRED listener
# (auth/uds_guard.py) is for same-host, non-networked local IPC and has no
# meaning between separate containers, which reach this service over the
# TCP listener (HEKB_HOST/HEKB_PORT) on the compose network instead.
# HEKB_AUDIT_PUBLIC_KEY_PATH (POST /experience's X-Audit-Signature
# governance) is intentionally not set here -- it must be supplied via the
# environment and a mounted key file at deploy time; no key is baked into
# this image.
CMD ["python", "store_service.py", "--no-uds"]
