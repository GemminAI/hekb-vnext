"""HEKB vNext: local transport authentication (SPEC-HEKB-REFACTOR-2026-v2.0
section 5.3, Step 2 of the refactor roadmap).

This is a NEW, separate layer from the existing `X-Audit-Signature` P-256
governance check in `store_service.py` (which attests to *who authored a
write's payload*). This layer instead answers *is this caller even allowed
to talk to this process at all* -- it applies to every request, including
reads and `/health`, not just `POST /experience`. No prior transport-auth
code existed in this repo before this file (verified: `store_service.py`
had no `Authorization`/Bearer/UDS handling before this change).

Two independent mechanisms, matching spec section 5.3:
  1. Ephemeral Bearer token (`hekb.token`, mode 0600) for the TCP loopback
     listener (`127.0.0.1`) -- dev/test convenience path.
  2. UDS peer-credential (`LOCAL_PEERCRED`) verification for the primary
     `hekb.sock` listener -- see `auth/uds_guard.py`.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import Header, HTTPException

TOKEN_BYTE_LENGTH = 32  # 256-bit ephemeral token


def generate_bearer_token(token_path: Path) -> str:
    """Creates a new random token, writes it to `token_path` with mode 0600
    (created with that mode from the start via os.open, not chmod'd after
    the fact -- avoids a window where the file is briefly world-readable)."""
    token = secrets.token_hex(TOKEN_BYTE_LENGTH)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(token_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode("ascii"))
    finally:
        os.close(fd)
    return token


def load_bearer_token(token_path: Path) -> str:
    return token_path.read_text().strip()


class BearerTokenGuard:
    """FastAPI dependency: raises 401 unless `Authorization: Bearer <token>`
    exactly matches the configured token. Fails closed -- if no token has
    been configured at all (`self.token is None`), every request is
    rejected rather than treated as 'auth not required'."""

    def __init__(self, token: str | None):
        self.token = token

    def __call__(self, authorization: str | None = Header(None)) -> None:
        if self.token is None:
            raise HTTPException(status_code=401, detail="local auth not configured; refusing all requests")
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
        presented = authorization[len("Bearer "):]
        if not secrets.compare_digest(presented, self.token):
            raise HTTPException(status_code=401, detail="invalid bearer token")
