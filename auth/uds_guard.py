"""HEKB vNext: Unix Domain Socket peer-credential authority guard.

SPEC-HEKB-REFACTOR-2026-v2.0 section 5.3: `hekb-vnext`'s primary local IPC
channel is a UDS at `~/Library/Application Support/MyLLMBuilder/hekb.sock`
(mode 0600), authorized via `LOCAL_PEERCRED` -- the connecting process's
real `uid` must match this process's own `uid` (the app owner). Any other
uid is disconnected immediately, before any request bytes are processed.

macOS/BSD specific (`LOCAL_PEERCRED`, `struct xucred` from
<sys/ucred.h>) -- this repo's own target environment is declared as
"macOS / iOS Local Ephemeral Runtime" (spec header), so no Linux
`SO_PEERCRED` fallback is implemented. Verified before writing this file:
Python's stdlib `socket` module does not expose `LOCAL_PEERCRED`/`SOL_LOCAL`
as named constants (those are POSIX/Linux-oriented; macOS's are BSD-only),
so their raw integer values are used directly, sourced from macOS's
<sys/un.h> (`SOL_LOCAL = 0`) and <sys/ucred.h> (`LOCAL_PEERCRED = 0x0001`).

This module implements the guard as a standalone asyncio UDS server that,
once a connection's peer uid is authorized, transparently relays bytes to
the real HTTP service's TCP loopback listener (`127.0.0.1:<port>`) -- a
local authenticating proxy, not a reimplementation of the HTTP layer.
Unauthorized connections are closed immediately without opening the
backend connection at all or reading any request bytes from the client
beyond what's needed to detect the peer credential (which requires none --
`LOCAL_PEERCRED` is available on the socket itself pre-accept-data).
"""
from __future__ import annotations

import asyncio
import os
import socket
import struct
from dataclasses import dataclass
from pathlib import Path

SOL_LOCAL = 0
LOCAL_PEERCRED = 0x0001
_XUCRED_PROBE_SIZE = 128  # generous; only the leading cr_version/cr_uid fields are parsed


class PeerCredentialError(RuntimeError):
    pass


def get_peer_uid(sock: socket.socket) -> int:
    """Reads the connecting peer's real uid via macOS LOCAL_PEERCRED.
    struct xucred layout: cr_version (u_int32), cr_uid (uid_t/uint32),
    cr_ngroups (short), cr_groups[NGROUPS] (gid_t[]) -- only the first two
    fields are needed here."""
    try:
        buf = sock.getsockopt(SOL_LOCAL, LOCAL_PEERCRED, _XUCRED_PROBE_SIZE)
    except OSError as exc:
        raise PeerCredentialError(f"LOCAL_PEERCRED getsockopt failed: {exc}") from exc
    if len(buf) < 8:
        raise PeerCredentialError(f"unexpected xucred buffer size: {len(buf)} bytes")
    _cr_version, cr_uid = struct.unpack_from("<II", buf, 0)
    return cr_uid


@dataclass
class UDSGuardConfig:
    socket_path: Path
    backend_host: str
    backend_port: int
    owner_uid: int | None = None  # defaults to os.getuid() if None

    def resolved_owner_uid(self) -> int:
        return self.owner_uid if self.owner_uid is not None else os.getuid()


async def _relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        writer.close()


async def _handle_connection(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    config: UDSGuardConfig,
) -> None:
    sock: socket.socket = client_writer.get_extra_info("socket")
    owner_uid = config.resolved_owner_uid()
    try:
        peer_uid = get_peer_uid(sock)
    except PeerCredentialError:
        client_writer.close()
        return

    if peer_uid != owner_uid:
        # Unauthorized uid: disconnect immediately, no backend connection opened,
        # no request bytes read.
        client_writer.close()
        return

    backend_reader, backend_writer = await asyncio.open_connection(config.backend_host, config.backend_port)
    await asyncio.gather(
        _relay(client_reader, backend_writer),
        _relay(backend_reader, client_writer),
        return_exceptions=True,
    )


async def serve_uds_guard(config: UDSGuardConfig) -> asyncio.AbstractServer:
    if config.socket_path.exists():
        config.socket_path.unlink()
    config.socket_path.parent.mkdir(parents=True, exist_ok=True)

    async def _handler(reader, writer):
        await _handle_connection(reader, writer, config)

    server = await asyncio.start_unix_server(_handler, path=str(config.socket_path))
    os.chmod(config.socket_path, 0o600)
    return server
