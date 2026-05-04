"""Helper daemon entry point: ``python -m meeting_scribe_helper``.

Listens on a Unix socket; one request, one response, close. Auth is
SO_PEERCRED — no shared secret, no token. The daemon is started by
:file:`/etc/systemd/system/meeting-scribe-helper.service` (root).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import pwd
import socket
import struct
import sys
import time
from pathlib import Path

from meeting_scribe_helper.protocol import (
    caller_authorized,
    encode_response,
    parse_request,
    redact_sensitive,
)
from meeting_scribe_helper.verbs import VERB_REGISTRY, VerbError

logger = logging.getLogger("meeting_scribe_helper")


DEFAULT_SOCKET = Path("/run/meeting-scribe/helper.sock")
DEFAULT_SERVICE_USER = "meeting-scribe"


def _resolve_service_uid(name: str) -> int:
    """Look up the service account UID. Falls back to current EUID for
    test fixtures where the named user doesn't exist."""
    try:
        return pwd.getpwnam(name).pw_uid
    except KeyError:
        return os.geteuid()


def _peer_uid_from_so_peercred(sock: socket.socket) -> int:
    """Extract the peer UID from SO_PEERCRED on a Unix socket.

    Linux returns ``struct ucred {pid, uid, gid}``. We pull the UID
    out via :mod:`struct`; this is the only auth check the helper
    performs.
    """
    SO_PEERCRED = 17  # Linux constant
    cred = sock.getsockopt(socket.SOL_SOCKET, SO_PEERCRED, struct.calcsize("3i"))
    _, uid, _ = struct.unpack("3i", cred)
    return uid


async def _handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    service_uid: int,
) -> None:
    """One request, one response, close."""
    sock = writer.get_extra_info("socket")
    try:
        peer_uid = _peer_uid_from_so_peercred(sock)
    except OSError as exc:
        logger.warning("peer cred lookup failed: %s", exc)
        writer.close()
        return

    # Allowed peers: UID 0 and UID meeting-scribe. Other UIDs cannot
    # invoke any verb regardless of which one they ask for.
    if peer_uid != 0 and peer_uid != service_uid:
        writer.write(encode_response(ok=False, error="uid_not_allowed"))
        await writer.drain()
        writer.close()
        return

    raw = await reader.readline()
    verb, args, request_id = parse_request(raw)
    if not verb:
        writer.write(
            encode_response(ok=False, request_id=request_id, error="invalid_request")
        )
        await writer.drain()
        writer.close()
        return

    spec = VERB_REGISTRY.get(verb)
    if spec is None:
        logger.info("unknown_verb peer_uid=%d verb=%r", peer_uid, verb)
        writer.write(
            encode_response(ok=False, request_id=request_id, error="unknown_verb")
        )
        await writer.drain()
        writer.close()
        return

    ok, err = caller_authorized(spec, caller_uid=peer_uid, service_uid=service_uid)
    if not ok:
        logger.info(
            "verb_denied peer_uid=%d verb=%r reason=%s",
            peer_uid,
            verb,
            err,
        )
        writer.write(
            encode_response(ok=False, request_id=request_id, error=err or "uid_not_allowed")
        )
        await writer.drain()
        writer.close()
        return

    redacted = redact_sensitive(args, spec.sensitive_keys)
    logger.info(
        "verb_invoked peer_uid=%d verb=%r request_id=%r args=%s",
        peer_uid,
        verb,
        request_id,
        redacted,
    )

    t0 = time.monotonic()
    try:
        result = await spec.handler(args)
        writer.write(
            encode_response(
                ok=True,
                request_id=request_id,
                result=result,
            )
        )
    except VerbError as ve:
        logger.warning(
            "verb_error peer_uid=%d verb=%r request_id=%r code=%s detail=%s",
            peer_uid,
            verb,
            request_id,
            ve.code,
            ve.detail,
        )
        writer.write(encode_response(ok=False, request_id=request_id, error=ve.code))
    except Exception as exc:  # noqa: BLE001 — log + return structured error
        logger.exception("verb_unhandled peer_uid=%d verb=%r request_id=%r", peer_uid, verb, request_id)
        writer.write(encode_response(ok=False, request_id=request_id, error="internal_error"))
    finally:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            "verb_done peer_uid=%d verb=%r request_id=%r elapsed_ms=%.1f",
            peer_uid,
            verb,
            request_id,
            elapsed_ms,
        )

    await writer.drain()
    writer.close()


async def serve(
    *,
    socket_path: Path,
    service_uid: int,
    socket_group: int | None = None,
) -> None:
    """Bind the Unix socket and run forever."""
    socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    if socket_path.exists():
        socket_path.unlink()

    server = await asyncio.start_unix_server(
        lambda r, w: _handle_connection(r, w, service_uid=service_uid),
        path=str(socket_path),
    )
    # Tighten permissions so only the meeting-scribe group can read/write
    # the socket. Root opens the socket; the service user connects through
    # its group membership.
    socket_path.chmod(0o660)
    if socket_group is not None:
        with contextlib.suppress(PermissionError):
            os.chown(str(socket_path), -1, socket_group)

    logger.info(
        "helper listening on %s (service_uid=%d)",
        socket_path,
        service_uid,
    )
    async with server:
        await server.serve_forever()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="meeting-scribe-helper")
    parser.add_argument(
        "--socket",
        default=str(DEFAULT_SOCKET),
        help="Unix socket path (default: %(default)s)",
    )
    parser.add_argument(
        "--service-user",
        default=DEFAULT_SERVICE_USER,
        help="username whose UID is allowed to call verbs (default: %(default)s)",
    )
    parser.add_argument(
        "--socket-group",
        default=None,
        help="GID to chown the socket to (default: leave unchanged)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)
    service_uid = _resolve_service_uid(args.service_user)
    socket_group: int | None = None
    if args.socket_group is not None:
        try:
            socket_group = int(args.socket_group)
        except ValueError:
            import grp

            socket_group = grp.getgrnam(args.socket_group).gr_gid
    try:
        asyncio.run(
            serve(
                socket_path=Path(args.socket),
                service_uid=service_uid,
                socket_group=socket_group,
            )
        )
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
