"""JSON-RPC wire protocol + verb dispatch + caller-UID gating.

The helper accepts one JSON-RPC request per connection (newline-
terminated JSON object), runs the named verb if it's allowlisted AND
the caller's UID is permitted for that verb, and writes one JSON
response. No connection multiplexing: one request, one response, close.

Wire format:

    request:  {"verb": "wifi.up", "args": {...}, "request_id": "abc"}
    response: {"ok": true,  "result": {...}, "request_id": "abc"}
              {"ok": false, "error": "...",   "request_id": "abc"}

Allowed verbs and their per-UID gating live in :data:`VERB_REGISTRY`.
Adding a verb is a deliberate code change — the registry is the
privilege manifest.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

logger = logging.getLogger(__name__)


# UID 0 = root (CLI via ``sudo meeting-scribe ...`` or direct invocation).
# UID = meeting-scribe = service user; verbs that mutate hotspot state
# reject this UID with ``cli_only_verb`` so a web-layer compromise can
# never reconfigure the AP.
ROOT_UID: int = 0


@dataclass(frozen=True)
class VerbSpec:
    """One registry entry per allowlisted verb.

    handler is an async callable invoked as ``await handler(args)``;
    ``args`` is the request's typed argument object after schema
    validation. ``allowed_uids`` is the set of caller UIDs the verb
    accepts; ``service_uid`` (the meeting-scribe service account UID) is
    resolved at daemon startup and added to the set for verbs that
    permit web-service calls.
    """

    handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
    allowed_uids: Literal["root_only", "root_and_service", "any"]
    sensitive_keys: frozenset[str] = frozenset()


def encode_response(
    *,
    ok: bool,
    request_id: str | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> bytes:
    """Encode a response to bytes (single line, newline terminator).

    Test helper + the daemon both use this; keeping it pure makes
    round-trip tests trivial.
    """
    body: dict[str, Any] = {"ok": ok}
    if request_id is not None:
        body["request_id"] = request_id
    if ok:
        body["result"] = result or {}
    else:
        body["error"] = error or "unspecified_error"
    return (json.dumps(body) + "\n").encode("utf-8")


def parse_request(raw: bytes) -> tuple[str | None, dict[str, Any], str | None]:
    """Decode a raw request line.

    Returns ``(verb, args, request_id)``. Malformed JSON or missing
    ``verb`` yields ``(None, {}, None)`` so the caller writes a
    structured ``invalid_request`` error.
    """
    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None, {}, None
    if not text:
        return None, {}, None
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return None, {}, None
    if not isinstance(body, dict):
        return None, {}, None
    verb = body.get("verb")
    args = body.get("args") or {}
    request_id = body.get("request_id")
    if not isinstance(verb, str):
        return None, args if isinstance(args, dict) else {}, request_id
    return verb, args if isinstance(args, dict) else {}, request_id


def caller_authorized(
    verb_spec: VerbSpec,
    *,
    caller_uid: int,
    service_uid: int,
) -> tuple[bool, str | None]:
    """Return ``(ok, error_code)`` for the per-verb caller-UID check.

    Mutating verbs (``allowed_uids="root_only"``) reject the service
    UID with ``cli_only_verb``; read-only verbs accept both.
    """
    if verb_spec.allowed_uids == "any":
        return True, None
    if caller_uid == ROOT_UID:
        return True, None
    if verb_spec.allowed_uids == "root_and_service" and caller_uid == service_uid:
        return True, None
    if verb_spec.allowed_uids == "root_only":
        return False, "cli_only_verb"
    return False, "uid_not_allowed"


def redact_sensitive(args: dict[str, Any], sensitive_keys: frozenset[str]) -> dict[str, Any]:
    """Replace sensitive args with ``<redacted>`` for log output.

    Wi-Fi passwords / admin secrets / etc. NEVER appear in plaintext or
    as a cross-restart-stable hash in the audit log. Per-boot HMAC
    correlation lives in :func:`audit_correlation` if "did this field
    change?" is needed.
    """
    out = dict(args)
    for key in sensitive_keys:
        if key in out:
            out[key] = "<redacted>"
    return out
