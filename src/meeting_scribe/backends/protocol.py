"""Wire contract for end-to-end Phase B GPU-call abort.

Every Phase B GPU HTTP call carries an opaque ``request_id`` so a recording
preempt can stop the in-flight inference at the GPU server, not just
locally. Both legs of the contract live here so the meeting-scribe client
and each backend service share one definition.

* Header injected on every request: ``X-Scribe-Request-Id: <rid>``.
* Same value also threaded through the query string as ``request_id=<rid>``
  for HTTP/1.1 path-only proxies that drop unknown headers.
* Cancellation endpoint: ``POST {base_url}/abort/{request_id}`` returning
  200 once the backend has yielded the GPU, 404 if the id is unknown.
* Backends MUST treat repeated aborts of the same id as no-ops (200 once;
  404 thereafter is also acceptable).

The lease's preempt path snapshots the rids registered for an in-flight
Phase B call and POSTs ``/abort/{rid}`` for each in parallel under one
shared monotonic deadline. See :class:`meeting_scribe.runtime.gpu_lease.GpuLease`.
"""

from __future__ import annotations

from typing import Final
from urllib.parse import quote

#: Canonical header name for the Phase B request id. Lowercase wins on the
#: wire (HTTP headers are case-insensitive) but we keep the casing here for
#: log/test readability.
REQUEST_ID_HEADER: Final[str] = "X-Scribe-Request-Id"

#: Query-parameter name carrying the same id, for path-only proxies.
REQUEST_ID_PARAM: Final[str] = "request_id"


def abort_url(base_url: str, request_id: str) -> str:
    """Return the canonical abort URL for the given backend + rid.

    ``base_url`` is the backend's root (e.g. ``http://127.0.0.1:8010``).
    Trailing slash is tolerated. The ``request_id`` is URL-quoted so a
    rogue caller can't smuggle path components into the URL.
    """
    base = base_url.rstrip("/")
    return f"{base}/abort/{quote(request_id, safe='')}"


def inject_rid(
    headers: dict[str, str] | None,
    params: dict[str, str] | None,
    request_id: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return updated ``(headers, params)`` with the rid threaded.

    Caller passes the existing dicts (or None); the function returns
    a copy with both the header and the query parameter set so the
    backend has at least one reliable channel to read the id.
    """
    h = dict(headers) if headers else {}
    p = dict(params) if params else {}
    h[REQUEST_ID_HEADER] = request_id
    p[REQUEST_ID_PARAM] = request_id
    return h, p
