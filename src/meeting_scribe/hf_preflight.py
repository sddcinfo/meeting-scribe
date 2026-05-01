"""Hugging Face credential pre-flight.

Validates that an `HF_TOKEN` (a) authenticates against the Hub, and
(b) can read every gated model meeting-scribe will later try to
download. The 2026-04-30 customer demo tripped on a missing EULA
acceptance after a 20-minute install — `hf download` died on a
generic 403 with no actionable hint. This module catches that case
up-front and tells the operator exactly which model URL to open.

Public surface:
    HfStatus            — per-model classification enum
    HfModelResult       — one row of the validation report
    ValidationReport    — aggregated report with .ok / .has_only_network_failures
    validate_hf_access  — entry point; runs whoami + per-model model_info

The contract for `ValidationReport.ok` is intentionally strict:
NETWORK_ERROR counts as non-OK so the customer-side blocking probe
(the authoritative source for `RuntimeManifest.model_revisions` in
the two-manifest model — see plans/steady-plotting-eich.md) cannot
silently degrade revision pinning. Callers that want to permit
transient network failures (the dev-box advisory site) check the
secondary `has_only_network_failures` predicate explicitly.
"""

from __future__ import annotations

import dataclasses
import time
from enum import StrEnum

from huggingface_hub import HfApi
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

# huggingface_hub raises ConnectionError-style exceptions through several
# different concrete classes depending on the underlying transport. Catch
# the broad `requests` family plus socket-level OSErrors so transient DNS
# / proxy / TLS hiccups land as `NETWORK_ERROR`, not a stack trace.
try:  # huggingface_hub vendors requests; fall through if unavailable.
    import requests.exceptions as _req_exc

    _NETWORK_EXC: tuple[type[BaseException], ...] = (
        _req_exc.ConnectionError,
        _req_exc.Timeout,
        _req_exc.SSLError,
        _req_exc.ProxyError,
        OSError,
    )
except ImportError:
    _NETWORK_EXC = (OSError,)


class HfStatus(StrEnum):
    OK = "ok"
    BAD_TOKEN = "bad_token"
    GATED_NOT_ACCEPTED = "gated_not_accepted"
    NOT_FOUND = "not_found"
    NETWORK_ERROR = "network_error"


@dataclasses.dataclass
class HfModelResult:
    """Per-model classification row."""

    model_id: str
    status: HfStatus
    detail: str = ""
    url: str = ""
    revision: str = ""

    def __post_init__(self) -> None:
        if not self.url:
            self.url = f"https://huggingface.co/{self.model_id}"


@dataclasses.dataclass
class ValidationReport:
    """Aggregated HF access validation."""

    token_prefix: str
    whoami: str | None
    results: list[HfModelResult] = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True iff EVERY result is OK.

        NETWORK_ERROR explicitly counts as non-OK. The customer-side
        probe is the sole authority for `RuntimeManifest.model_revisions`,
        so an unreachable HF must block — degrading the contract here
        would silently drop revision pinning.
        """
        return all(r.status is HfStatus.OK for r in self.results)

    @property
    def has_only_network_failures(self) -> bool:
        """True iff at least one result is NETWORK_ERROR and every
        non-OK result is NETWORK_ERROR.

        Lets the LOCAL/advisory caller (`meeting-scribe setup` interactive
        prompt) warn-and-continue specifically for transient network
        issues, without ever softening the contract for token/EULA
        failures.
        """
        return any(r.status is HfStatus.NETWORK_ERROR for r in self.results) and all(
            r.status in (HfStatus.OK, HfStatus.NETWORK_ERROR) for r in self.results
        )

    def render(self) -> str:
        """Human-readable failure report. Skips OK rows."""
        if self.ok:
            return f"[OK] HF_TOKEN ({self.token_prefix}) verified for {len(self.results)} model(s)."
        lines: list[str] = []
        if self.whoami is None and any(r.status is HfStatus.BAD_TOKEN for r in self.results):
            lines.append(
                " ✗ HF_TOKEN is invalid or revoked (whoami → 401 Unauthorized).\n"
                "   Mint a new READ token at https://huggingface.co/settings/tokens\n"
                "   then re-run."
            )
            return "\n\n".join(lines)
        for r in self.results:
            if r.status is HfStatus.OK:
                continue
            if r.status is HfStatus.GATED_NOT_ACCEPTED:
                lines.append(
                    f" ✗ {r.model_id} — your token can't read this model.\n"
                    f"   Likely cause: you haven't accepted the model's terms.\n"
                    f"   Open {r.url} in a browser,\n"
                    f'   click "Agree and access repository", then re-run.'
                )
            elif r.status is HfStatus.NOT_FOUND:
                lines.append(
                    f" ✗ {r.model_id} — repo not found on Hugging Face (typo in a recipe?).\n"
                    f"   Verify {r.url} resolves before re-running."
                )
            elif r.status is HfStatus.NETWORK_ERROR:
                detail = (r.detail[:120] + "…") if len(r.detail) > 120 else r.detail
                lines.append(
                    f" ✗ {r.model_id} — could not validate (network error: {detail}).\n"
                    f"   Check DNS / outbound HTTPS / proxy CA bundle on this host.\n"
                    f"   {r.url} should resolve from here."
                )
            elif r.status is HfStatus.BAD_TOKEN:
                lines.append(
                    f" ✗ {r.model_id} — auth failed for token {self.token_prefix}.\n"
                    f"   Token may be valid but lacks read scope. Verify at\n"
                    f"   https://huggingface.co/settings/tokens"
                )
        return "\n\n".join(lines) if lines else "(no failing rows)"


def _classify_exception(exc: BaseException) -> tuple[HfStatus, str]:
    """Map an HfApi exception to (status, detail)."""
    if isinstance(exc, GatedRepoError):
        return HfStatus.GATED_NOT_ACCEPTED, str(exc)
    if isinstance(exc, RepositoryNotFoundError):
        return HfStatus.NOT_FOUND, str(exc)
    if isinstance(exc, HfHubHTTPError):
        # 401 = bad token; some 403s are also "no access" (gated handled
        # above, but some endpoints raise plain HfHubHTTPError on 403).
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 401:
            return HfStatus.BAD_TOKEN, str(exc)
        if status == 403:
            return HfStatus.GATED_NOT_ACCEPTED, str(exc)
        return HfStatus.NETWORK_ERROR, str(exc)
    if isinstance(exc, _NETWORK_EXC):
        return HfStatus.NETWORK_ERROR, repr(exc)
    return HfStatus.NETWORK_ERROR, repr(exc)


def _token_prefix(token: str) -> str:
    if not token:
        return "(empty)"
    head = token[:6]
    return f"{head}…" if len(token) > 6 else head


def validate_hf_access(
    token: str,
    model_ids: list[str],
    *,
    timeout: float = 10.0,
    network_retries: int = 3,
    network_backoff_s: float = 5.0,
) -> ValidationReport:
    """Validate `token` against every entry in `model_ids`.

    Performs one whoami() call up front (catches BAD_TOKEN once and
    populates `report.whoami` so the renderer can emit a single
    top-of-report bad-token banner instead of N per-model rows). Then
    one model_info() per model. The returned commit SHA from each
    successful model_info is captured into `HfModelResult.revision`
    so the orchestrator can freeze a `RuntimeManifest`.

    Network errors retry up to `network_retries` times with linear
    backoff (`network_backoff_s` × attempt-index). After exhaustion
    the row is `NETWORK_ERROR`. The function takes no
    `block_on_network_error` kwarg — policy is the caller's job
    via `report.ok` / `report.has_only_network_failures`.
    """
    api = HfApi(token=token, endpoint=None)
    report = ValidationReport(token_prefix=_token_prefix(token), whoami=None)

    # Step 1: whoami. If the token is bad, emit one BAD_TOKEN row per
    # model so the report renderer's BAD_TOKEN banner fires once and
    # callers' `.ok` predicate evaluates correctly.
    whoami: str | None = None
    whoami_err: HfStatus | None = None
    whoami_detail = ""
    for attempt in range(1, network_retries + 1):
        try:
            info = api.whoami()
            whoami = info.get("name") if isinstance(info, dict) else str(info)
            break
        except BaseException as e:
            status, detail = _classify_exception(e)
            if status is HfStatus.NETWORK_ERROR and attempt < network_retries:
                time.sleep(network_backoff_s * attempt)
                continue
            whoami_err = status
            whoami_detail = detail
            break
    report.whoami = whoami

    if whoami_err is not None:
        for mid in model_ids:
            report.results.append(
                HfModelResult(model_id=mid, status=whoami_err, detail=whoami_detail)
            )
        return report

    # Step 2: per-model model_info. Each model retries `network_retries`
    # times independently — one model's flaky DNS shouldn't poison the
    # rest of the report.
    for mid in model_ids:
        result_status: HfStatus | None = None
        result_detail = ""
        result_revision = ""
        for attempt in range(1, network_retries + 1):
            try:
                info = api.model_info(mid, timeout=timeout)
                result_status = HfStatus.OK
                result_revision = getattr(info, "sha", "") or ""
                break
            except BaseException as e:
                status, detail = _classify_exception(e)
                if status is HfStatus.NETWORK_ERROR and attempt < network_retries:
                    time.sleep(network_backoff_s * attempt)
                    continue
                result_status = status
                result_detail = detail
                break
        assert result_status is not None  # loop always terminates with a status
        report.results.append(
            HfModelResult(
                model_id=mid,
                status=result_status,
                detail=result_detail,
                revision=result_revision,
            )
        )
    return report
