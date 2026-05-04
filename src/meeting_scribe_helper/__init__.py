"""Privileged root-owned helper daemon for meeting-scribe.

Separate top-level package — distinct import root from
``meeting_scribe`` — so the helper has a minimal import surface and
cannot accidentally import the full server. Wire-protocol contract +
caller-UID enforcement live in :mod:`meeting_scribe_helper.protocol`;
typed verb implementations in :mod:`meeting_scribe_helper.verbs`.

The daemon listens on ``/run/meeting-scribe/helper.sock`` (mode 0o660,
group ``meeting-scribe``) and authenticates peers via SO_PEERCRED. Two
caller UIDs are accepted: ``0`` (root, for local recovery and direct
CLI invocation via ``sudo meeting-scribe ...``) and the
``meeting-scribe`` service user. Other UIDs are rejected with a
structured error.

The web service connects via the helper client at
``meeting_scribe.helper_client``; the CLI uses the same client so both
go through one code path. The web service's sudoers grant is removed
in the cutover that wires this in.
"""
