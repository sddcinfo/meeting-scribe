#!/usr/bin/env python3
"""Probe the live audio-routing API end-to-end without the admin cookie.

We bypass ``_require_admin_response`` in-process (this is a diagnostic
script, not a request through the wire) and exercise both endpoints
with the same payloads the admin UI's ``admin-audio-card.js`` would
send. Surfaces the exact ``selection`` shape the UI receives back so
you can see whether the persisted ``mic_node`` actually appears in
``devices.sources`` (no "(missing)" badge) or not.

Usage::

    python3 scripts/probe_audio_routing.py            # read-only enum + selection
    python3 scripts/probe_audio_routing.py --post-mic  # round-trip a real Poly node
    python3 scripts/probe_audio_routing.py --post-sink # round-trip a real Poly sink

Read-only mode prints a "MATCHES" / "MISSING" line per persisted
selection so the divergence is impossible to miss.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def _probe() -> dict:
    from meeting_scribe.audio.audio_routing import (
        enumerate_audio_devices,
        get_routing_settings,
    )
    from meeting_scribe.server_support.settings_store import _load_settings_override

    devices = await enumerate_audio_devices()
    selection = get_routing_settings(_load_settings_override())
    return {"devices": devices, "selection": selection}


def _check_match(label: str, persisted_value: str, devices: list[dict]) -> None:
    if not persisted_value:
        print(f"  {label}: <empty> → default sink/source")
        return
    match = next((d for d in devices if d["node_name"] == persisted_value), None)
    if match is None:
        print(f"  {label}: MISSING — '{persisted_value}'")
        print("    → not in pw-dump, would render as italicized '(missing)' in UI")
        candidates = [d["node_name"] for d in devices]
        for c in candidates[:5]:
            print(f"    candidate: {c}")
    else:
        print(f"  {label}: MATCHES — '{persisted_value}'")
        print(f"    description: {match['description']}")
        print(f"    device_class: {match['device_class']}")
        print(f"    is_default: {match.get('is_default')}")


async def _post_route(*, mic_node: str | None, sink_node: str | None) -> None:
    """Direct call into the route handler, skipping the cookie gate.

    Mirrors what /api/admin/audio/route POST does except the
    ``_require_admin_response`` shortcut returns ``None`` (admin OK).
    Useful for verifying the persistence + reconcile path without
    needing the admin password.
    """
    from meeting_scribe.routes import admin_audio

    # Patch the gate for the duration of the call.
    original_gate = admin_audio._require_admin_response
    admin_audio._require_admin_response = lambda req: None

    class _StubRequest:
        async def json(self) -> dict:
            body = {}
            if mic_node is not None:
                body["mic_node"] = mic_node
            if sink_node is not None:
                body["sink_node"] = sink_node
            return body

    try:
        resp = await admin_audio.audio_route_post(_StubRequest())
    finally:
        admin_audio._require_admin_response = original_gate

    body = json.loads(resp.body.decode())
    print("POST response:", json.dumps(body, indent=2))


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--post-mic",
        nargs="?",
        const="@poly",
        help="POST a real mic node to /api/admin/audio/route. "
        "Pass a node.name or @poly to auto-pick the Poly USB source.",
    )
    parser.add_argument(
        "--post-sink",
        nargs="?",
        const="@poly",
        help="POST a real sink node. Pass a node.name or @poly.",
    )
    args = parser.parse_args()

    result = await _probe()
    devices = result["devices"]
    selection = result["selection"]

    print("=== Live device enumeration ===")
    print(f"sources: {len(devices['sources'])}, sinks: {len(devices['sinks'])}")
    for src in devices["sources"]:
        star = " ★" if src.get("is_default") else ""
        print(f"  [src/{src['device_class']}] {src['description']}{star}")
        print(f"    node_name: {src['node_name']}")
    for sink in devices["sinks"]:
        star = " ★" if sink.get("is_default") else ""
        print(f"  [sink/{sink['device_class']}] {sink['description']}{star}")
        print(f"    node_name: {sink['node_name']}")

    print("\n=== Persisted selection ===")
    print(f"mic_active: {selection['mic_active']}")
    _check_match("mic_node", selection["mic_node"], devices["sources"])
    _check_match("sink_node", selection["sink_node"], devices["sinks"])

    def _resolve_alias(alias: str | None, kind: str) -> str | None:
        if alias is None:
            return None
        if alias == "@poly":
            collection = devices["sources"] if kind == "source" else devices["sinks"]
            for d in collection:
                if "Plantronics" in d["description"] or "Poly" in d["description"]:
                    return d["node_name"]
            return None
        return alias

    if args.post_mic is not None:
        node = _resolve_alias(args.post_mic, "source")
        if not node:
            print("\n@poly source not found; skipping --post-mic", file=sys.stderr)
        else:
            print(f"\n=== POST /api/admin/audio/route mic_node={node} ===")
            await _post_route(mic_node=node, sink_node=None)
    if args.post_sink is not None:
        node = _resolve_alias(args.post_sink, "sink")
        if not node:
            print("\n@poly sink not found; skipping --post-sink", file=sys.stderr)
        else:
            print(f"\n=== POST /api/admin/audio/route sink_node={node} ===")
            await _post_route(mic_node=None, sink_node=node)

    if args.post_mic is not None or args.post_sink is not None:
        print("\n=== Re-probe after POST ===")
        result = await _probe()
        devices = result["devices"]
        selection = result["selection"]
        _check_match("mic_node", selection["mic_node"], devices["sources"])
        _check_match("sink_node", selection["sink_node"], devices["sinks"])


if __name__ == "__main__":
    asyncio.run(_main())
