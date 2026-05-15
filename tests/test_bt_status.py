from __future__ import annotations

import subprocess

import pytest

from meeting_scribe import bt


@pytest.mark.asyncio
async def test_bt_status_includes_known_unpaired_devices(monkeypatch):
    async def fake_run(argv: list[str], *, timeout: float = 10.0):
        cmd = tuple(argv)
        if cmd == ("bluetoothctl", "show"):
            return subprocess.CompletedProcess(argv, 0, stdout="Powered: yes\n", stderr="")
        if cmd == ("bluetoothctl", "devices"):
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="Device 10:54:D2:C4:76:0B Dell WL5024 Headset\n",
                stderr="",
            )
        if cmd == ("bluetoothctl", "devices", "Paired"):
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if cmd == ("bluetoothctl", "devices", "Connected"):
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="Device 10:54:D2:C4:76:0B Dell WL5024 Headset\n",
                stderr="",
            )
        if cmd == ("bluetoothctl", "info", "10:54:D2:C4:76:0B"):
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=(
                    "Device 10:54:D2:C4:76:0B (public)\n"
                    "\tName: Dell WL5024 Headset\n"
                    "\tPaired: no\n"
                    "\tTrusted: yes\n"
                    "\tConnected: yes\n"
                    "\tUUID: Headset (00001108-0000-1000-8000-00805f9b34fb)\n"
                    "\tUUID: Audio Sink (0000110b-0000-1000-8000-00805f9b34fb)\n"
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(bt, "_run", fake_run)

    snap = await bt.bt_status_sync()

    assert snap["powered"] is True
    assert snap["known_count"] == 1
    assert snap["paired_count"] == 0
    assert snap["connected_count"] == 1
    assert snap["devices"] == [
        {
            "mac": "10:54:D2:C4:76:0B",
            "name": "Dell WL5024 Headset",
            "paired": False,
            "connected": True,
            "trusted": True,
            "audio": True,
        }
    ]


@pytest.mark.asyncio
async def test_reconcile_connected_audio_outputs_nudges_missing_sink(monkeypatch):
    calls: list[tuple[str, str]] = []
    settings: dict[str, object] = {"bt_input_active": False}
    resolve_count = 0

    async def fake_status():
        return {
            "devices": [
                {
                    "mac": "10:54:D2:C4:76:0B",
                    "name": "Dell WL5024 Headset",
                    "connected": True,
                    "audio": True,
                    "blocked": False,
                }
            ]
        }

    async def fake_resolve(mac: str):
        nonlocal resolve_count
        resolve_count += 1
        return (None, None) if resolve_count == 1 else (None, "bluez_output.10_54_D2_C4_76_0B.1")

    async def fake_connect(mac: str) -> None:
        calls.append(("connect", mac))

    async def fake_choose(mac: str, *, want: str) -> str:
        calls.append(("choose", want))
        return "a2dp-sink"

    async def fake_set_profile(mac: str, profile: str, *, timeout: float = 6.0) -> None:
        calls.append(("profile", profile))

    def fake_load():
        return dict(settings)

    def fake_save(update: dict[str, object]) -> None:
        settings.update(update)

    monkeypatch.setattr(bt, "bt_status_sync", fake_status)
    monkeypatch.setattr(bt, "bt_resolve_nodes", fake_resolve)
    monkeypatch.setattr(bt, "bt_connect", fake_connect)
    monkeypatch.setattr(bt, "bt_choose_profile", fake_choose)
    monkeypatch.setattr(bt, "bt_set_profile", fake_set_profile)
    monkeypatch.setattr(bt, "_last_audio_reconcile_by_mac", {})
    monkeypatch.setattr(
        "meeting_scribe.server_support.settings_store._load_settings_override",
        fake_load,
    )
    monkeypatch.setattr(
        "meeting_scribe.server_support.settings_store._save_settings_override",
        fake_save,
    )

    result = await bt.reconcile_connected_audio_outputs(reason="test")

    assert calls == [
        ("connect", "10:54:D2:C4:76:0B"),
        ("choose", "a2dp"),
        ("profile", "a2dp-sink"),
    ]
    assert result["updated"] == [
        {"mac": "10:54:D2:C4:76:0B", "sink_node": "bluez_output.10_54_D2_C4_76_0B.1"}
    ]
    assert settings["audio_admin_tts_sink_node"] == "bluez_output.10_54_D2_C4_76_0B.1"


@pytest.mark.asyncio
async def test_reconcile_connected_audio_outputs_does_not_scan(monkeypatch):
    commands: list[list[str]] = []

    async def fake_run(argv: list[str], *, timeout: float = 10.0):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    async def fake_status():
        return {"devices": []}

    monkeypatch.setattr(bt, "_run", fake_run)
    monkeypatch.setattr(bt, "bt_status_sync", fake_status)
    monkeypatch.setattr(
        "meeting_scribe.server_support.settings_store._load_settings_override",
        lambda: {},
    )
    monkeypatch.setattr(
        "meeting_scribe.server_support.settings_store._save_settings_override",
        lambda update: None,
    )

    await bt.reconcile_connected_audio_outputs(reason="test")

    assert not any(cmd[:3] == ["bluetoothctl", "--timeout", "10"] for cmd in commands)
    assert not any(cmd[:2] == ["bluetoothctl", "scan"] for cmd in commands)
