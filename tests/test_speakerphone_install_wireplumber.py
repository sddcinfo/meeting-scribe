"""Tests for the WirePlumber pin rule installed by
``meeting-scribe speakerphone install``.

The rule body lives as a string constant in
``meeting_scribe.cli.speakerphone`` and is copied to
``~/.config/wireplumber/main.lua.d/51-sp325-pin.lua`` on install. The
SP325 default-sink reliability story (2026-05-13) depends on this
rule landing exactly as written — a stale or malformed copy puts the
device back into the "Off" profile and silently regresses the admin
SPA audio surface to a Plantronics / HDMI fallback.

These tests pin:

* The contract elements the rule MUST contain (pro-audio profile,
  SP325 vid:pid card-name matcher, priority bump).
* The install path is XDG_CONFIG_HOME-relative + idempotent (running
  install twice writes the same bytes once).
* Uninstall removes the file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meeting_scribe.cli import speakerphone as sp_cli


def test_wp_rule_body_targets_pro_audio_profile() -> None:
    """The rule must force the SP325 onto the Pro Audio profile.

    Pro Audio is the only profile that exposes BOTH the pro-input-0
    source (mic capture for ASR) AND the pro-output-0 sink (room TTS
    playback). Analog-stereo is sink-only; pro-audio is the only
    duplex option on the SP325.
    """
    body = sp_cli._WP_RULE_BODY
    assert '["device.profile"] = "pro-audio"' in body
    # Also disable WP's auto-profile chooser so the explicit profile
    # above is not overwritten on subsequent attach events.
    assert '["api.acp.auto-profile"] = false' in body


def test_wp_rule_body_matches_sp325_card_name() -> None:
    """The rule must match the SP325 family's ALSA card-name slug.

    The 413c:8223 USB device exposes its card name as
    ``alsa_card.usb-Dell_Inc._Dell_SP325_Speakerphone_…`` — the rule
    matcher must hit that prefix. SP3022 (8205) and the legacy 8222
    share the same product-family branding and so the same prefix.
    """
    body = sp_cli._WP_RULE_BODY
    assert "alsa_card.usb-Dell_Inc._Dell_SP325_Speakerphone" in body
    # Both node-name matchers (input + output) must be present so the
    # priority bump applies to BOTH directions.
    assert "alsa_output.usb-Dell_Inc._Dell_SP325_Speakerphone" in body
    assert "alsa_input.usb-Dell_Inc._Dell_SP325_Speakerphone" in body


def test_wp_rule_body_boosts_priority() -> None:
    """The rule must bump session + driver priority on SP325 nodes.

    WirePlumber's default-node election picks the highest priority
    node. The bump must be high enough to beat the built-in HDMI sink
    (~1000-1500) and the analog-stereo profile's own 1009.
    """
    body = sp_cli._WP_RULE_BODY
    assert '["priority.session"] = 2000' in body
    assert '["priority.driver"] = 2000' in body


def test_wp_rule_path_is_xdg_relative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``XDG_CONFIG_HOME`` must drive the install location.

    Test-rig isolation: bootstrap should not write to a real home dir.
    The helper must respect XDG so containerised installs (which
    typically set XDG_CONFIG_HOME=/etc/...) land the file in the right
    place.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    expected = tmp_path / "wireplumber" / "main.lua.d" / "51-sp325-pin.lua"
    assert sp_cli._wp_rule_path() == expected


def test_install_wireplumber_rule_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Running install twice writes the canonical body once.

    Idempotency is the contract bootstrap.sh / fresh GB10 setup relies
    on — re-running the install command after every pull must not
    churn the file or restart wireplumber when nothing has changed.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # Block the wireplumber restart that the helper would otherwise
    # attempt — irrelevant in unit-test scope.
    import subprocess as _subprocess

    monkeypatch.setattr(
        _subprocess,
        "run",
        lambda *a, **kw: type("P", (), {"returncode": 0})(),
    )

    assert sp_cli._install_wireplumber_rule(quiet=True) is True
    path = sp_cli._wp_rule_path()
    assert path.exists()
    first_body = path.read_text()
    assert first_body == sp_cli._WP_RULE_BODY

    # Second install — file already up-to-date; body must not drift.
    assert sp_cli._install_wireplumber_rule(quiet=True) is True
    assert path.read_text() == first_body


def test_uninstall_wireplumber_rule_removes_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import subprocess as _subprocess

    monkeypatch.setattr(
        _subprocess,
        "run",
        lambda *a, **kw: type("P", (), {"returncode": 0})(),
    )

    path = sp_cli._wp_rule_path()
    sp_cli._install_wireplumber_rule(quiet=True)
    assert path.exists()

    sp_cli._uninstall_wireplumber_rule(quiet=True)
    assert not path.exists()
