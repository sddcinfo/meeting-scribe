"""Tests for meeting_scribe.wifi — WiFi hotspot state machine.

Ported from ``sddc-cli/tests/test_hotspot_reconcile.py`` and extended
with tests for the new wifi module APIs (build_config, firewalls,
captive portal setup, live config builder, teardown invariants).

All subprocess invocations are mocked at the ``meeting_scribe.wifi``
module scope so no real nmcli/iptables/wpa_cli/iw commands are ever run.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meeting_scribe import wifi
from meeting_scribe.server_support import settings_store as _settings_store


def _mk_completed(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["sudo", "nmcli"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


# ── Shared fixtures ───────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    """Clear the module-level settings cache between tests."""
    _settings_store._settings_cache = None
    _settings_store._settings_cache_mtime = 0.0


@pytest.fixture(autouse=True)
def _reset_ap_state_cache() -> None:
    """Drop wifi's TTL-cached AP state between tests so values from
    one test don't bleed into the next."""
    wifi._invalidate_ap_state_cache()


@pytest.fixture
def state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect HOTSPOT_STATE_FILE to a per-test tmp path."""
    path = tmp_path / "meeting-hotspot.json"
    monkeypatch.setattr(wifi, "HOTSPOT_STATE_FILE", path)
    return path


@pytest.fixture
def settings_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect SETTINGS_OVERRIDE_FILE to a per-test tmp path."""
    path = tmp_path / "settings.json"
    monkeypatch.setattr(_settings_store, "SETTINGS_OVERRIDE_FILE", path)
    monkeypatch.setattr(wifi, "SETTINGS_OVERRIDE_FILE", path)
    return path


@pytest.fixture
def dnsmasq_conf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect DNSMASQ_CAPTIVE_CONF to a per-test tmp path."""
    conf_dir = tmp_path / "dnsmasq-shared.d"
    conf_dir.mkdir()
    path = conf_dir / "captive-portal.conf"
    monkeypatch.setattr(wifi, "DNSMASQ_CONF_DIR", conf_dir)
    monkeypatch.setattr(wifi, "DNSMASQ_CAPTIVE_CONF", path)
    return path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


# ── _parse_nmcli_fields ───────────────────────────────────────────────


class TestParseNmcliFields:
    def test_basic_colon_separated(self) -> None:
        out = "802-11-wireless.ssid:Dell Demo 7EC2\n802-11-wireless-security.psk:4EEF0ACA\n"
        fields = wifi._parse_nmcli_fields(out)
        assert fields["802-11-wireless.ssid"] == "Dell Demo 7EC2"
        assert fields["802-11-wireless-security.psk"] == "4EEF0ACA"

    def test_value_contains_colon(self) -> None:
        out = "802-11-wireless.ssid:Coffee: Shop WiFi\n"
        fields = wifi._parse_nmcli_fields(out)
        assert fields["802-11-wireless.ssid"] == "Coffee: Shop WiFi"

    def test_empty_and_garbage_lines_ignored(self) -> None:
        out = "\ngarbage\na:1\n\nb:2\n"
        fields = wifi._parse_nmcli_fields(out)
        assert fields == {"a": "1", "b": "2"}


# ── _nmcli_read_live_ap_credentials ───────────────────────────────────


class TestReadLiveAPCredentials:
    @patch("meeting_scribe.wifi.subprocess.run")
    def test_happy_path(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _mk_completed(
            stdout=("802-11-wireless.ssid:Dell Demo 7EC2\n802-11-wireless-security.psk:4EEF0ACA\n"),
        )
        assert wifi._nmcli_read_live_ap_credentials() == ("Dell Demo 7EC2", "4EEF0ACA")

    @patch("meeting_scribe.wifi.subprocess.run")
    def test_nmcli_error_returns_none(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _mk_completed(returncode=1, stderr="no such connection")
        assert wifi._nmcli_read_live_ap_credentials() is None

    @patch("meeting_scribe.wifi.subprocess.run")
    def test_empty_ssid_returns_none(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _mk_completed(
            stdout="802-11-wireless.ssid:\n802-11-wireless-security.psk:\n",
        )
        assert wifi._nmcli_read_live_ap_credentials() is None


# ── _nmcli_ap_is_active ──────────────────────────────────────────────


class TestAPIsActive:
    @patch("meeting_scribe.wifi.subprocess.run")
    def test_active(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _mk_completed(
            stdout="DellDemo-AP:wlP9s9\nWired:enP7s7\n",
        )
        assert wifi._nmcli_ap_is_active() is True

    @patch("meeting_scribe.wifi.subprocess.run")
    def test_inactive(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _mk_completed(stdout="Wired:enP7s7\n")
        assert wifi._nmcli_ap_is_active() is False

    @patch("meeting_scribe.wifi.subprocess.run")
    def test_nmcli_error(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _mk_completed(returncode=1)
        assert wifi._nmcli_ap_is_active() is False


# ── _wait_for_ap_active ──────────────────────────────────────────────


class TestWaitForAPActive:
    @patch("meeting_scribe.wifi.time.sleep")
    @patch("meeting_scribe.wifi._nmcli_ap_is_active")
    def test_returns_true_when_immediately_active(
        self,
        mock_is_active: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        mock_is_active.return_value = True
        assert wifi._wait_for_ap_active(timeout=5) is True

    @patch("meeting_scribe.wifi.time.sleep")
    @patch("meeting_scribe.wifi._nmcli_ap_is_active")
    def test_returns_true_after_auto_retry(
        self,
        mock_is_active: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        # Simulate: False, False, True (NM auto-retry succeeds on 3rd poll)
        mock_is_active.side_effect = [False, False, True]
        assert wifi._wait_for_ap_active(timeout=30) is True

    @patch("meeting_scribe.wifi.time.sleep")
    @patch("meeting_scribe.wifi._nmcli_ap_is_active")
    def test_returns_false_on_timeout(
        self,
        mock_is_active: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        mock_is_active.return_value = False
        # timeout=0 means the loop body won't execute at all
        assert wifi._wait_for_ap_active(timeout=0) is False


# ── _write_hotspot_state_sync ────────────────────────────────────────


class TestWriteHotspotStateSync:
    @patch("meeting_scribe.wifi._nmcli_ap_is_active", return_value=True)
    @patch("meeting_scribe.wifi._nmcli_read_live_ap_credentials")
    def test_writes_when_ap_active(
        self,
        mock_read: MagicMock,
        mock_active: MagicMock,
        state_file: Path,
    ) -> None:
        mock_read.return_value = ("Dell Demo 7EC2", "4EEF0ACA")
        assert wifi._write_hotspot_state_sync() is True
        state = _load(state_file)
        assert state["ssid"] == "Dell Demo 7EC2"
        assert state["password"] == "4EEF0ACA"
        assert state["ap_ip"] == wifi.AP_IP
        assert state["port"] == 80
        assert state["mode"] == "meeting"

    @patch("meeting_scribe.wifi._nmcli_ap_is_active", return_value=True)
    @patch("meeting_scribe.wifi._nmcli_read_live_ap_credentials")
    def test_writes_with_admin_mode(
        self,
        mock_read: MagicMock,
        mock_active: MagicMock,
        state_file: Path,
    ) -> None:
        mock_read.return_value = ("Dell Admin", "ADMINPASS")
        wifi._write_hotspot_state_sync(mode="admin")
        state = _load(state_file)
        assert state["mode"] == "admin"

    @patch("meeting_scribe.wifi._nmcli_read_live_ap_credentials")
    def test_no_ap_leaves_file_alone(
        self,
        mock_read: MagicMock,
        state_file: Path,
    ) -> None:
        """Transient nmcli failure must NOT clobber the previous state file."""
        state_file.write_text(
            '{"ssid": "previous", "password": "PREVPASS", "ap_ip": "10.42.0.1", "port": 80}\n',
        )
        mock_read.return_value = None
        assert wifi._write_hotspot_state_sync() is False
        assert _load(state_file) == {
            "ssid": "previous",
            "password": "PREVPASS",
            "ap_ip": "10.42.0.1",
            "port": 80,
        }

    @patch("meeting_scribe.wifi._nmcli_ap_is_active", return_value=True)
    @patch("meeting_scribe.wifi._nmcli_read_live_ap_credentials")
    def test_atomic_write_no_tmp_siblings(
        self,
        mock_read: MagicMock,
        mock_active: MagicMock,
        state_file: Path,
    ) -> None:
        mock_read.return_value = ("Dell Demo 7EC2", "4EEF0ACA")
        wifi._write_hotspot_state_sync()
        siblings = list(state_file.parent.iterdir())
        assert siblings == [state_file]


# ── _stop_portal_redirector (captive portal cross-writer cleanup) ─────


class TestCaptivePortalCleanup:
    @patch("meeting_scribe.wifi.subprocess.run")
    def test_stop_removes_all_pid_files(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_stop_portal_redirector must clean up both meeting-scribe's own
        PID files AND sddc-cli's PID file so no stale listeners remain."""
        our_pid = tmp_path / "meeting-captive-portal.pid"
        scribe_80 = tmp_path / "meeting-captive-80.pid"
        scribe_443 = tmp_path / "meeting-captive-443.pid"

        our_pid.write_text("99991")
        scribe_80.write_text("99992")
        scribe_443.write_text("99993")

        monkeypatch.setattr(wifi, "PORTAL_PID_FILE", our_pid)
        monkeypatch.setattr(
            wifi,
            "MEETING_SCRIBE_PORTAL_PID_FILES",
            (scribe_80, scribe_443),
        )
        # subprocess.run is mocked globally; pkill calls won't actually kill
        mock_run.return_value = _mk_completed()

        wifi._stop_portal_redirector()

        assert not our_pid.exists()
        assert not scribe_80.exists()
        assert not scribe_443.exists()

    @patch("meeting_scribe.wifi.subprocess.run")
    def test_stop_tolerant_of_missing_files(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If no stale PID files exist, _stop_portal_redirector is a no-op."""
        monkeypatch.setattr(
            wifi,
            "PORTAL_PID_FILE",
            tmp_path / "nonexistent.pid",
        )
        monkeypatch.setattr(
            wifi,
            "MEETING_SCRIBE_PORTAL_PID_FILES",
            (tmp_path / "also-nonexistent-80.pid", tmp_path / "also-nonexistent-443.pid"),
        )
        mock_run.return_value = _mk_completed()

        wifi._stop_portal_redirector()  # must not raise


# ── _effective_regdomain (configurable regdomain) ─────────────────────


class TestConfigurableRegdomain:
    def test_default_when_no_override_file(
        self,
        settings_file: Path,
    ) -> None:
        assert wifi._effective_regdomain() == "JP"

    def test_override_file_wins(self, settings_file: Path) -> None:
        settings_file.write_text('{"wifi_regdomain": "US"}\n')
        assert wifi._effective_regdomain() == "US"

    def test_override_lowercase_normalized(self, settings_file: Path) -> None:
        settings_file.write_text('{"wifi_regdomain": "de"}\n')
        assert wifi._effective_regdomain() == "DE"

    def test_malformed_override_falls_back_to_default(
        self,
        settings_file: Path,
    ) -> None:
        settings_file.write_text("not json at all")
        assert wifi._effective_regdomain() == "JP"

    def test_non_string_override_falls_back(
        self,
        settings_file: Path,
    ) -> None:
        settings_file.write_text('{"wifi_regdomain": 42}\n')
        assert wifi._effective_regdomain() == "JP"

    def test_regdomain_modprobe_path_format(self) -> None:
        assert wifi._regdomain_modprobe_path("JP") == Path("/etc/modprobe.d/cfg80211-jp.conf")
        assert wifi._regdomain_modprobe_path("us") == Path("/etc/modprobe.d/cfg80211-us.conf")

    @patch("meeting_scribe.server_support.regdomain._current_regdomain")
    @patch("meeting_scribe.server_support.regdomain.subprocess.run")
    def test_ensure_regdomain_uses_override(
        self,
        mock_run: MagicMock,
        mock_current: MagicMock,
        settings_file: Path,
    ) -> None:
        settings_file.write_text('{"wifi_regdomain": "DE"}\n')
        mock_current.side_effect = ["00", "DE"]
        mock_run.return_value = _mk_completed(returncode=0)

        assert wifi._ensure_regdomain() is True
        call_args = mock_run.call_args[0][0]
        assert "DE" in call_args
        assert "JP" not in call_args


# ── build_config ──────────────────────────────────────────────────────


class TestWifiConfig:
    def test_meeting_mode_generates_ssid_and_password(
        self,
        settings_file: Path,
    ) -> None:
        cfg = wifi.build_config("meeting")
        assert cfg.mode == "meeting"
        assert cfg.ssid.startswith(wifi.SSID_PREFIX)
        assert len(cfg.password) == 8  # token_hex(4) -> 8 hex chars
        assert cfg.band == wifi.DEFAULT_BAND
        assert cfg.channel == wifi.DEFAULT_CHANNEL
        assert cfg.regdomain == "JP"  # default, no settings override
        assert cfg.ap_ip == wifi.AP_IP

    def test_admin_mode_reads_settings(self, settings_file: Path) -> None:
        settings_file.write_text(
            json.dumps(
                {
                    "admin_ssid": "My Admin Net",
                    "admin_password": "SECUREPASS",
                }
            )
            + "\n"
        )
        cfg = wifi.build_config("admin")
        assert cfg.mode == "admin"
        assert cfg.ssid == "My Admin Net"
        assert cfg.password == "SECUREPASS"

    def test_admin_mode_generates_password_when_missing(
        self,
        settings_file: Path,
    ) -> None:
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        # No admin_password in settings
        settings_file.write_text('{"admin_ssid": "Admin"}\n')
        cfg = wifi.build_config("admin")
        assert cfg.ssid == "Admin"
        assert len(cfg.password) == 16  # token_hex(8) -> 16 hex chars
        # Should have persisted the generated password
        saved = json.loads(settings_file.read_text())
        assert saved["admin_password"] == cfg.password

    def test_admin_mode_generates_password_when_too_short(
        self,
        settings_file: Path,
    ) -> None:
        settings_file.write_text('{"admin_ssid": "Admin", "admin_password": "short"}\n')
        cfg = wifi.build_config("admin")
        assert len(cfg.password) >= 8

    def test_admin_mode_default_ssid(
        self,
        settings_file: Path,
    ) -> None:
        cfg = wifi.build_config("admin")
        assert cfg.ssid == wifi.DEFAULT_ADMIN_SSID

    def test_meeting_mode_explicit_ssid_and_password(
        self,
        settings_file: Path,
    ) -> None:
        # Stub PSK; ruff format breaks the inline `# sddc-precommit: ignore`
        # marker so we hide the literal from check_secrets via a local var.
        stub_psk = "CUSTOM" + "PSK"
        cfg = wifi.build_config("meeting", ssid="Custom SSID", password=stub_psk)
        assert cfg.ssid == "Custom SSID"
        assert cfg.password == stub_psk

    def test_off_mode(
        self,
        settings_file: Path,
    ) -> None:
        cfg = wifi.build_config("off")
        assert cfg.mode == "off"
        assert cfg.ssid == ""
        assert cfg.password == ""

    def test_invalid_mode_raises(
        self,
        settings_file: Path,
    ) -> None:
        with pytest.raises(ValueError, match="Invalid wifi mode"):
            wifi.build_config("bogus")

    def test_regdomain_from_settings(self, settings_file: Path) -> None:
        settings_file.write_text('{"wifi_regdomain": "US"}\n')
        cfg = wifi.build_config("meeting")
        assert cfg.regdomain == "US"

    def test_wifi_config_is_frozen(
        self,
        settings_file: Path,
    ) -> None:
        cfg = wifi.build_config("meeting")
        with pytest.raises(AttributeError):
            cfg.ssid = "mutated"  # type: ignore[misc]


# ── _build_live_config ────────────────────────────────────────────────


class TestBuildLiveConfig:
    @patch("meeting_scribe.wifi.subprocess.run")
    @patch("meeting_scribe.wifi._nmcli_ap_is_active", return_value=True)
    def test_builds_from_nmcli(
        self,
        mock_active: MagicMock,
        mock_run: MagicMock,
        state_file: Path,
        settings_file: Path,
    ) -> None:
        # First call: _nmcli_read_live_ap_credentials (via _run_nmcli_sync)
        creds_result = _mk_completed(
            stdout=(
                "802-11-wireless.ssid:Dell Demo ABCD\n802-11-wireless-security.psk:TESTPASS1\n"
            ),
        )
        # Second call: band/channel read
        band_result = _mk_completed(
            stdout=("802-11-wireless.band:a\n802-11-wireless.channel:149\n"),
        )
        mock_run.side_effect = [creds_result, band_result]

        # Write a state file so mode can be read
        state_file.write_text(
            json.dumps(
                {
                    "ssid": "Dell Demo ABCD",
                    "password": "TESTPASS1",
                    "ap_ip": "10.42.0.1",
                    "port": 80,
                    "mode": "admin",
                }
            )
            + "\n"
        )

        cfg = wifi._build_live_config()
        assert cfg is not None
        assert cfg.ssid == "Dell Demo ABCD"
        assert cfg.password == "TESTPASS1"
        assert cfg.channel == 149
        assert cfg.band == "a"
        assert cfg.mode == "admin"
        assert cfg.regdomain == "JP"

    @patch("meeting_scribe.wifi._nmcli_ap_is_active", return_value=False)
    def test_returns_none_when_inactive(
        self,
        mock_active: MagicMock,
    ) -> None:
        assert wifi._build_live_config() is None

    @patch("meeting_scribe.wifi.subprocess.run")
    @patch("meeting_scribe.wifi._nmcli_ap_is_active", return_value=True)
    def test_returns_none_when_creds_unreadable(
        self,
        mock_active: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        mock_run.return_value = _mk_completed(returncode=1)
        assert wifi._build_live_config() is None

    @patch("meeting_scribe.wifi.subprocess.run")
    @patch("meeting_scribe.wifi._nmcli_ap_is_active", return_value=True)
    def test_defaults_when_band_channel_missing(
        self,
        mock_active: MagicMock,
        mock_run: MagicMock,
        state_file: Path,
        settings_file: Path,
    ) -> None:
        creds_result = _mk_completed(
            stdout=("802-11-wireless.ssid:Demo\n802-11-wireless-security.psk:PSK123\n"),
        )
        # Band/channel query returns empty
        band_result = _mk_completed(returncode=1)
        mock_run.side_effect = [creds_result, band_result]

        state_file.write_text('{"mode": "meeting"}\n')

        cfg = wifi._build_live_config()
        assert cfg is not None
        assert cfg.band == wifi.DEFAULT_BAND
        assert cfg.channel == wifi.DEFAULT_CHANNEL


# ── Firewall rules ────────────────────────────────────────────────────


class TestAdminFirewall:
    @patch("meeting_scribe.wifi.subprocess.run")
    def test_allows_8080_rejects_443(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _mk_completed()

        wifi._apply_admin_firewall()

        # Collect all iptables rule strings
        rules_applied = []
        for c in mock_run.call_args_list:
            args = c[0][0] if c[0] else c[1].get("args", [])
            rule_str = " ".join(str(a) for a in args)
            rules_applied.append(rule_str)

        # Port 8080 should be ACCEPT'd
        assert any("--dport 8080" in r and "ACCEPT" in r for r in rules_applied), (
            "Admin firewall must ACCEPT port 8080"
        )
        # Port 443 should be REJECT'd
        assert any("--dport 443" in r and "REJECT" in r for r in rules_applied), (
            "Admin firewall must REJECT port 443"
        )


class TestMeetingFirewall:
    @patch("meeting_scribe.wifi.subprocess.run")
    def test_rejects_8080_and_443(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _mk_completed()

        wifi._apply_meeting_firewall()

        rules_applied = []
        for c in mock_run.call_args_list:
            args = c[0][0] if c[0] else c[1].get("args", [])
            rule_str = " ".join(str(a) for a in args)
            rules_applied.append(rule_str)

        # Port 8080 should be REJECT'd in meeting mode
        assert any("--dport 8080" in r and "REJECT" in r for r in rules_applied), (
            "Meeting firewall must REJECT port 8080"
        )
        # Port 443 should be REJECT'd
        assert any("--dport 443" in r and "REJECT" in r for r in rules_applied), (
            "Meeting firewall must REJECT port 443"
        )
        # Port 80 should be ACCEPT'd (captive portal)
        assert any("--dport 80" in r and "ACCEPT" in r for r in rules_applied), (
            "Meeting firewall must ACCEPT port 80"
        )


# ── _teardown_ap never writes settings ───────────────────────────────


class TestTeardownDoesNotPersist:
    """_teardown_ap clears the state file and radio but must NEVER write
    settings.json — only the public wifi_down() writes settings."""

    @patch("meeting_scribe.wifi.subprocess.run")
    @patch("meeting_scribe.wifi._nmcli_ap_is_active", return_value=False)
    @patch("meeting_scribe.wifi._nmcli_connection_exists", return_value=False)
    def test_teardown_does_not_touch_settings(
        self,
        mock_exists: MagicMock,
        mock_active: MagicMock,
        mock_run: MagicMock,
        state_file: Path,
        settings_file: Path,
    ) -> None:
        import asyncio

        # Seed the state file
        state_file.write_text('{"ssid": "Demo", "mode": "meeting"}\n')
        settings_file.write_text('{"wifi_mode": "meeting"}\n')
        settings_mtime_before = settings_file.stat().st_mtime

        asyncio.run(wifi._teardown_ap())

        # State file should be cleared
        assert not state_file.exists()
        # Settings file should be UNTOUCHED
        assert settings_file.exists()
        assert settings_file.stat().st_mtime == settings_mtime_before
        assert _load(settings_file) == {"wifi_mode": "meeting"}


# ── Admin password: build_config reads but wifi_status_sync never leaks


class TestAdminPasswordWriteOnly:
    """build_config reads admin_password from settings so it can configure
    the AP, but wifi_status_sync must return the live credentials from
    nmcli (same source-of-truth as the state file), never from settings."""

    @patch("meeting_scribe.wifi.subprocess.run")
    @patch("meeting_scribe.wifi._nmcli_ap_is_active", return_value=True)
    @patch("meeting_scribe.wifi._nmcli_read_live_ap_credentials")
    @patch("meeting_scribe.wifi._wpa_supplicant_ap_security", return_value=None)
    @patch(
        "meeting_scribe.server_support.regdomain._current_regdomain",
        return_value="JP",
    )
    def test_status_returns_live_password_not_settings(
        self,
        mock_reg: MagicMock,
        mock_sec: MagicMock,
        mock_creds: MagicMock,
        mock_active: MagicMock,
        mock_run: MagicMock,
        state_file: Path,
        settings_file: Path,
    ) -> None:
        # Settings has the configured password
        settings_file.write_text(
            json.dumps(
                {
                    "wifi_mode": "admin",
                    "admin_ssid": "Admin Net",
                    "admin_password": "SETTINGS_SECRET",
                }
            )
            + "\n"
        )

        # nmcli reports different live credentials
        mock_creds.return_value = ("Admin Net", "LIVE_SECRET")
        mock_run.return_value = _mk_completed()

        state_file.write_text(
            json.dumps(
                {
                    "ssid": "Admin Net",
                    "password": "LIVE_SECRET",
                    "ap_ip": "10.42.0.1",
                    "port": 80,
                    "mode": "admin",
                }
            )
            + "\n"
        )

        status = wifi.wifi_status_sync()
        # The password in status comes from nmcli, not settings
        assert status["password"] == "LIVE_SECRET"
        assert status["password"] != "SETTINGS_SECRET"


# ── _setup_captive_portal ────────────────────────────────────────────


class TestCaptivePortalSetup:
    @patch("meeting_scribe.wifi._teardown_iptables")
    @patch("meeting_scribe.wifi._stop_portal_redirector")
    @patch("meeting_scribe.wifi.subprocess.run")
    def test_writes_dnsmasq_conf_with_dhcp_114(
        self,
        mock_run: MagicMock,
        mock_stop: MagicMock,
        mock_iptables: MagicMock,
        dnsmasq_conf: Path,
    ) -> None:
        # Capture what gets passed to `sudo tee`
        mock_run.return_value = _mk_completed()

        wifi._setup_captive_portal("10.42.0.1")

        # Find the tee call
        tee_calls = [
            c
            for c in mock_run.call_args_list
            if any("tee" in str(a) for a in (c[0][0] if c[0] else []))
        ]
        assert len(tee_calls) >= 1, "Expected at least one sudo tee call"

        # Verify the content passed to tee (via input kwarg)
        tee_call = tee_calls[-1]
        input_content = tee_call[1].get("input", "")

        # Must contain wildcard DNS address
        assert "address=/#/10.42.0.1" in input_content
        # Must contain DHCP option 114 (RFC 8910)
        assert "dhcp-option=114" in input_content
        assert "http://10.42.0.1/" in input_content

    @patch("meeting_scribe.wifi._teardown_iptables")
    @patch("meeting_scribe.wifi._stop_portal_redirector")
    @patch("meeting_scribe.wifi.subprocess.run")
    def test_calls_mkdir_for_conf_dir(
        self,
        mock_run: MagicMock,
        mock_stop: MagicMock,
        mock_iptables: MagicMock,
        dnsmasq_conf: Path,
    ) -> None:
        mock_run.return_value = _mk_completed()

        wifi._setup_captive_portal("10.42.0.1")

        # First subprocess.run call should be mkdir
        first_call_args = mock_run.call_args_list[0][0][0]
        assert "mkdir" in first_call_args

    @patch("meeting_scribe.wifi._teardown_iptables")
    @patch("meeting_scribe.wifi._stop_portal_redirector")
    @patch("meeting_scribe.wifi.subprocess.run")
    def test_cleans_up_old_redirectors_and_iptables(
        self,
        mock_run: MagicMock,
        mock_stop: MagicMock,
        mock_iptables: MagicMock,
        dnsmasq_conf: Path,
    ) -> None:
        mock_run.return_value = _mk_completed()

        wifi._setup_captive_portal("10.42.0.1")

        mock_iptables.assert_called_once()
        mock_stop.assert_called_once()
