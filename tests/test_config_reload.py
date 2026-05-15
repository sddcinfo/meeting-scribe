"""Tests for runtime_config hot-reload — the Qwen3.6 plan's rollback axis.

Phase 1 prereq #2 of ~/.claude/plans/sprightly-prancing-valiant.md.  The
plan's Phase 6 cutover and Phase 7 rollback both rely on flipping
``translate_url`` / ``slide_translate_url`` / ``slide_use_json_schema`` /
``slide_stats_dir`` without a process restart.  These tests are the
belt-and-suspenders proof that the SIGHUP path actually propagates new
values to consumers before the cutover runbook trusts it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from meeting_scribe import runtime_config as rc


@pytest.fixture
def isolated_rc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[rc._RuntimeConfig]:
    """Install a fresh runtime-config singleton backed by a tmp file.

    Keeps tests from clobbering the real ~/.local/share/meeting-scribe
    config the dev box might have set.
    """
    store = tmp_path / "runtime-config.json"
    fresh = rc._RuntimeConfig(path=store)
    rc.install_singleton(fresh)
    yield fresh
    # No cleanup: the next test's fixture rebuilds a fresh singleton.


class TestAllowlist:
    """Writes only land for keys in _ALLOWED_KEYS."""

    def test_known_key_accepted(self, isolated_rc: rc._RuntimeConfig) -> None:
        isolated_rc.set("translate_url", "http://localhost:8012")
        assert isolated_rc.get("translate_url") == "http://localhost:8012"

    def test_slide_stats_dir_accepted(self, isolated_rc: rc._RuntimeConfig) -> None:
        # New key added for Phase 1 — ensures the allowlist was actually bumped.
        isolated_rc.set("slide_stats_dir", "/tmp/qwen36-shadow/slide_stats")
        assert isolated_rc.get("slide_stats_dir") == "/tmp/qwen36-shadow/slide_stats"

    def test_unknown_key_rejected(self, isolated_rc: rc._RuntimeConfig) -> None:
        # Typo-catching — a misspelled key must fail LOUDLY at set() time
        # instead of silently shadowing the static default forever.
        with pytest.raises(KeyError, match="unknown key 'translation_url'"):
            isolated_rc.set("translation_url", "http://localhost:8012")

    def test_unset_rejected_for_unknown_key(self, isolated_rc: rc._RuntimeConfig) -> None:
        with pytest.raises(KeyError, match="unknown key"):
            isolated_rc.unset("translation_url")


class TestPersistence:
    """set() writes to disk; reload_from_disk() picks up external changes."""

    def test_set_persists_to_disk(self, isolated_rc: rc._RuntimeConfig) -> None:
        isolated_rc.set("translate_url", "http://localhost:8012")
        on_disk = json.loads(isolated_rc.path.read_text())
        assert on_disk == {"translate_url": "http://localhost:8012"}

    def test_reload_picks_up_external_write(self, isolated_rc: rc._RuntimeConfig) -> None:
        # Simulate `meeting-scribe config set` writing the file
        # out-of-process — the server only sees it after SIGHUP.
        isolated_rc.path.parent.mkdir(parents=True, exist_ok=True)
        isolated_rc.path.write_text(json.dumps({"translate_url": "http://localhost:8012"}))

        # Before reload: in-memory dict is empty.
        assert isolated_rc.get("translate_url") is None

        # After reload (what the SIGHUP handler calls): value visible.
        isolated_rc.reload_from_disk()
        assert isolated_rc.get("translate_url") == "http://localhost:8012"

    def test_reload_drops_stale_out_of_allowlist_key(self, isolated_rc: rc._RuntimeConfig) -> None:
        # A file written by an older server version with a key that's
        # since been removed must NOT silently re-appear.
        isolated_rc.path.parent.mkdir(parents=True, exist_ok=True)
        isolated_rc.path.write_text(
            json.dumps(
                {
                    "translate_url": "http://localhost:8012",
                    "some_removed_key": "stale",
                }
            )
        )
        isolated_rc.reload_from_disk()
        snap = isolated_rc.as_dict()
        assert snap == {"translate_url": "http://localhost:8012"}

    def test_missing_file_is_empty_not_error(self, tmp_path: Path) -> None:
        # First boot: no persistence file yet.  Must not crash.
        fresh = rc._RuntimeConfig(path=tmp_path / "does-not-exist.json")
        assert fresh.as_dict() == {}

    def test_malformed_file_is_empty_not_error(self, tmp_path: Path) -> None:
        p = tmp_path / "runtime-config.json"
        p.write_text("not json {{{")
        fresh = rc._RuntimeConfig(path=p)
        assert fresh.as_dict() == {}


class TestConsumersReadFresh:
    """Per-request callers see the new value without a process restart.

    Mirrors the Phase 6 cutover flow: operator writes the file via
    `meeting-scribe config set`, sends SIGHUP, next request reads the
    new URL.  We simulate the whole sequence in-process.
    """

    def test_get_default_returns_fallback_when_unset(self, isolated_rc: rc._RuntimeConfig) -> None:
        # Pattern used by translate_vllm.py + _slide_translate_fn:
        #   rc.get("translate_url", static_default)
        assert rc.get("translate_url", "http://localhost:8010") == "http://localhost:8010"

    def test_get_returns_runtime_value_after_set(self, isolated_rc: rc._RuntimeConfig) -> None:
        rc.get("translate_url", "http://localhost:8010")  # no-op, just for symmetry
        isolated_rc.set("translate_url", "http://localhost:8012")
        # Same call, different answer — this is what "hot-reload" means
        # from a consumer's perspective.
        assert rc.get("translate_url", "http://localhost:8010") == "http://localhost:8012"

    def test_unset_falls_back_to_static_default(self, isolated_rc: rc._RuntimeConfig) -> None:
        isolated_rc.set("translate_url", "http://localhost:8012")
        assert rc.get("translate_url", "http://fallback") == "http://localhost:8012"

        isolated_rc.unset("translate_url")
        assert rc.get("translate_url", "http://fallback") == "http://fallback"

    def test_sighup_equivalent_reload_propagates(self, isolated_rc: rc._RuntimeConfig) -> None:
        # Simulate the full cross-process flow:
        #   1. CLI writes file out-of-process.
        #   2. Server receives SIGHUP → calls reload_from_disk().
        #   3. Next consumer read sees the new value.
        #
        # This is the ONE integration proof the plan's Phase 6 cutover
        # step 6 needs: "kill -HUP $(meeting-scribe pid)" must actually
        # flip the endpoint before resume-translation fires.
        assert rc.get("slide_translate_url", "http://localhost:8010") == "http://localhost:8010"

        isolated_rc.path.parent.mkdir(parents=True, exist_ok=True)
        isolated_rc.path.write_text(json.dumps({"slide_translate_url": "http://localhost:8012"}))

        rc.reload_from_disk()  # what the SIGHUP handler does

        assert rc.get("slide_translate_url", "http://localhost:8010") == "http://localhost:8012"


class TestSlideStatsPathResolution:
    """slide_stats_dir override redirects where _emit_deck_stats writes."""

    def test_default_path_when_unset(self, isolated_rc: rc._RuntimeConfig) -> None:
        from meeting_scribe.slides.job import _DEFAULT_STATS_DIR, _resolve_stats_path

        assert _resolve_stats_path() == _DEFAULT_STATS_DIR / "slide-translation-stats.jsonl"

    def test_override_redirects(self, isolated_rc: rc._RuntimeConfig, tmp_path: Path) -> None:
        from meeting_scribe.slides.job import _resolve_stats_path

        shadow_dir = tmp_path / "qwen36-shadow" / "slide_stats"
        isolated_rc.set("slide_stats_dir", str(shadow_dir))
        assert _resolve_stats_path() == shadow_dir / "slide-translation-stats.jsonl"

    def test_override_read_fresh_per_call(
        self, isolated_rc: rc._RuntimeConfig, tmp_path: Path
    ) -> None:
        # Mid-eval-run retarget scenario: path must reflect the latest
        # set(), not a cached value captured at import time.
        from meeting_scribe.slides.job import _resolve_stats_path

        first = tmp_path / "first"
        second = tmp_path / "second"

        isolated_rc.set("slide_stats_dir", str(first))
        assert _resolve_stats_path().parent == first

        isolated_rc.set("slide_stats_dir", str(second))
        assert _resolve_stats_path().parent == second
