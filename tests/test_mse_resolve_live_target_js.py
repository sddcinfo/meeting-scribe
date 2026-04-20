"""Wrapper that runs the JS resolveLiveTarget tests inside pytest.

The canonical implementation of _resolveLiveTarget lives inline in
``static/guest.html``; the testable extraction lives in
``static/js/mse-audio.js``. The full test surface runs under Node's
built-in test runner. This wrapper exists so a single ``pytest``
invocation also exercises that suite.

Skipped (not failed) when Node isn't on PATH so the rest of the Python
suite still runs in environments without a JS runtime.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "tests" / "js" / "mse-audio.test.mjs"


def test_mse_resolve_live_target_suite_passes_under_node():
    if shutil.which("node") is None:
        pytest.skip("node not on PATH; skipping resolveLiveTarget JS suite")
    if not TEST_FILE.exists():
        pytest.fail(f"missing JS test file: {TEST_FILE}")

    result = subprocess.run(
        ["node", "--test", str(TEST_FILE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "resolveLiveTarget JS suite FAILED — see static/js/mse-audio.js "
            "and tests/js/mse-audio.test.mjs.\n\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
