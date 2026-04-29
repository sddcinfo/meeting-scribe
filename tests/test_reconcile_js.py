"""Wrapper that runs the JS meeting-reconcile regression tests inside pytest.

The reconciler lives in ``static/js/meeting-reconcile.js`` and owns every
client-side live-meeting state transition (rehydration, recorder
ownership, banner precedence, AbortController timeout). The full test
surface lives in ``tests/js/reconcile.test.mjs`` and runs under Node's
built-in test runner. This wrapper exists so a single ``pytest``
invocation also exercises that suite — it would be too easy to miss
"oh, you also need to run node --test separately" otherwise.

Skipped (not failed) when Node isn't on PATH so the rest of the Python
suite still runs in environments without a JS runtime.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "tests" / "js" / "reconcile.test.mjs"


def test_reconcile_regression_suite_passes_under_node():
    if shutil.which("node") is None:
        pytest.skip("node not on PATH; skipping reconcile JS regression suite")
    if not TEST_FILE.exists():
        pytest.fail(f"missing JS test file: {TEST_FILE}")

    # 60 s ceiling — the checkStatus timeout test waits ~5 s on its own;
    # the rest of the suite finishes in milliseconds.
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
            "reconcile JS regression suite FAILED — see "
            "static/js/meeting-reconcile.js and the comment block at "
            "the top of tests/js/reconcile.test.mjs.\n\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
