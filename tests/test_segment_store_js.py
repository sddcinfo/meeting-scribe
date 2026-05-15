"""Wrapper that runs the JS SegmentStore regression tests inside pytest.

The SegmentStore lives in `static/js/segment-store.js` and has been the
source of three production bugs where transcript dimensions silently
disappeared (translation dropped by furigana race, furigana dropped by
speaker catch-up race, speakers wiped by a higher-rev furigana update).
The full test surface lives in `tests/js/segment-store.test.mjs` and runs
under Node's built-in test runner. This wrapper exists so a single
`pytest` invocation also exercises that suite — it would be too easy to
miss "oh, you also need to run node --test separately" otherwise.

Skipped (not failed) when Node isn't on PATH so the rest of the Python
suite still runs in environments without a JS runtime.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "tests" / "js" / "segment-store.test.mjs"


def test_segment_store_regression_suite_passes_under_node():
    if shutil.which("node") is None:
        pytest.skip("node not on PATH; skipping SegmentStore JS regression suite")
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
            "SegmentStore JS regression suite FAILED — see static/js/segment-store.js "
            "and the comment block at the top of tests/js/segment-store.test.mjs.\n\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
