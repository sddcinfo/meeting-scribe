"""Test server management — starts an isolated server for integration tests.

The test server:
- Runs on port 9080 (separate from production on 8080)
- Uses a temporary meetings directory (cleaned up after tests)
- Connects to the SAME model backends (ports 8000-8003) — no GPU duplication
- Starts automatically via pytest fixture, stops on teardown

Usage:
    pytest -m integration          # Runs only integration tests against test server
    pytest                         # Runs only unit tests (fast, no server needed)
    pytest -m "not integration"    # Same as above, explicit
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
TEST_PORT = 9080
_test_server_proc = None
_test_meetings_dir = None


def _wait_for_server(port: int, timeout: int = 120) -> bool:
    """Wait for the test server to become responsive."""
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for _ in range(timeout):
        try:
            with httpx.Client(verify=False, timeout=3) as c:
                r = c.get(f"https://127.0.0.1:{port}/api/status")
                if r.status_code == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def start_test_server() -> subprocess.Popen:
    """Start an isolated test server on TEST_PORT."""
    global _test_server_proc, _test_meetings_dir

    _test_meetings_dir = tempfile.mkdtemp(prefix="scribe-test-meetings-")

    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python3"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["SCRIBE_PORT"] = str(TEST_PORT)
    env["SCRIBE_HOST"] = "0.0.0.0"
    env["SCRIBE_PROFILE"] = "gb10"
    env["SCRIBE_MEETINGS_DIR"] = _test_meetings_dir
    # Preserve empty meetings so integration tests that do
    # start → stop → GET /api/meetings/{id}/* can fetch the meeting after
    # stop. Production leaves this unset, so zero-event carcasses are
    # still cleaned up on real meetings.
    env["SCRIBE_PRESERVE_EMPTY_ON_STOP"] = "1"

    # Load .env for HF_TOKEN etc.
    dotenv = PROJECT_ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                env.setdefault(key.strip(), val.strip())

    cmd = [
        str(venv_python),
        "-m",
        "uvicorn",
        "meeting_scribe.server:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(TEST_PORT),
        "--log-level",
        "warning",
    ]

    # TLS certs
    ssl_key = PROJECT_ROOT / "certs" / "key.pem"
    ssl_cert = PROJECT_ROOT / "certs" / "cert.pem"
    if ssl_key.exists() and ssl_cert.exists():
        cmd += ["--ssl-keyfile", str(ssl_key), "--ssl-certfile", str(ssl_cert)]

    log_file = Path(tempfile.gettempdir()) / "meeting-scribe-test.log"
    with open(log_file, "w") as log_f:
        _test_server_proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(PROJECT_ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    if not _wait_for_server(TEST_PORT):
        stop_test_server()
        pytest.fail(f"Test server failed to start on port {TEST_PORT}. Check logs: {log_file}")

    return _test_server_proc


def stop_test_server():
    """Stop the test server and clean up."""
    global _test_server_proc, _test_meetings_dir

    if _test_server_proc:
        try:
            os.kill(_test_server_proc.pid, signal.SIGTERM)
            _test_server_proc.wait(timeout=10)
        except ProcessLookupError, subprocess.TimeoutExpired:
            try:
                os.kill(_test_server_proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _test_server_proc = None

    if _test_meetings_dir:
        import shutil

        shutil.rmtree(_test_meetings_dir, ignore_errors=True)
        _test_meetings_dir = None

    # Fixture is defined in conftest.py — this module provides only the helpers
