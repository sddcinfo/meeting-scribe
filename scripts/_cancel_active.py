"""POST /api/meeting/cancel against the running server (with admin auth)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / "src"))
    from meeting_scribe.setup_state import _mint_admin_password

    pw = _mint_admin_password()
    jar = tempfile.NamedTemporaryFile(prefix="ms_jar_", suffix=".txt", delete=False)  # noqa: SIM115 — name reused after close
    jar.close()

    subprocess.run(
        [
            "curl",
            "-k",
            "-sS",
            "-c",
            jar.name,
            "-b",
            jar.name,
            "-d",
            f"password={pw}",
            "https://127.0.0.1:443/api/admin/authorize",
            "-o",
            "/dev/null",
        ],
        check=True,
    )

    out = subprocess.check_output(
        [
            "curl",
            "-k",
            "-sS",
            "-b",
            jar.name,
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "https://127.0.0.1:443/api/meeting/cancel",
        ],
        timeout=10,
    ).decode()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
