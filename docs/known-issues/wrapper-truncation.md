# Console-script wrapper truncation — ROOT-CAUSED 2026-05-15

## Symptom (historical, now fixed)

`.venv/bin/meeting-scribe` would intermittently land as an 11-byte file
containing only `#!/bin/sh\n` after some sequence of operations
involving `pip install -e .`. When in this state:

- `meeting-scribe …` invocations exited 0 immediately with no output.
- The systemd unit (when it was invoking the wrapper) failed with
  `Result: protocol` — the empty shell script exited 0 before sending
  `READY=1`, breaking `Type=notify` start.
- Other operator commands silently no-op'd.

Observed truncation dates: 2026-05-07, 2026-05-12, 2026-05-15.

## Root cause

`tests/test_cli_install_service.py` was writing the pathological
`#!/bin/sh\n` stub directly into the **real** `.venv/bin/meeting-scribe`.

Two offending sites in that file:

1. The `fake_home` fixture (line 36) used
   `Path(__file__).resolve().parents[1] / ".venv" / "bin"` instead of
   `tmp_path`, then wrote `#!/bin/sh\n` to `meeting-scribe` if the
   wrapper didn't already exist.
2. `test_install_service_bails_when_venv_entry_missing` (line 200)
   `unlink()`-ed the real wrapper, ran the bail-on-missing assertion,
   then "restored" it by writing `#!/bin/sh\n` — leaving exactly the
   truncated state we kept blaming on pip.

The earlier "pip install -e . race" hypothesis was wrong. The tests
were the source. Every `pytest tests/test_cli_install_service.py` run
guaranteed the wrapper landed in the broken state.

## Fix

`tests/test_cli_install_service.py` now:

- Builds a fake venv under `tmp_path` in the `fake_home` fixture.
- Monkeypatches `meeting_scribe.cli.install_service._venv_bin` to
  return that tmp-path location.
- Mutates only the fake wrapper in
  `test_install_service_bails_when_venv_entry_missing`.

Real `.venv/bin/meeting-scribe` is never touched by tests again.

## Mitigation in service path (still in place)

The systemd unit still renders `<venv>/bin/python3 -m meeting_scribe …`
rather than the console-script wrapper — this remains a useful belt &
suspenders since user-shell invocations of `meeting-scribe …` always
go through the wrapper. `tests/test_install_service_python_module.py`
locks the contract.

## Operator-facing repair (kept for forward compat)

If `meeting-scribe …` ever exits 0 with no output again:

```sh
mise exec -- python scripts/repair_venv.py
```

`scripts/repair_venv.py` rebuilds the wrapper via
`pip install --force-reinstall -e .` and re-detects python-version
drift between `mise.toml` and the existing `.venv`.

## Related

- Locking test: `tests/test_install_service_python_module.py`
- Repair entry point: `scripts/repair_venv.py`
- Bootstrap call site: `bootstrap.sh` (self-heals on every bootstrap)
