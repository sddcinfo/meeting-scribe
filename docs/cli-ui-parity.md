# CLI ↔ Admin UI Parity Matrix

The contract for v1.0: every operator-facing CLI subcommand has an
equivalent admin UI affordance. The CLI remains authoritative; the UI
is a typed-form wrapper over the same primitives.

CI lints this file against `src/meeting_scribe/cli/` and the admin
routes — adding a new operator-facing CLI subcommand without a matching
UI surface fails the gate. See
`scripts/check_cli_ui_parity.py` (lint).

## Categories

* **operator-facing** — operators run during demos / setup / recovery.
  100 % UI coverage required for the v1.0 ship.
* **daemon-control** — systemd-managed lifecycle (`start`, `stop`,
  `restart`, `setup`, `doctor`, `install-service`). UI surfaces *output*
  (e.g. `doctor` results in a diag card) but the runner stays CLI.
* **diagnostics** — read-only state reads (`status`, `version`).
  Surfaced via the diagnostics card.
* **dev-only** — benchmarks, library scripts, profile probes. Excluded
  from the parity gate.
* **cli-only** — operations that intentionally live in the CLI because
  exposing them on the web would create a footgun (e.g. WiFi reconfig
  that strands the admin session).

## Matrix

| CLI subcommand                     | Category         | UI surface                                              | Admin API                              | Status |
|------------------------------------|------------------|---------------------------------------------------------|----------------------------------------|--------|
| `start`                            | daemon-control   | n/a                                                     | n/a                                    | ✅      |
| `stop`                             | daemon-control   | n/a                                                     | n/a                                    | ✅      |
| `restart`                          | daemon-control   | n/a                                                     | n/a                                    | ✅      |
| `setup`                            | daemon-control   | n/a — provisioning (root)                               | n/a                                    | ✅      |
| `doctor`                           | daemon-control   | Diagnostics card (live result)                          | `GET /api/admin/diag/audio`            | ✅      |
| `install-service`                  | daemon-control   | n/a                                                     | n/a                                    | ✅      |
| `status`                           | diagnostics      | Header status pill                                      | `GET /api/status`                      | ✅      |
| `version`                          | diagnostics      | Footer version line                                     | `GET /api/status`                      | ✅      |
| `wifi up`                          | cli-only         | (read-only banner — see §C.0a)                          | n/a                                    | ✅      |
| `wifi down`                        | cli-only         | (read-only banner)                                      | n/a                                    | ✅      |
| `wifi status`                      | operator-facing  | WiFi card (read-only)                                   | `GET /api/admin/wifi/status`           | ⏳      |
| `bt pair`                          | operator-facing  | BT card → "Pair new device" modal                       | `POST /api/admin/bt/pair`              | ⏳      |
| `bt connect`                       | operator-facing  | BT card → device row "Connect" button                   | `POST /api/admin/bt/connect`           | ⏳      |
| `bt disconnect`                    | operator-facing  | BT card → device row "Disconnect" button                | `POST /api/admin/bt/disconnect`        | ⏳      |
| `bt forget`                        | operator-facing  | BT card → device row "Forget" + confirmation            | `POST /api/admin/bt/forget`            | ⏳      |
| `bt status`                        | diagnostics      | BT card status row                                      | `GET /api/admin/bt/status`             | ⏳      |
| `bt profile`                       | diagnostics      | BT card → device row tooltip                            | `GET /api/admin/bt/status`             | ⏳      |
| `bt nodes`                         | diagnostics      | (CLI only — internal debug)                             | n/a                                    | ✅      |
| `meetings start`                   | operator-facing  | Header logo-mark click                                  | `POST /api/meetings/<id>/start`        | ✅      |
| `meetings stop`                    | operator-facing  | Header logo-mark click (stop)                           | `POST /api/meetings/<id>/stop`         | ✅      |
| `meetings finalize`                | operator-facing  | Meeting list → Finalize action                          | `POST /api/meetings/<id>/finalize`     | ✅      |
| `meetings cleanup`                 | operator-facing  | Meeting list → bulk-prune action                        | `POST /api/admin/meetings/cleanup`     | ⏳      |
| `terminal access`                  | operator-facing  | Settings → "Show terminal secret" reveal                | `GET /api/admin/terminal-access`       | ✅      |
| `config get`                       | operator-facing  | Settings panel                                          | `GET /api/admin/config`                | ✅      |
| `config set`                       | operator-facing  | Settings panel                                          | `POST /api/admin/config`               | ✅      |
| `validate`                         | diagnostics      | (CLI only — fixture-driven)                             | n/a                                    | ✅      |
| `validate-customer`                | diagnostics      | (CLI only)                                              | n/a                                    | ✅      |
| `precommit *`                      | dev-only         | n/a                                                     | n/a                                    | ✅      |
| `bench *` / `benchmark *`          | dev-only         | n/a                                                     | n/a                                    | ✅      |
| `library *`                        | dev-only         | n/a                                                     | n/a                                    | ✅      |
| `hf-probe`                         | dev-only         | n/a                                                     | n/a                                    | ✅      |
| `versions`                         | diagnostics      | Footer                                                  | `GET /api/status`                      | ✅      |
| `gb10 *`                           | dev-only         | n/a                                                     | n/a                                    | ✅      |
| `demo-smoke`                       | dev-only         | n/a                                                     | n/a                                    | ✅      |
| `queue *`                          | diagnostics      | (CLI only — internal queue inspection)                  | n/a                                    | ✅      |
| `export-cert`                      | operator-facing  | n/a — local TTY only (Plan §TLS Trust Anchor)           | n/a                                    | ✅      |
| `cert-fingerprint`                 | operator-facing  | Bootstrap-page identity surface                         | n/a (rendered on page)                 | ✅      |
| `trust-install --from-pem`         | operator-facing  | n/a — runs on admin laptop                              | n/a                                    | ✅      |
| `trust-install --confirm-fingerprint` | operator-facing | n/a — runs on admin laptop                              | n/a                                    | ✅      |
| `trust-uninstall`                  | operator-facing  | n/a — runs on admin laptop                              | n/a                                    | ✅      |
| `appliance-info`                   | diagnostics      | Bootstrap-page known-appliances check                   | n/a                                    | ✅      |
| `logs`                             | diagnostics      | Diagnostics card — log tail                             | `GET /api/admin/logs`                  | ✅      |
| `health`                           | diagnostics      | Diagnostics card                                        | `GET /api/status`                      | ✅      |
| `diagnose`                         | diagnostics      | Diagnostics card                                        | `GET /api/admin/diag/audio`            | ✅      |
| `containers`                       | diagnostics      | Backend pills (status only)                             | `GET /api/status`                      | ✅      |
| `reload`                           | daemon-control   | n/a                                                     | n/a                                    | ✅      |
| `drain`                            | daemon-control   | n/a — drain queue then stop                             | n/a                                    | ✅      |
| `reprocess`                        | operator-facing  | Meeting list → Reprocess action                         | `POST /api/admin/meetings/<id>/reprocess` | ⏳   |
| `reprocess-summaries`              | operator-facing  | Meeting list → Re-summarize action                      | `POST /api/admin/meetings/<id>/resummarize` | ⏳ |
| `full-reprocess`                   | operator-facing  | Meeting list → Full-reprocess action                    | `POST /api/admin/meetings/<id>/full-reprocess` | ⏳ |
| `uninstall-service`                | daemon-control   | n/a                                                     | n/a                                    | ✅      |
| `kill-tmux`                        | dev-only         | n/a — recovery hammer                                   | n/a                                    | ✅      |
| `pause-translation`                | operator-facing  | Settings panel → translation toggle                     | `POST /api/admin/translation/pause`    | ⏳      |
| `resume-translation`               | operator-facing  | Settings panel → translation toggle                     | `POST /api/admin/translation/resume`   | ⏳      |
| `unset`                            | operator-facing  | Settings panel → reset value                            | `POST /api/admin/config`               | ✅      |
| `bench`                            | dev-only         | n/a                                                     | n/a                                    | ✅      |
| `benchmark-translate`              | dev-only         | n/a                                                     | n/a                                    | ✅      |
| `benchmark-install-deps`           | dev-only         | n/a                                                     | n/a                                    | ✅      |

Legend: ✅ landed · ⏳ pending (Phase 15 admin UI BT/WiFi cards)

## CI lint

The lint script in `scripts/check_cli_ui_parity.py` walks the click
group tree, classifies each command per the rules above (the
classification key is a metadata dict embedded next to each
`@cli.command(...)` decorator), and:

* Fails when a new `operator-facing` command lacks a matching admin
  API entry in this file.
* Warns when a CLI command exists but is unclassified — forces the
  PR author to make the call explicitly.

Run locally via `scripts/run-pytest.sh tests/test_cli_ui_parity.py`
(unit-tested by re-deriving the classification + asserting the matrix
is in sync with the actual CLI surface).
