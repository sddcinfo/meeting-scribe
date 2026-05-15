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
| `factory-reset`                    | daemon-control   | Settings → "Factory reset" (destructive button)         | `POST /api/admin/factory-reset`        | ✅      |
| `doctor`                           | daemon-control   | Diagnostics card (live result)                          | `GET /api/admin/diag/audio`            | ✅      |
| `install-service`                  | daemon-control   | n/a                                                     | n/a                                    | ✅      |
| `status`                           | diagnostics      | Header status pill                                      | `GET /api/status`                      | ✅      |
| `version`                          | diagnostics      | Footer version line                                     | `GET /api/status`                      | ✅      |
| `wifi up`                          | cli-only         | (read-only banner — see §C.0a)                          | n/a                                    | ✅      |
| `wifi down`                        | cli-only         | (read-only banner)                                      | n/a                                    | ✅      |
| `wifi status`                      | operator-facing  | WiFi card (read-only)                                   | `GET /api/admin/wifi/status`           | ⏳      |
| `wifi wan up`                      | operator-facing  | WAN card → "Bring up active WAN" button                 | `POST /api/admin/wan/up`               | ✅      |
| `wifi wan down`                    | operator-facing  | WAN card → "Tear down WAN" button                       | `POST /api/admin/wan/down`             | ✅      |
| `wifi wan status`                  | diagnostics      | WAN card per-iface state pills                          | `GET /api/admin/wan/status`            | ✅      |
| `wifi wan scan`                    | operator-facing  | WAN card → "Scan" button                                | `GET /api/admin/wan/scan`              | ✅      |
| `wifi wan profiles ls`             | operator-facing  | WAN card → "Saved profiles" list                        | `GET /api/admin/wan/profiles`          | ✅      |
| `wifi wan profiles add`            | operator-facing  | WAN card → "Add a profile" form                         | `POST /api/admin/wan/profiles`         | ✅      |
| `wifi wan profiles rm`             | operator-facing  | WAN card → profile row "Delete" button                  | `DELETE /api/admin/wan/profiles/{id}`  | ✅      |
| `wifi wan profiles set-active`     | operator-facing  | WAN card → profile row "Set active" button              | `POST /api/admin/wan/profiles/{id}/set-active` | ✅ |
| `wifi wan mode get`                | diagnostics      | WAN card → "Egress mode" row (radio + source label)     | `GET /api/admin/wan/mode`              | ✅      |
| `wifi wan mode set`                | operator-facing  | WAN card → "Egress mode" radio (block/gateway/captive)  | `PUT /api/admin/wan/mode`              | ✅      |
| `bt pair`                          | operator-facing  | BT card → "Pair new device" modal                       | `POST /api/admin/bt/pair`              | ⏳      |
| `bt connect`                       | operator-facing  | BT card → device row "Connect" button                   | `POST /api/admin/bt/connect`           | ⏳      |
| `bt disconnect`                    | operator-facing  | BT card → device row "Disconnect" button                | `POST /api/admin/bt/disconnect`        | ⏳      |
| `bt forget`                        | operator-facing  | BT card → device row "Forget" + confirmation            | `POST /api/admin/bt/forget`            | ⏳      |
| `bt status`                        | diagnostics      | BT card status row                                      | `GET /api/admin/bt/status`             | ⏳      |
| `bt profile`                       | diagnostics      | BT card → device row tooltip                            | `GET /api/admin/bt/status`             | ⏳      |
| `bt nodes`                         | diagnostics      | (CLI only — internal debug)                             | n/a                                    | ✅      |
| `audio`                            | operator-facing  | Settings + setup meeting-audio cards                    | `GET /api/admin/audio/*`               | ✅      |
| `devices`                          | operator-facing  | Settings + setup meeting-audio cards                    | `GET /api/admin/audio/devices`         | ✅      |
| `route`                            | operator-facing  | Settings + setup meeting-audio cards                    | `GET/POST /api/admin/audio/route`      | ✅      |
| `interpretation`                   | operator-facing  | Live meeting TTS controls + settings/setup controls     | `GET/POST /api/admin/audio/interpretation` | ✅  |
| `meetings start`                   | operator-facing  | Header logo-mark click                                  | `POST /api/meetings/<id>/start`        | ✅      |
| `meetings stop`                    | operator-facing  | Header logo-mark click (stop)                           | `POST /api/meetings/<id>/stop`         | ✅      |
| `meetings finalize`                | operator-facing  | Meeting list → Finalize action                          | `POST /api/meetings/<id>/finalize`     | ✅      |
| `finalize status`                  | operator-facing  | (CLI-only — Phase B observability)                      | `GET /api/admin/finalize/status`       | ✅      |
| `finalize retry`                   | operator-facing  | Background-finalize toast → Reprocess button            | `POST /api/meetings/<id>/reprocess`    | ✅      |
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
| `speakerphone`                     | cli-only         | (group root; see subcommands below)                     | n/a                                    | ✅      |
| `speakerphone install`             | daemon-control   | n/a — udev rule + systemd unit install (root)           | n/a                                    | ✅      |
| `speakerphone uninstall`           | daemon-control   | n/a — udev rule + systemd unit removal (root)           | n/a                                    | ✅      |
| `speakerphone listen`              | daemon-control   | n/a — long-running listener daemon (systemd unit)       | n/a                                    | ✅      |
| `speakerphone test`                | dev-only         | Hardware tab → "Test LEDs" / "Test feedback" buttons    | `POST /api/admin/speakerphone/led-test`| ✅      |
| `speakerphone compliance`          | operator-facing  | Hardware tab → wideband compliance chip                 | `GET /api/admin/speakerphone/wideband-status` | ✅ |
| `speakerphone set-wideband`        | operator-facing  | Hardware tab → "Re-apply wideband" button               | `POST /api/admin/speakerphone/wideband-apply` | ✅ |
| `speakerphone capture-descriptor`  | dev-only         | n/a — HID descriptor dump for protocol reverse-eng      | n/a                                    | ✅      |
| `speakerphone detect-rate`         | dev-only         | n/a — PipeWire native-rate detection probe              | n/a                                    | ✅      |
| `speakerphone benchmark`           | dev-only         | n/a — SP325 wideband sweep harness                      | n/a                                    | ✅      |
| `kiosk install`                    | daemon-control   | n/a — cage + chromium systemd unit install (root)       | n/a                                    | ✅      |
| `kiosk uninstall`                  | daemon-control   | n/a — kiosk unit removal (root)                         | n/a                                    | ✅      |
| `kiosk up`                         | operator-facing  | HDMI Settings tab → "Start kiosk" button                | `POST /api/admin/kiosk/up`             | ⏳      |
| `kiosk down`                       | operator-facing  | HDMI Settings tab → "Stop kiosk" button                 | `POST /api/admin/kiosk/down`           | ⏳      |
| `kiosk status`                     | diagnostics      | HDMI Settings tab → HDMI status row                     | `GET /api/admin/kiosk/status`          | ✅      |
| `kiosk mint-nonce`                 | cli-only         | n/a — bootstrap nonce for the kiosk cookie exchange     | `POST /api/admin/kiosk/mint-nonce`     | ✅      |
| `kiosk run-runtime`                | cli-only         | n/a — hidden systemd entry point inside cage session    | n/a                                    | ✅      |

Legend: ✅ landed · ⏳ pending

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
