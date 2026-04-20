# Test viewports — static pages

Every static page under `static/` (primarily `how-it-works.html`, but
also `guest.html`, `reader.html`, `portal.html`) is QA'd against the
logical viewports in the table below.  All CSS breakpoints are
expressed in **logical pixels**; the browser's media-query layer
translates logical → physical via device pixel ratio (DPR), so no
hand-rolled DPR handling is needed.

## Canonical device list

| Device | Logical viewport (portrait × landscape) | DPR | Purpose |
|---|---|---:|---|
| Desktop workstation | 1440 × 900 | 1× | Reference for full-width layout, two-column bilingual rendering, topology diagram horizontal. |
| iPad Pro 11" M4 (2024) | 834 × 1194 · 1194 × 834 | 2× | Largest in-production tablet; exercises the tablet-tier breakpoint without topology collapse at 1194 landscape. |
| iPad Pro 9.7" (2016) | 768 × 1024 · 1024 × 768 | 2× | Low-end iPad still in active use.  Landscape 1024 is the breakpoint at which the topology diagram collapses to a single column. |
| iPad (10th gen, 2022) | 820 × 1180 · 1180 × 820 | 2× | Base-model tablet; same breakpoint band as iPad Pro 11" M4. |
| iPhone 17 Pro Max | 440 × 956 | 3× | Largest in-production phone (per Brad's device).  Sits between the 480 px "small phone" and 768 px "tablet portrait" bands, so it has its own `min-width: 416px and max-width: 480px` block. |
| iPhone 16 / 15 / 14 | 390 × 844 | 3× | Sub-440 phones caught by the generic `max-width: 480px` block. |
| Pixel 8 / Pixel 7 | 412 × 915 | ~2.6× | Large Android phone; caught by the iPhone 17 Pro Max block or the generic `max-width: 480px` depending on exact width. |

Notes:

* "Logical pixels" is what `window.innerWidth` reports and what media queries see.  A 1024 × 768 logical iPad Pro 9.7" is 2048 × 1536 physical; CSS doesn't care about the physical number.
* No raster images are shipped with these pages — every icon is an inline SVG and the grain overlay is a data-URI noise pattern — so DPR doesn't change asset sharpness.
* If you add a new tablet breakpoint, check iPad Pro 11" M4 (1194 landscape) doesn't fall between two rules and lose both.

## Breakpoints in `how-it-works.css`

The CSS has a matching set of media queries.  Keep this table in sync
when you add a breakpoint.

| Media query | Triggers on | What changes |
|---|---|---|
| (default) | Desktop ≥ 1025 px | Full two-column bilingual, horizontal topology with connector wires, full flow panels. |
| `@media (max-width: 1024px)` | iPad Pro 9.7" landscape, iPad portrait | `.page` tightens to `--space-lg` padding; topology collapses to a single column (the schematic connectors hide); glossary header stacks. |
| `@media (max-width: 768px)` | Phones, tablet portrait | Bilingual stacks to 1-column; stage cards tighten; `.flow` + `.lifecycle` switch to vertical stack with downward CSS-arrow connectors; mockup shrinks. |
| `@media (max-width: 640px)` | Small phones | Bilingual `.bi` drops borders; `.bi` already at 1-column. |
| `@media (min-width: 416px) and (max-width: 480px)` | iPhone 17 Pro Max portrait | `.topology` tightens padding; `topo-service` grid re-flows port under label; memory bar height bumped to 22 px for touch readability. |
| `@media (max-width: 480px)` | Sub-iPhone-17 phones | Mockup grid stacks vertically, nav shrinks further. |

## Safe area (iOS)

The HTML declares `viewport-fit=cover`.  `.page` applies
`env(safe-area-inset-*)` to all four sides via `padding`, with the
bottom also carrying the base `--space-xl` so content never hides
behind Safari's bottom tab bar or the iPhone home indicator.  If a new
full-bleed component is added (one that overrides `.page`'s padding),
re-apply the inset on that component.

## How to capture a real-device viewport

`how-it-works.js` beacons a one-shot JSON to `/api/diag/listener` on
page load containing `viewport_w`, `viewport_h`, `dpr`,
`orientation`, and a short user-agent string.  Retrieve via:

```bash
curl -sk https://localhost:8080/api/diag/listeners | jq '.[] | select(.page == "how-it-works")'
```

Use the returned viewport + DPR to confirm which breakpoint the real
device falls into.  If the device sits between two breakpoints, update
this doc and add a targeted media-query block in
`static/css/how-it-works.css`.
