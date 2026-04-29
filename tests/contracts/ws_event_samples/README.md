# WS event samples

One canonical sample message per server-emitted event ``type``.
Hand-curated — NOT auto-generated from a schema, because schema validity
≠ shape the server actually emits.

Used by:

* ``tests/test_ws_event_types_coverage.py`` — asserts every type in
  ``meeting_scribe.ws.event_types.WsEventType`` has a sample file here.
* ``tests/js/ws-event-handler-coverage.test.mjs`` — replays each sample
  through both popout and admin WS message handlers, asserting exactly
  one named handler ran (and not the catch-all default).

Adding a new type? Add the slug to ``WsEventType``, write the sample
here, and add named handlers in both client cascades
(``scribe-app.js:3071`` admin and ``scribe-app.js:5137`` popout).
