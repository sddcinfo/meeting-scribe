"""Hotspot subsystem — captive portal probes + WiFi AP control.

The hotspot is the SSID-rotating AP that scribe brings up so guests
can join with their phones, scan a QR, and load the guest viewer
without typing a URL. Two layers in this package:

* ``captive_portal`` — captive-portal probe routes (Apple iOS/macOS,
  Android/Chrome, Windows NCSI, Firefox, RFC 8910 ``/api/captive``)
  plus the public ``/api/meeting/wifi`` GET that returns SSID +
  password + QR codes for the join screen. Pure read-only routes
  backed by the in-process captive-ack set
  (``server_support.captive_ack``) and the ``hotspot-state.json``
  file written by the AP-control path.

* ``ap_control`` — bring-up / tear-down / rotation of the AP itself
  (still in ``server.py`` as of this commit). Will move once the
  WiFi-state-machine seam is clean.
"""
