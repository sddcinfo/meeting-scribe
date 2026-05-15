"""Quick health check.

Hits the admin listener on ``https://127.0.0.1:8080/api/status``. The
admin listener binds both loopback and the management IP, so loopback
always works locally. TLS verification is disabled because the admin
cert is self-signed.
"""

import json
import ssl
import urllib.request

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE

try:
    resp = urllib.request.urlopen(
        "https://127.0.0.1:8080/api/status",
        timeout=5,
        context=_ctx,
    )
    data = json.loads(resp.read())
    print(
        f"Server UP — ASR:{data['backends']['asr']} Translate:{data['backends']['translate']} Meeting:{data['meeting']['state']}"
    )
except Exception as e:
    print(f"Server DOWN: {e}")
