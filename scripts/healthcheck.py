"""Quick health check."""
import urllib.request
import json

try:
    resp = urllib.request.urlopen("http://localhost:8080/api/status")
    data = json.loads(resp.read())
    print(f"Server UP — ASR:{data['backends']['asr']} Translate:{data['backends']['translate']} Meeting:{data['meeting']['state']}")
except Exception as e:
    print(f"Server DOWN: {e}")
