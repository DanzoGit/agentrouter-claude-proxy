"""
Verify the proxy handles a client disconnect cleanly: it must close the upstream
stream, increment client_disconnects, and keep serving subsequent requests.
"""

import json
import os

import httpx

PROXY = os.environ.get("TEST_PROXY_URL", "http://127.0.0.1:8789")
# Deliberately fake. These tests never touch a real credential or a real upstream.
H = {"authorization": "Bearer test-token-not-a-real-key",
     "anthropic-version": "2023-06-01", "content-type": "application/json"}


def stats():
    return httpx.get(f"{PROXY}/_stats", timeout=10).json()


before = stats()

body = {"model": "scenario:slow_valid", "max_tokens": 64, "stream": True,
        "messages": [{"role": "user", "content": "hi"}]}

frames = 0
try:
    with httpx.stream("POST", f"{PROXY}/v1/messages", headers=H, json=body,
                      timeout=60) as r:
        print(f"  upstream HTTP status     : {r.status_code}")
        for chunk in r.iter_text():
            frames += chunk.count("\n\n")
            if frames >= 4:
                print(f"  abandoning stream after ~{frames} frames (client disconnect)")
                break  # exiting the context manager aborts the connection
except Exception as exc:
    print(f"  stream aborted with {type(exc).__name__}")

after = stats()
delta = after.get("client_disconnects", 0) - before.get("client_disconnects", 0)
print(f"  client_disconnects delta : {delta}")

# Proxy must still be healthy and able to serve a new request.
probe = httpx.post(f"{PROXY}/v1/messages", headers=H, timeout=60,
                   json={"model": "scenario:nonstream_valid", "max_tokens": 8,
                         "messages": [{"role": "user", "content": "hi"}]})
alive = probe.status_code == 200
print(f"  proxy still serving      : {'yes' if alive else 'NO'} (HTTP {probe.status_code})")

ok = delta >= 1 and alive
print(f"  FINAL                    : {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
