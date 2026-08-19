"""Deterministic end-to-end tests for reliable buffered streaming."""

import os

import httpx

PROXY = os.environ["TEST_RELIABLE_PROXY_URL"]
MOCK = os.environ["TEST_MOCK_URL"]
H = {"authorization": "Bearer test-token-not-a-real-key",
     "anthropic-version": "2023-06-01", "content-type": "application/json"}


def request(scenario: str, stream: bool = True) -> httpx.Response:
    return httpx.post(f"{PROXY}/v1/messages", headers=H, timeout=60,
                      json={"model": f"scenario:{scenario}", "max_tokens": 64,
                            "stream": stream,
                            "messages": [{"role": "user", "content": "hi"}]})


def attempts(scenario: str) -> int:
    return httpx.get(f"{MOCK}/_mock_stats", timeout=10).json().get(scenario, 0)


def main() -> int:
    httpx.post(f"{MOCK}/_mock_reset", timeout=10)
    checks = []

    health = httpx.get(f"{PROXY}/_health", timeout=10).json()
    checks.append(("config visible", health.get("stream_mode") == "reliable"
                   and health.get("reliable_max_bytes") == 4096))

    upstream = httpx.post(f"{MOCK}/v1/messages", headers=H, timeout=60,
                          json={"model": "scenario:byte_fixture", "stream": True})
    expected = upstream.content
    out = request("byte_fixture")
    checks.append(("complete SSE byte-identical", out.status_code == 200
                   and out.content == expected and attempts("byte_fixture") == 2))
    checks.append(("tool/thinking/unicode fidelity", b"toolu_01ABCDEFGHIJKLMNOPQRSTUV" in out.content
                   and b"thinking_delta" in out.content and b"\\u96ea" in out.content))

    out = request("remote_break")
    checks.append(("first break then success", out.status_code == 200
                   and b"message_stop" in out.content and attempts("remote_break") == 2
                   and out.content.count(b"message_start") == 2))

    out = request("flaky")
    checks.append(("two incomplete then success", out.status_code == 200
                   and attempts("flaky") == 3 and out.content.count(b"message_stop") == 2))

    out = request("always_break")
    checks.append(("all breaks no partial leak", out.status_code == 502
                   and attempts("always_break") == 3
                   and b"content_block_delta" not in out.content))

    out = request("truncated_post")
    checks.append(("EOF without stop retried", out.status_code == 502
                   and attempts("truncated_post") == 3
                   and b"partial answer" not in out.content))

    out = request("tail_glued")
    checks.append(("EOF multi-event tail", out.status_code == 200
                   and b"message_stop" in out.content))

    out = request("tail_truncated")
    checks.append(("malformed tail rejected", out.status_code == 502
                   and b"message_st" not in out.content))

    out = request("limit")
    checks.append(("buffer cap no commit", out.status_code == 502
                   and attempts("limit") == 3 and b"x" * 32 not in out.content))

    out = request("large_valid")
    checks.append(("large valid below cap", out.status_code == 200
                   and len(out.content) > 2000 and b"message_stop" in out.content))

    for scenario, status in (("http_429_saturated_hint", 429),
                             ("http_400_content_blocked", 400),
                             ("http_400_effort_thinking", 400),
                             ("http_401", 401), ("http_403_model", 403)):
        out = request(scenario)
        checks.append((f"{scenario} one attempt", out.status_code == status
                       and attempts(scenario) == 1))
        if scenario == "http_429_saturated_hint":
            checks.append(("Retry-After preserved", out.headers.get("retry-after") == "42"))

    stats = httpx.get(f"{PROXY}/_stats", timeout=10).json()
    checks.append(("reliable counters", stats["reliable_stream_recovered"] >= 2
                   and stats["reliable_stream_exhausted"] >= 3
                   and stats["post_commit_failures"] == 0))

    failed = 0
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        failed += not ok
    print(f"Reliable stream: {len(checks) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
