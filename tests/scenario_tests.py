"""
Validate the proxy's compatibility logic against a mock upstream that
reproduces AgentRouter's observed failure modes.

  mock upstream : 127.0.0.1:8788
  proxy         : 127.0.0.1:8789  (AGENTROUTER_UPSTREAM=http://127.0.0.1:8788)
"""

import json
import os
import sys

import httpx

PROXY = os.environ.get("TEST_PROXY_URL", "http://127.0.0.1:8789")
MOCK = os.environ.get("TEST_MOCK_URL", "http://127.0.0.1:8788")
# Deliberately fake. These tests never touch a real credential or a real upstream.
H = {"authorization": "Bearer test-token-not-a-real-key",
     "anthropic-version": "2023-06-01", "content-type": "application/json"}
TOOL_ID = "toolu_01ABCDEFGHIJKLMNOPQRSTUV"

# name, scenario, stream, exp_status, exp_attempts, exp_events(None=skip),
# text_substr, want_tool_id, min_dropped
CASES = [
    ("empty 200 body, non-stream, always fails", "nonstream_empty", False, 502, 3, None, None, False, 0),
    ("null body, non-stream", "nonstream_null", False, 502, 3, None, None, False, 0),
    ("empty 200 twice then valid -> RECOVERS", "nonstream_flaky", False, 200, 3, None, "recovered", False, 0),
    ("valid non-stream", "nonstream_valid", False, 200, 1, None, "OK", False, 0),
    ("empty SSE stream, always", "empty_sse", True, 502, 3, None, None, False, 0),
    ("keep-alive pings only", "keepalive_only", True, 502, 3, None, None, False, 0),
    ("billing/quota metadata only", "billing_only", True, 502, 3, None, None, False, 2),
    # the mock emits two text deltas ("O", "K"), so two content_block_delta
    ("empty stream twice then valid -> RECOVERS", "flaky", True, 200, 3,
     ["message_start", "content_block_start", "content_block_delta",
      "content_block_delta", "content_block_stop", "message_delta",
      "message_stop"], "OK", False, 0),
    ("data:null + [DONE] + billing junk dropped", "null_data", True, 200, 1,
     ["message_start", "content_block_start", "content_block_delta",
      "content_block_delta", "content_block_stop", "message_delta",
      "message_stop"], "OK", False, 3),
    ("truncated BEFORE content -> retried", "truncated_pre", True, 502, 3, None, None, False, 0),
    ("truncated AFTER content -> NO replay, error event", "truncated_post", True, 200, 1, None,
     "partial answer", False, 0),
    ("tool_use IDs preserved byte-exact", "valid_tool", True, 200, 1, None, "OK", True, 0),
    ("upstream SSE error event forwarded, not retried", "error_event", True, 200, 1, None, None, False, 0),
    ("HTTP 401 -> no retry, forwarded", "http_401", False, 401, 1, None, None, False, 0),
    ("HTTP 400 -> no retry, forwarded verbatim", "http_400", False, 400, 1, None, None, False, 0),
    ("HTTP 502 -> retried, then reported", "http_502", False, 502, 3, None, None, False, 0),
    ("HTTP 503 no channel -> reported as availability", "http_503_nochannel", False, 503, 3, None, None, False, 0),
    ("malformed gzip handled safely", "gzip_bad", False, 502, 3, None, None, False, 0),
    # --- HTTP 200 bodies Claude Code's own validator rejects ----------------
    # Claude Code requires content:array + model:string + usage:object on a
    # non-streaming /v1/messages reply. Anything it would reject must be
    # retried here and reported as 502 -- never forwarded as a 200.
    ("no usage field -> retried, never forwarded as 200", "nonstream_no_usage", False, 502, 3, None, None, False, 0),
    ("usage not an object -> retried", "nonstream_usage_not_object", False, 502, 3, None, None, False, 0),
    ("usage null is accepted (typeof null === object)", "nonstream_usage_null", False, 200, 1, None, "OK", False, 0),
    ("no model field -> retried", "nonstream_no_model", False, 502, 3, None, None, False, 0),
    ("model not a string -> retried", "nonstream_model_not_string", False, 502, 3, None, None, False, 0),
    ("content is a string, not an array -> retried", "nonstream_content_string", False, 502, 3, None, None, False, 0),
    ("content is an object, not an array -> retried", "nonstream_content_object", False, 502, 3, None, None, False, 0),
    ("missing content -> existing behavior preserved", "nonstream_no_content", False, 502, 3, None, None, False, 0),
    ("empty content array -> existing behavior preserved", "nonstream_empty_content", False, 502, 3, None, None, False, 0),
    ("no usage once, then valid -> RECOVERS", "nonstream_flaky_usage", False, 200, 2, None, "recovered", False, 0),
    ("HTTP 200 error object -> 502, upstream message kept", "nonstream_error_object", False, 502, 1, None, None, False, 0),
]


def stats():
    try:
        return httpx.get(f"{PROXY}/_stats", timeout=10).json()
    except Exception:
        return {}


def consume(scenario, stream):
    body = {"model": f"scenario:{scenario}", "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}]}
    events, text, raw = [], [], ""
    if stream:
        body["stream"] = True
        with httpx.stream("POST", f"{PROXY}/v1/messages", headers=H, json=body,
                          timeout=120) as r:
            status = r.status_code
            buf = ""
            for chunk in r.iter_text():
                buf += chunk
                raw += chunk
                while "\n\n" in buf:
                    frame, buf = buf.split("\n\n", 1)
                    if not frame.strip():
                        continue
                    et, payload = "", None
                    for line in frame.split("\n"):
                        if line.startswith("event:"):
                            et = line[6:].strip()
                        elif line.startswith("data:"):
                            try:
                                payload = json.loads(line[5:].strip())
                                et = payload.get("type", et) if isinstance(payload, dict) else et
                            except Exception:
                                pass
                    events.append(et)
                    if et == "content_block_delta" and isinstance(payload, dict):
                        d = payload.get("delta", {})
                        text.append(d.get("text") or d.get("partial_json") or "")
    else:
        r = httpx.post(f"{PROXY}/v1/messages", headers=H, json=body, timeout=120)
        status, raw = r.status_code, r.text
        try:
            data = r.json()
            for b in (data.get("content") or []):
                if b.get("type") == "text":
                    text.append(b.get("text", ""))
        except Exception:
            pass
    return status, [e for e in events if e != "ping"], "".join(text), raw


def main():
    httpx.post(f"{MOCK}/_mock_reset", timeout=10)
    passed = failed = 0

    for (name, scen, stream, exp_status, exp_att, exp_events,
         substr, want_tool, min_drop) in CASES:
        before = stats()
        try:
            status, events, text, raw = consume(scen, stream)
        except Exception as exc:
            print(f"[FAIL] {name}\n       exception {type(exc).__name__}: {str(exc)[:160]}")
            failed += 1
            continue
        after = stats()
        attempts = httpx.get(f"{MOCK}/_mock_stats", timeout=10).json().get(scen, 0)
        dropped = after.get("dropped_sse_frames", 0) - before.get("dropped_sse_frames", 0)
        retries = after.get("retries", 0) - before.get("retries", 0)

        errs = []
        if status != exp_status:
            errs.append(f"status {status} != {exp_status}")
        if attempts != exp_att:
            errs.append(f"upstream attempts {attempts} != {exp_att}")
        if exp_events is not None and events != exp_events:
            errs.append(f"events {events} != {exp_events}")
        if substr and substr not in text and substr not in raw:
            errs.append(f"missing text {substr!r}")
        if want_tool and TOOL_ID not in raw:
            errs.append("tool_use ID missing or altered")
        if dropped < min_drop:
            errs.append(f"dropped {dropped} < {min_drop}")
        if scen == "truncated_post" and "upstream_stream_incomplete" not in raw:
            errs.append("no error event after post-commit truncation")
        if scen == "error_event" and "overloaded_error" not in raw:
            errs.append("upstream error event not forwarded")
        # The invalid body itself must never reach the client, and the proxy
        # must say which field was missing.
        if scen == "nonstream_no_usage":
            if "missing_usage_field" not in raw:
                errs.append("no missing_usage_field diagnostic in proxy error body")
            if "billing" in raw:
                errs.append("invalid upstream body leaked to the client")
        # A genuine upstream error object must keep its own message, and must
        # not be retried -- but must not arrive under HTTP 200 either.
        if scen == "nonstream_error_object" and "overloaded_error" not in raw:
            errs.append("upstream error message not preserved")
        if scen == "nonstream_flaky_usage":
            recovered = (after.get("retries_successful", 0)
                         - before.get("retries_successful", 0))
            if recovered < 1:
                errs.append(f"retries_successful incremented by {recovered}, expected >= 1")

        if errs:
            print(f"[FAIL] {name}")
            for e in errs:
                print(f"       - {e}")
            failed += 1
        else:
            bits = [f"HTTP {status}", f"attempts={attempts}", f"retries={retries}"]
            if dropped:
                bits.append(f"dropped={dropped}")
            if events:
                bits.append(f"{len(events)} events")
            print(f"[PASS] {name}\n       {', '.join(bits)}")
            passed += 1

    # request-body null stripping
    print()
    body = {"model": "scenario:echo_keys", "max_tokens": 8, "metadata": None,
            "temperature": None, "top_k": None, "system": "keep me",
            "messages": [{"role": "user", "content": "hi"}]}
    r = httpx.post(f"{PROXY}/v1/messages", headers=H, json=body, timeout=60)
    keys = json.loads(r.json()["content"][0]["text"])
    stripped_ok = not ({"metadata", "temperature", "top_k"} & set(keys))
    kept_ok = {"system", "messages", "model", "max_tokens"} <= set(keys)
    if stripped_ok and kept_ok:
        print(f"[PASS] null request fields stripped, non-null preserved\n       upstream saw: {keys}")
        passed += 1
    else:
        print(f"[FAIL] null-field stripping\n       upstream saw: {keys}")
        failed += 1

    print(f"\n{'=' * 64}\nscenario results: {passed} passed, {failed} failed")
    print("proxy counters:", json.dumps(
        {k: v for k, v in stats().items() if isinstance(v, int)}, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
