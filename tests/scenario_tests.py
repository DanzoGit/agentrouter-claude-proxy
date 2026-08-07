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
    ("valid JSON as text/plain -> 200, relabelled application/json", "nonstream_text_plain", False, 200, 1, None, "OK", False, 0),
    # --- trailing SSE frame at EOF ------------------------------------------
    # A stream that ends without its final blank-line separator can leave one
    # or more COMPLETE events in the buffer. Those must be forwarded. A tail
    # that is genuinely incomplete must still be dropped.
    ("trailing events glued at EOF -> message_stop forwarded", "tail_glued", True, 200, 1,
     ["message_start", "content_block_start", "content_block_delta",
      "content_block_delta", "content_block_stop", "message_delta",
      "message_stop"], "OK", False, 0),
    ("lone trailing message_stop, no final blank line", "tail_lone", True, 200, 1,
     ["message_start", "content_block_start", "content_block_delta",
      "content_block_delta", "content_block_stop", "message_delta",
      "message_stop"], "OK", False, 0),
    ("truncated trailing frame -> dropped, NOT repaired", "tail_truncated", True, 200, 1,
     None, None, False, 1),
    # --- rate limiting and permanent 4xx ------------------------------------
    # A 429 is a real, transient upstream condition: it is forwarded as a 429 on
    # the first attempt (never replayed internally, which would multiply load on
    # a saturated upstream) and the client owns the retry loop. The permanent
    # 4xx below must keep their own status -- converting any of them into a 429
    # would turn a hard failure into an infinite retry.
    ("429 saturated -> real 429, not retried", "http_429_saturated", False, 429, 1, None, None, False, 0),
    ("429 saturated with upstream Retry-After", "http_429_saturated_hint", False, 429, 1, None, None, False, 0),
    ("ordinary 429 rate limit -> forwarded", "http_429_plain", False, 429, 1, None, None, False, 0),
    ("400 effort requires thinking -> not retried", "http_400_effort_thinking", False, 400, 1, None, None, False, 0),
    ("400 content blocked -> not converted", "http_400_content_blocked", False, 400, 1, None, None, False, 0),
    ("403 model authorization -> unchanged", "http_403_model", False, 403, 1, None, None, False, 0),
    ("model text about saturation is not a rate limit", "nonstream_talks_about_saturation",
     False, 200, 1, None, "back off", False, 0),
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

    # --- Content-Type normalization -----------------------------------------
    # AgentRouter serves some valid non-streaming messages as text/plain. The
    # Anthropic SDK only calls response.json() when the content-type names JSON,
    # so Claude Code's validator received a raw string and its first test,
    # `typeof body === "object"`, failed -- a structurally perfect message was
    # reported as "an empty or malformed response (HTTP 200)". The proxy now
    # relabels such a response as application/json. Nothing else may change:
    # this checks upstream really sent text/plain, that the body was already a
    # valid Anthropic message, that we relabel it, and that not one byte of the
    # body moved.
    print()
    body = {"model": "scenario:nonstream_text_plain", "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}]}
    # 1. straight from the mock upstream: what did AgentRouter actually send?
    up = httpx.post(f"{MOCK}/v1/messages", headers=H, json=body, timeout=60)
    up_ct = up.headers.get("content-type", "")
    # 2. through the proxy
    r = httpx.post(f"{PROXY}/v1/messages", headers=H, json=body, timeout=60)
    out_ct_list = r.headers.get_list("content-type")
    out_ct = r.headers.get("content-type", "")
    parsed = r.json()
    fields_ok = bool(isinstance(parsed.get("content"), list) and parsed["content"]
                     and isinstance(parsed.get("model"), str)
                     and isinstance(parsed.get("usage"), dict))

    errs = []
    if "text/plain" not in up_ct.lower():
        errs.append(f"mock upstream content-type {up_ct!r} is not text/plain -- "
                    f"this test no longer models the observed failure")
    if r.status_code != up.status_code or r.status_code != 200:
        errs.append(f"status {r.status_code} != upstream {up.status_code} (expected 200)")
    if not fields_ok:
        errs.append("body is not a structurally valid Anthropic message")
    if "application/json" not in out_ct.lower():
        errs.append(f"outgoing content-type {out_ct!r} is not application/json -- "
                    f"the SDK will not parse it")
    if len(out_ct_list) != 1:
        errs.append(f"{len(out_ct_list)} content-type headers on the wire: {out_ct_list}")
    if r.content != up.content:
        errs.append(f"body bytes changed: {len(up.content)} upstream -> "
                    f"{len(r.content)} downstream")
    if errs:
        print("[FAIL] text/plain -> application/json normalization")
        for e in errs:
            print(f"       - {e}")
        failed += 1
    else:
        print("[PASS] valid JSON served as text/plain is relabelled application/json, "
              "body byte-identical")
        print(f"       upstream content-type   : {up_ct}")
        print(f"       outgoing content-type   : {out_ct}  ({len(out_ct_list)} header)")
        print(f"       body structurally valid : {fields_ok} "
              f"(content:array, model:str, usage:object all present)")
        print(f"       body bytes              : {len(up.content)} upstream == "
              f"{len(r.content)} downstream, identical: {r.content == up.content}")
        print(f"       SDK would JSON-parse    : True  <-- FKu now receives an object")
        passed += 1

    # A JSON content-type must be forwarded untouched -- normalization applies
    # only when the upstream header would defeat the client's parser.
    r2 = httpx.post(f"{PROXY}/v1/messages", headers=H,
                    json={"model": "scenario:nonstream_valid", "max_tokens": 64,
                          "messages": [{"role": "user", "content": "hi"}]},
                    timeout=60)
    ct2 = r2.headers.get_list("content-type")
    if len(ct2) == 1 and "application/json" in ct2[0].lower() and r2.status_code == 200:
        print("[PASS] already-JSON content-type passes through untouched, single header")
        passed += 1
    else:
        print(f"[FAIL] JSON passthrough altered: status {r2.status_code}, "
              f"content-type {ct2}")
        failed += 1

    # --- trailing SSE frame at EOF: exact counter behavior -------------------
    # The production symptom was "dropped trailing SSE frame: type=message_stop"
    # immediately followed by "post-commit failure: upstream closed after N
    # frames without message_stop". A complete message_stop was being discarded
    # because the tail held two glued events, so the client saw a truncation
    # error at the end of an otherwise perfect response. These checks pin the
    # counters, not just the status code.
    print()
    for scen, label, want_stop, want_pcf, want_drop in [
        ("tail_glued", "glued trailing events at EOF", True, 0, 0),
        ("tail_lone", "lone trailing message_stop at EOF", True, 0, 0),
        ("tail_truncated", "truncated trailing frame (negative control)", False, 1, 1),
        ("truncated_post", "real mid-stream truncation (must stay a failure)", False, 1, 0),
    ]:
        before = stats()
        # These scenarios already ran once in CASES, so the mock's attempt
        # counter is cumulative -- compare the delta, not the absolute.
        att_before = httpx.get(f"{MOCK}/_mock_stats", timeout=10).json().get(scen, 0)
        status, events, text, raw = consume(scen, True)
        after = stats()
        d_pcf = after.get("post_commit_failures", 0) - before.get("post_commit_failures", 0)
        d_drop = after.get("dropped_sse_frames", 0) - before.get("dropped_sse_frames", 0)
        attempts = (httpx.get(f"{MOCK}/_mock_stats", timeout=10).json().get(scen, 0)
                    - att_before)
        got_stop = "message_stop" in events
        incomplete = "upstream_stream_incomplete" in raw

        errs = []
        if status != 200:
            errs.append(f"status {status} != 200")
        if got_stop != want_stop:
            errs.append(f"message_stop forwarded={got_stop}, expected {want_stop}")
        if d_pcf != want_pcf:
            errs.append(f"post_commit_failures +{d_pcf}, expected +{want_pcf}")
        if d_drop != want_drop:
            errs.append(f"dropped_sse_frames +{d_drop}, expected +{want_drop}")
        if want_stop and incomplete:
            errs.append("emitted upstream_stream_incomplete despite a valid message_stop")
        if not want_stop and not incomplete:
            errs.append("no upstream_stream_incomplete error event on a real truncation")
        # Nothing may be invented to make a stream look complete.
        if not want_stop and '"type": "message_stop"' in raw.replace('"type":"message_stop"',
                                                                    '"type": "message_stop"'):
            errs.append("a message_stop appeared that upstream never completed -- synthesized")
        if attempts != 1:
            errs.append(f"upstream attempts {attempts} != 1 -- request was replayed after commit")

        if errs:
            print(f"[FAIL] {label}")
            for e in errs:
                print(f"       - {e}")
            failed += 1
        else:
            print(f"[PASS] {label}")
            print(f"       message_stop forwarded : {got_stop}")
            print(f"       post_commit_failures   : +{d_pcf}")
            print(f"       dropped_sse_frames     : +{d_drop}")
            print(f"       upstream attempts      : {attempts} (no replay after commit)")
            passed += 1

    # --- 429 / Retry-After, and the 4xx that must never be reclassified ------
    # The production symptom was "All providers are saturated; retry shortly"
    # followed by "Retrying in 0s - attempt 4/10": upstream sent a real 429 with
    # no Retry-After, so the SDK rescheduled immediately and hammered an already
    # saturated upstream. The fix is only the missing hint. These checks pin
    # what must NOT change with it: the status, every body byte, and the fact
    # that a 4xx is answered on the first attempt and never converted.
    print()
    for (scen, label, want_status, want_ra, want_added, want_429, want_sat,
         want_effort) in [
        ("http_429_saturated", "429 saturated, no upstream hint -> Retry-After added",
         429, "15", 1, 1, 1, 0),
        ("http_429_saturated_hint", "429 saturated, upstream Retry-After preserved",
         429, "42", 0, 1, 1, 0),
        ("http_429_plain", "ordinary 429 rate limit stays retryable",
         429, "15", 1, 1, 0, 0),
        ("http_400_content_blocked", "400 content blocked -> never a 429",
         400, None, 0, 0, 0, 0),
        ("http_400_effort_thinking", "400 effort/thinking -> never a 429, never retried",
         400, None, 0, 0, 0, 1),
        ("http_401", "401 auth -> unchanged", 401, None, 0, 0, 0, 0),
        ("http_403_model", "403 model authorization -> unchanged", 403, None, 0, 0, 0, 0),
        ("nonstream_talks_about_saturation",
         "model text about saturation is not a capacity failure", 200, None, 0, 0, 0, 0),
    ]:
        req = {"model": f"scenario:{scen}", "max_tokens": 16,
               "messages": [{"role": "user", "content": "hi"}]}
        before = stats()
        att_before = httpx.get(f"{MOCK}/_mock_stats", timeout=10).json().get(scen, 0)
        r = httpx.post(f"{PROXY}/v1/messages", headers=H, json=req, timeout=60)
        after = stats()
        attempts = (httpx.get(f"{MOCK}/_mock_stats", timeout=10).json().get(scen, 0)
                    - att_before)
        # Fetch the upstream's own bytes independently: the client must receive
        # exactly what upstream produced, not a rewritten or invented body.
        direct = httpx.post(f"{MOCK}/v1/messages", json=req, timeout=30)

        got_ra = r.headers.get("retry-after")
        d = {k: after.get(k, 0) - before.get(k, 0) for k in (
            "retry_after_added", "upstream_429", "upstream_429_saturated",
            "effort_thinking_validation_errors", "rate_limit_converted",
            "upstream_400", "upstream_401", "upstream_403", "retries")}

        errs = []
        if r.status_code != want_status:
            errs.append(f"status {r.status_code} != {want_status} "
                        f"-- upstream status was not preserved")
        if r.content != direct.content:
            errs.append("body was not forwarded byte-for-byte")
        if got_ra != want_ra:
            errs.append(f"Retry-After {got_ra!r} != {want_ra!r}")
        for key, want in (("retry_after_added", want_added),
                          ("upstream_429", want_429),
                          ("upstream_429_saturated", want_sat),
                          ("effort_thinking_validation_errors", want_effort),
                          ("upstream_400", 1 if want_status == 400 else 0),
                          ("upstream_401", 1 if want_status == 401 else 0),
                          ("upstream_403", 1 if want_status == 403 else 0)):
            if d[key] != want:
                errs.append(f"{key} +{d[key]}, expected +{want}")
        # Nothing is converted, and a permanent 4xx is never replayed.
        if d["rate_limit_converted"] != 0:
            errs.append(f"rate_limit_converted +{d['rate_limit_converted']} "
                        f"-- no status is converted into a 429")
        if attempts != 1:
            errs.append(f"upstream attempts {attempts} != 1 -- a 4xx was retried")
        if d["retries"] != 0:
            errs.append(f"retries +{d['retries']} -- a 4xx must not be retried")

        if errs:
            print(f"[FAIL] {label}")
            for e in errs:
                print(f"       - {e}")
            failed += 1
        else:
            print(f"[PASS] {label}")
            print(f"       status / Retry-After   : {r.status_code} / {got_ra!r}")
            print(f"       body byte-identical    : True ({len(r.content)} bytes)")
            print(f"       upstream attempts      : {attempts} (never retried)")
            print(f"       429 / saturated / added: +{d['upstream_429']} / "
                  f"+{d['upstream_429_saturated']} / +{d['retry_after_added']}")
            passed += 1

    # --- structural diagnostics must never carry content ---------------------
    # describe_request_shape is what makes claude_effort_requires_thinking
    # diagnosable. It reads field NAMES and small enum values only; this pins
    # that promise against a request stuffed with every kind of sensitive value.
    print()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from proxy import describe_request_shape

    SECRETS = ["SUPER_SECRET_SYSTEM_PROMPT", "user asked about payroll",
               "sk-ant-not-a-real-key", "read_file", "C:/private/salaries.xlsx",
               "internal chain of thought", "tool result payload"]
    probe = json.dumps({
        "model": "claude-opus-5", "stream": True,
        "system": "SUPER_SECRET_SYSTEM_PROMPT",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "user asked about payroll C:/private/salaries.xlsx"},
            {"type": "tool_result", "content": "tool result payload"}]}],
        "tools": [{"name": "read_file", "description": "SUPER_SECRET_SYSTEM_PROMPT"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024,
                     "text": "internal chain of thought"},
        "output_config": {"effort": "high"},
        "metadata": {"api_key": "sk-ant-not-a-real-key"},
    }).encode()
    shape = describe_request_shape(probe)
    # The exact combination upstream rejects: effort present, thinking absent.
    rejected_shape = describe_request_shape(json.dumps({
        "model": "claude-opus-5", "output_config": {"effort": "high"},
        "messages": []}).encode())

    errs = []
    leaked = [s for s in SECRETS if s in shape]
    if leaked:
        errs.append(f"diagnostic leaked content: {leaked}")
    for required in ("model=claude-opus-5", "stream=True", "thinking_present=True",
                     "thinking.type=enabled", "thinking.budget_tokens_present=True",
                     "output_config.effort_present=True", "output_config.effort=high"):
        if required not in shape:
            errs.append(f"diagnostic missing {required}")
    for required in ("thinking_present=False", "output_config.effort=high"):
        if required not in rejected_shape:
            errs.append(f"rejected-shape diagnostic missing {required}")
    if describe_request_shape(b"not json") != "unparseable":
        errs.append("non-JSON body was not handled safely")

    if errs:
        print("[FAIL] request-shape diagnostics are structure-only")
        for e in errs:
            print(f"       - {e}")
        failed += 1
    else:
        print("[PASS] request-shape diagnostics are structure-only")
        print(f"       leaked content values  : none of {len(SECRETS)} probes")
        print(f"       logged for a good req  : {shape}")
        print(f"       logged for the reject  : {rejected_shape}")
        passed += 1

    print(f"\n{'=' * 64}\nscenario results: {passed} passed, {failed} failed")
    print("proxy counters:", json.dumps(
        {k: v for k, v in stats().items() if isinstance(v, int)}, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
