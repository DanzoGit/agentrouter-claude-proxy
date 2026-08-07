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

    # --- content-blocked: classification, preservation, recovery -------------
    # The production symptom is Claude Code showing "API Error: 400
    # content-blocked" mid-session. That is a real AgentRouter moderation
    # decision, so the entire job here is to NOT interfere with it: keep the
    # 400, keep every body byte, never retry, never convert -- while recording
    # enough content-free structure to say where the conversation was when it
    # happened. These checks pin both halves.
    print()
    for scen, label, want_ct_count in [
        ("http_400_content_blocked", "400 content-blocked (JSON envelope)", 1),
        ("http_400_content_blocked_bare", "400 content-blocked (bare token body)", 1),
        ("http_400_other", "400 unrelated schema error is NOT content-blocked", 0),
        ("http_400_effort_thinking", "400 effort/thinking is NOT content-blocked", 0),
        ("http_429_saturated", "429 saturated is NOT content-blocked", 0),
        ("http_401", "401 auth is NOT content-blocked", 0),
        ("http_403_model", "403 model access is NOT content-blocked", 0),
    ]:
        req = {"model": f"scenario:{scen}", "max_tokens": 16,
               "messages": [{"role": "user", "content": "hi"}]}
        before = stats()
        att_before = httpx.get(f"{MOCK}/_mock_stats", timeout=10).json().get(scen, 0)
        r = httpx.post(f"{PROXY}/v1/messages", headers=H, json=req, timeout=60)
        after = stats()
        attempts = (httpx.get(f"{MOCK}/_mock_stats", timeout=10).json().get(scen, 0)
                    - att_before)
        direct = httpx.post(f"{MOCK}/v1/messages", json=req, timeout=30)
        d = {k: after.get(k, 0) - before.get(k, 0) for k in (
            "upstream_400_content_blocked", "content_blocked_events",
            "retry_after_added", "retries", "rate_limit_converted")}

        errs = []
        if r.status_code != direct.status_code:
            errs.append(f"status {r.status_code} != upstream {direct.status_code}")
        if r.content != direct.content:
            errs.append("body was not forwarded byte-for-byte")
        if d["upstream_400_content_blocked"] != want_ct_count:
            errs.append(f"upstream_400_content_blocked +{d['upstream_400_content_blocked']}, "
                        f"expected +{want_ct_count}")
        if d["content_blocked_events"] != want_ct_count:
            errs.append(f"content_blocked_events +{d['content_blocked_events']}, "
                        f"expected +{want_ct_count}")
        if attempts != 1:
            errs.append(f"upstream attempts {attempts} != 1 -- the request was retried")
        if d["retries"] != 0:
            errs.append(f"retries +{d['retries']} -- content-blocked must never be retried")
        if d["rate_limit_converted"] != 0:
            errs.append("status was converted -- a 400 must stay a 400")
        # Only a 429 ever gets a Retry-After; a moderation 400 must not look
        # retryable, or Claude Code's retry loop would replay a hard rejection.
        if want_ct_count and r.headers.get("retry-after") is not None:
            errs.append(f"Retry-After {r.headers.get('retry-after')!r} injected on a 400")
        if want_ct_count and d["retry_after_added"] != 0:
            errs.append(f"retry_after_added +{d['retry_after_added']} on a 400")

        if errs:
            print(f"[FAIL] {label}")
            for e in errs:
                print(f"       - {e}")
            failed += 1
        else:
            print(f"[PASS] {label}")
            print(f"       status                 : {r.status_code} (upstream "
                  f"{direct.status_code}, preserved)")
            print(f"       body byte-identical    : True ({len(r.content)} bytes)")
            print(f"       classified as blocked  : +{d['upstream_400_content_blocked']}")
            print(f"       attempts / retries     : {attempts} / +{d['retries']} "
                  f"(never retried)")
            passed += 1

    # Claude Code's watchdog is configured for up to 30 retries. That loop must
    # never be handed a moderation rejection: the proxy answers a content-blocked
    # 400 on the first attempt, with no Retry-After to schedule against.
    print()
    att_before = httpx.get(f"{MOCK}/_mock_stats", timeout=10).json().get(
        "http_400_content_blocked_bare", 0)
    rb = httpx.post(f"{PROXY}/v1/messages", headers=H,
                    json={"model": "scenario:http_400_content_blocked_bare",
                          "max_tokens": 16,
                          "messages": [{"role": "user", "content": "hi"}]}, timeout=60)
    att_after = httpx.get(f"{MOCK}/_mock_stats", timeout=10).json().get(
        "http_400_content_blocked_bare", 0)
    errs = []
    if att_after - att_before != 1:
        errs.append(f"upstream saw {att_after - att_before} attempts, expected exactly 1")
    if rb.headers.get("retry-after") is not None:
        errs.append("Retry-After present -- the SDK would schedule a retry")
    if rb.status_code != 400:
        errs.append(f"status {rb.status_code} != 400")
    if errs:
        print("[FAIL] content-blocked never enters the retry loop")
        for e in errs:
            print(f"       - {e}")
        failed += 1
    else:
        print("[PASS] content-blocked never enters the retry loop "
              "(CLAUDE_CODE_MAX_RETRIES=30 is not engaged)")
        print(f"       upstream attempts      : {att_after - att_before} of a possible 30")
        print(f"       Retry-After            : {rb.headers.get('retry-after')!r} "
              f"(nothing to schedule against)")
        passed += 1

    # --- recovery checkpoints: growth, lanes, bounds, and blocked matching ----
    # A checkpoint is written only when upstream ACCEPTS a turn, so it records
    # the last state known to be acceptable. These checks drive real traffic
    # through the proxy and read the result back from /_recovery.
    print()

    def recovery():
        return httpx.get(f"{PROXY}/_recovery", timeout=10).json()

    def turn(n_messages, scen="nonstream_valid", session=None, tag="hi"):
        """One /v1/messages call with a conversation of n_messages messages."""
        msgs = []
        for i in range(n_messages):
            msgs.append({"role": "user" if i % 2 == 0 else "assistant",
                         "content": f"{tag} {i}"})
        hdrs = dict(H)
        if session:
            hdrs["x-session-id"] = session
        return httpx.post(f"{PROXY}/v1/messages", headers=hdrs,
                          json={"model": f"scenario:{scen}", "max_tokens": 16,
                                "messages": msgs}, timeout=60)

    before = stats()
    turn(3, session="sess-alpha")
    turn(5, session="sess-alpha")          # alpha grows
    turn(7, session="sess-beta")           # a concurrent conversation
    after = stats()
    rec = recovery()
    saved = after.get("recovery_checkpoints_saved", 0) - before.get(
        "recovery_checkpoints_saved", 0)

    errs = []
    if saved != 3:
        errs.append(f"recovery_checkpoints_saved +{saved}, expected +3")
    if rec.get("last_good", {}).get("message_count") != 7:
        errs.append(f"last_good message_count "
                    f"{rec.get('last_good', {}).get('message_count')} != 7")
    if rec.get("last_good", {}).get("session_id") != "sess-beta":
        errs.append(f"last_good session {rec.get('last_good', {}).get('session_id')!r} "
                    f"!= 'sess-beta' -- lanes overwrote each other")
    if rec.get("status") not in ("ready", "content_blocked"):
        errs.append(f"status {rec.get('status')!r} unexpected after successful turns")
    if errs:
        print("[FAIL] successful turns create per-lane last-good checkpoints")
        for e in errs:
            print(f"       - {e}")
        failed += 1
    else:
        print("[PASS] successful turns create per-lane last-good checkpoints")
        print(f"       checkpoints saved      : +{saved} (3 turns, 2 conversations)")
        print(f"       last good              : {rec['last_good']['message_count']} messages, "
              f"lane={rec['last_good']['lane_kind']}")
        passed += 1

    # A blocked request must be matched against ITS OWN lane's last-good turn,
    # not merely the most recent one globally.
    turn(9, session="sess-alpha")                     # alpha's last good = 9
    turn(11, session="sess-beta")                     # beta is more recent
    blocked = turn(13, scen="http_400_content_blocked", session="sess-alpha")
    rec = recovery()
    lb = rec.get("last_blocked") or {}
    lg = rec.get("last_good") or {}

    errs = []
    if blocked.status_code != 400:
        errs.append(f"blocked status {blocked.status_code} != 400")
    if rec.get("status") != "content_blocked":
        errs.append(f"/_recovery status {rec.get('status')!r} != 'content_blocked'")
    if lb.get("blocked_message_count") != 13:
        errs.append(f"blocked_message_count {lb.get('blocked_message_count')} != 13")
    if lb.get("last_good_message_count") != 9:
        errs.append(f"last_good_message_count {lb.get('last_good_message_count')} != 9 "
                    f"-- matched the wrong lane")
    if lg.get("message_count") != 9:
        errs.append(f"last_good served {lg.get('message_count')} != 9 -- /_recovery "
                    f"did not report the blocked lane's checkpoint")
    if lb.get("session_id") != "sess-alpha":
        errs.append(f"blocked session {lb.get('session_id')!r} != 'sess-alpha'")
    if not lb.get("structural_change"):
        errs.append("structural_change=False despite 9 -> 13 messages")
    if lb.get("proxy_retry") is not False:
        errs.append(f"proxy_retry {lb.get('proxy_retry')!r} -- must be False")
    if lb.get("request_modified") is not False:
        errs.append(f"request_modified {lb.get('request_modified')!r} -- must be False")
    if lb.get("upstream_request_id") != "req_mock_blocked_001":
        errs.append(f"upstream_request_id {lb.get('upstream_request_id')!r} not captured")
    if errs:
        print("[FAIL] a blocked request finds its own lane's last-good checkpoint")
        for e in errs:
            print(f"       - {e}")
        failed += 1
    else:
        print("[PASS] a blocked request finds its own lane's last-good checkpoint")
        print(f"       last good / blocked    : {lb['last_good_message_count']} -> "
              f"{lb['blocked_message_count']} messages (own lane, not the newest)")
        print(f"       structural change      : {lb['structural_change']}")
        print(f"       proxy retried/modified : {lb['proxy_retry']} / {lb['request_modified']}")
        print(f"       upstream request id    : {lb['upstream_request_id']}")
        passed += 1

    # Requests without a session header still get separate lanes when their
    # structure differs, and the store stays bounded under sustained load.
    from proxy import MAX_RECOVERY_CHECKPOINTS

    for i in range(MAX_RECOVERY_CHECKPOINTS + 12):
        # Distinct tool_definition_count -> distinct structural lane.
        httpx.post(f"{PROXY}/v1/messages", headers=H,
                   json={"model": "scenario:nonstream_valid", "max_tokens": 16,
                         "messages": [{"role": "user", "content": "hi"}],
                         "tools": [{"name": f"t{j}", "description": "d",
                                    "input_schema": {"type": "object"}}
                                   for j in range(i + 1)]}, timeout=60)
    rec = recovery()
    held = rec.get("checkpoints_held", -1)
    if held > MAX_RECOVERY_CHECKPOINTS or held <= 0:
        print(f"[FAIL] checkpoint store is bounded")
        print(f"       - holding {held}, limit {MAX_RECOVERY_CHECKPOINTS}")
        failed += 1
    else:
        print("[PASS] checkpoint store is bounded (oldest lanes evicted first)")
        print(f"       lanes created          : {MAX_RECOVERY_CHECKPOINTS + 12} distinct")
        print(f"       checkpoints held       : {held} / {MAX_RECOVERY_CHECKPOINTS}")
        passed += 1

    # --- recovery state must never carry content -----------------------------
    # Same canary technique as the request-shape probe: send a request stuffed
    # with sensitive values, then serialize the ENTIRE recovery state and assert
    # not one of them survives anywhere in it.
    print()
    CB_SECRETS = ["CONTENT_BLOCKED_CANARY_SYSTEM", "payroll spreadsheet question",
                  "sk-ant-not-a-real-key-2", "delete_everything",
                  "C:/private/contracts.docx", "secret chain of thought",
                  "tool result canary payload", "https://internal.example/secret"]
    canary_req = {
        "model": "scenario:http_400_content_blocked", "max_tokens": 16,
        "system": "CONTENT_BLOCKED_CANARY_SYSTEM",
        "messages": [{"role": "user", "content": [
            {"type": "text",
             "text": "payroll spreadsheet question C:/private/contracts.docx "
                     "https://internal.example/secret"},
            {"type": "tool_result", "tool_use_id": "toolu_canary",
             "content": "tool result canary payload"}]}],
        "tools": [{"name": "delete_everything", "description": "sk-ant-not-a-real-key-2",
                   "input_schema": {"type": "object"}}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "output_config": {"effort": "high"},
        "metadata": {"api_key": "sk-ant-not-a-real-key-2",
                     "note": "secret chain of thought"},
    }
    httpx.post(f"{PROXY}/v1/messages", headers=H, json=canary_req, timeout=60)
    rec_raw = httpx.get(f"{PROXY}/_recovery", timeout=10).text
    stats_raw = httpx.get(f"{PROXY}/_stats", timeout=10).text
    rec_json = json.loads(rec_raw)

    errs = []
    for blob, where in ((rec_raw, "/_recovery"), (stats_raw, "/_stats")):
        leaked = [s for s in CB_SECRETS if s in blob]
        if leaked:
            errs.append(f"{where} leaked: {leaked}")
    # Field names that would imply content storage.
    for banned in ('"content"', '"system"', '"prompt"', '"text":', '"tools"',
                   '"tool_use_id"', '"authorization"', '"api_key"', 'toolu_canary'):
        if banned in rec_raw:
            errs.append(f"/_recovery exposes {banned}")
    # It must still be USEFUL: structure has to be there.
    lb = rec_json.get("last_blocked") or {}
    for required in ("blocked_message_count", "structural_change", "proxy_retry"):
        if required not in lb:
            errs.append(f"/_recovery lost required field {required}")
    if not rec_json.get("suggested_action"):
        errs.append("/_recovery has no suggested_action")
    # Byte counts are aggregates, never the bytes themselves.
    if lb.get("text_bytes") in (None, 0):
        errs.append("text_bytes missing -- aggregate sizes should still be recorded")

    if errs:
        print("[FAIL] recovery state is structure-only")
        for e in errs:
            print(f"       - {e}")
        failed += 1
    else:
        print("[PASS] recovery state is structure-only")
        print(f"       canaries leaked        : none of {len(CB_SECRETS)} "
              f"(checked /_recovery and /_stats)")
        print(f"       content field names    : absent")
        print(f"       still diagnostic       : {lb['blocked_message_count']} messages, "
              f"{lb['text_bytes']} text bytes (aggregate only)")
        passed += 1

    # Model output that merely TALKS about content-blocked must never be
    # classified. The detector only ever sees an upstream error status, so an
    # HTTP 200 carrying those words cannot reach it -- this pins that.
    before = stats()
    r = httpx.post(f"{PROXY}/v1/messages", headers=H,
                   json={"model": "scenario:nonstream_says_content_blocked",
                         "max_tokens": 32,
                         "messages": [{"role": "user", "content": "hi"}]}, timeout=60)
    after = stats()
    d_cb = after.get("upstream_400_content_blocked", 0) - before.get(
        "upstream_400_content_blocked", 0)
    d_ev = after.get("content_blocked_events", 0) - before.get("content_blocked_events", 0)
    errs = []
    if r.status_code != 200:
        errs.append(f"status {r.status_code} != 200")
    if "content-blocked" not in r.text:
        errs.append("the model's own wording was not delivered to the client")
    if d_cb or d_ev:
        errs.append(f"assistant text classified as moderation: "
                    f"+{d_cb} blocked, +{d_ev} events")
    if errs:
        print("[FAIL] model text containing 'content-blocked' is not a moderation event")
        for e in errs:
            print(f"       - {e}")
        failed += 1
    else:
        print("[PASS] model text containing 'content-blocked' is not a moderation event")
        print(f"       status                 : {r.status_code}, text delivered verbatim")
        print(f"       classifier fired       : +{d_cb} (only an error status can trigger it)")
        passed += 1

    # The helper may only name Claude Code behavior that actually exists. This
    # asserts the suggested action sticks to documented flags, and never claims
    # to reopen a specific turn -- no CLI flag does that.
    from proxy import suggested_action

    with_session = suggested_action({"session_id": "abc-123", "message_count": 9,
                                     "timestamp": "2026-08-08T00:00:00Z"})
    without = suggested_action({"session_id": None, "message_count": 9,
                                "timestamp": "2026-08-08T00:00:00Z"})
    none_at_all = suggested_action(None)
    helper = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "recover-content-blocked.ps1"),
        encoding="utf-8").read()

    errs = []
    if "--resume abc-123" not in with_session:
        errs.append("a known session id is not offered to --resume")
    if "--fork-session" not in with_session:
        errs.append("--fork-session is not offered")
    if "--continue" not in without:
        errs.append("--continue is not offered when no session id is known")
    if "--resume" not in none_at_all:
        errs.append("the session picker is not offered when nothing is known")
    # Invented flags. --rewind does not exist; /rewind is an in-session command.
    for invented in ("--rewind", "--restore", "--undo", "--replay", "--rollback",
                     "--resume-at", "--from-message", "--truncate"):
        for blob, where in ((with_session, "suggested_action"), (without, "suggested_action"),
                            (helper, "recover-content-blocked.ps1")):
            if invented in blob:
                errs.append(f"{where} names unsupported flag {invented}")
    # The helper must not touch conversation storage.
    for forbidden in ("Remove-Item", "Set-Content", "Out-File", ".jsonl", "history.db",
                      "__store.db"):
        if forbidden in helper:
            errs.append(f"helper writes or reads conversation storage: {forbidden}")
    if errs:
        print("[FAIL] recovery guidance uses only supported Claude Code behavior")
        for e in errs:
            print(f"       - {e}")
        failed += 1
    else:
        print("[PASS] recovery guidance uses only supported Claude Code behavior")
        print(f"       with a session id      : --resume <id>, --fork-session")
        print(f"       without one            : --continue / --resume picker")
        print(f"       invented flags         : none; helper never writes session files")
        passed += 1

    print(f"\n{'=' * 64}\nscenario results: {passed} passed, {failed} failed")
    print("proxy counters:", json.dumps(
        {k: v for k, v in stats().items() if isinstance(v, int)}, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
