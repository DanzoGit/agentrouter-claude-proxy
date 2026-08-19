"""
Mock upstream that reproduces AgentRouter's observed failure modes so the
proxy's normalization + retry logic can be validated end-to-end without
depending on the real service.

Scenario is selected via the request body's "model" field: "scenario:<name>".
Attempts per scenario are counted so tests can assert exact retry counts.
"""

import asyncio
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI()
ATTEMPTS: dict[str, int] = {}

TOOL_ID = "toolu_01ABCDEFGHIJKLMNOPQRSTUV"
MSG_ID = "msg_01XYZmockmessage000000"

BYTE_FIXTURE = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_bytes","type":"message","role":"assistant","model":"mock","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"block":{"type":"thinking","thinking":""}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"considering"}}\n\n'
    b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
    b'event: content_block_start\ndata: {"type":"content_block_start","index":1,"block":{"type":"text","text":""}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Unicode: \\u96ea"}}\n\n'
    b'event: content_block_stop\ndata: {"type":"content_block_stop","index":1}\n\n'
    b'event: content_block_start\ndata: {"type":"content_block_start","index":2,"block":{"type":"tool_use","id":"toolu_01ABCDEFGHIJKLMNOPQRSTUV","name":"fixture_tool","input":{}}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":\\".\\"}"}}\n\n'
    b'event: content_block_stop\ndata: {"type":"content_block_stop","index":2}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":9}}\n\n'
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# Sentinel: drop a key from the base message entirely rather than overwrite it.
DROP = object()


def nonstream_message(**overrides):
    """
    A valid non-streaming Anthropic message, optionally broken in exactly one
    way. Claude Code validates such a reply before using it and requires
    content:array + model:string + usage:object, so each override below models
    one shape it would reject.
    """
    base = {
        "type": "message", "id": MSG_ID, "role": "assistant",
        "model": "mock", "stop_reason": "end_turn", "stop_sequence": None,
        "content": [{"type": "text", "text": "OK"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    for key, value in overrides.items():
        if value is DROP:
            base.pop(key, None)
        else:
            base[key] = value
    return base


def valid_frames(with_tool: bool = False):
    yield sse("message_start", {
        "type": "message_start",
        "message": {"id": MSG_ID, "type": "message", "role": "assistant",
                    "model": "claude-opus-4-8", "content": [],
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0}},
    })
    yield sse("content_block_start", {
        "type": "content_block_start", "index": 0,
        "block": {"type": "text", "text": ""}})
    for piece in ("O", "K"):
        yield sse("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}})
    yield sse("content_block_stop", {"type": "content_block_stop", "index": 0})

    if with_tool:
        yield sse("content_block_start", {
            "type": "content_block_start", "index": 1,
            "block": {"type": "tool_use", "id": TOOL_ID, "name": "list_files",
                      "input": {}}})
        yield sse("content_block_delta", {
            "type": "content_block_delta", "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"path":"."}'}})
        yield sse("content_block_stop", {"type": "content_block_stop", "index": 1})

    yield sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use" if with_tool else "end_turn",
                  "stop_sequence": None},
        "usage": {"output_tokens": 5}})
    yield sse("message_stop", {"type": "message_stop"})


@app.get("/_mock_stats")
async def mock_stats():
    return JSONResponse(ATTEMPTS)


@app.post("/_mock_reset")
async def mock_reset():
    ATTEMPTS.clear()
    return JSONResponse({"reset": True})


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def handler(request: Request, path: str):
    raw = await request.body()
    try:
        body = json.loads(raw) if raw else {}
    except Exception:
        body = {}

    model = str(body.get("model", ""))
    scenario = model.split("scenario:", 1)[1] if "scenario:" in model else "valid"
    ATTEMPTS[scenario] = ATTEMPTS.get(scenario, 0) + 1
    n = ATTEMPTS[scenario]

    # ---- non-stream scenarios ---------------------------------------------
    if scenario == "echo_keys":
        return JSONResponse({
            "id": MSG_ID, "type": "message", "role": "assistant",
            "model": "mock", "stop_reason": "end_turn",
            "content": [{"type": "text",
                         "text": json.dumps(sorted(body.keys()))}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    if scenario == "nonstream_valid":
        return JSONResponse({
            "id": MSG_ID, "type": "message", "role": "assistant", "model": "mock",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "OK"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    if scenario == "nonstream_empty":
        return Response(content=b"", status_code=200, media_type="application/json")

    if scenario == "nonstream_null":
        return Response(content=b"null", status_code=200, media_type="application/json")

    if scenario == "nonstream_flaky":
        if n < 3:
            return Response(content=b"", status_code=200, media_type="application/json")
        return JSONResponse({
            "id": MSG_ID, "type": "message", "role": "assistant", "model": "mock",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "recovered"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    # ---- HTTP 200 bodies Claude Code's own validator rejects ---------------
    # AgentRouter has been observed returning a "billing" block and no "usage"
    # at all. Claude Code then rejects the reply as
    # "API returned an empty or malformed response (HTTP 200)".
    if scenario == "nonstream_no_usage":
        return JSONResponse({
            "type": "message",
            "id": "msg_test",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "OK"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "billing": {"example": True},
        })

    if scenario == "nonstream_flaky_usage":
        # First attempt omits usage; every later attempt is valid.
        if n < 2:
            return JSONResponse({
                "type": "message", "id": "msg_test", "role": "assistant",
                "model": "claude-opus-5",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "billing": {"example": True},
            })
        return JSONResponse(nonstream_message(
            content=[{"type": "text", "text": "recovered"}]))

    if scenario == "nonstream_usage_not_object":
        return JSONResponse(nonstream_message(usage="1 input token"))

    if scenario == "nonstream_usage_null":
        # typeof null === "object" in JavaScript, so Claude Code accepts this.
        return JSONResponse(nonstream_message(usage=None))

    if scenario == "nonstream_no_model":
        return JSONResponse(nonstream_message(model=DROP))

    if scenario == "nonstream_model_not_string":
        return JSONResponse(nonstream_message(model=12345))

    if scenario == "nonstream_content_string":
        return JSONResponse(nonstream_message(content="OK"))

    if scenario == "nonstream_content_object":
        return JSONResponse(nonstream_message(
            content={"type": "text", "text": "OK"}))

    if scenario == "nonstream_no_content":
        return JSONResponse(nonstream_message(content=DROP))

    if scenario == "nonstream_empty_content":
        return JSONResponse(nonstream_message(content=[]))

    if scenario == "nonstream_error_object":
        # A real upstream error delivered under HTTP 200.
        return JSONResponse({
            "type": "error",
            "error": {"type": "overloaded_error", "message": "upstream overloaded"},
        })

    if scenario == "nonstream_text_plain":
        # HTTP 200, a structurally perfect Anthropic message, but served as
        # text/plain -- the shape AgentRouter was observed using. This
        # deliberately models the upstream response that the proxy must relabel
        # as application/json without changing the body bytes.
        return Response(content=json.dumps(nonstream_message()).encode(),
                        status_code=200,
                        media_type="text/plain; charset=utf-8")

    if scenario == "gzip_bad":
        return Response(content=b"this is definitely not gzip", status_code=200,
                        headers={"content-encoding": "gzip"},
                        media_type="application/json")

    if scenario == "http_503_nochannel":
        return JSONResponse(
            {"error": {"message": "no available channel for model claude-opus-4-8"}},
            status_code=503)

    if scenario == "http_502":
        return JSONResponse({"error": {"message": "bad gateway"}}, status_code=502)

    if scenario == "http_401":
        return JSONResponse({"error": {"message": "unauthorized"}}, status_code=401)

    if scenario == "http_400":
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error",
                                        "message": "max_tokens is required"}},
            status_code=400)

    # ---- rate limiting / capacity ------------------------------------------
    # The observed saturation shape: a real 429, and no Retry-After at all. The
    # missing header is what makes Claude Code reschedule immediately.
    if scenario == "http_429_saturated":
        return JSONResponse(
            {"type": "error", "error": {"type": "rate_limit_error",
                                        "message": "All providers are saturated; "
                                                   "retry shortly"}},
            status_code=429)

    if scenario == "http_429_saturated_hint":
        # Same failure, but upstream supplies its own delay. It must survive.
        return JSONResponse(
            {"type": "error", "error": {"type": "rate_limit_error",
                                        "message": "All providers are saturated; "
                                                   "retry shortly"}},
            status_code=429, headers={"retry-after": "42"})

    if scenario == "http_429_plain":
        # An ordinary per-key rate limit: still a 429, but not saturation.
        return JSONResponse(
            {"type": "error", "error": {"type": "rate_limit_error",
                                        "message": "rate limit exceeded for this key"}},
            status_code=429)

    # ---- permanent 4xx that must never be converted or retried --------------
    if scenario == "http_400_effort_thinking":
        return JSONResponse(
            {"type": "error",
             "error": {"type": "invalid_request_error",
                       "message": "output_config.effort: requires thinking.type "
                                  "to be enabled",
                       "rule_id": "claude_effort_requires_thinking"}},
            status_code=400)

    if scenario == "http_400_content_blocked":
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error",
                                        "message": "content blocked by policy"}},
            status_code=400,
            headers={"request-id": "req_mock_blocked_001"})

    # The shape Claude Code actually renders as "API Error: 400 content-blocked":
    # a bare token, not a JSON error envelope. Detection must survive it, and the
    # body must still reach the client byte-for-byte.
    if scenario == "http_400_content_blocked_bare":
        return Response(content=b"content-blocked", status_code=400,
                        media_type="text/plain",
                        headers={"request-id": "req_mock_blocked_002"})

    # A 400 that merely mentions moderation in passing is a different failure and
    # must not be classified as content-blocked.
    if scenario == "http_400_other":
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error",
                                        "message": "max_tokens must be positive"}},
            status_code=400)

    if scenario == "http_403_model":
        return JSONResponse(
            {"type": "error", "error": {"type": "permission_error",
                                        "message": "not authorized to access model"}},
            status_code=403)

    # A model reply that merely *discusses* saturation. Structural matching must
    # not mistake assistant content for a capacity failure.
    if scenario == "nonstream_talks_about_saturation":
        return JSONResponse(nonstream_message(content=[
            {"type": "text",
             "text": "When all providers are saturated you should back off."}]))

    # The same trap for moderation: an ordinary HTTP 200 answer that quotes the
    # exact wording of a rejection. Classifying this would let the model's own
    # output fabricate a moderation event.
    if scenario == "nonstream_says_content_blocked":
        return JSONResponse(nonstream_message(content=[
            {"type": "text",
             "text": "If you see API Error: 400 content-blocked, the upstream "
                     "content_filter rejected the request."}]))

    # ---- streaming scenarios ----------------------------------------------
    async def gen_empty_sse():
        return
        yield  # pragma: no cover

    async def gen_keepalive_only():
        for _ in range(3):
            yield sse("ping", {"type": "ping"})
            await asyncio.sleep(0.02)

    async def gen_billing_only():
        yield sse("ping", {"type": "ping"})
        yield sse("billing_summary", {"type": "billing_summary", "cost": 0.002})
        yield sse("quota_update", {"type": "quota_update", "remaining": 900})

    async def gen_null_data():
        yield "data: null\n\n"
        yield "\n\n"
        yield "data: [DONE]\n\n"
        yield sse("billing_summary", {"type": "billing_summary", "cost": 0.001})
        for f in valid_frames():
            yield f

    async def gen_truncated_pre_content():
        yield sse("message_start", {
            "type": "message_start",
            "message": {"id": MSG_ID, "type": "message", "role": "assistant",
                        "model": "mock", "content": [], "stop_reason": None,
                        "usage": {"input_tokens": 1, "output_tokens": 0}}})
        # dies before any content block

    async def gen_truncated_post_content():
        yield sse("message_start", {
            "type": "message_start",
            "message": {"id": MSG_ID, "type": "message", "role": "assistant",
                        "model": "mock", "content": [], "stop_reason": None,
                        "usage": {"input_tokens": 1, "output_tokens": 0}}})
        yield sse("content_block_start", {
            "type": "content_block_start", "index": 0,
            "block": {"type": "text", "text": ""}})
        yield sse("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "partial answer"}})
        # dies mid-stream, no message_stop

    async def gen_flaky():
        if n < 3:
            return
        for f in valid_frames():
            yield f

    async def gen_valid():
        for f in valid_frames():
            yield f

    async def gen_valid_tool():
        for f in valid_frames(with_tool=True):
            yield f

    async def gen_slow_valid():
        # Commits quickly, then stays open long enough for a client to abandon it.
        frames = list(valid_frames())
        for f in frames[:3]:
            yield f
        for _ in range(40):
            await asyncio.sleep(0.25)
            yield sse("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "."}})
        for f in frames[3:]:
            yield f

    async def gen_remote_break():
        frames = list(valid_frames())
        for f in frames[:6]:
            yield f
        if n < 2:
            # httpx surfaces a closed response body as a protocol/read failure.
            raise RuntimeError("simulated RemoteProtocolError")
        for f in frames[6:]:
            yield f

    async def gen_always_break():
        for f in list(valid_frames())[:6]:
            yield f
        raise RuntimeError("simulated RemoteProtocolError")

    async def gen_byte_fixture():
        yield BYTE_FIXTURE

    async def gen_large_valid():
        frames = list(valid_frames())
        for f in frames[:2]:
            yield f
        yield sse("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "L" * 2000}})
        for f in frames[3:]:
            yield f

    async def gen_limit():
        yield ("event: message_start\ndata: {\"type\": \"message_start\"}\n\n" +
               "x" * 5000)

    async def gen_error_event():
        yield sse("error", {"type": "error",
                            "error": {"type": "overloaded_error",
                                      "message": "upstream overloaded"}})

    # ---- trailing-frame / EOF framing --------------------------------------
    # Observed in production: a stream ends without its final blank-line
    # separator, leaving one or more complete events in the parser buffer.

    async def gen_tail_glued():
        """
        The production bug: the last two events arrive glued by a SINGLE
        newline and the stream ends with no trailing blank line. Both are
        complete and valid; parsed as one frame they are two JSON documents
        joined by a newline, which does not parse.
        """
        frames = list(valid_frames())
        for f in frames[:5]:          # through content_block_stop, well-formed
            yield f
        yield ('event: message_delta\n'
               'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", '
               '"stop_sequence": null}, "usage": {"output_tokens": 5}}\n'
               'event: message_stop\n'
               'data: {"type": "message_stop"}')

    async def gen_tail_lone():
        """A single complete message_stop with no trailing blank line."""
        frames = list(valid_frames())
        for f in frames[:6]:          # through message_delta, well-formed
            yield f
        yield 'event: message_stop\ndata: {"type": "message_stop"}'

    async def gen_tail_truncated():
        """
        Negative control: the tail is genuinely incomplete -- the JSON payload
        is cut mid-object. It must stay dropped and must never be repaired or
        turned into a message_stop.
        """
        frames = list(valid_frames())
        for f in frames[:5]:
            yield f
        yield 'event: message_stop\ndata: {"type": "message_st'

    streams = {
        "empty_sse": gen_empty_sse,
        "keepalive_only": gen_keepalive_only,
        "billing_only": gen_billing_only,
        "null_data": gen_null_data,
        "truncated_pre": gen_truncated_pre_content,
        "truncated_post": gen_truncated_post_content,
        "flaky": gen_flaky,
        "valid": gen_valid,
        "valid_tool": gen_valid_tool,
        "slow_valid": gen_slow_valid,
        "remote_break": gen_remote_break,
        "always_break": gen_always_break,
        "byte_fixture": gen_byte_fixture,
        "large_valid": gen_large_valid,
        "limit": gen_limit,
        "error_event": gen_error_event,
        "tail_glued": gen_tail_glued,
        "tail_lone": gen_tail_lone,
        "tail_truncated": gen_tail_truncated,
    }

    if scenario == "empty_body_stream":
        return Response(content=b"", status_code=200, media_type="application/json")

    gen = streams.get(scenario, gen_valid)
    return StreamingResponse(gen(), status_code=200, media_type="text/event-stream")
