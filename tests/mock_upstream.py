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


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


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

    async def gen_error_event():
        yield sse("error", {"type": "error",
                            "error": {"type": "overloaded_error",
                                      "message": "upstream overloaded"}})

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
        "error_event": gen_error_event,
    }

    if scenario == "empty_body_stream":
        return Response(content=b"", status_code=200, media_type="application/json")

    gen = streams.get(scenario, gen_valid)
    return StreamingResponse(gen(), status_code=200, media_type="text/event-stream")
