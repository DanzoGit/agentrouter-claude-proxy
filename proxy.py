"""
proxy.py -- Local Anthropic Messages API compatibility proxy for AgentRouter.

Purpose
-------
Claude Code sometimes receives "API returned an empty or malformed response
(HTTP 200)" from AgentRouter. This proxy sits between Claude Code and
https://agentrouter.org and fixes ONLY transport/compatibility problems:

  * Upstream returns HTTP 200 with an empty body, a null body, or an SSE
    stream that never produces a usable Anthropic event. The proxy detects
    this BEFORE any bytes are committed to Claude Code, and transparently
    retries the original request with exponential backoff.
  * Junk SSE frames (empty frames, `data: null`, `data: [DONE]`,
    billing/quota-only metadata) are dropped so they cannot corrupt the
    Anthropic event stream.
  * httpx transparently decompresses gzip/deflate/br, so stale
    `content-encoding` / `content-length` response headers are stripped
    (leaving them makes the client try to gunzip already-plain bytes -- a
    classic source of "malformed response").

Hard guarantees
---------------
  * Event payloads are NEVER rewritten. Frames are forwarded byte-for-byte or
    dropped whole. tool_use / tool_result IDs therefore cannot be altered.
  * Model output is NEVER fabricated. No synthetic message_stop, no synthetic
    tool results, no mocked endpoints.
  * A failed upstream response is NEVER converted into a fake success.
    Upstream status codes and error bodies are forwarded.
  * Once a byte has been forwarded to the client, the request is NEVER
    replayed (that would duplicate text or tool calls). The failure is logged
    and surfaced as a real Anthropic `error` event instead.
  * No User-Agent spoofing, no filter bypass, no prompt modification.
  * API keys are never logged and never hardcoded.
  * The listener is loopback-only. LISTEN_HOST is deliberately not
    configurable by environment.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import os
import time
from typing import Any, AsyncIterator, Iterable, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# ----------------------------------------------------------------------------
# Configuration (all overridable by environment, no secrets in source)
# ----------------------------------------------------------------------------

UPSTREAM = os.environ.get("AGENTROUTER_UPSTREAM", "https://agentrouter.org").rstrip("/")

# Loopback only. This is a security boundary, not a preference: the proxy
# forwards whatever credential the client presents, so a LAN-reachable
# listener would let any host on the network spend your API quota.
LISTEN_HOST = "127.0.0.1"
PORT = int(os.environ.get("PROXY_PORT", "8787"))

# attempt 1 -> 500ms -> attempt 2 -> 1000ms -> attempt 3
MAX_ATTEMPTS = int(os.environ.get("PROXY_MAX_ATTEMPTS", "3"))
BACKOFF_BASE_MS = int(os.environ.get("PROXY_BACKOFF_BASE_MS", "500"))

# How long to wait for the first *usable* Anthropic content event before
# declaring the stream empty. Reasoning models can be slow to first token.
PRIME_TIMEOUT_S = float(os.environ.get("PROXY_PRIME_TIMEOUT_S", "120"))

CONNECT_TIMEOUT_S = float(os.environ.get("PROXY_CONNECT_TIMEOUT_S", "15"))
READ_TIMEOUT_S = float(os.environ.get("PROXY_READ_TIMEOUT_S", "600"))

VERBOSE = os.environ.get("PROXY_VERBOSE", "0") not in ("0", "", "false", "False")

# ----------------------------------------------------------------------------
# Anthropic SSE vocabulary
# ----------------------------------------------------------------------------

# Events that make up a well-formed Anthropic message stream.
ANTHROPIC_EVENTS = {
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
    "ping",
    "error",
}

# Seeing one of these proves the upstream is actually producing assistant or
# tool content -- this is the point at which we commit the stream to the client.
COMMIT_EVENTS = {"content_block_start", "content_block_delta"}

# Non-Anthropic bookkeeping frames some gateways interleave. Claude Code cannot
# parse them; they are dropped (type is logged, payload is not).
JUNK_TYPE_MARKERS = ("billing", "usage_summary", "credit", "quota", "balance")

# Top-level request fields that upstream schema validation rejects when
# explicitly null. Only these exact keys are stripped, and only at the top
# level -- message content is never touched.
NULLABLE_STRIP_KEYS = (
    "metadata",
    "system",
    "tools",
    "tool_choice",
    "stop_sequences",
    "temperature",
    "top_p",
    "top_k",
    "thinking",
    "service_tier",
)

# Connection-management headers, per RFC 7230 section 6.1. They describe a
# single hop and must not be relayed to the next one.
_RFC7230_CONNECTION_HEADERS = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
})

# Headers this proxy must recompute rather than relay, because it terminates
# and re-originates the request: the framing belongs to our own connection,
# and the encoding negotiation is ours to make (see accept-encoding below).
_PROXY_OWNED_HEADERS = frozenset({"host", "content-length", "accept-encoding"})

HOP_BY_HOP = _RFC7230_CONNECTION_HEADERS | _PROXY_OWNED_HEADERS

# httpx already decoded the body; forwarding these would corrupt it.
STRIP_RESPONSE_HEADERS = HOP_BY_HOP | {"content-encoding"}

SENSITIVE_HEADERS = {"authorization", "x-api-key", "api-key", "cookie", "set-cookie",
                     "proxy-authorization", "x-goog-api-key"}

# ----------------------------------------------------------------------------
# Counters
# ----------------------------------------------------------------------------

STATS: dict[str, int] = {
    "total_requests": 0,
    "successful_requests": 0,
    "empty_200_responses": 0,
    "malformed_streams": 0,
    "retries": 0,
    "retries_successful": 0,
    "upstream_5xx": 0,
    "upstream_4xx": 0,
    "dropped_sse_frames": 0,
    "post_commit_failures": 0,
    "client_disconnects": 0,
    "failed_requests": 0,
}

_STARTED_AT = time.time()


def bump(key: str, n: int = 1) -> None:
    STATS[key] = STATS.get(key, 0) + n


# ----------------------------------------------------------------------------
# Logging (never emits credentials)
# ----------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[proxy] {msg}", flush=True)


def vlog(msg: str) -> None:
    if VERBOSE:
        print(f"[proxy] {msg}", flush=True)


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Header names preserved, sensitive values replaced by a length hint."""
    out: dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in SENSITIVE_HEADERS:
            out[k] = f"<redacted len={len(v)}>"
        else:
            out[k] = v
    return out


def key_fingerprint(key: str) -> str:
    """Non-reversible-enough hint for debugging. Never the key itself."""
    if not key:
        return "<none>"
    return f"<len={len(key)} tail=...{key[-4:]}>"


def preview(body: bytes, limit: int = 160) -> str:
    """
    Sanitized diagnostic for an unusable body: structural shape only.

    Never returns prompt or completion text -- an unusable upstream body is
    typically an error page or a gateway notice, but it could contain echoed
    request content, so only a classification and the head of the payload with
    whitespace collapsed is emitted.
    """
    if not body:
        return "<empty>"
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        return f"<{len(body)} undecodable bytes>"

    stripped = text.strip()
    lowered = stripped[:400].lower()
    if lowered.startswith("<!doctype") or lowered.startswith("<html") or "<html" in lowered[:200]:
        return "<html document>"
    if stripped in ("null", "NULL"):
        return "<literal null>"

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return f"<json object, keys={sorted(parsed.keys())[:8]}>"
        return f"<json {type(parsed).__name__}>"
    except (json.JSONDecodeError, ValueError):
        pass

    collapsed = " ".join(stripped.split())
    return f"<non-json {len(body)}B: {collapsed[:limit]!r}>"


def body_parses_as_json(body: bytes) -> bool:
    """Diagnostic: does this body parse as JSON at all? Shape only, no content."""
    try:
        json.loads((body or b"").decode("utf-8"))
        return True
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return False


def client_will_json_parse(content_type: str) -> bool:
    """
    Mirror the Anthropic SDK's decision to call response.json().

    The SDK gates parsing on the content-type naming JSON; for anything else it
    hands the caller the raw text. Claude Code then runs a validator that begins
    with `typeof body === "object"`, so a perfectly valid message served as
    text/plain is rejected as an "empty or malformed response". This is a
    diagnostic only -- nothing is rewritten on the basis of it.
    """
    ct = (content_type or "").lower()
    return "application/json" in ct or "+json" in ct


# ----------------------------------------------------------------------------
# SSE framing
# ----------------------------------------------------------------------------


class SSEFramer:
    """
    Incremental byte-stream -> SSE-frame splitter.

    Owns both the UTF-8 decoder and the carry-over buffer so that a multi-byte
    character or a frame boundary landing mid-chunk is handled in one place.
    Callers get whole frames and never touch partial text.

    Frame text is returned without its trailing blank-line separator and is
    otherwise unmodified, which is what keeps forwarding byte-exact.
    """

    SEPARATOR = "\n\n"

    def __init__(self) -> None:
        self._decode = codecs.getincrementaldecoder("utf-8")(errors="replace").decode
        self._carry = ""

    def feed(self, chunk: bytes) -> list[str]:
        """Absorb one network chunk, return every complete frame it finished."""
        self._carry += self._decode(chunk).replace("\r\n", "\n")
        if self.SEPARATOR not in self._carry:
            return []
        pieces = self._carry.split(self.SEPARATOR)
        # The final piece is by definition not yet terminated: carry it over.
        self._carry = pieces.pop()
        return [p for p in pieces if p.strip()]

    def flush(self) -> Optional[str]:
        """
        Surrender any unterminated trailing text once the stream has ended.

        Returns None when nothing meaningful remains, so callers can treat a
        clean end and a whitespace-only remainder identically.
        """
        remainder, self._carry = self._carry, ""
        return remainder if remainder.strip() else None


def split_trailing_frames(remainder: str) -> list[str]:
    """
    Split an unterminated tail into the individual SSE events it contains.

    A stream that ends without its final blank-line separator can leave more
    than one complete event in the buffer, glued by a single newline:

        event: message_delta
        data: {"type": "message_delta", ...}
        event: message_stop
        data: {"type": "message_stop"}

    Treated as one frame that is two JSON documents joined by a newline, which
    does not parse -- so a complete, valid message_stop was dropped and the
    stream was reported as having ended without one.

    A new event starts at an `event:` line that follows a `data:` line; a raw
    line can never begin with "event:" from inside a payload, because JSON
    escapes newlines rather than emitting them literally. Multi-line `data:`
    payloads stay with their own event.

    This only separates candidates. Each one still goes through parse_frame and
    drop_reason unchanged, so a genuinely truncated or malformed tail is
    dropped exactly as before and nothing is ever synthesized.
    """
    frames: list[str] = []
    current: list[str] = []
    for line in remainder.split("\n"):
        if line.startswith("event:") and any(l.startswith("data:") for l in current):
            frames.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        frames.append("\n".join(current))
    return [f for f in frames if f.strip()]


def encode_frame(text: str) -> bytes:
    """Re-attach the SSE separator and encode. The payload itself is untouched."""
    return (text.rstrip("\n") + SSEFramer.SEPARATOR).encode("utf-8")


def parse_frame(frame: str) -> tuple[str, Optional[Any], bool]:
    """
    Parse one raw SSE frame.

    Returns (event_type, parsed_data_or_None, data_was_parseable_json).
    The frame text itself is never modified by this function.
    """
    event_name = ""
    data_lines: list[str] = []

    for line in frame.split("\n"):
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    raw_data = "\n".join(data_lines).strip()

    if not raw_data:
        return event_name, None, False
    if raw_data == "[DONE]":
        return event_name or "[DONE]", None, False

    try:
        parsed = json.loads(raw_data)
    except (json.JSONDecodeError, ValueError):
        return event_name, None, False

    if isinstance(parsed, dict):
        event_name = parsed.get("type") or event_name

    return event_name, parsed, True


def drop_reason(event_type: str, data: Any, parseable: bool) -> Optional[str]:
    """
    Decide whether a frame must be dropped.

    Returns a short reason string, or None to forward the frame untouched.
    Unknown-but-well-formed event types are forwarded (forward compatibility);
    only clearly non-Anthropic noise is dropped.
    """
    if event_type == "[DONE]":
        return "openai_style_done_sentinel"
    if not parseable and data is None:
        if not event_type:
            return "empty_or_unparseable_frame"
        return "non_json_data"
    if data is None:
        return "data_null"
    if not isinstance(data, dict):
        return "data_not_an_object"

    lowered = (event_type or "").lower()
    for marker in JUNK_TYPE_MARKERS:
        if marker in lowered:
            return f"non_anthropic_metadata:{marker}"

    if not event_type:
        return "missing_event_type"

    return None


def error_frame(message: str, err_type: str = "api_error") -> bytes:
    """
    A real Anthropic-shaped `error` event. This reports a transport failure --
    it carries no model output and is only ever emitted after the upstream has
    already broken mid-stream.
    """
    payload = {"type": "error", "error": {"type": err_type, "message": message}}
    return f"event: error\ndata: {json.dumps(payload)}\n\n".encode("utf-8")


# ----------------------------------------------------------------------------
# Request preparation
# ----------------------------------------------------------------------------


def resolve_api_key() -> str:
    for var in ("ANTHROPIC_AUTH_TOKEN", "AGENTROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val.strip()
    return ""


def build_upstream_headers(client_headers: dict[str, str]) -> dict[str, str]:
    """
    Forward the client's own headers. The real Claude Code User-Agent is passed
    through unchanged -- no impersonation of any other client.
    """
    headers = {k: v for k, v in client_headers.items() if k.lower() not in HOP_BY_HOP}

    # Ask for an uncompressed body: removes the entire class of malformed
    # gzip/deflate/br failures at the source.
    headers["accept-encoding"] = "identity"

    lower = {k.lower() for k in headers}

    # AgentRouter authenticates with a Bearer token; Claude Code may send only
    # x-api-key. Mirror it across without altering the credential.
    if "authorization" not in lower:
        api_key = ""
        for k, v in headers.items():
            if k.lower() == "x-api-key":
                api_key = v
                break
        if not api_key:
            api_key = resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    if "anthropic-version" not in lower:
        headers["anthropic-version"] = "2023-06-01"

    return headers


def sanitize_body(body: bytes) -> tuple[bytes, list[str], bool]:
    """
    Strip only explicitly-null top-level fields known to break upstream schema
    validation. Messages, system prompt text, tools and tool_choice values are
    never rewritten -- a key is either removed (when its value is exactly null)
    or left completely untouched.

    Returns (body, stripped_keys, stream_requested).
    """
    if not body:
        return body, [], False

    try:
        data = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return body, [], False

    if not isinstance(data, dict):
        return body, [], False

    stream_requested = data.get("stream") is True

    stripped = [k for k in NULLABLE_STRIP_KEYS if k in data and data[k] is None]
    if not stripped:
        return body, [], stream_requested

    for k in stripped:
        del data[k]

    return json.dumps(data).encode("utf-8"), stripped, stream_requested


# ----------------------------------------------------------------------------
# Upstream attempt outcome
# ----------------------------------------------------------------------------


class UpstreamStalled(Exception):
    """No bytes arrived from the upstream before the deadline."""


class UpstreamBroke(Exception):
    """The upstream connection failed mid-stream."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


class AttemptResult:
    """Outcome of a single upstream attempt."""

    def __init__(
        self,
        *,
        ok: bool,
        reason: str = "",
        transient: bool = False,
        status: int = 0,
        response: Optional[httpx.Response] = None,
        client: Optional[httpx.AsyncClient] = None,
        body: Optional[bytes] = None,
        primed: Optional[list[str]] = None,
        framer: Optional[SSEFramer] = None,
        stream_iter: Optional[AsyncIterator[bytes]] = None,
        headers: Optional[dict[str, str]] = None,
        is_sse: bool = False,
    ) -> None:
        self.ok = ok
        self.reason = reason
        self.transient = transient
        self.status = status
        self.response = response
        self.client = client
        self.body = body
        # Frame texts consumed while proving the stream usable. Held as text,
        # not bytes, so the payload is encoded exactly once on the way out.
        self.primed = primed or []
        self.framer = framer
        self.stream_iter = stream_iter
        self.headers = headers or {}
        self.is_sse = is_sse


async def close_quietly(response: Optional[httpx.Response], client: Optional[httpx.AsyncClient]) -> None:
    for obj in (response, client):
        if obj is None:
            continue
        try:
            await obj.aclose()
        except Exception:
            pass


async def deadline_chunks(
    stream_iter: AsyncIterator[bytes],
    deadline: float,
) -> AsyncIterator[bytes]:
    """
    Yield upstream chunks until the iterator ends or `deadline` passes.

    Failure modes are raised as typed exceptions rather than signalled in a
    return value, so the consuming loop only ever deals with real data.
    """
    while True:
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise UpstreamStalled()
        try:
            chunk = await asyncio.wait_for(stream_iter.__anext__(), timeout=budget)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            raise UpstreamStalled() from None
        except (httpx.HTTPError, OSError) as exc:
            raise UpstreamBroke(type(exc).__name__) from exc
        yield chunk


# ----------------------------------------------------------------------------
# Streaming: prime the upstream before committing anything to the client
# ----------------------------------------------------------------------------


async def prime_stream(response: httpx.Response, client: httpx.AsyncClient) -> AttemptResult:
    """
    Read the upstream SSE stream until we can prove it is usable.

    "Usable" means a content_block_start or content_block_delta has arrived --
    i.e. the upstream is genuinely producing assistant or tool content. Frames
    consumed while proving this are buffered verbatim and replayed to the
    client in order, so nothing is lost.

    An upstream `error` event is a real error: it is committed immediately and
    forwarded rather than retried.
    """
    framer = SSEFramer()
    stream_iter = response.aiter_bytes().__aiter__()
    primed: list[str] = []
    seen: list[str] = []

    def commit(reason: str) -> AttemptResult:
        return AttemptResult(ok=True, reason=reason, status=response.status_code,
                             response=response, client=client, primed=primed,
                             framer=framer, stream_iter=stream_iter,
                             headers=dict(response.headers), is_sse=True)

    def abandon(reason: str, counter: str) -> AttemptResult:
        bump(counter)
        return AttemptResult(ok=False, reason=reason, transient=True,
                             status=response.status_code, response=response,
                             client=client)

    source = deadline_chunks(stream_iter, time.monotonic() + PRIME_TIMEOUT_S)
    try:
        async for chunk in source:
            for text in framer.feed(chunk):
                etype, data, parseable = parse_frame(text)
                seen.append(etype or "<untyped>")

                reason = drop_reason(etype, data, parseable)
                if reason:
                    bump("dropped_sse_frames")
                    log(f"dropped SSE frame: type={etype or '<none>'} reason={reason}")
                    continue

                primed.append(text)

                if etype == "error":
                    kind = ""
                    if isinstance(data, dict) and isinstance(data.get("error"), dict):
                        kind = str(data["error"].get("type", ""))
                    log(f"upstream sent an SSE error event (type={kind or 'unknown'}) "
                        f"-- forwarding as-is, not retrying")
                    return commit("upstream_error_event")

                if etype in COMMIT_EVENTS:
                    log("valid Anthropic SSE detected")
                    log("forwarding stream")
                    return commit("valid_sse")
    except UpstreamStalled:
        log(f"invalid response: no content event within {PRIME_TIMEOUT_S:.0f}s "
            f"(saw: {seen or 'nothing'})")
        return abandon("stream_timeout_no_content", "malformed_streams")
    except UpstreamBroke as exc:
        log(f"invalid response: upstream stream broke while priming ({exc.kind})")
        return abandon("stream_broken_while_priming", "malformed_streams")
    finally:
        await source.aclose()

    # Stream ended before any usable content event.
    if framer.flush() is not None:
        vlog("trailing partial frame discarded at end of empty stream")

    if not seen:
        log("invalid response: empty stream (upstream closed with zero SSE frames)")
        return abandon("empty_stream", "empty_200_responses")

    if all(t in ("ping", "<untyped>") for t in seen):
        log(f"invalid response: keep-alive only, no Anthropic events ({len(seen)} frames)")
        return abandon("keepalive_only", "malformed_streams")

    ordered = ",".join(dict.fromkeys(seen))
    if "message_stop" in seen or "message_delta" in seen:
        log(f"invalid response: stream completed with no assistant content (events: {ordered})")
        return abandon("no_usable_content", "malformed_streams")

    log(f"invalid response: stream ended before message_stop with no content (events: {ordered})")
    return abandon("truncated_before_content", "malformed_streams")


# ----------------------------------------------------------------------------
# Non-streaming validation
# ----------------------------------------------------------------------------


def validate_json_response(status: int, body: bytes, is_messages: bool) -> tuple[bool, str]:
    """
    Validate a non-streaming upstream body. Only HTTP 200 is validated for
    shape; non-2xx bodies are real errors and are forwarded verbatim.
    """
    if status != 200:
        return True, "non_200_forwarded_verbatim"

    if not body or not body.strip():
        bump("empty_200_responses")
        return False, "empty_body"

    try:
        data = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        bump("malformed_streams")
        return False, f"unparseable_json_body(bytes={len(body)})"

    if data is None:
        bump("empty_200_responses")
        return False, "null_body"

    if not is_messages:
        return True, "ok"

    if not isinstance(data, dict):
        bump("malformed_streams")
        return False, "body_not_an_object"

    if "error" in data and data.get("type") == "error":
        # A real upstream error delivered with a 200. It is not retried -- a
        # genuine rejection returns the same answer every time -- but it is not
        # a usable message either, so it must not reach the client as a 200.
        # The caller forwards this body verbatim under HTTP 502 so the
        # upstream's own message survives.
        return False, "upstream_error_object"

    # Claude Code validates a non-streaming /v1/messages reply before using it
    # and requires content:array + model:string + usage:object. Anything it
    # would reject is rejected here too, so the request is retried while a
    # clean retry is still possible instead of failing in the client as
    # "empty or malformed response (HTTP 200)". Nothing is ever fabricated to
    # make a body pass -- a missing field is a failure, not something to fill in.
    content = data.get("content")
    if content is None:
        bump("malformed_streams")
        return False, "missing_content_field"
    if not isinstance(content, list):
        bump("malformed_streams")
        return False, "content_not_an_array"
    if len(content) == 0:
        bump("empty_200_responses")
        return False, "empty_content_array"

    if "model" not in data:
        bump("malformed_streams")
        return False, "missing_model_field"
    if not isinstance(data["model"], str):
        bump("malformed_streams")
        return False, "model_not_a_string"

    if "usage" not in data:
        bump("malformed_streams")
        return False, "missing_usage_field"
    # Claude Code's test is `typeof usage === "object"`, which is also true for
    # null in JavaScript, so a null usage is accepted here rather than retried.
    # Being stricter than the client would reject replies it would have used.
    if data["usage"] is not None and not isinstance(data["usage"], dict):
        bump("malformed_streams")
        return False, "usage_not_an_object"

    return True, "ok"


# ----------------------------------------------------------------------------
# Attempt loop
# ----------------------------------------------------------------------------


async def attempt_upstream(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
    stream_expected: bool,
    is_messages: bool,
) -> AttemptResult:
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
        follow_redirects=False,
    )
    response: Optional[httpx.Response] = None

    try:
        request = client.build_request(method, url, headers=headers, content=body or None)
        response = await client.send(request, stream=True)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
            httpx.WriteTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError,
            httpx.ReadError, httpx.WriteError) as exc:
        await close_quietly(response, client)
        log(f"connection failure: {type(exc).__name__}")
        return AttemptResult(ok=False, reason=f"connect:{type(exc).__name__}", transient=True)
    except Exception as exc:
        await close_quietly(response, client)
        log(f"unexpected transport failure: {type(exc).__name__}")
        return AttemptResult(ok=False, reason=f"transport:{type(exc).__name__}", transient=False)

    status = response.status_code
    log(f"HTTP {status}")

    # --- Authentication / permission: report, never work around ------------
    if status in (401, 403):
        raw = await read_body_safe(response)
        response_headers = dict(response.headers)
        await close_quietly(response, client)
        bump("upstream_4xx")
        log(f"authentication/permission failure from upstream (HTTP {status}) -- "
            f"not retrying, not bypassing")
        return AttemptResult(ok=True, reason="auth_failure", status=status,
                             body=raw, headers=response_headers, is_sse=False)

    # --- Server-side failures: retry only transient classes -----------------
    if status >= 500:
        bump("upstream_5xx")
        raw = await read_body_safe(response)
        response_headers = dict(response.headers)
        await close_quietly(response, client)
        transient = status in (502, 503, 504)
        detail = classify_503(raw) if status == 503 else ""
        if detail:
            log(f"upstream availability problem (HTTP 503): {detail}")
        return AttemptResult(ok=False, reason=f"http_{status}{(':' + detail) if detail else ''}",
                             transient=transient, status=status, body=raw,
                             headers=response_headers)

    if 400 <= status < 500:
        bump("upstream_4xx")
        raw = await read_body_safe(response)
        response_headers = dict(response.headers)
        await close_quietly(response, client)
        return AttemptResult(ok=True, reason=f"http_{status}", status=status, body=raw,
                             headers=response_headers, is_sse=False)

    content_type = response.headers.get("content-type", "")
    is_sse = "text/event-stream" in content_type.lower()

    # --- Streaming path -----------------------------------------------------
    if stream_expected and is_sse:
        return await prime_stream(response, client)

    if stream_expected and not is_sse:
        raw = await read_body_safe(response)
        response_headers = dict(response.headers)
        await close_quietly(response, client)
        if not raw or not raw.strip():
            bump("empty_200_responses")
            log("invalid response: empty body (HTTP 200, stream was requested)")
            return AttemptResult(ok=False, reason="empty_body_stream_requested",
                                 transient=True, status=status, body=raw,
                                 headers=response_headers)
        bump("malformed_streams")
        log(f"invalid response: stream requested but upstream sent "
            f"content-type={content_type or '<none>'} ({len(raw)} bytes)")
        return AttemptResult(ok=False, reason="non_sse_for_stream_request",
                             transient=True, status=status, body=raw,
                             headers=response_headers)

    # --- Non-streaming path -------------------------------------------------
    raw = await read_body_safe(response)
    response_headers = dict(response.headers)
    await close_quietly(response, client)

    valid, reason = validate_json_response(status, raw, is_messages)
    if not valid:
        log(f"invalid response: {reason}")
        # A genuine upstream error object is a real rejection, not a transport
        # glitch -- retrying it only repeats the same answer.
        return AttemptResult(ok=False, reason=reason,
                             transient=reason != "upstream_error_object",
                             status=status, body=raw, headers=response_headers)

    return AttemptResult(ok=True, reason=reason, status=status, body=raw,
                         headers=response_headers, is_sse=False)


async def read_body_safe(response: httpx.Response) -> bytes:
    """
    Read a body without letting a decompression or truncation error escape.
    httpx handles gzip/deflate/br transparently; a corrupt encoding surfaces
    here as a DecodingError and is reported as an empty body so the caller
    treats the attempt as invalid rather than crashing.
    """
    try:
        return await response.aread()
    except httpx.DecodingError as exc:
        log(f"upstream body failed to decompress ({type(exc).__name__}) -- "
            f"treating as invalid")
        return b""
    except (httpx.HTTPError, OSError) as exc:
        log(f"upstream body read failed ({type(exc).__name__}) -- treating as invalid")
        return b""


def classify_503(body: bytes) -> str:
    """Recognise the 'no channel available' shape. Reported, never repaired."""
    try:
        text = body.decode("utf-8", errors="replace").lower()
    except Exception:
        return ""
    for marker in ("no available channel", "no channel", "channel not found",
                   "model not available", "no available model"):
        if marker in text:
            return "no available channel/model for the requested model"
    return ""


async def run_with_retries(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
    stream_expected: bool,
    is_messages: bool,
) -> AttemptResult:
    last: Optional[AttemptResult] = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"upstream attempt {attempt}")
        result = await attempt_upstream(
            method=method, url=url, headers=headers, body=body,
            stream_expected=stream_expected, is_messages=is_messages,
        )
        last = result

        if result.ok:
            if attempt > 1:
                bump("retries_successful")
                log(f"recovered on attempt {attempt}")
            return result

        await close_quietly(result.response, result.client)
        result.response = None
        result.client = None

        if not result.transient:
            log(f"non-transient failure ({result.reason}) -- not retrying")
            return result

        if attempt >= MAX_ATTEMPTS:
            log(f"all {MAX_ATTEMPTS} attempts failed ({result.reason})")
            return result

        delay_ms = BACKOFF_BASE_MS * (2 ** (attempt - 1))
        bump("retries")
        log(f"retrying in {delay_ms}ms")
        await asyncio.sleep(delay_ms / 1000.0)

    return last or AttemptResult(ok=False, reason="no_attempt_made")


# ----------------------------------------------------------------------------
# Client-facing stream
# ----------------------------------------------------------------------------


class StreamTracker:
    """Records how a stream terminated, so the caller can tell truncation apart
    from a legitimate end."""

    def __init__(self) -> None:
        self.forwarded = 0
        self.saw_message_stop = False
        self.saw_error_event = False

    def note(self, event_type: str) -> None:
        if event_type == "message_stop":
            self.saw_message_stop = True
        elif event_type == "error":
            self.saw_error_event = True

    @property
    def terminated_cleanly(self) -> bool:
        # An upstream `error` event is a legitimate terminal state -- the stream
        # is complete, it just ended in an error rather than message_stop.
        return self.saw_message_stop or self.saw_error_event


async def forward_stream(result: AttemptResult) -> AsyncIterator[bytes]:
    """
    Replay the primed frames, then continue the live stream.

    From the first yielded byte the request is committed: it is never replayed,
    because doing so could duplicate assistant text or tool calls. A later
    failure is logged and surfaced as an Anthropic `error` event.
    """
    response = result.response
    client = result.client
    framer = result.framer
    stream_iter = result.stream_iter

    tracker = StreamTracker()
    started = time.monotonic()
    failure: Optional[str] = None

    async def frames() -> AsyncIterator[str]:
        """Primed frames first, then whatever the live connection still has."""
        for text in result.primed:
            yield text

        while True:
            try:
                chunk = await stream_iter.__anext__()
            except StopAsyncIteration:
                break
            except (httpx.HTTPError, OSError) as exc:
                raise UpstreamBroke(type(exc).__name__) from exc
            for text in framer.feed(chunk):
                yield text

        leftover = framer.flush()
        if leftover is not None:
            # EOF without the final blank line. The tail may still hold one or
            # more complete events -- typically the closing message_stop. Offer
            # each to the same validation every other frame gets; an incomplete
            # or malformed one is dropped there, as before.
            for text in split_trailing_frames(leftover):
                yield text

    source = frames()
    try:
        async for text in source:
            etype, data, parseable = parse_frame(text)

            reason = drop_reason(etype, data, parseable)
            if reason:
                bump("dropped_sse_frames")
                log(f"dropped SSE frame: type={etype or '<none>'} reason={reason}")
                continue

            tracker.note(etype)
            tracker.forwarded += 1
            yield encode_frame(text)

        if not tracker.terminated_cleanly:
            failure = (f"upstream closed after {tracker.forwarded} frames "
                       f"without message_stop")

    except UpstreamBroke as exc:
        failure = f"upstream stream broke after {tracker.forwarded} frames ({exc.kind})"
    except asyncio.CancelledError:
        # Client vanished while we were awaiting an upstream chunk.
        bump("client_disconnects")
        log("client disconnected -- closing upstream stream")
        raise
    except GeneratorExit:
        # Client vanished while we were handing a frame to the server; the
        # generator is being closed rather than cancelled.
        bump("client_disconnects")
        log("client disconnected -- closing upstream stream")
        raise
    except Exception as exc:
        failure = f"proxy stream error ({type(exc).__name__})"
    finally:
        # Cleanup must survive cancellation. A plain `await` here would itself
        # be cancelled before the upstream socket is released, leaking one
        # upstream connection per abandoned request.
        async def release() -> None:
            try:
                await source.aclose()
            except Exception:
                pass
            await close_quietly(response, client)

        closer = asyncio.ensure_future(release())
        try:
            await asyncio.shield(closer)
        except asyncio.CancelledError:
            # `closer` still runs to completion; the in-flight exception resumes.
            pass

    elapsed = time.monotonic() - started

    if failure:
        bump("post_commit_failures")
        bump("failed_requests")
        log(f"post-commit failure: {failure}")
        log("not replaying the request (already-forwarded output must not be "
            "duplicated) -- emitting an error event instead")
        # Not model output: an explicit transport-failure signal.
        yield error_frame(
            f"Upstream stream ended abnormally: {failure}. The proxy did not "
            f"replay the request to avoid duplicating output. Please retry.",
            "upstream_stream_incomplete",
        )
    else:
        bump("successful_requests")
        log(f"stream complete: {tracker.forwarded} frames in {elapsed:.1f}s, "
            f"message_stop=yes")


# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------

app = FastAPI(title="AgentRouter compatibility proxy", docs_url=None, redoc_url=None)


def passthrough_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in STRIP_RESPONSE_HEADERS}


@app.get("/_stats")
async def stats() -> JSONResponse:
    return JSONResponse(
        {
            **STATS,
            "uptime_seconds": round(time.time() - _STARTED_AT, 1),
            "upstream": UPSTREAM,
            "max_attempts": MAX_ATTEMPTS,
            "backoff_base_ms": BACKOFF_BASE_MS,
            "prime_timeout_s": PRIME_TIMEOUT_S,
        }
    )


@app.get("/_health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "upstream": UPSTREAM,
                         "listen": f"{LISTEN_HOST}:{PORT}"})


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "status": "running",
            "upstream": UPSTREAM,
            "endpoints": ["/v1/messages", "/v1/messages?beta=true", "/_stats", "/_health"],
        }
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str) -> Response:
    bump("total_requests")

    raw_body = await request.body()
    body, stripped, stream_requested = sanitize_body(raw_body)

    query = request.url.query
    url = f"{UPSTREAM}/{path}" + (f"?{query}" if query else "")
    is_messages = path.rstrip("/") == "v1/messages"

    log("=" * 62)
    log(f"{request.method} /{path}{('?' + query) if query else ''} "
        f"-> {url}  stream={stream_requested}")
    if stripped:
        log(f"stripped null request fields: {', '.join(stripped)}")

    headers = build_upstream_headers(dict(request.headers))
    if VERBOSE:
        vlog(f"upstream headers: {redact_headers(headers)}")

    if not any(k.lower() == "authorization" for k in headers):
        log("no credential available (client sent none and no ANTHROPIC_AUTH_TOKEN / "
            "AGENTROUTER_API_KEY in environment)")

    result = await run_with_retries(
        method=request.method,
        url=url,
        headers=headers,
        body=body,
        stream_expected=stream_requested,
        is_messages=is_messages,
    )

    # ---- Failure after all attempts: report honestly ------------------------
    if not result.ok:
        bump("failed_requests")
        await close_quietly(result.response, result.client)

        # Forward the upstream body verbatim ONLY when the upstream itself
        # reported an error status. An invalid HTTP 200 must never be passed
        # through as a 200 -- that is the exact failure Claude Code reports as
        # "empty or malformed response (HTTP 200)".
        if result.status >= 400 and result.body:
            log(f"final failure: {result.reason} -- forwarding upstream "
                f"HTTP {result.status} body verbatim")
            return Response(
                content=result.body,
                status_code=result.status,
                media_type=result.headers.get("content-type", "application/json"),
            )

        # An upstream error object delivered under HTTP 200 is a real error
        # message, so its text is forwarded verbatim -- but never under the 200
        # that Claude Code would reject as malformed.
        if result.reason == "upstream_error_object" and result.body:
            log("final failure: upstream_error_object -- forwarding upstream "
                "error body under HTTP 502")
            return Response(
                content=result.body,
                status_code=502,
                media_type=result.headers.get("content-type", "application/json"),
            )

        if result.status == 200 and result.body:
            # Sanitized diagnostics: shape only, never forwarded to the client.
            log(f"discarding unusable HTTP 200 body "
                f"({len(result.body)} bytes, content-type="
                f"{result.headers.get('content-type', '<none>')}) -- "
                f"preview: {preview(result.body)}")

        detail = describe_failure(result)
        log(f"final failure: {result.reason}")
        return JSONResponse(
            {"type": "error",
             "error": {"type": "api_error",
                       "message": detail,
                       "proxy_reason": result.reason,
                       "proxy_attempts": MAX_ATTEMPTS}},
            status_code=result.status if result.status >= 400 else 502,
        )

    # ---- Success: streaming -------------------------------------------------
    if result.is_sse and result.stream_iter is not None:
        out_headers = passthrough_headers(result.headers)
        out_headers["cache-control"] = "no-cache"
        return StreamingResponse(
            forward_stream(result),
            status_code=result.status,
            headers=out_headers,
            media_type="text/event-stream",
        )

    # ---- Success: non-streaming (incl. forwarded 4xx) -----------------------
    normalize_content_type = False
    if result.status == 200:
        bump("successful_requests")
        log(f"non-stream response OK ({len(result.body or b'')} bytes)")
        if is_messages:
            # Diagnostic only: headers and shape, never body content. Records
            # whether the client's SDK will actually JSON-parse what we forward.
            # If it will not, Claude Code's validator sees a string instead of
            # an object and rejects a structurally perfect message.
            upstream_ct = result.headers.get("content-type", "<none>")
            outgoing_ct = result.headers.get("content-type", "application/json")
            json_parsed = body_parses_as_json(result.body)
            log(f"non-stream diagnostics: "
                f"upstream_content_type={upstream_ct} "
                f"json_parsed={json_parsed} "
                f"validation={result.reason} "
                f"outgoing_content_type={outgoing_ct} "
                f"client_will_json_parse={client_will_json_parse(outgoing_ct)} "
                f"bytes={len(result.body or b'')}")
            # A body only reaches here by passing validate_json_response, so it
            # is a well-formed Anthropic message. AgentRouter has been observed
            # serving such a message as text/plain; the Anthropic SDK then skips
            # response.json() and hands Claude Code a raw string, whose
            # validator fails on `typeof body === "object"` and reports "an
            # empty or malformed response (HTTP 200)". Relabel it. Only the
            # header changes -- the body bytes are forwarded untouched.
            normalize_content_type = json_parsed and not client_will_json_parse(outgoing_ct)
    else:
        bump("failed_requests")
        log(f"forwarding upstream HTTP {result.status} verbatim ({result.reason})")

    out_headers = passthrough_headers(result.headers)
    media_type = result.headers.get("content-type", "application/json")
    if normalize_content_type:
        # Starlette ignores media_type entirely when the headers mapping already
        # carries a content-type, so the upstream value has to be dropped here
        # rather than overridden below. Removing it case-insensitively also
        # collapses any duplicate the upstream may have sent, which guarantees
        # exactly one content-type on the wire.
        out_headers = {k: v for k, v in out_headers.items()
                       if k.lower() != "content-type"}
        media_type = "application/json"
        log(f"content-type normalized for client SDK: "
            f"{result.headers.get('content-type', '<none>')} -> application/json "
            f"(body unchanged, {len(result.body or b'')} bytes)")

    return Response(
        content=result.body or b"",
        status_code=result.status,
        headers=out_headers,
        media_type=media_type,
    )


def describe_failure(result: AttemptResult) -> str:
    reason = result.reason or "unknown"
    if reason.startswith("http_503"):
        if "no available channel" in reason:
            return ("AgentRouter upstream availability problem: no available "
                    "channel/model for the requested model. This is an upstream "
                    "capacity/routing issue, not a client error.")
        return ("AgentRouter returned HTTP 503 on every attempt -- upstream "
                "availability problem.")
    if reason.startswith("http_5"):
        return f"AgentRouter returned {reason.replace('http_', 'HTTP ')} on every attempt."
    if reason.startswith("connect:"):
        return (f"Could not reach {UPSTREAM} after {MAX_ATTEMPTS} attempts "
                f"({reason.split(':', 1)[1]}).")
    if reason in ("empty_stream", "empty_body", "empty_body_stream_requested", "null_body"):
        return (f"AgentRouter returned HTTP 200 with an empty response on all "
                f"{MAX_ATTEMPTS} attempts. No model output was produced upstream; "
                f"the proxy will not fabricate one.")
    if reason in ("keepalive_only", "no_usable_content", "truncated_before_content",
                  "empty_content_array", "missing_content_field"):
        return (f"AgentRouter returned HTTP 200 but never produced usable assistant "
                f"content on any of {MAX_ATTEMPTS} attempts ({reason}).")
    if reason in ("content_not_an_array", "missing_model_field", "model_not_a_string",
                  "missing_usage_field", "usage_not_an_object"):
        return (f"AgentRouter returned HTTP 200 with a body that is not a valid "
                f"Anthropic message on all {MAX_ATTEMPTS} attempts ({reason}). "
                f"Claude Code checks the same fields and would reject it as an "
                f"empty or malformed response, so the proxy reports the failure "
                f"instead of forwarding it. The missing field is not invented.")
    if reason.startswith("unparseable_json_body") or reason == "non_sse_for_stream_request":
        return (f"AgentRouter returned a malformed response on all {MAX_ATTEMPTS} "
                f"attempts ({reason}).")
    return f"Upstream request failed after {MAX_ATTEMPTS} attempts ({reason})."


if __name__ == "__main__":
    import uvicorn

    if not resolve_api_key():
        log("warning: no ANTHROPIC_AUTH_TOKEN / AGENTROUTER_API_KEY in environment "
            "-- relying on the credential Claude Code sends per-request")
    else:
        log(f"credential loaded from environment {key_fingerprint(resolve_api_key())}")

    log(f"upstream = {UPSTREAM}")
    log(f"listening on http://{LISTEN_HOST}:{PORT} (loopback only)")
    uvicorn.run(app, host=LISTEN_HOST, port=PORT, log_level="warning", access_log=False)
