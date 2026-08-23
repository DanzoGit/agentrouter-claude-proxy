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
import hashlib
import json
import os
import re
import secrets
import time
from collections import OrderedDict, deque
from email.utils import parsedate_tz
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Optional
from urllib.parse import unquote_plus, urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

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

STREAM_MODE = os.environ.get("PROXY_STREAM_MODE", "normal").strip().lower()
if STREAM_MODE not in ("normal", "reliable"):
    STREAM_MODE = "normal"
RELIABLE_MAX_BYTES = int(os.environ.get("PROXY_RELIABLE_MAX_BYTES", str(16 * 1024 * 1024)))

VERBOSE = os.environ.get("PROXY_VERBOSE", "0") not in ("0", "", "false", "False")

# Advertised to the client when the upstream rate-limits us but sends no
# Retry-After of its own. Without a hint the Anthropic SDK schedules its next
# attempt immediately -- the "Retrying in 0s" the terminal reports -- which
# hammers an upstream that is already out of capacity.
DEFAULT_RETRY_AFTER_S = int(os.environ.get("PROXY_RETRY_AFTER_S", "15"))

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

# The wording upstream uses when every provider behind it is busy. Matched only
# against an error envelope's own message fields -- never against assistant
# content -- so a reply that merely discusses saturation cannot be mistaken for
# one. Recognition is diagnostic: the status, the body and the retry decision
# are identical whether or not a 429 matches.
# Bounded number of last-known-good structural checkpoints kept in memory.
# Small on purpose: recovery only ever needs the most recent accepted turn per
# conversation, and an unbounded store would grow for the life of the process.
MAX_RECOVERY_CHECKPOINTS = int(
    os.environ.get("PROXY_MAX_RECOVERY_CHECKPOINTS", "32"))

# Explicit moderation wording only, matched against the upstream error envelope
# and never against assistant output. Deliberately excludes anything that could
# also describe capacity, rate limiting, schema validation, authentication or
# model access -- those are different failures with different handling.
CONTENT_BLOCKED_MARKERS = (
    "content-blocked",
    "content_blocked",
    "content blocked",
    "blocked by content policy",
    "content policy violation",
    "content_filter",
    "content_policy",
    "flagged by moderation",
)

# Headers a client could use to name a conversation. Claude Code is not known to
# send any of them; the list exists so that if one ever appears the recovery
# lane becomes a real session id instead of a structural hash. Only the opaque
# value is used, and it is never treated as content.
SESSION_HEADER_CANDIDATES = ("x-session-id", "x-claude-session-id",
                             "anthropic-session-id", "x-conversation-id")

SATURATION_MARKERS = (
    "all providers are saturated",
    "providers are saturated",
    "no available provider",
    "provider capacity",
    "all channels are busy",
)

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

# A header the list above does not know by name is still a credential if it
# reads like one. Matched as a substring so x-auth-token or x-vendor-secret is
# covered without enumerating every vendor's spelling; "key" alone is left out
# deliberately, or idempotency-key would be redacted for nothing.
_SENSITIVE_HEADER_HINTS = ("authorization", "api-key", "apikey", "token", "secret",
                           "password", "credential", "cookie", "signature")

# Query parameters that carry a credential rather than a routing detail. The
# NAME decides and the value is never inspected, so the redaction is predictable
# and a value that happens to look harmless cannot slip through.
SENSITIVE_QUERY_KEYS = frozenset({
    "api_key", "apikey", "key", "access_token", "refresh_token", "id_token",
    "token", "auth", "authorization", "secret", "client_secret", "password",
    "passwd", "pwd", "signature", "sig", "credential", "credentials",
})

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
    # Paths a browser asks for on its own, answered here instead of upstream.
    # Deliberately outside total_requests: they are not API traffic, and folding
    # them in would inflate the figure /_stats reports as "requests".
    "local_404_responses": 0,
    "failed_requests": 0,
    # 4xx broken out by kind. upstream_4xx above stays the aggregate.
    "upstream_400": 0,
    "upstream_401": 0,
    "upstream_403": 0,
    "upstream_429": 0,
    "upstream_429_saturated": 0,
    "upstream_other_4xx": 0,
    "retry_after_added": 0,
    # Reserved: no upstream status other than 429 is known to be a retryable
    # rate limit, so nothing is ever converted. Exposed so a future signature
    # can be counted without changing the shape of /_stats.
    "rate_limit_converted": 0,
    "effort_thinking_validation_errors": 0,
    # Content-blocked recovery. upstream_400_content_blocked counts upstream
    # responses; content_blocked_events counts recovery snapshots recorded.
    # A 400 is never retried, so in practice they move together.
    "upstream_400_content_blocked": 0,
    "content_blocked_events": 0,
    "content_blocked_sessions": 0,
    "recovery_checkpoints_saved": 0,
    "recovery_checkpoint_hits": 0,
    "recovery_checkpoint_misses": 0,
    "reliable_stream_requests": 0,
    "reliable_stream_completed": 0,
    "reliable_stream_retry_attempts": 0,
    "reliable_stream_recovered": 0,
    "reliable_stream_exhausted": 0,
    "reliable_stream_remote_protocol_errors": 0,
    "reliable_stream_incomplete_eof": 0,
    "reliable_stream_buffer_limit_exceeded": 0,
    "reliable_stream_bytes_buffered": 0,
    "reliable_stream_client_disconnects": 0,
}

_STARTED_AT = time.time()


def bump(key: str, n: int = 1) -> None:
    STATS[key] = STATS.get(key, 0) + n


# ----------------------------------------------------------------------------
# Logging (never emits credentials)
# ----------------------------------------------------------------------------

# /_events serves the proxy's own log lines out of this ring buffer rather than
# off disk: no filesystem path becomes reachable over HTTP, and the feed reads
# the same whether start-proxy.ps1 launched the proxy or a bare uvicorn did.
# Nothing lands here that print() would not already have written to the
# terminal, minus the masking below.
EVENT_BUFFER = max(50, int(os.environ.get("PROXY_EVENT_BUFFER", "600")))

_EVENTS: deque[dict[str, Any]] = deque(maxlen=EVENT_BUFFER)
_EVENT_SEQ = 0

# key_fingerprint() prints the last four characters of a credential. That is
# fine in a terminal the operator already owns; it is not something to serve
# over HTTP, so it is masked on the way into the buffer.
#
# The last pattern is a net rather than the first line of defence. A credential
# arriving as a query parameter is redacted where the request line is built (see
# redact_query), which keeps it out of the terminal and out of logs/proxy.log as
# well; this catches any later line that interpolates a name=value pair without
# going through that helper. The name survives the substitution: knowing that a
# caller passed api_key is the diagnosable half, its value never is.
_SECRET_MASKS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"tail=\.\.\.[^\s>]{1,8}"), "tail=...****"),
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{4,}"), "sk-ant-****"),
    # Other gateways issue sk-... without the vendor segment. Placed after the
    # pattern above so an Anthropic key still reads as one, and long enough that
    # no ordinary word starting with "sk-" is caught.
    (re.compile(r"(?<!\w)sk-[A-Za-z0-9_\-]{16,}"), "sk-****"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{4,}"), "Bearer ****"),
    # Basic holds "user:password" in base64, so the scheme name is not the part
    # worth keeping quiet -- what follows it is. Without this, the name=value net
    # below stops at the space and leaves the credential itself in place.
    (re.compile(r"(?i)\bbasic\s+[A-Za-z0-9+/=]{8,}"), "Basic ****"),
    # (?<!\w) keeps max_tokens=1024 and stream=True readable -- an underscore or
    # letter in front means this is a longer word, not one of these names -- while
    # still reaching the value in x-api-key=... and refresh-token=...
    (re.compile(r"(?i)(?<!\w)(api[-_]?key|access[-_]?token|refresh[-_]?token"
                r"|id[-_]?token|client[-_]?secret|authorization|auth|token"
                r"|secret|password|passwd|pwd|signature|credentials?)"
                r"=([^&\s\"']+)"), r"\1=****"),
)

# What a reader groups a line under. First match wins, so failure phrasings come
# before the generic status lines they would otherwise be swallowed by.
_EVENT_KINDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?:GET|POST|PUT|PATCH|DELETE|OPTIONS) \S+ ->"), "request"),
    (re.compile(r"^(?:final failure|all \d+ attempts failed|non-transient failure"
                r"|no credential available|post-commit|HTTP 5\d\d)"), "fail"),
    (re.compile(r"^(?:retrying in|invalid response|discarding unusable"
                r"|client disconnected|upstream \d+ carried no usable"
                r"|forwarding upstream HTTP|HTTP 4\d\d|content blocked)"), "warn"),
    (re.compile(r"^(?:HTTP 2\d\d|stream complete|buffered stream complete"
                r"|valid Anthropic SSE"
                r"|forwarding stream|non-stream response OK"
                r"|recovered on attempt)"), "ok"),
)

_STREAM_DONE_RE = re.compile(r"^stream complete: (\d+) frames in ([\d.]+)s")
_BUFFERED_DONE_RE = re.compile(r"^buffered stream complete: (\d+) bytes in ([\d.]+)s")
_HTTP_STATUS_RE = re.compile(r"^HTTP (\d{3})")
_REQUEST_RE = re.compile(r"^(GET|POST|PUT|PATCH|DELETE|OPTIONS) (\S+) ->")


def mask_secrets(msg: str) -> str:
    for pattern, replacement in _SECRET_MASKS:
        msg = pattern.sub(replacement, msg)
    return msg


def record_event(msg: str) -> None:
    """Keep a masked, classified copy of a log line for /_events."""
    global _EVENT_SEQ
    if msg.startswith("=========="):  # separator: useful in a terminal, noise in a feed
        return

    text = mask_secrets(msg)
    kind = "info"
    for pattern, name in _EVENT_KINDS:
        if pattern.match(text):
            kind = name
            break

    _EVENT_SEQ += 1
    event: dict[str, Any] = {"seq": _EVENT_SEQ, "t": round(time.time(), 3),
                             "kind": kind, "text": text}

    # Extras worth having without re-parsing the text downstream: a finished
    # stream carries how long it took, a request line its method and path, a
    # status line its code.
    done = _STREAM_DONE_RE.match(text)
    if done:
        event["frames"] = int(done.group(1))
        event["seconds"] = float(done.group(2))
    buffered = _BUFFERED_DONE_RE.match(text)
    if buffered:
        event["bytes"] = int(buffered.group(1))
        event["seconds"] = float(buffered.group(2))
    status = _HTTP_STATUS_RE.match(text)
    if status:
        event["status"] = int(status.group(1))
    request = _REQUEST_RE.match(text)
    if request:
        event["method"] = request.group(1)
        event["path"] = request.group(2)

    _EVENTS.append(event)


def log(msg: str) -> None:
    record_event(msg)
    print(f"[proxy] {msg}", flush=True)


def vlog(msg: str) -> None:
    if VERBOSE:
        record_event(msg)
        print(f"[proxy] {msg}", flush=True)


def is_sensitive_header(name: str) -> bool:
    """Whether this header's value must never be printed."""
    lowered = name.lower()
    return (lowered in SENSITIVE_HEADERS
            or any(hint in lowered for hint in _SENSITIVE_HEADER_HINTS))


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """
    Header names preserved, sensitive values replaced by a length hint.

    A name this does not recognize as a credential still gets its value run
    through mask_secrets: location and referer carry a full URL, and a URL is
    allowed to carry ?api_key=. Ordinary values -- content-type, a byte count, a
    request id -- match none of those patterns and come back unchanged.
    """
    out: dict[str, str] = {}
    for k, v in headers.items():
        if is_sensitive_header(k):
            out[k] = f"<redacted len={len(v)}>"
        else:
            out[k] = mask_secrets(v)
    return out


# Parameters are separated by "&", and historically also by ";" -- a form
# modern parsers no longer split on, but one an intermediary may still honour.
# Splitting on both keeps a value from hiding behind the separator this code
# happens not to recognize; the capturing group means the separators come back
# in the output exactly as they arrived.
_QUERY_SPLIT_RE = re.compile(r"([&;])")


def query_key(name: str) -> str:
    """
    A parameter name reduced to the form SENSITIVE_QUERY_KEYS is written in.

    The name on the wire may be percent-encoded, and the server decodes it before
    it means anything: api%5Fkey, api+key and API-KEY all reach the upstream as
    api_key. Classifying the raw spelling therefore lets an encoded name walk a
    credential straight through, so the name is decoded first -- for the decision
    only. What gets printed keeps the spelling the caller used.
    """
    return re.sub(r"[-\s]+", "_", unquote_plus(name).strip().lower())


def redact_query(query: str) -> str:
    """
    A query string safe to print: credential values gone, parameter names kept.

    The upstream still receives the query verbatim -- this is only what the
    terminal, logs/proxy.log and /_events get to see. Names survive because they
    are the diagnosable half: that a caller passed api_key at all is worth
    seeing, its value never is. Only the names in SENSITIVE_QUERY_KEYS are
    touched, so beta=true stays readable.

    The placeholder holds no spaces on purpose: a request line is parsed back
    into method and path by whitespace (see _REQUEST_RE), and a length hint like
    the one redact_headers uses would split the path in two.
    """
    if not query:
        return query
    # Odd positions are the separators the regex captured; even ones are the
    # parameters between them.
    tokens = _QUERY_SPLIT_RE.split(query)
    for i in range(0, len(tokens), 2):
        name, sep, value = tokens[i].partition("=")
        if sep and value and query_key(name) in SENSITIVE_QUERY_KEYS:
            tokens[i] = f"{name}=****"
    return "".join(tokens)


def safe_upstream(url: str) -> str:
    """
    The upstream address in a form that is safe to print.

    A base URL is normally credential-free, but nothing stops an operator from
    configuring one that carries userinfo or a key in the query string -- and
    this string is printed at startup, on every request line, and served by
    /_health, /_stats and /. The address the requests actually go to is built
    from UPSTREAM itself; only what gets shown passes through here.

    Splitting this by hand was wrong for a base URL with no path: everything
    after the host, query included, landed in the authority and was printed
    verbatim, so https://gw.example?api-key=... leaked. urlsplit knows where a
    query starts whether or not a path precedes it.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        # An address urlsplit refuses (an unclosed IPv6 bracket, say) is one
        # httpx will refuse too, so the proxy is not going anywhere either way.
        # Printing the raw string to explain that would defeat the whole point
        # of this function, so it stays unprinted and .env is where to look.
        return "<unparsable upstream address>"

    netloc = parts.netloc
    if "@" in netloc:  # user:password@host
        netloc = f"****@{netloc.rsplit('@', 1)[1]}"
    # A fragment never reaches a server and has no business holding a secret,
    # but it is printed, so it is treated like the query rather than trusted.
    return urlunsplit((parts.scheme, netloc, parts.path,
                       redact_query(parts.query), redact_query(parts.fragment)))


# Computed once: every place that prints the upstream uses this, never UPSTREAM.
UPSTREAM_SAFE = safe_upstream(UPSTREAM)


def key_fingerprint(key: str) -> str:
    """Non-reversible-enough hint for debugging. Never the key itself."""
    if not key:
        return "<none>"
    return f"<len={len(key)} tail=...{key[-4:]}>"


def preview(body: bytes) -> str:
    """
    Sanitized diagnostic for an unusable body: structural shape only.

    Never returns prompt or completion text. An unusable upstream body is
    typically an error page or a gateway notice, but it could carry echoed
    request content or a partial completion, so nothing here quotes it: each
    branch reports a classification and measurements, and the fallback for a
    shape nobody anticipated does the same rather than printing the head.
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

    # A truncated SSE stream is not JSON, so without this it would fall through
    # to the raw head below -- and the head of a stream is where completion text
    # lives. Event names answer the diagnostic question ("how far did it get?")
    # without reading a single data payload.
    if stripped.startswith("event:") or stripped.startswith("data:"):
        names: list[str] = []
        frames = 0
        for line in stripped.split("\n"):
            if line.startswith("event:"):
                names.append(line[6:].strip())
            elif line.startswith("data:"):
                frames += 1
        ordered = ",".join(list(dict.fromkeys(n for n in names if n))[:8]) or "<unnamed>"
        return f"<sse {len(body)}B, frames={frames}, events={ordered}>"

    # Anything else. This branch used to print the head of the payload with
    # whitespace collapsed, which is precisely where an echoed prompt or a
    # partial completion would appear -- the one shape that is unclassified is
    # also the one nothing is known about, so quoting it was the weakest link in
    # a function whose whole promise is that it never quotes. Measurements answer
    # the diagnostic question -- how big, one line or many, text or binary, plain
    # ASCII or not -- without reproducing a character of the body.
    # U+FFFD is what errors="replace" left behind, one per byte sequence that is
    # not valid UTF-8; a body mostly made of them is not text at all.
    undecodable = stripped.count("\ufffd")
    if undecodable * 10 > len(stripped):
        return f"<binary {len(body)}B, {undecodable} undecodable sequences>"
    lines = stripped.count("\n") + 1
    words = len(stripped.split())
    shape = "markup" if stripped.startswith("<") else "text"
    charset = "ascii" if stripped.isascii() else "non-ascii"
    return (f"<non-json {len(body)}B, {shape}, {charset}, "
            f"lines={lines}, words={words}>")


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


def json_body_expected(content_type: str) -> bool:
    """
    Whether a body under this content-type is still owed a JSON shape.

    Genuine API replies arrive as application/json and, on some routes, as
    text/plain -- AgentRouter has been observed serving a well-formed message
    that way -- so both stay under the JSON contract, as does a reply that names
    no type at all. A body that names some other type is not an API reply: the
    upstream serves /favicon.ico as image/x-icon and a landing page as text/html.
    Judging those as malformed JSON condemns a sound response.
    """
    ct = (content_type or "").split(";")[0].strip().lower()
    if not ct:
        return True
    return ct in ("application/json", "text/plain") or ct.endswith("+json")


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


async def reliable_stream_body(response: httpx.Response, client: httpx.AsyncClient) -> AttemptResult:
    """Buffer one complete SSE attempt before allowing a client commit."""
    stream_iter = response.aiter_bytes().__aiter__()
    carry = b""
    accepted: list[bytes] = []
    seen: list[str] = []
    saw_message_stop = False
    saw_content = False
    total = 0
    deadline = time.monotonic() + READ_TIMEOUT_S

    def invalid(reason: str, counter: str = "malformed_streams") -> AttemptResult:
        bump(counter)
        return AttemptResult(ok=False, reason=reason, transient=True,
                             status=response.status_code, response=response,
                             client=client, headers=dict(response.headers))

    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return invalid("reliable_stream_timeout")
            try:
                chunk = await asyncio.wait_for(stream_iter.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                return invalid("reliable_stream_timeout")
            except (httpx.HTTPError, OSError) as exc:
                if isinstance(exc, httpx.RemoteProtocolError):
                    bump("reliable_stream_remote_protocol_errors")
                return invalid(f"reliable_stream_broke:{type(exc).__name__}")

            total += len(chunk)
            bump("reliable_stream_bytes_buffered", len(chunk))
            if total > RELIABLE_MAX_BYTES:
                return invalid("reliable_stream_buffer_limit_exceeded", "reliable_stream_buffer_limit_exceeded")
            carry += chunk
            while b"\n\n" in carry:
                raw_frame, carry = carry.split(b"\n\n", 1)
                if not raw_frame.strip():
                    continue
                text = raw_frame.decode("utf-8", errors="replace")
                etype, data, parseable = parse_frame(text)
                seen.append(etype or "<untyped>")
                reason = drop_reason(etype, data, parseable)
                if reason:
                    bump("dropped_sse_frames")
                    log(f"dropped SSE frame: type={etype or '<none>'} reason={reason}")
                    continue
                accepted.append(raw_frame + b"\n\n")
                if etype == "message_stop":
                    saw_message_stop = True
                elif etype in COMMIT_EVENTS:
                    saw_content = True

        if carry.strip():
            # Preserve the existing EOF-tail behavior, while retaining the
            # original bytes for each complete candidate.
            tail_text = carry.decode("utf-8", errors="replace")
            candidates = split_trailing_frames(tail_text)
            for candidate in candidates:
                etype, data, parseable = parse_frame(candidate)
                reason = drop_reason(etype, data, parseable)
                if reason:
                    bump("dropped_sse_frames")
                    continue
                accepted.append(candidate.encode("utf-8") + b"\n\n")
                if etype == "message_stop":
                    saw_message_stop = True
                elif etype in COMMIT_EVENTS:
                    saw_content = True

        if not saw_message_stop or not saw_content:
            return invalid("reliable_stream_incomplete_eof", "reliable_stream_incomplete_eof")
        body = b"".join(accepted)
        bump("reliable_stream_completed")
        return AttemptResult(ok=True, reason="reliable_stream_complete", status=200,
                             body=body, headers=dict(response.headers), is_sse=True)
    finally:
        await close_quietly(response, client)


# ----------------------------------------------------------------------------
# Non-streaming validation
# ----------------------------------------------------------------------------


def validate_json_response(status: int, body: bytes, is_messages: bool,
                           content_type: str = "") -> tuple[bool, str]:
    """
    Validate a non-streaming upstream body. Only HTTP 200 is validated for
    shape; non-2xx bodies are real errors and are forwarded verbatim.
    """
    if status != 200:
        return True, "non_200_forwarded_verbatim"

    # A path outside the JSON API may legitimately answer with something else,
    # and the catch-all route forwards every path. Parsing /favicon.ico's
    # image/x-icon as JSON marks a sound 200 invalid, spends all three attempts
    # re-fetching the same bytes and answers 502 -- which a client reads as a
    # dead endpoint, and it reads it precisely while probing whether the endpoint
    # is alive. The API's own replies are still held to the contract below.
    if not is_messages and not json_body_expected(content_type):
        return True, "non_json_content_type_forwarded_verbatim"

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
        classify_client_error(status, raw)
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
        classify_client_error(status, raw)
        return AttemptResult(ok=True, reason=f"http_{status}", status=status, body=raw,
                             headers=response_headers, is_sse=False)

    content_type = response.headers.get("content-type", "")
    is_sse = "text/event-stream" in content_type.lower()

    # --- Streaming path -----------------------------------------------------
    if stream_expected and is_sse:
        if STREAM_MODE == "reliable":
            return await reliable_stream_body(response, client)
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

    valid, reason = validate_json_response(status, raw, is_messages, content_type)
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


def upstream_error_text(body: bytes) -> str:
    """
    The upstream error envelope's own wording, lowercased and concatenated.

    Only the fields of a JSON *error object* are read. Assistant content is
    never inspected, so a reply that happens to discuss rate limits or
    saturation can never be classified as one.
    """
    try:
        data = json.loads((body or b"").decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""

    parts: list[str] = []
    err = data.get("error")
    if isinstance(err, dict):
        for key in ("message", "type", "code", "rule_id", "detail", "param"):
            value = err.get(key)
            if isinstance(value, str):
                parts.append(value)
    elif isinstance(err, str):
        parts.append(err)
    for key in ("message", "detail", "rule_id"):
        value = data.get(key)
        if isinstance(value, str):
            parts.append(value)
    return " ".join(parts).lower()


def is_saturation(body: bytes) -> bool:
    """Whether a 429 carries the upstream's provider-capacity wording."""
    text = upstream_error_text(body)
    return any(marker in text for marker in SATURATION_MARKERS)


def is_effort_thinking_error(body: bytes) -> bool:
    """
    Recognise the permanent 400 that rejects output_config.effort unless
    thinking is enabled. Retrying a schema rejection only repeats it, and
    converting it into a 429 would hide a real client/upstream incompatibility
    behind an infinite retry loop -- so this is recorded and forwarded, never
    acted on.
    """
    text = upstream_error_text(body)
    if "claude_effort_requires_thinking" in text:
        return True
    return "output_config.effort" in text and "thinking" in text


def retry_after_is_usable(value: Optional[str]) -> bool:
    """
    Whether an upstream Retry-After can be forwarded untouched. Both RFC 9110
    forms count: delta-seconds, and an HTTP-date.
    """
    if not value:
        return False
    raw = value.strip()
    try:
        return int(raw) >= 0
    except ValueError:
        return parsedate_tz(raw) is not None


def classify_client_error(status: int, body: bytes) -> None:
    """
    Record what kind of 4xx arrived. Counters and log lines only -- the status
    code, the body bytes and the (non-)retry decision are all left exactly as
    they were.
    """
    if status == 400:
        bump("upstream_400")
        if is_effort_thinking_error(body):
            bump("effort_thinking_validation_errors")
            log("upstream rejected output_config.effort because thinking is not "
                "enabled (claude_effort_requires_thinking) -- permanent schema "
                "error, forwarded unchanged and never retried")
        elif is_content_blocked(status, body):
            bump("upstream_400_content_blocked")
            log("upstream rejected the request as content-blocked (HTTP 400) -- "
                "a real moderation decision, forwarded verbatim and never "
                "retried; the proxy records only structural recovery metadata")
    elif status == 401:
        bump("upstream_401")
    elif status == 403:
        bump("upstream_403")
    elif status == 429:
        bump("upstream_429")
        if is_saturation(body):
            bump("upstream_429_saturated")
            log("upstream reports provider saturation (HTTP 429) -- forwarding "
                "the real 429; the client owns the retry schedule")
    else:
        bump("upstream_other_4xx")


def describe_request_shape(body: bytes) -> str:
    """
    Structural summary of an outgoing request, for diagnosing schema rejections
    like claude_effort_requires_thinking.

    Field NAMES and small enum values only. Prompt text, system prompt, tool
    names, tool arguments, tool results, thinking text, file paths and the raw
    body are excluded by construction: nothing below reads a value that could
    carry user content.
    """
    try:
        data = json.loads((body or b"").decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return "unparseable"
    if not isinstance(data, dict):
        return "not_a_json_object"

    model = data.get("model")
    parts = [
        f"model={model if isinstance(model, str) else '<none>'}",
        f"stream={data.get('stream') is True}",
        f"top_level_keys={','.join(sorted(data.keys()))}",
        f"thinking_present={'thinking' in data}",
    ]

    thinking = data.get("thinking")
    if isinstance(thinking, dict):
        t_type = thinking.get("type")
        parts.append(
            f"thinking.type={t_type if isinstance(t_type, str) else '<non-string>'}")
        parts.append(f"thinking.budget_tokens_present={'budget_tokens' in thinking}")

    parts.append(f"output_config_present={'output_config' in data}")
    output_config = data.get("output_config")
    if isinstance(output_config, dict):
        parts.append(
            f"output_config_keys={','.join(sorted(output_config.keys()))}")
        parts.append(f"output_config.effort_present={'effort' in output_config}")
        effort = output_config.get("effort")
        if isinstance(effort, str):
            parts.append(f"output_config.effort={effort}")
    return " ".join(parts)


def is_content_blocked(status: int, body: bytes) -> bool:
    """
    Whether a 400 is AgentRouter's moderation rejection.

    Only ever consulted for HTTP 400, and it reads the upstream *error
    envelope*, never assistant output -- which arrives under HTTP 200 and so
    cannot reach this function at all. A reply discussing the phrase
    "content-blocked" is therefore never classified as one.

    The non-JSON branch exists because the observed rejection reaches the client
    as `API Error: 400 content-blocked`, i.e. a bare token rather than a JSON
    envelope. It is deliberately strict: the entire body, minus punctuation and
    whitespace, must itself be the marker. A body that merely *contains* the
    phrase somewhere is not matched.
    """
    if status != 400:
        return False

    text = upstream_error_text(body)
    if text:
        return any(marker in text for marker in CONTENT_BLOCKED_MARKERS)

    # No parseable error envelope: fall back to an exact whole-body token.
    try:
        raw = (body or b"").decode("utf-8", errors="replace").strip()
    except Exception:
        return False
    if len(raw) > 64:
        return False
    normalized = raw.strip("\"' \t\r\n.").lower()
    return normalized in CONTENT_BLOCKED_MARKERS


# ----------------------------------------------------------------------------
# Content-blocked recovery: bounded, structural, content-free
# ----------------------------------------------------------------------------

# Per-process salt. Fingerprints are used to group a conversation's turns and to
# show that structure changed between the last accepted turn and the rejected
# one. Salting them means the digests are meaningless outside this process and
# cannot be compared against precomputed hashes of known text.
_LANE_SALT = secrets.token_bytes(16)

# lane -> last known good checkpoint. OrderedDict gives FIFO eviction.
_CHECKPOINTS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_LAST_BLOCKED: Optional[dict[str, Any]] = None
_BLOCKED_LANES: set[str] = set()


def _digest(*parts: str) -> str:
    h = hashlib.sha256(_LANE_SALT)
    for part in parts:
        h.update(part.encode("utf-8", errors="replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def session_id_from_headers(headers: dict[str, str]) -> Optional[str]:
    """
    A client-supplied conversation id, if one exists.

    Claude Code is not known to send one. When absent the caller falls back to a
    structural lane, so recovery still works -- it just cannot name the session.
    """
    for key, value in (headers or {}).items():
        if key.lower() in SESSION_HEADER_CANDIDATES:
            candidate = (value or "").strip()
            if candidate and len(candidate) <= 128:
                return candidate
    return None


def request_structure(body: bytes) -> dict[str, Any]:
    """
    Safe structural metadata for one request.

    Every value below is a count, a byte total, a boolean, a small enum, or a
    salted digest. No prompt text, system text, thinking text, tool name, tool
    argument, tool result, filename, command or URL is read, and the raw body is
    never retained -- the returned dict is the only thing that outlives the call.
    """
    out: dict[str, Any] = {
        "parsed": False, "model": None, "stream": False, "message_count": 0,
        "role_sequence": [], "block_type_sequence": [], "text_bytes": 0,
        "tool_result_bytes": 0, "tool_use_count": 0, "tool_result_count": 0,
        "tool_result_error_count": 0, "thinking_present": False,
        "thinking_type": None, "output_config_present": False,
        "output_config_effort": None, "system_present": False,
        "system_bytes": 0, "tool_definition_count": 0,
    }
    try:
        data = json.loads((body or b"").decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return out
    if not isinstance(data, dict):
        return out
    out["parsed"] = True

    model = data.get("model")
    out["model"] = model if isinstance(model, str) else None
    out["stream"] = data.get("stream") is True

    system = data.get("system")
    out["system_present"] = "system" in data
    if isinstance(system, str):
        out["system_bytes"] = len(system.encode("utf-8", errors="replace"))
    elif isinstance(system, list):
        out["system_bytes"] = sum(
            len(str(b.get("text", "")).encode("utf-8", errors="replace"))
            for b in system if isinstance(b, dict))

    tools = data.get("tools")
    if isinstance(tools, list):
        out["tool_definition_count"] = len(tools)

    thinking = data.get("thinking")
    out["thinking_present"] = "thinking" in data
    if isinstance(thinking, dict):
        t_type = thinking.get("type")
        out["thinking_type"] = t_type if isinstance(t_type, str) else None

    output_config = data.get("output_config")
    out["output_config_present"] = "output_config" in data
    if isinstance(output_config, dict):
        effort = output_config.get("effort")
        out["output_config_effort"] = effort if isinstance(effort, str) else None

    messages = data.get("messages")
    if not isinstance(messages, list):
        return out
    out["message_count"] = len(messages)

    roles: list[str] = []
    block_types: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            roles.append("?")
            continue
        role = msg.get("role")
        roles.append(role if isinstance(role, str) else "?")

        content = msg.get("content")
        if isinstance(content, str):
            block_types.append("text")
            out["text_bytes"] += len(content.encode("utf-8", errors="replace"))
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            block_types.append(btype if isinstance(btype, str) else "?")
            if btype == "text":
                out["text_bytes"] += len(
                    str(block.get("text", "")).encode("utf-8", errors="replace"))
            elif btype == "thinking":
                out["text_bytes"] += len(
                    str(block.get("thinking", "")).encode("utf-8", errors="replace"))
            elif btype == "tool_use":
                out["tool_use_count"] += 1
            elif btype == "tool_result":
                out["tool_result_count"] += 1
                if block.get("is_error") is True:
                    out["tool_result_error_count"] += 1
                payload = block.get("content")
                if isinstance(payload, str):
                    out["tool_result_bytes"] += len(
                        payload.encode("utf-8", errors="replace"))
                elif isinstance(payload, list):
                    out["tool_result_bytes"] += sum(
                        len(str(p.get("text", "")).encode("utf-8", errors="replace"))
                        for p in payload if isinstance(p, dict))

    out["role_sequence"] = roles[-24:]
    out["block_type_sequence"] = block_types[-40:]
    return out


def lane_key(structure: dict[str, Any], session: Optional[str]) -> str:
    """
    The bucket a turn belongs to.

    A real session id is used when the client supplies one. Otherwise the lane
    is a salted digest of the conversation's opening shape -- model plus the
    role/size profile of its first turns -- which stays stable as the
    conversation grows and differs between concurrent conversations. It is
    derived from counts and lengths, never from text.
    """
    if session:
        return f"session:{session}"
    roles = structure.get("role_sequence") or []
    return "shape:" + _digest(
        str(structure.get("model")),
        ",".join(roles[:2]),
        str(structure.get("system_bytes", 0)),
        str(structure.get("tool_definition_count", 0)),
    )


def structural_fingerprint(structure: dict[str, Any]) -> str:
    """A digest of the whole current shape, for showing that it changed."""
    return _digest(
        str(structure.get("model")),
        ",".join(structure.get("role_sequence") or []),
        ",".join(structure.get("block_type_sequence") or []),
        str(structure.get("message_count", 0)),
        str(structure.get("text_bytes", 0)),
        str(structure.get("tool_result_bytes", 0)),
    )


def checkpoint_view(structure: dict[str, Any], session: Optional[str],
                    lane: str, request_id: str, ts: float,
                    message_stop: Optional[bool]) -> dict[str, Any]:
    """The safe record stored and served. Counts, sizes, enums and digests."""
    return {
        "timestamp": _iso(ts),
        "proxy_request_id": request_id,
        "session_id": session,
        "session_id_source": "client_header" if session else None,
        "lane": lane,
        "lane_kind": "session" if session else "structural",
        "model": structure.get("model"),
        "stream": structure.get("stream"),
        "message_count": structure.get("message_count"),
        "role_sequence": structure.get("role_sequence"),
        "block_type_sequence": structure.get("block_type_sequence"),
        "text_bytes": structure.get("text_bytes"),
        "tool_result_bytes": structure.get("tool_result_bytes"),
        "system_bytes": structure.get("system_bytes"),
        "tool_use_count": structure.get("tool_use_count"),
        "tool_result_count": structure.get("tool_result_count"),
        "tool_result_error_count": structure.get("tool_result_error_count"),
        "tool_definition_count": structure.get("tool_definition_count"),
        "thinking_present": structure.get("thinking_present"),
        "thinking_type": structure.get("thinking_type"),
        "output_config_present": structure.get("output_config_present"),
        "output_config_effort": structure.get("output_config_effort"),
        "structural_fingerprint": structural_fingerprint(structure),
        "message_stop_observed": message_stop,
        "success": True,
    }


def save_checkpoint(pending: Optional[dict[str, Any]],
                    message_stop: Optional[bool]) -> None:
    """
    Record a turn the upstream accepted as this lane's last known good state.

    Called only after success. Bounded by MAX_RECOVERY_CHECKPOINTS with FIFO
    eviction, so a long-lived proxy cannot grow without limit.
    """
    if not pending:
        return
    structure = pending.get("structure") or {}
    if not structure.get("parsed"):
        return
    lane = pending["lane"]
    _CHECKPOINTS[lane] = checkpoint_view(
        structure, pending.get("session"), lane, pending["request_id"],
        pending["ts"], message_stop)
    _CHECKPOINTS.move_to_end(lane)
    while len(_CHECKPOINTS) > MAX_RECOVERY_CHECKPOINTS:
        _CHECKPOINTS.popitem(last=False)
    bump("recovery_checkpoints_saved")


def record_content_blocked(pending: Optional[dict[str, Any]],
                           upstream_request_id: Optional[str]) -> None:
    """
    Snapshot a moderation rejection next to the lane's last accepted turn.

    Diagnostic only. The 400, its body and the decision not to retry are
    unaffected by anything here.
    """
    global _LAST_BLOCKED
    bump("content_blocked_events")
    if not pending:
        return

    structure = pending.get("structure") or {}
    lane = pending["lane"]
    last_good = _CHECKPOINTS.get(lane)
    if last_good is None:
        bump("recovery_checkpoint_misses")
    else:
        bump("recovery_checkpoint_hits")
    if lane not in _BLOCKED_LANES:
        _BLOCKED_LANES.add(lane)
        bump("content_blocked_sessions")

    current_fp = structural_fingerprint(structure)
    _LAST_BLOCKED = {
        "timestamp": _iso(pending["ts"]),
        "proxy_request_id": pending["request_id"],
        "upstream_request_id": upstream_request_id,
        "session_id": pending.get("session"),
        "lane": lane,
        "lane_kind": "session" if pending.get("session") else "structural",
        "model": structure.get("model"),
        "blocked_message_count": structure.get("message_count"),
        "last_good_message_count": (last_good or {}).get("message_count"),
        "blocked_fingerprint": current_fp,
        "last_good_fingerprint": (last_good or {}).get("structural_fingerprint"),
        "structural_change": bool(
            last_good and last_good.get("structural_fingerprint") != current_fp),
        "role_sequence": structure.get("role_sequence"),
        "last_good_role_sequence": (last_good or {}).get("role_sequence"),
        "block_type_sequence": structure.get("block_type_sequence"),
        "last_good_block_type_sequence": (last_good or {}).get("block_type_sequence"),
        "text_bytes": structure.get("text_bytes"),
        "last_good_text_bytes": (last_good or {}).get("text_bytes"),
        "tool_result_bytes": structure.get("tool_result_bytes"),
        "last_good_tool_result_bytes": (last_good or {}).get("tool_result_bytes"),
        "tool_use_count": structure.get("tool_use_count"),
        "last_good_tool_use_count": (last_good or {}).get("tool_use_count"),
        "tool_result_count": structure.get("tool_result_count"),
        "last_good_tool_result_count": (last_good or {}).get("tool_result_count"),
        "tool_result_error_count": structure.get("tool_result_error_count"),
        "thinking_present": structure.get("thinking_present"),
        "thinking_type": structure.get("thinking_type"),
        "output_config_present": structure.get("output_config_present"),
        "output_config_effort": structure.get("output_config_effort"),
        "have_last_good_checkpoint": last_good is not None,
        "proxy_retry": False,
        "request_modified": False,
    }

    log("[CONTENT_BLOCKED] "
        f"request_id={pending['request_id']} "
        f"upstream_request_id={upstream_request_id or '<none>'} "
        f"session={pending.get('session') or '<none:structural-lane>'} "
        f"lane={lane} "
        f"last_good_messages={(last_good or {}).get('message_count', '<none>')} "
        f"blocked_messages={structure.get('message_count')} "
        f"structural_change={'yes' if _LAST_BLOCKED['structural_change'] else 'no'} "
        f"proxy_retry=no request_modified=no")


def suggested_action(last_good: Optional[dict[str, Any]]) -> str:
    """
    The supported Claude Code action to recover with.

    Only flags this CLI actually documents are named: -c/--continue,
    -r/--resume, --fork-session. The proxy cannot select a turn for the user --
    Claude Code exposes no flag to resume at a specific message -- so it does
    not pretend to.
    """
    if last_good is None:
        return ("No accepted turn was recorded for this conversation. Start a new "
                "Claude Code session, or run `claude --resume` to pick an earlier "
                "session from the interactive picker.")
    session = last_good.get("session_id")
    where = (f"The last turn AgentRouter accepted had "
             f"{last_good.get('message_count')} messages at "
             f"{last_good.get('timestamp')}.")
    if session:
        return (f"{where} Resume it with `claude --resume {session}`, or "
                f"`claude --resume {session} --fork-session` to branch instead of "
                f"continuing in place. In the session, rewind with /rewind and "
                f"resend without the rejected content.")
    return (f"{where} Claude Code does not expose its session id to the proxy, so "
            f"the proxy cannot name it. Recover with `claude --continue` for the "
            f"most recent conversation in this directory, or `claude --resume` to "
            f"choose from the picker; then use /rewind to step back to before the "
            f"rejected message and resend it differently. The proxy neither "
            f"retried nor altered the blocked request.")


def upstream_request_id(headers: dict[str, str]) -> Optional[str]:
    """
    The upstream's own trace id for a rejected request, if it sent one.

    Opaque identifier only -- useful when asking AgentRouter about a specific
    rejection, and carries no conversation content.
    """
    for name in ("request-id", "x-request-id", "cf-ray", "x-trace-id"):
        for key, value in (headers or {}).items():
            if key.lower() == name and value:
                return value[:128]
    return None


def recovery_state() -> dict[str, Any]:
    """Payload for /_recovery. Safe metadata only -- see checkpoint_view."""
    last_good = next(reversed(_CHECKPOINTS.values()), None) if _CHECKPOINTS else None
    if _LAST_BLOCKED is not None:
        lane_good = _CHECKPOINTS.get(_LAST_BLOCKED.get("lane", ""))
        if lane_good is not None:
            last_good = lane_good
        status = "content_blocked"
    elif last_good is not None:
        status = "ready"
    else:
        status = "no_checkpoint"
    return {
        "status": status,
        "last_good": last_good,
        "last_blocked": _LAST_BLOCKED,
        "suggested_action": suggested_action(last_good),
        "checkpoints_held": len(_CHECKPOINTS),
        "max_checkpoints": MAX_RECOVERY_CHECKPOINTS,
        "session_id_available": bool(last_good and last_good.get("session_id")),
        "notice": ("Structural metadata only. This endpoint never stores or "
                   "returns prompts, system text, thinking text, tool names, "
                   "tool arguments, tool results, filenames or raw bodies."),
    }


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

    if stream_expected and STREAM_MODE == "reliable":
        bump("reliable_stream_requests")

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
                if stream_expected and STREAM_MODE == "reliable":
                    bump("reliable_stream_recovered")
                log(f"recovered on attempt {attempt}")
            return result

        await close_quietly(result.response, result.client)
        result.response = None
        result.client = None

        if not result.transient:
            log(f"non-transient failure ({result.reason}) -- not retrying")
            return result

        if attempt >= MAX_ATTEMPTS:
            if stream_expected and STREAM_MODE == "reliable":
                bump("reliable_stream_exhausted")
            log(f"all {MAX_ATTEMPTS} attempts failed ({result.reason})")
            return result

        delay_ms = BACKOFF_BASE_MS * (2 ** (attempt - 1))
        bump("retries")
        if stream_expected and STREAM_MODE == "reliable":
            bump("reliable_stream_retry_attempts")
        log(f"retrying in {delay_ms}ms")
        await asyncio.sleep(delay_ms / 1000.0)

    return last or AttemptResult(ok=False, reason="no_attempt_made")


async def run_reliable_request(request: Request, **kwargs: Any) -> Optional[AttemptResult]:
    """Cancel a buffered upstream attempt as soon as its client disconnects."""
    upstream = asyncio.create_task(run_with_retries(**kwargs))

    async def disconnected() -> None:
        while not await request.is_disconnected():
            await asyncio.sleep(0.05)

    watcher = asyncio.create_task(disconnected())
    done, _ = await asyncio.wait((upstream, watcher), return_when=asyncio.FIRST_COMPLETED)
    if upstream in done:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        return upstream.result()

    upstream.cancel()
    await asyncio.gather(upstream, return_exceptions=True)
    bump("reliable_stream_client_disconnects")
    log("client disconnected while reliable stream was buffering -- upstream cancelled")
    return None


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


async def forward_stream(result: AttemptResult,
                         pending: Optional[dict[str, Any]] = None) -> AsyncIterator[bytes]:
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
        # A stream is only known good once it has completed, so the checkpoint
        # is written here rather than when the response headers arrived.
        save_checkpoint(pending, True)
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
    payload: dict[str, Any] = {
        **STATS,
        "uptime_seconds": round(time.time() - _STARTED_AT, 1),
        "upstream": UPSTREAM_SAFE,
        "max_attempts": MAX_ATTEMPTS,
        "backoff_base_ms": BACKOFF_BASE_MS,
        "prime_timeout_s": PRIME_TIMEOUT_S,
        "stream_mode": STREAM_MODE,
        "reliable_max_bytes": RELIABLE_MAX_BYTES,
    }
    # STATS holds counters only; the timestamp is merged in here so bump() stays
    # integer-typed. Structural metadata, never content.
    if _LAST_BLOCKED is not None:
        payload["last_content_blocked_at"] = _LAST_BLOCKED.get("timestamp")
    return JSONResponse(payload)


@app.get("/_recovery")
async def recovery() -> JSONResponse:
    """
    Where the last accepted turn was, and what supported action recovers it.

    Safe metadata only -- counts, byte totals, enums and salted digests. No
    prompts, no tool data, no credentials, no raw bodies.
    """
    return JSONResponse(recovery_state())


@app.get("/_health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "upstream": UPSTREAM_SAFE,
                         "listen": f"{LISTEN_HOST}:{PORT}",
                         "stream_mode": STREAM_MODE,
                         "reliable_max_bytes": RELIABLE_MAX_BYTES})


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "status": "running",
            "upstream": UPSTREAM_SAFE,
            "endpoints": ["/v1/messages", "/v1/messages?beta=true", "/_stats",
                          "/_health", "/_recovery", "/_events", "/_ui"],
        }
    )


@app.get("/_events")
async def events(after: int = 0, limit: int = 200) -> JSONResponse:
    """
    Masked log lines as a feed, oldest first.

    `after` is the highest seq the caller already holds, so polling costs one
    small payload rather than the whole buffer. `origin` changes when the proxy
    restarts, which tells the caller its history no longer belongs to this
    process. `gap` is true when the ring buffer discarded lines the caller
    never saw.
    """
    limit = max(1, min(limit, EVENT_BUFFER))
    after = max(0, after)
    selected = [e for e in _EVENTS if e["seq"] > after][-limit:]
    return JSONResponse({
        "origin": round(_STARTED_AT, 3),
        "seq": _EVENT_SEQ,
        "buffered": len(_EVENTS),
        "capacity": EVENT_BUFFER,
        "gap": bool(after and _EVENTS and _EVENTS[0]["seq"] > after + 1),
        "events": selected,
    })


# Served from a file rather than a string constant so the panel can be edited
# and reloaded with F5, without restarting a proxy that is carrying live traffic.
_UI_PATH = Path(__file__).resolve().parent / "ui" / "panel.html"
_UI_CACHE: dict[str, Any] = {"mtime": None, "html": ""}

_UI_MISSING = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Panel missing</title>
<style>body{background:#0d1620;color:#d5e2ee;font:14px ui-monospace,Consolas,monospace;
padding:48px;line-height:1.7}code{color:#e3a441}</style></head><body>
<p>The panel file is not installed.</p>
<p>Expected it at <code>ui/panel.html</code>, next to <code>proxy.py</code>.</p>
<p>The proxy itself is unaffected. Machine-readable status stays at
<code>/_health</code>, <code>/_stats</code>, <code>/_recovery</code> and
<code>/_events</code>.</p></body></html>"""


@app.get("/_ui", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    """The monitoring panel."""
    try:
        mtime = _UI_PATH.stat().st_mtime
    except OSError:
        return HTMLResponse(_UI_MISSING, status_code=503)

    if _UI_CACHE["mtime"] != mtime:
        _UI_CACHE["html"] = _UI_PATH.read_text(encoding="utf-8")
        _UI_CACHE["mtime"] = mtime
    return HTMLResponse(_UI_CACHE["html"], headers={"cache-control": "no-store"})


# Only the API surface this proxy exists for is relayed; everything else is
# answered here and never leaves the machine.
#
# The catch-all below used to forward whatever path it was given, which turned a
# browser's own curiosity into outbound traffic: pointing a tab at the proxy is
# enough to make it ask for /favicon.ico, and Chrome with its devtools open also
# asks for /.well-known/appspecific/com.chrome.devtools.json. The upstream answers
# those with an HTML error page that validate_json_response cannot parse, so a
# single probe became three attempts, a failed_requests and three
# malformed_streams -- all attributed to API traffic that never happened.
#
# Listing the probes would have been the wrong shape for the fix: such a list only
# ever covers the ones already seen, and the next browser, extension or link
# checker arrives with a name that is not on it. So the test is inverted -- a path
# is relayed only when it looks like the API being relayed. Anything else is a
# local 404 regardless of what it is called, which is the only version of this
# that cannot be outgrown. If a future API surface lives outside /v1, adding its
# prefix here is what makes it reachable; until then a 404 in the log names the
# exact path that was refused.
_API_PREFIXES = ("v1/",)


def is_api_path(path: str) -> bool:
    """Whether this path belongs to the API surface this proxy relays."""
    return path.strip("/").lower().startswith(_API_PREFIXES)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str) -> Response:
    # Answered before total_requests is touched, so a browser looking for a
    # favicon leaves no mark on the API figures. One line in the log rather than
    # a request block: enough to explain a 404 someone is looking at, not enough
    # to bury the traffic it sits between.
    if not is_api_path(path):
        bump("local_404_responses")
        log(f"local 404: /{path} (not an API path, not forwarded)")
        return JSONResponse(
            {"type": "error",
             "error": {"type": "not_found_error",
                       "message": f"/{path} is not an API path. This proxy relays "
                                  f"/v1/... calls to the upstream; see / for what "
                                  f"it serves."}},
            status_code=404)

    bump("total_requests")
    started = time.monotonic()

    raw_body = await request.body()
    body, stripped, stream_requested = sanitize_body(raw_body)

    query = request.url.query
    url = f"{UPSTREAM}/{path}" + (f"?{query}" if query else "")
    is_messages = path.rstrip("/") == "v1/messages"

    # The upstream gets `url`; every line below gets the redacted twin. A client
    # that puts its credential in the query string would otherwise write it to
    # the terminal, to logs/proxy.log and -- through record_event -- to /_events.
    safe_suffix = f"?{redact_query(query)}" if query else ""

    log("=" * 62)
    log(f"{request.method} /{path}{safe_suffix} "
        f"-> {UPSTREAM_SAFE}/{path}{safe_suffix}  stream={stream_requested}")
    if stripped:
        log(f"stripped null request fields: {', '.join(stripped)}")
    if is_messages:
        # Structure only -- see describe_request_shape. This is what makes a
        # schema rejection like claude_effort_requires_thinking diagnosable
        # without ever recording a prompt, a tool, or a body.
        log(f"request shape: {describe_request_shape(body)}")

    headers = build_upstream_headers(dict(request.headers))
    if VERBOSE:
        vlog(f"upstream headers: {redact_headers(headers)}")

    # Structural recovery context for this turn. Built from counts, lengths and
    # enums only (see request_structure); no prompt, tool or thinking text is
    # read out of the body. Stored only if the upstream accepts the request.
    pending: Optional[dict[str, Any]] = None
    if is_messages:
        structure = request_structure(body)
        session = session_id_from_headers(dict(request.headers))
        pending = {
            "structure": structure,
            "session": session,
            "lane": lane_key(structure, session),
            "request_id": _digest(str(time.time_ns()), str(id(request))),
            "ts": time.time(),
        }

    if not any(k.lower() == "authorization" for k in headers):
        log("no credential available (client sent none and no ANTHROPIC_AUTH_TOKEN / "
            "AGENTROUTER_API_KEY in environment)")

    retry_args = {"method": request.method, "url": url, "headers": headers,
                  "body": body, "stream_expected": stream_requested,
                  "is_messages": is_messages}
    if stream_requested and STREAM_MODE == "reliable":
        result = await run_reliable_request(request, **retry_args)
        if result is None:
            return Response(status_code=499)
    else:
        result = await run_with_retries(**retry_args)

    # A moderation rejection is recorded once, before any response branch, so
    # that every path out of here is covered. This only writes diagnostics --
    # the status, the body and the no-retry decision are untouched.
    if is_messages and is_content_blocked(result.status, result.body or b""):
        record_content_blocked(pending, upstream_request_id(result.headers))

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
    if result.is_sse and result.body is not None:
        bump("successful_requests")
        save_checkpoint(pending, True)
        # The buffered mode returned without a word in the log until now, so a
        # completed turn left no trace and no timing anywhere. Frames are not
        # counted on this path -- the stream arrives as one body -- so the line
        # reports the bytes handed to the client instead.
        log(f"buffered stream complete: {len(result.body)} bytes in "
            f"{time.monotonic() - started:.1f}s")
        out_headers = passthrough_headers(result.headers)
        out_headers["cache-control"] = "no-cache"
        return Response(content=result.body, status_code=result.status,
                        headers=out_headers, media_type="text/event-stream")

    if result.is_sse and result.stream_iter is not None:
        out_headers = passthrough_headers(result.headers)
        out_headers["cache-control"] = "no-cache"
        return StreamingResponse(
            forward_stream(result, pending),
            status_code=result.status,
            headers=out_headers,
            media_type="text/event-stream",
        )

    # ---- Success: non-streaming (incl. forwarded 4xx) -----------------------
    normalize_content_type = False
    if result.status == 200:
        bump("successful_requests")
        # Upstream accepted this exact structure: it becomes the lane's last
        # known good state to recover to if a later turn is blocked.
        save_checkpoint(pending, None)
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

    # A 429 with no Retry-After leaves the SDK to pick its own delay, and Claude
    # Code then reschedules immediately -- the "Retrying in 0s" seen in the
    # terminal -- which hammers an upstream that is already out of capacity.
    # Supplying the missing hint is the whole fix: the status code and every
    # body byte are forwarded exactly as the upstream sent them, and the client
    # still owns the retry loop.
    if result.status == 429:
        upstream_hint = next((v for k, v in out_headers.items()
                              if k.lower() == "retry-after"), None)
        if retry_after_is_usable(upstream_hint):
            log(f"preserving upstream Retry-After: {upstream_hint}")
        else:
            out_headers = {k: v for k, v in out_headers.items()
                           if k.lower() != "retry-after"}
            out_headers["retry-after"] = str(DEFAULT_RETRY_AFTER_S)
            bump("retry_after_added")
            log(f"upstream 429 carried no usable Retry-After -- advertising "
                f"{DEFAULT_RETRY_AFTER_S}s so the client backs off")

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
        return (f"Could not reach {UPSTREAM_SAFE} after {MAX_ATTEMPTS} attempts "
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

    log(f"upstream = {UPSTREAM_SAFE}")
    log(f"listening on http://{LISTEN_HOST}:{PORT} (loopback only)")
    uvicorn.run(app, host=LISTEN_HOST, port=PORT, log_level="warning", access_log=False)
