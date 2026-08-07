<#
    recover-content-blocked.ps1 -- show where a content-blocked rejection
    happened and how to get back to work.

    AgentRouter can reject a request with HTTP 400 content-blocked. That is a
    real moderation decision by the upstream: the proxy forwards it verbatim,
    never retries it, and never alters the request. What the proxy does keep is
    a small structural record of the last turn AgentRouter accepted, so you can
    see how far the conversation got before the rejection.

    This script only READS. It:
      - queries http://127.0.0.1:8787/_recovery
      - prints the last accepted turn and the rejection next to it
      - prints the supported Claude Code command to recover with, and can run
        it for you with -Resume

    It deliberately does NOT:
      - edit or read Claude Code's conversation files or database
      - delete, rewrite or resend the blocked message
      - invent a session id, or use any flag Claude Code does not document

    Usage:
      .\recover-content-blocked.ps1              # report only
      .\recover-content-blocked.ps1 -Resume      # report, then resume
      .\recover-content-blocked.ps1 -Port 8790
      .\recover-content-blocked.ps1 -Json        # raw safe payload
#>

[CmdletBinding()]
param(
    [int]$Port = 8787,
    # Launch the supported resume command instead of only printing it.
    [switch]$Resume,
    # Resume into a new session id rather than continuing the original.
    [switch]$ForkSession,
    [switch]$Json
)

$ErrorActionPreference = 'Stop'
$uri = "http://127.0.0.1:$Port/_recovery"

try {
    $state = Invoke-RestMethod -Uri $uri -TimeoutSec 10
}
catch {
    Write-Host "Could not reach the proxy at $uri" -ForegroundColor Red
    Write-Host "  $($_.Exception.Message)"
    Write-Host ""
    Write-Host "Is it running? Check with:  Invoke-RestMethod http://127.0.0.1:$Port/_health"
    exit 1
}

if ($Json) {
    $state | ConvertTo-Json -Depth 8
    exit 0
}

function Write-Field {
    param([string]$Label, $Value)
    if ($null -eq $Value -or "$Value" -eq '') { $Value = '<none>' }
    if ($Value -is [array]) { $Value = ($Value -join ' ') }
    Write-Host ("  {0,-26} {1}" -f $Label, $Value)
}

Write-Host ""
Write-Host "AgentRouter proxy -- content-blocked recovery" -ForegroundColor Cyan
Write-Host ("-" * 62)

switch ($state.status) {
    'no_checkpoint' {
        Write-Host "STATUS  no accepted turn recorded yet" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "The proxy has not seen a successful /v1/messages turn since it"
        Write-Host "started, so it has nothing to point you back to."
    }
    'ready' {
        Write-Host "STATUS  ready -- no content-blocked rejection recorded" -ForegroundColor Green
    }
    'content_blocked' {
        Write-Host "STATUS  content-blocked rejection recorded" -ForegroundColor Red
    }
    default {
        Write-Host "STATUS  $($state.status)"
    }
}

if ($state.last_good) {
    Write-Host ""
    Write-Host "LAST TURN AGENTROUTER ACCEPTED" -ForegroundColor Green
    Write-Field 'time'              $state.last_good.timestamp
    Write-Field 'model'             $state.last_good.model
    Write-Field 'messages'          $state.last_good.message_count
    Write-Field 'tool_use blocks'   $state.last_good.tool_use_count
    Write-Field 'tool_results'      $state.last_good.tool_result_count
    Write-Field 'structure'         $state.last_good.structural_fingerprint
    Write-Field 'session id'        $state.last_good.session_id
}

if ($state.last_blocked) {
    $b = $state.last_blocked
    Write-Host ""
    Write-Host "REJECTED REQUEST" -ForegroundColor Red
    Write-Field 'time'                $b.timestamp
    Write-Field 'messages at block'   $b.blocked_message_count
    Write-Field 'messages last good'  $b.last_good_message_count
    Write-Field 'structure changed'   $(if ($b.structural_change) { 'yes' } else { 'no' })
    Write-Field 'upstream request id' $b.upstream_request_id
    Write-Field 'proxy retried it'    $(if ($b.proxy_retry) { 'yes' } else { 'no' })
    Write-Field 'request modified'    $(if ($b.request_modified) { 'yes' } else { 'no' })
}

Write-Host ""
Write-Host "WHAT TO DO" -ForegroundColor Cyan
Write-Host "  $($state.suggested_action)"
Write-Host ""
Write-Host "  Note: the rejection came from AgentRouter's moderation, not from the"
Write-Host "  proxy. Resuming replays the conversation up to a point you choose --"
Write-Host "  it does not make the rejected content acceptable. Rephrase or drop"
Write-Host "  whatever was rejected before sending it again."
Write-Host ""
Write-Host ("-" * 62)
Write-Host ("checkpoints held: {0}/{1}   ({2})" -f `
    $state.checkpoints_held, $state.max_checkpoints, $state.notice)

if (-not $Resume) {
    if ($state.status -eq 'content_blocked') {
        Write-Host ""
        Write-Host "Re-run with -Resume to launch the resume command above." -ForegroundColor Yellow
    }
    exit 0
}

# ---- -Resume: run only a documented Claude Code command ---------------------
$claude = Get-Command claude -ErrorAction SilentlyContinue
if (-not $claude) {
    Write-Host ""
    Write-Host "claude is not on PATH -- run the command above yourself." -ForegroundColor Red
    exit 1
}

$sessionId = $null
if ($state.last_good) { $sessionId = $state.last_good.session_id }

# Claude Code exposes no flag to resume at a specific message, so this never
# claims to reopen the exact turn -- it opens the conversation and leaves the
# choice of how far to step back to you (/rewind, inside the session).
if ($sessionId) {
    $claudeArgs = @('--resume', $sessionId)
    if ($ForkSession) { $claudeArgs += '--fork-session' }
}
else {
    # No session id reached the proxy, so the picker is the honest choice:
    # --continue would guess at the most recent conversation in this directory.
    $claudeArgs = @('--resume')
    if ($ForkSession) { $claudeArgs += '--fork-session' }
    Write-Host ""
    Write-Host "No session id was available to the proxy -- opening the session picker." -ForegroundColor Yellow
    if ($state.last_good) {
        Write-Host ("Pick the conversation that had about {0} messages at {1}." -f `
            $state.last_good.message_count, $state.last_good.timestamp)
    }
}

Write-Host ""
Write-Host ("Running: claude {0}" -f ($claudeArgs -join ' ')) -ForegroundColor Cyan
Write-Host ""
& $claude.Source @claudeArgs
exit $LASTEXITCODE
