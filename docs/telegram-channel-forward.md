# Telegram channel forwarding workflow

Forward an initial snapshot of a source channel/supergroup into a target channel/supergroup. Post-forward full verification is skipped by default; opt in with `--verify-after` or run it separately with `--verify-only`. Source and target IDs must differ and begin with `-100`.

## Setup

Install `uv`. Set these environment variables or put them in the repository root `.env`:

```dotenv
TELEGRAM_API_ID=123456
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_STRING_SESSION=your_authorized_string_session
```

The session must belong to an account allowed to read the source and publish to the target. Protected forwarding is not bypassed. Nothing makes the target private automatically: check the selected target's visibility and members yourself.

## Run

```bash
uv run workflows/forward_telegram_channel.py \
  --source-id=-1001234567890 \
  --target-id=-1009876543210
```

On macOS, optionally prefix the command with `caffeinate -i`. Fixed policy: at most **30 messages per forwarding request**, **108 seconds after each request**. No rate arguments are exposed. API latency, reconciliation, server backoff and cooldowns add time, so throughput is below the theoretical 1,000 messages/hour. No rate guarantees immunity from limits.

Each completed cycle snapshots the latest source message ID. An interrupted cycle resumes its existing cutoff. After completion, rerunning the same command extends the cutoff and forwards only messages with IDs greater than the previous cutoff. Old-message edits/deletions are not synchronized. When requested, verification covers the entire accumulated migration. Service messages cannot be forwarded. Existing destination history is left untouched; verification covers only the new destination range.

## Resume / recover

Stop with Ctrl+C. Keep the artifact directory and rerun the identical command to resume. State is keyed by source/target pair under `artifacts/telegram_channel_forward/`. Each invocation gets a separate `runs/<timestamp-id>/summary.json`, while progress is shared for that pair.

- Intent (source IDs, original random IDs and fingerprints) is atomically persisted before a Telegram write.
- Partial delivery is reconciled against the exact destination prefix, then only the remaining IDs are retried.
- Server-requested waits are persisted and honored across restarts.
- Network failures during forwarding reconnect at most six times; other read/network errors stop safely and can be resumed.
- Server-busy requests retry only after waiting/reconciliation; four consecutive attempts without progress stop.
- One process per pair/state is enforced with OS locks.

Old uncertain requests require explicit authorization:

```bash
uv run workflows/forward_telegram_channel.py \
  --source-id=-1001234567890 --target-id=-1009876543210 --recover-stale
```

This verifies remaining source content, waits 30 seconds and rechecks the target. It assumes no destination messages were manually deleted. Server deduplication retention is not assumed infinite, and exactly-once delivery cannot be guaranteed after arbitrary deletions or external writes. Do not modify the target during a job.

## Verification and evidence

By default, forwarding completion does not trigger a full scan. Add `--verify-after` to invoke the atom again in read-only verification mode after forwarding. Per-request delivery reconciliation always remains enabled. The mandatory pre-forward verification for legacy imports is unchanged. It compares exact text and document/photo media IDs, including multiplicity and full ordering. Forwarded timestamps, entity auto-parsing and web previews are intentionally excluded. This is not a byte-for-byte media download audit or a complete validation of all possible Telegram media types.

Files in the pair directory:

- `progress.json`: snapshot cutoff, last confirmed source ID, confirmed count and pending intent.
- `source-evidence.jsonl`, `target-evidence.jsonl`: message IDs and content fingerprints.
- `verification.json`: counts, missing/extra counts, ordered equality and verification time.
- `runs/.../summary.json`: invocation summary and atom results.

If forwarding completed but verification was interrupted, run:

```bash
uv run workflows/forward_telegram_channel.py \
  --source-id=-1001234567890 --target-id=-1009876543210 --verify-only
```

A completed pair automatically performs an incremental cycle on a normal invocation; with no new posts it finishes without a full scan unless verification was explicitly requested. Keep the same artifacts root to preserve deduplication history. A different `--artifacts-root` starts a new job and does **not** deduplicate against existing destination history.

## Import the audited legacy migration

On the FIRST workflow invocation for this pair, explicitly import the completed old migration:

```bash
caffeinate -i uv run workflows/forward_telegram_channel.py \
  --source-id=-1001234567890 --target-id=-1009876543210 \
  --import-legacy artifacts/legacy_telegram_forward
```

The importer validates the completed legacy state, successful audit, mapping and evidence hashes. It imports the recorded source cutoff and historical destination range, then performs a fresh live full verification BEFORE any new forwarding. Import verification failure leaves a persistent safety gate blocking forwarding; inspect the mismatch and use `--verify-only` to clear the gate after successful verification. Never delete the imported state to bypass a failed audit.

Subsequent invocations omit `--import-legacy` and use the same source/target IDs. Only posts after the last completed cutoff are sent. Import refuses to overwrite an existing workflow state. Starting this pair WITHOUT importing on its first invocation would duplicate historical messages.

## Implementation

- `workflows/forward_telegram_channel.py`: paths, pair-level lock, atom subprocess orchestration and summaries.
- `atoms/forward_telegram_channel.py`: standalone PEP 723 script; Telegram forwarding, persistence, reconciliation and verification.
- `tests/test_forward_telegram_channel.py`: offline unit/contract tests. No real Telegram sends are performed by tests.
