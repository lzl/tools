# Telegram channel forwarding workflow

Forward an initial snapshot of a source channel/supergroup into a target channel/supergroup, then perform a full read-only content verification. Source and target IDs must differ and begin with `-100`.

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

The initial run snapshots the latest source message ID. Later posts are deliberately outside this job. Service messages cannot be forwarded. Existing destination history is left untouched; verification covers only the new destination range.

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

After forwarding completes, the workflow invokes the atom again in read-only verification mode. It compares exact text and document/photo media IDs, including multiplicity and full ordering. Forwarded timestamps, entity auto-parsing and web previews are intentionally excluded. This is not a byte-for-byte media download audit or a complete validation of all possible Telegram media types.

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

A completed pair is not automatically replayed on a normal invocation. To run an intentionally separate snapshot, use another `--artifacts-root`, knowing this does **not** deduplicate against previous jobs or existing destination messages.

**Legacy migration warning:** this workflow does not import state from `artifacts/legacy_telegram_forward/`. Running it against the previously migrated pair creates a NEW job and would forward the old content again. Do not run that pair as a new job unless duplicate forwarding is intentional.

## Implementation

- `workflows/forward_telegram_channel.py`: paths, pair-level lock, atom subprocess orchestration and summaries.
- `atoms/forward_telegram_channel.py`: standalone PEP 723 script; Telegram forwarding, persistence, reconciliation and verification.
- `tests/test_forward_telegram_channel.py`: offline unit/contract tests. No real Telegram sends are performed by tests.
