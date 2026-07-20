# Telegram Channel Backup

Backup one Telegram channel. Export only messages first saved by current run.

## Need

Create `.env` in repo root.

```dotenv
TELEGRAM_API_ID=123456
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_STRING_SESSION=your_authorized_string_session
```

Get API ID/hash from `https://my.telegram.org/apps`. String session must belong
to Telegram account able to read target channel. Keep `.env` and session secret.

Need `uv`.

## Run

```bash
uv run workflows/backup_telegram_channel_markdown.py -1001445373305
```

Use public channel username too.

```bash
uv run workflows/backup_telegram_channel_markdown.py example_channel
```

First run fetch all channel history. Later run fetch only IDs newer than saved
maximum ID. Workflow prints connect, channel resolve, sync start, every 100
messages, atom finish. Progress goes stderr. `--json` keeps stdout one JSON
object.

```bash
uv run workflows/backup_telegram_channel_markdown.py -1001445373305 --json
```

## Files

Persistent backup DB:

```text
data/telegram_channel_backup.sqlite3
```

Each run gets new directory:

```text
artifacts/<timestamp>-<id>/
  new_message_ids.json
  outputs/new_messages.md
```

`new_messages.md` contains only rows newly inserted during current run. First
run treats full history as new local backup. Later no-new run writes valid
Markdown with `No new messages.`.

Choose paths:

```bash
uv run workflows/backup_telegram_channel_markdown.py -1001445373305 \
  --database data/bitcoin.sqlite3 \
  --artifacts-root artifacts/telegram \
  --output exports/latest.md
```

`--output` moves Markdown only. Run manifest stays under artifacts directory.

## Atoms

Run atoms alone when needed.

```bash
uv run atoms/backup_telegram_channel.py -1001445373305 \
  --database data/telegram_channel_backup.sqlite3 \
  --new-messages-json artifacts/manual/new_message_ids.json \
  --json
```

```bash
uv run atoms/export_telegram_messages_markdown.py \
  --database data/telegram_channel_backup.sqlite3 \
  --channel-id 1445373305 \
  --message-ids-json artifacts/manual/new_message_ids.json \
  --output artifacts/manual/new_messages.md \
  --json
```

`--full` makes backup atom scan whole channel. Existing DB rows stay unchanged.
Export still includes only rows newly inserted by that run.

## Problems

`TELEGRAM_API_ID is required`: add `.env`, or export vars in shell.

`Telegram session is not authorized`: create authorized StringSession. Account
must join private channel.

`Input did not resolve to a Telegram channel`: check username or numeric ID.
Private supergroup/channel IDs commonly start `-100`.

Ctrl+C stops current atom. SQLite commits every 100 scanned messages. Run again;
incremental backup continues after saved maximum message ID.
