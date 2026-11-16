# /// script
# requires-python = ">=3.11"
# dependencies = ["telethon>=1.41.2,<2", "python-dotenv>=1.0"]
# ///
"""Resumable channel forwarding with persistent intent and read-only verification."""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import time

from dotenv import load_dotenv
from telethon import TelegramClient, functions, types
from telethon.sessions import StringSession

REQUEST_SIZE = 30
INTERVAL = 108


def log(message):
    print(message, file=sys.stderr, flush=True)


def save(path, state):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def fingerprint(message):
    media = message.media
    obj = getattr(media, 'document', None) or getattr(media, 'photo', None)
    # Web previews, entity parsing and forwarded timestamps can legitimately change.
    data = [message.raw_text or '', getattr(obj, 'id', None)]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()


def check_prefix(observed, expected):
    if len(observed) > len(expected) or observed != expected[:len(observed)]:
        raise RuntimeError('Destination differs from pending prefix; manual reconciliation required')
    return len(observed)


async def run(args):
    path = args.state
    state = json.loads(path.read_text()) if path.exists() else None
    load_dotenv(args.env_file)
    client = TelegramClient(StringSession(os.environ['TELEGRAM_STRING_SESSION']),
                            int(os.environ['TELEGRAM_API_ID']), os.environ['TELEGRAM_API_HASH'],
                            flood_sleep_threshold=0, request_retries=0, raise_last_call_error=True)
    await client.connect()
    try:
        if not await client.is_user_authorized():
            raise RuntimeError('Telegram session is unauthorized')
        source = await client.get_entity(args.source_id)
        target = await client.get_entity(args.target_id)
        if not isinstance(source, types.Channel) or not isinstance(target, types.Channel):
            raise ValueError('Source and target must be Telegram channels/supergroups')
        if source.noforwards:
            raise RuntimeError('Source prohibits forwarding; will not bypass protection')
        if state:
            if (state['source_id'], state['target_id']) != (args.source_id, args.target_id):
                raise ValueError('State belongs to a different source/target pair')
        else:
            last_source = await client.get_messages(source, limit=1)
            last_target = await client.get_messages(target, limit=1)
            state = {'version': 1, 'source_id': args.source_id, 'target_id': args.target_id,
                     'cutoff': last_source[0].id if last_source else 0,
                     'target_baseline': last_target[0].id if last_target else 0,
                     'cursor': 0, 'forwarded': 0, 'status': 'running', 'not_before': 0}
            save(path, state)

        async def wait_deadline():
            delay = state.get('not_before', 0) - time.time()
            if delay > 0:
                log(f'Waiting {delay:.0f}s (persisted request spacing / Telegram cooldown)')
                await asyncio.sleep(delay)

        async def reconcile():
            pending = state['pending']
            observed = [m async for m in client.iter_messages(target, reverse=True,
                          min_id=pending['baseline'], wait_time=2) if isinstance(m, types.Message)]
            if any(not m.fwd_from for m in observed):
                raise RuntimeError('External destination writes detected; manual reconciliation required')
            count = check_prefix([fingerprint(m) for m in observed], pending['fingerprints'])
            if count:
                state['forwarded'] += count
                state['cursor'] = pending['ids'][count-1]
                pending['baseline'] = observed[-1].id
                for key in ('ids', 'random_ids', 'fingerprints'):
                    pending[key] = pending[key][count:]
                if not pending['ids']:
                    state.pop('pending')
                save(path, state)
                log(f"Confirmed {count}; total={state['forwarded']}; source ID={state['cursor']}")
            return 'pending' not in state

        async def deliver():
            await wait_deadline()
            if await reconcile():
                return
            pending = state['pending']
            if time.time() - pending['created_at'] > 3600:
                if not args.recover_stale:
                    raise RuntimeError('Old uncertain request: rerun with --recover-stale after ensuring no destination messages were deleted')
                originals = await client.get_messages(source, ids=pending['ids'])
                if any(not isinstance(m, types.Message) for m in originals) or [fingerprint(m) for m in originals] != pending['fingerprints']:
                    raise RuntimeError('Source changed; manual reconciliation required')
                await asyncio.sleep(30)
                if await reconcile():
                    return
                state['pending']['created_at'] = time.time()
                save(path, state)
            failures = reconnects = 0
            while 'pending' in state:
                await wait_deadline()
                pending = state['pending']
                before = len(pending['ids'])
                # Save a spacing deadline before the write so interruption cannot bypass it.
                state['not_before'] = time.time() + INTERVAL
                save(path, state)
                log(f"Forwarding {before} messages with original deduplication IDs")
                try:
                    await client(functions.messages.ForwardMessagesRequest(
                        from_peer=source, to_peer=target, id=pending['ids'],
                        random_id=pending['random_ids'], silent=True))
                    state['not_before'] = time.time() + INTERVAL
                except Exception as exc:
                    if hasattr(exc, 'seconds'):
                        state['not_before'] = time.time() + max(INTERVAL, exc.seconds + 2)
                        log(f'Telegram cooldown: {exc.seconds}s')
                    elif isinstance(exc, (ConnectionError, TimeoutError, OSError)):
                        reconnects += 1
                        if reconnects > 6:
                            raise
                        await asyncio.sleep(10)
                        await client.disconnect()
                        await client.connect()
                    elif type(exc).__name__ in ('WorkerBusyTooLongRetryError', 'ServerError', 'TimedOutError', 'RpcCallFailError'):
                        state['not_before'] = time.time() + max(INTERVAL, 60 * 2**failures)
                        log(f'Server busy: {type(exc).__name__}; waiting before reconciliation')
                    else:
                        raise
                finally:
                    save(path, state)
                await wait_deadline()
                if await reconcile():
                    return
                failures = failures + 1 if len(state['pending']['ids']) == before else 0
                if failures >= 4:
                    raise RuntimeError('No progress after four attempts; pending retained, not blindly retried')

        if args.mode == 'forward':
            if state.get('import_requires_verification'):
                raise RuntimeError('Imported state requires successful --verify-only before forwarding')
            if state.get('status') == 'completed':
                latest = await client.get_messages(source, limit=1)
                cutoff = latest[0].id if latest else 0
                if cutoff <= state['cutoff']:
                    return state
                # Previous cutoff, not last regular message: service-only tails must not replay.
                state['cursor'] = state['cutoff']
                state['cutoff'] = cutoff
            state['status'] = 'running'
            save(path, state)
            if 'pending' in state:
                await deliver()
            batch = []
            async def send_batch():
                latest = await client.get_messages(target, limit=1)
                state['pending'] = {'ids': [m.id for m in batch],
                    'random_ids': [secrets.randbits(63) for _ in batch],
                    'fingerprints': [fingerprint(m) for m in batch],
                    'baseline': latest[0].id if latest else 0, 'created_at': time.time()}
                save(path, state)
                await deliver()
            async for message in client.iter_messages(source, reverse=True, min_id=state['cursor'], max_id=state['cutoff']+1, wait_time=2):
                if not isinstance(message, types.Message):
                    continue
                batch.append(message)
                if len(batch) == REQUEST_SIZE:
                    await send_batch()
                    batch = []
            if batch:
                await send_batch()
            state['status'] = 'completed'
            save(path, state)
            return state

        if 'pending' in state or state['status'] != 'completed':
            raise RuntimeError('Forwarding must complete before verification')
        datasets = []
        for label, peer, min_id, max_id in [('source', source, 0, state['cutoff']+1),
                                            ('target', target, state['target_baseline'], 0)]:
            rows = []
            with (path.parent/f'{label}-evidence.jsonl').open('w') as handle:
                async for message in client.iter_messages(peer, reverse=True, min_id=min_id, max_id=max_id, wait_time=2):
                    if not isinstance(message, types.Message):
                        continue
                    row = {'id': message.id, 'fingerprint': fingerprint(message)}
                    rows.append(row)
                    handle.write(json.dumps(row)+'\n')
            datasets.append(rows)
        a,b = datasets
        ca,cb = Counter(r['fingerprint'] for r in a),Counter(r['fingerprint'] for r in b)
        result = {'status': 'verified', 'source_count':len(a),'target_count':len(b),
                  'missing':sum((ca-cb).values()), 'extra':sum((cb-ca).values()),
                  'ordered_equal':[r['fingerprint'] for r in a]==[r['fingerprint'] for r in b],
                  'cutoff':state['cutoff'], 'verified_at':datetime.now(timezone.utc).isoformat()}
        save(path.parent/'verification.json',result)
        if result['missing'] or result['extra'] or not result['ordered_equal'] or len(b)!=state['forwarded']:
            raise RuntimeError('Verification failed; inspect verification.json and evidence files')
        if state.pop('import_requires_verification', False):
            save(path, state)
        return result
    finally:
        await client.disconnect()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-id',type=int,required=True)
    parser.add_argument('--target-id',type=int,required=True)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--env-file',type=Path,required=True)
    parser.add_argument('--mode',choices=['forward','verify'],default='forward')
    parser.add_argument('--recover-stale',action='store_true')
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args()
    if args.source_id==args.target_id:
        parser.error('Source and target must differ')
    args.state.parent.mkdir(parents=True,exist_ok=True)
    try:
        with args.state.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=asyncio.run(run(args))
        print(json.dumps(result,ensure_ascii=False) if args.json else result['status'])
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        log(f'{type(exc).__name__}: {exc}')
        return 1

if __name__=='__main__':
    raise SystemExit(main())
