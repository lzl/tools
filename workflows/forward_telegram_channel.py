# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Forward and resume channel snapshots, with optional full verification."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]


def run_atom(arguments):
    atom=ROOT/'atoms/forward_telegram_channel.py'
    process=subprocess.run(['uv','run',str(atom),*arguments,'--json'],
                           stdout=subprocess.PIPE,text=True)  # stderr is live/inherited
    if process.returncode:
        raise RuntimeError(f'{atom} failed with exit code {process.returncode}; see stderr above')
    result=json.loads(process.stdout)
    if not isinstance(result,dict):
        raise RuntimeError('Atom returned invalid JSON contract')
    return result


def import_legacy(directory, destination, source_id, target_id):
    """Import only an audited completed legacy job; verification occurs before any writes."""
    import hashlib
    legacy=json.loads((directory/'progress.json').read_text())
    audit=json.loads((directory/'audit-summary.json').read_text())
    if (legacy['source'],legacy['target']) != (source_id,target_id):
        raise ValueError('Legacy source/target mismatch')
    if legacy['status']!='completed' or 'pending' in legacy:
        raise ValueError('Legacy job must be complete with no pending intent')
    if (audit['missing_count'] or audit['extra_count'] or not audit['ordered_content_identical']
            or audit['target_forwarded_count']!=legacy['forwarded']
            or audit['last_pair']['source_id']!=legacy['cutoff_message_id']):
        raise ValueError('Legacy audit does not prove a complete matching snapshot')
    for name, digest in audit['evidence_sha256'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Legacy audit evidence changed: {name}')
    mapping=[json.loads(line) for line in (directory/'audit-mapping.jsonl').read_text().splitlines()]
    if len(mapping)!=legacy['forwarded'] or not all(r['content_matches'] for r in mapping):
        raise ValueError('Invalid legacy mapping')
    baseline=mapping[0]['target_id']-1
    imported={'version':1,'source_id':source_id,'target_id':target_id,
              'cutoff':legacy['cutoff_message_id'],'cursor':legacy['cutoff_message_id'],
              'target_baseline':baseline,'forwarded':legacy['forwarded'],
              'status':'completed','not_before':0,'import_requires_verification':True,'legacy_import':str(directory.resolve())}
    destination.write_text(json.dumps(imported,ensure_ascii=False,indent=2))


def execute(args):
    pair=args.artifacts_root.resolve()/f'{args.source_id}_to_{args.target_id}'
    pair.mkdir(parents=True,exist_ok=True)
    with (pair/'workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=pair/'progress.json'
        if args.verify_only and not state.exists():
            raise ValueError('No previous state to verify')
        if args.import_legacy:
            if state.exists():
                raise ValueError('Cannot import over existing workflow state')
            import_legacy(args.import_legacy, state, args.source_id, args.target_id)
        run_dir=pair/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
        run_dir.mkdir(parents=True)
        common=['--source-id',str(args.source_id),'--target-id',str(args.target_id),
                '--state',str(state),'--env-file',str(ROOT/'.env')]
        if args.recover_stale:
            common.append('--recover-stale')
        result={'artifacts_dir':str(run_dir),'state':str(state)}
        print(f'Progress: {state}',file=sys.stderr,flush=True)
        try:
            if args.import_legacy:
                # Require a fresh live full verification before extending the imported snapshot.
                result['import_verification']=run_atom([*common,'--mode','verify'])
            if not args.verify_only:
                result['forward']=run_atom([*common,'--mode','forward'])
            if args.verify_only or args.verify_after:
                print('Starting full verification...',file=sys.stderr,flush=True)
                result['verification']=run_atom([*common,'--mode','verify'])
                result['status']='verified'
            else:
                result['status']='completed'
                result['verification_skipped']=True
        except Exception as exc:
            result['status']='failed'
            result['error']=str(exc)
            (run_dir/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
            raise
        (run_dir/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-id',type=int,required=True)
    parser.add_argument('--target-id',type=int,required=True)
    parser.add_argument('--artifacts-root',type=Path,default=ROOT/'artifacts/telegram_channel_forward')
    parser.add_argument('--recover-stale',action='store_true',help='Authorize verified recovery of old uncertain writes; assumes no destination deletions')
    verification=parser.add_mutually_exclusive_group()
    verification.add_argument('--verify-only',action='store_true',help='Only run full verification')
    verification.add_argument('--verify-after',action='store_true',help='Run full verification after forwarding (skipped by default)')
    parser.add_argument('--import-legacy',type=Path,help='Import an audited legacy job directory, then verify live before forwarding')
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args()
    if args.source_id==args.target_id:
        parser.error('Source and target must differ')
    if not str(args.source_id).startswith('-100') or not str(args.target_id).startswith('-100'):
        parser.error('Use Telegram channel IDs beginning with -100')
    try:
        result=execute(args)
    except (Exception,KeyboardInterrupt) as exc:
        print(f'Error: {exc}',file=sys.stderr)
        return 1
    message='Completed and verified' if result['status']=='verified' else 'Forwarding completed (post-forward full verification skipped)'
    print(json.dumps(result,ensure_ascii=False) if args.json else f"{message}. Summary: {result['artifacts_dir']}/summary.json")
    return 0

if __name__=='__main__':
    raise SystemExit(main())
