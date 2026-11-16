# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Forward a channel snapshot, resume safely, then verify all transferred content."""
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


def execute(args):
    pair=args.artifacts_root.resolve()/f'{args.source_id}_to_{args.target_id}'
    pair.mkdir(parents=True,exist_ok=True)
    with (pair/'workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=pair/'progress.json'
        if args.verify_only and not state.exists():
            raise ValueError('No previous state to verify')
        if state.exists():
            previous=json.loads(state.read_text())
            if previous.get('status')=='completed' and not args.verify_only:
                raise ValueError('This pair already completed. Use --verify-only, or a different --artifacts-root for an explicitly new job. Do not restart blindly.')
        run_dir=pair/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
        run_dir.mkdir(parents=True)
        common=['--source-id',str(args.source_id),'--target-id',str(args.target_id),
                '--state',str(state),'--env-file',str(ROOT/'.env')]
        if args.recover_stale:
            common.append('--recover-stale')
        result={'artifacts_dir':str(run_dir),'state':str(state)}
        print(f'Progress: {state}',file=sys.stderr,flush=True)
        try:
            if not args.verify_only:
                result['forward']=run_atom([*common,'--mode','forward'])
            result['verification']=run_atom([*common,'--mode','verify'])
            result['status']='verified'
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
    parser.add_argument('--verify-only',action='store_true')
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
    print(json.dumps(result,ensure_ascii=False) if args.json else f"Completed and verified. Summary: {result['artifacts_dir']}/summary.json")
    return 0

if __name__=='__main__':
    raise SystemExit(main())
