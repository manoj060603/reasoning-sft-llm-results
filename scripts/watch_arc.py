"""Read-only monitor for one ARC experiment job; no submissions or restarts."""
import argparse
import datetime
import json
import re
import subprocess
import time
from pathlib import Path


def tail(path, limit=8192):
    if not path.exists():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-limit))
        return stream.read().decode('utf-8',errors='replace')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True)
    parser.add_argument('--root',default='.')
    parser.add_argument('--interval',type=int,default=60)
    parser.add_argument('--hours',type=float,default=4)
    parser.add_argument('--log',help='log path relative to the project root')
    args=parser.parse_args()
    root=Path(args.root);deadline=time.monotonic()+args.hours*3600
    while time.monotonic()<deadline:
        status=subprocess.check_output(['squeue','-h','-j',args.job,'-o','%T %M'],text=True).strip()
        log=tail(root/args.log if args.log else root/'logs'/f'pilot_{args.job}.log')
        progress=re.findall(r'Processed prompts:.*?(\d+)/([0-9]+)',log)
        training={}
        for path in sorted((root/'runs').glob('*/train.jsonl')):
            lines=tail(path).strip().splitlines()
            if lines:
                try:training[path.parent.name]=json.loads(lines[-1])['step']
                except (ValueError,KeyError):pass
        scores={str(path.parent.name):json.loads(path.read_text())
                for path in sorted((root/'results').glob('*/metrics.json'))}
        print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            job=args.job,status=status or 'not in queue',training_steps=training,
            generation_progress=progress[-1] if progress else None,scores=scores)),flush=True)
        if not status:
            break
        time.sleep(max(1,args.interval))


if __name__=='__main__':main()
