"""Group repeated model trace kernels without publishing a multi-MB event list."""
import argparse
import hashlib
import json
from pathlib import Path


def compact(source):
    out={k:v for k,v in source.items() if k!='backends'}
    out['backends']={}
    for backend,record in source['backends'].items():
        groups={}
        for event in record['kernels']:
            group=groups.setdefault(event['name'],dict(name=event['name'],count=0,
                summed_duration_us=0,launch_configurations=[]))
            group['count']+=1;group['summed_duration_us']+=event['duration_us']
            config={k:event['args'][k] for k in ('grid','block','registers per thread','shared memory') if k in event['args']}
            if config not in group['launch_configurations']: group['launch_configurations'].append(config)
        assert sum(g['count'] for g in groups.values())==record['kernel_launches']
        out['backends'][backend]=dict(kernel_launches=record['kernel_launches'],
            summed_kernel_us=record['summed_kernel_us'],
            kernels_by_name=sorted(groups.values(),key=lambda g:-g['summed_duration_us']))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();data=Path(args.input).read_bytes();out=compact(json.loads(data))
    out['raw_summary_sha256']=hashlib.sha256(data).hexdigest()
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(out,indent=2))


if __name__=='__main__': main()
