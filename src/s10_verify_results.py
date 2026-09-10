"""CPU-only integrity checks for the published S10 experiment artifacts."""
import json
from pathlib import Path

from s10_analyze import summarize_run


def main():
    root=Path(__file__).resolve().parents[1]/"results/s10"
    runs=[summarize_run(p.parent) for p in sorted((root/"runs").glob("*/launch.json"))]
    assert len(runs)==21 and all(r['accepted'] for r in runs), 'expected 21 formal runs'
    assert sum(r['steps'] for r in runs)==2100
    groups={}
    for row in runs:
        key=(row['backend'],row['world'],row.get('stage'),row.get('tp'),row.get('pp'))
        groups.setdefault(key,[]).append(row['block'])
        assert row['warmup']==20 and row['global_tokens']==8192 and row['sequence']==512
        assert row['model']=='qwen7b'
        for p in (root/'runs'/row['run']).glob('rank-*.json'):
            rank=json.loads(p.read_text())
            if row['backend']=='zero':assert rank['deepspeed']=='0.17.5'
            else:assert rank['upstream_commit']=='23e00ed0963c35382dfe8a5a94fb3cda4d21e133'
    expected={('zero',2,3,None,None),('zero',4,1,None,None),('zero',4,2,None,None),
              ('zero',4,3,None,None),('megatron',4,None,2,2),
              ('megatron',4,None,4,1),('megatron',4,None,1,4)}
    assert set(groups)==expected and all(sorted(blocks)==[0,1,2] for blocks in groups.values())
    capacity=list((root/'capacity').glob('*/launch.json'))
    assert len(capacity)==12
    capacity_passes=0
    for path in capacity:
        launch=json.loads(path.read_text())
        if launch['returncode']==0:
            capacity_passes+=1
            assert all(json.loads((path.parent/f'rank-{rank}.json').read_text())['complete']
                       for rank in range(launch['case']['world']))
        else:
            assert 'out of memory' in (path.parent/'process.txt').read_text().lower(),path
    assert capacity_passes==7
    checks=[json.loads(p.read_text()) for p in (root/'validation').rglob('stage-*-rank-*.json')]
    assert len(checks)==18
    assert {(r['world'],r['stage'],r['rank']) for r in checks}=={
        (world,stage,rank) for world in (2,4) for stage in (1,2,3) for rank in range(world)}
    assert all(r['passed'] and r['deepspeed']=='0.17.5' and
               r['relative_update_l2_error']<=.05 and r['gradient_norm_relative_error']<=.01 for r in checks)
    profiles=json.loads((root/'profile/summary.json').read_text())
    assert len(profiles)==24 and len({r['trace'] for r in profiles})==24
    assert all(r['gpu_kernel_count']>0 and r['nccl_kernel_count']>0 for r in profiles)
    published=json.loads((root/'analysis/runs.json').read_text())
    accepted={r['run']:r for r in published if r['accepted']}
    assert set(accepted)=={r['run'] for r in runs}
    for r in runs:
        for key in ('tokens_per_second','median_step_seconds','p95_step_seconds','max_rank_peak_allocated_gib'):
            assert accepted[r['run']][key]==r[key],(r['run'],key)
    print('PASS: 21 runs / 2100 updates; 18 numerical records; 24 GPU/NCCL profiles')


if __name__=='__main__':main()
