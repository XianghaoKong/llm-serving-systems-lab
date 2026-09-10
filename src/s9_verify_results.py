"""CPU-only integrity checks for published S9 records; does not rerun CUDA."""
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1];base=root/'results'/'s9'
    counts={'formal-norm-swiglu':3600,'formal-swiglu-wide':1200,'formal-w4-fp32acc':320}
    cells=collections.defaultdict(list)
    for name,count in counts.items():
        folder=base/'runs'/name
        assert json.loads((folder/'complete.json').read_text())['complete']
        records=[json.loads(line) for line in (folder/'measurements.jsonl').read_text().splitlines()]
        assert len(records)==count,(name,len(records))
        for r in records:
            assert r['correctness'] and len(r['samples_us'])==30
            assert all(math.isfinite(v) and v>0 for v in r['samples_us'])
            assert r['p50_us']==statistics.median(r['samples_us'])
            key=tuple(r[k] for k in ('op','rows','width','dtype','backend','phase'))
            cells[key].append(r)
    assert len(cells)==1024
    assert all(len(rs)==5 and {r['block'] for r in rs}=={1,2,3,4,5} for rs in cells.values())
    summary=json.loads((base/'summary'/'summary.json').read_text())
    assert len(summary)==1024
    for r in summary:
        key=tuple(r[k] for k in ('op','rows','width','dtype','backend','phase'))
        assert r['median_p50_us']==statistics.median(x['p50_us'] for x in cells[key])
    for backend in ('triton','tilelang'):
        check=json.loads((base/'correctness'/f'correctness-{backend}.json').read_text())
        assert check['complete'] and len(check['cases'])==18 and all(r['passed'] for r in check['cases'])
    profile=json.loads((base/'profile'/'summary.json').read_text())
    assert len(profile['cases'])==44 and all(r['kernel_launches']>0 for r in profile['cases'])
    model=base/'model'
    assert json.loads((model/'complete.json').read_text())=={'complete':True,'requests':1200}
    records=[json.loads(s) for s in (model/'requests.jsonl').read_text().splitlines()]
    assert len(records)==1200
    assert len({(r['block'],r['request_id'],r['backend']) for r in records})==1200
    assert len({r['request_id'] for r in records})==200
    assert collections.Counter(r['block'] for r in records)=={1:400,2:400,3:400}
    assert collections.Counter(r['backend'] for r in records)=={'eager':600,'triton_compatible':600}
    manifest={r['request_id']:r for r in json.loads((model/'request_manifest.json').read_text())}
    for r in records:
        assert r['output_steps']==32 and len(r['generated_ids'])==32 and r['greedy_token_matches']==32
        assert r['logit_max_abs_error']==0 and r['logit_nrmse']==0
        assert r['input_tokens']==int(manifest[r['request_id']]['input_tokens'])
        assert r['category']==manifest[r['request_id']]['workload_category']
    checksums=json.loads((base/'environment'/'manifest-checksums.json').read_text())
    original=(root/'workloads'/'final'/'public_workload_manifest.csv').read_bytes()
    assert hashlib.sha256(original.replace(b'\r\n',b'\n')).hexdigest()==checksums['lf_sha256']
    assert json.loads((model/'environment.json').read_text())['manifest_sha256']==checksums['deployed_byte_sha256']
    mp=json.loads((model/'model-profile-summary.json').read_text())
    for r in mp['backends'].values():
        assert sum(g['count'] for g in r['kernels_by_name'])==r['kernel_launches']>0
        assert math.isclose(sum(g['summed_duration_us'] for g in r['kernels_by_name']),r['summed_kernel_us'],rel_tol=1e-9)
    print('PASS: 5120 operator records / 1024 cells; 1200 model calls; 44 operator + 2 model profiles')


if __name__=='__main__': main()
