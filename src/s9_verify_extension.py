"""CPU-only verification of the published S9 MLIR/decode-graph extension."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / 'results/s9/extensions/mlir-decode-20261102'
    manifest = read(base / 'manifest.json')
    files = {p.relative_to(base).as_posix() for p in base.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.name != 'manifest.json'}
    assert files == set(manifest['sha256_lf']), 'artifact inventory changed'
    for name, expected in manifest['sha256_lf'].items():
        assert digest(base / name) == expected, name

    env = read(base / 'host-graph/environment.json')
    for name, expected in env['source_sha256'].items():
        assert digest(root / 'src' / name) == expected, f'measured source changed: {name}'
    assert digest(base / 'baseline/s9_mlir_adapter_before.py') == env['before_sha256']
    rows = [json.loads(s) for s in (base / 'host-graph/model-records.jsonl').read_text().splitlines()]
    assert len(rows) == 150
    assert len({(r['length'], r['block'], r['backend']) for r in rows}) == 150
    assert {r['length'] for r in rows} == {33, 346, 4096}
    backends = env['backends']
    assert set(backends) == {'eager', 'compatible', 'before', 'fast', 'graph'}
    for r in rows:
        assert len(r['forward_event_ms']) == 32 and r['fallback_calls'] == 0
        assert r['graph_replays'] == (31 if r['backend'] == 'graph' else 0)
        for value in [r['model_loop_wall_ms'], *r['forward_event_ms']]:
            assert math.isfinite(value) and value > 0
    summaries = read(base / 'host-graph/models.json')
    assert {s['length'] for s in summaries} == {33, 346, 4096}
    for summary in summaries:
        selected = [r for r in rows if r['length'] == summary['length']]
        groups = {b: {r['backend']: r for r in selected if r['block'] == b}
                  for b in range(1, 11)}
        orders = []
        for group in groups.values():
            assert set(group) == set(backends)
            order = next(iter(group.values()))['order']
            assert len(order) == 5 and set(order) == set(backends)
            assert all(r['order'] == order for r in group.values())
            orders.append(order)
        for position in range(5):
            assert Counter(o[position] for o in orders) == {b: 2 for b in backends}
        adjacent = Counter((a, b) for o in orders for a, b in zip(o, o[1:]))
        assert adjacent == {(a, b): 2 for a in backends for b in backends if a != b}
        for a in backends:
            for b in backends:
                if a != b:
                    assert sum(o.index(a) < o.index(b) for o in orders) == 5
        for backend, metrics in summary['timing'].items():
            for metric, value in metrics.items():
                measured = statistics.median(g[backend][metric] for g in groups.values())
                assert math.isclose(measured, value, rel_tol=1e-12)
        for name, comparison in summary['comparisons'].items():
            candidate, baseline = name.split('_vs_')
            for metric, value in comparison['median_percent'].items():
                changes = [100 * (g[candidate][metric] / g[baseline][metric] - 1)
                           for g in groups.values()]
                assert math.isclose(statistics.median(changes), value, abs_tol=1e-10)
            faster = sum(g[candidate]['model_loop_wall_ms'] < g[baseline]['model_loop_wall_ms']
                         for g in groups.values())
            assert faster == comparison['faster_wall_blocks'] == 10

    positions = 0
    for case in read(base / 'host-graph/correctness.json'):
        assert set(case['paths']) == {'compatible', 'before', 'fast', 'graph', 'graph_second_request'}
        for path in case['paths'].values():
            assert len(path['positions']) == path['greedy_matches'] == 32
            for result in path['positions']:
                assert result['finite'] and result['different'] == 0
                assert result['max_abs'] == result['max_ulp'] == result['nrmse'] == 0
                positions += 1
    complete = read(base / 'host-graph/complete.json')
    assert complete['passed'] and complete['fallbacks'] == 0
    assert complete['model_requests'] == 150 and complete['correctness_positions'] == positions == 480
    profile = read(base / 'host-graph/profile-summary.json')
    assert profile['kernel_launches'] == sum(profile['kernels_by_name'].values()) == 769
    assert profile['rmsnorm_kernels'] == 57
    assert profile['runtime_calls']['cudaGraphLaunch'] == profile['runtime_calls']['cudaMemcpyAsync'] == 1
    validation = read(base / 'numerics/validation.json')
    assert validation['passed'] and len(validation['cases']) == 17
    print('PASS: S9 extension hashes; 150 balanced requests; recomputed medians/paired ratios; '
          '480 exact logit positions; zero fallbacks; 769-kernel graph observation')


if __name__ == '__main__':
    main()
