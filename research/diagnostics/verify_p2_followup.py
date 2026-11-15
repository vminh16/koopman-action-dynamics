"""Verify the follow-up's imported sources, E05 records and paired run histories.

Complements p2_followup --stage recompute, which reloads all trained checkpoints.
"""
import json
import torch
from research.baseline.common import ROOT, sha256_file
from research.diagnostics.p2_followup import OUT, OLD, save, make_model, matrices
from research.diagnostics.run_p2_diagnostics import evaluate_e05_system


def verify():
    torch.set_num_threads(1)
    source = json.loads((OUT/'source_snapshot.json').read_text())
    sources = {**source['files'], **source['additional_files']}
    for path, expected in sources.items():
        assert sha256_file(ROOT/path) == expected, f'Source changed: {path}'
    e05 = json.loads((OUT/'e05_refresh.json').read_text())
    assert sha256_file(OLD) == e05['historical_report_sha256']
    for name, entry in e05['entries'].items():
        system = name.split('-s')[0]
        assert sha256_file(ROOT/entry['config']) == entry['config_sha256']
        actual = evaluate_e05_system(system, entry['config'], entry['checkpoint']['path'])
        assert actual == {k: v for k, v in entry.items() if k != 'config_sha256'}, name
    records = {}
    output_ranks = []
    for path in OUT.glob('*/result.json'):
        r = json.loads(path.read_text())
        assert r['status'] == 'complete'
        assert r['windows'] == 600 and len(r['window_samples']) == 600
        assert r['updates'] == r['epochs']*10
        assert len(r['history']) == r['epochs']
        assert [row['epoch'] for row in r['history']] == list(range(1, r['epochs']+1))
        assert all(row['lr'] > 0 for row in r['history'])
        model = make_model(r['seed'], 'general' in r['arm'])
        model.load_state_dict(torch.load(path.parent/'checkpoint.pt', weights_only=True), strict=True)
        A, B = (v.detach().double() for v in matrices(model))
        blocks = [B]
        for _ in range(1, 2):
            blocks.append(A @ blocks[-1])
        rank = int(torch.linalg.matrix_rank(torch.cat(blocks, dim=1)[:2], atol=1e-10))
        assert rank == 2, (path, rank)
        output_ranks.append(rank)
        records[(r['a'], r['arm'], r['epochs'], r['seed'])] = r
    comparisons = 0
    for (a, arm, epochs, seed), r in records.items():
        baseline = records[(a, 'baseline', epochs, seed)]
        for key in ('train', 'eval', 'window_samples', 'shuffle_seed'):
            assert r[key] == baseline[key], (a, arm, seed, key)
        if epochs == 100:
            assert r['history'][:25] == records[(a, arm, 25, seed)]['history']
            comparisons += 1
    checkpoint_check = json.loads((OUT/'recompute.json').read_text())
    assert checkpoint_check['verdict'] == 'PASS' and checkpoint_check['runs'] == len(records) == 140
    historical = json.loads((OUT/'historical_recompute.json').read_text())
    assert historical['verdict'] == 'PASS'
    result = {'verdict': 'PASS', 'source_files_checked': len(sources), 'e05_records_recomputed': len(e05['entries']),
              'completed_runs': len(records), 'updates': sum(r['updates'] for r in records.values()),
              'paired_windows_and_shuffle_verified': True, 'long_run_prefixes_identical': comparisons,
              'historical_report_unchanged': True, 'float64_output_rank_2_runs': len(output_ranks)}
    save(OUT/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
