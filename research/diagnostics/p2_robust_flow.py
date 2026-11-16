"""Multi-seed configuration and coordinate controls; reuses the frozen P2 runner."""
import argparse
import json
import platform
import statistics
from pathlib import Path

import numpy as np
import torch

from research.baseline.common import ROOT, sha256_file, git_provenance
from research.diagnostics import p2_config_check as c

OUT = ROOT/'outputs/diagnosis/p2_robust_flow_2026-11-16'
DOC = ROOT/'docs/superpowers/2026-11-16-p2-robust-flow.md'
DEV = (42,43,44)
CONFIRM = (52,53,54,55,56)


def load(path):
    return json.loads(path.read_text())


def shared_probe(seed):
    data,_=c.generate(seed,'validation',False)
    x=torch.cat([torch.cat([r['x0'],r['x_seq'][:-1]]) for r in data]).double()
    ids=torch.randperm(len(x),generator=torch.Generator().manual_seed(4242+seed))[:64]
    return x[ids]


def probe_metric(model,record):
    x=shared_probe(record['seed'])
    a=record['a']
    G=torch.stack([torch.cos(a*x[:,0]),torch.sin(a*x[:,0])],1)
    B=c.matrices(model)[1][:2,0].detach().double()
    response=float((G-B).square().sum(1).mean())
    floor=float((G-G.mean(0)).square().sum(1).mean())
    gap=float((B-G.mean(0)).square().sum())
    assert abs(response-floor-gap)<1e-10
    train=torch.load(ROOT/record['train']['path'],weights_only=True)
    states=torch.cat([torch.cat([r['x0'],r['x_seq'][:-1]]) for r in train]).double()
    scale=states.std(0).clamp_min(1e-8)
    nearest=torch.cdist(x/scale,states/scale).min(1).values
    return {'response':response,'floor':floor,'excess_fitting_error':gap,
            'normalized_nearest_train_distance_p95':float(torch.quantile(nearest,.95)),
            'coverage_below_3':bool(torch.quantile(nearest,.95)<=3)}


def evaluate(record,epoch,split):
    entry=record['checkpoints'][str(epoch)]
    assert sha256_file(ROOT/entry['path'])==entry['sha256']
    model=c.make_model(record['seed'],record['general'])
    model.load_state_dict(torch.load(ROOT/entry['path'],weights_only=True),strict=True)
    data,identity=c.generate(record['seed'],split,record['compatible'],record['a'])
    result={'metrics':c.metric(model,data,record['compatible'],record['a']),'data':identity}
    if not record['compatible']:result['shared_probe']=probe_metric(model,record)
    return result


def run(name,cfg,seed,compatible=True,general=False,a=0.):
    return c.train_run(name,cfg,seed,compatible=compatible,general=general,a=a)


def selection(development):
    ranking=[]
    for lr in (.001,.01):
        rows=[r for r in development if r['cfg']['lr']==lr]
        assert sorted(r['seed'] for r in rows)==list(DEV)
        finite=[r for r in rows if r['status']=='complete']
        metrics=[r['final_validation_or_tiny_train'] for r in finite]
        passed=sum(m['passed'] for m in metrics)
        key=(len(DEV)-passed,max((m['response'] for m in metrics),default=float('inf')),
             statistics.mean(m['H15_ratio_to_ignore_input'] for m in metrics) if metrics else float('inf'),lr)
        ranking.append((key,lr,passed))
    key,lr,passed=min(ranking)
    return {'cfg':{'lr':lr,'scheduler':'constant','epochs':1000,'noenc':False},
            'development_pass_count':passed,'development_qualified':passed==len(DEV),
            'rule':'fewest failed dev seeds, worst response, mean H15/ignore-input ratio, lower LR'}


def gate(records,compatible,general=False):
    rows=[r for r in records if r['compatible']==compatible and r['general']==general and r['a']==0.]
    assert sorted(r['seed'] for r in rows)==list(CONFIRM)
    return all(r['status']=='complete' and r['final_validation_or_tiny_train']['passed'] for r in rows)


def snapshot():
    if (OUT/'source_snapshot.json').exists():
        saved=load(OUT/'source_snapshot.json')
        for name,digest in saved['files'].items():assert sha256_file(ROOT/name)==digest,name
        return
    (OUT/'protocol_before_run.md').write_text(DOC.read_text())
    paths=['research/diagnostics/p2_robust_flow.py','research/diagnostics/p2_config_check.py',
           'research/diagnostics/p2_followup.py','research/diagnostics/run_p2_diagnostics.py',
           'research/baseline/common.py','research/benchmarks/rotation_family.py',
           'models/encoder.py','models/model_builder.py','models/koopman_operator.py',
           'my_utils/dataset.py','training/train.py','training/loss.py']
    c.save(OUT/'source_snapshot.json',{'files':{p:sha256_file(ROOT/p) for p in paths},
        'protocol_sha256':sha256_file(OUT/'protocol_before_run.md'),'git':git_provenance(),
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                   'device':'cpu','threads':1,'training_dtype':'float32','metrics_dtype':'float64'}})


def execute():
    snapshot();c.verify()
    development=[run(f'dev-lr{lr}-s{seed}',{'lr':lr,'scheduler':'constant','epochs':1000,'noenc':False},seed)
                 for lr in (.001,.01) for seed in DEV]
    chosen=selection(development);c.save(OUT/'selected_config.json',chosen)
    records=[]
    for seed in CONFIRM:
        for kind,compatible,general in [('compatible',True,False),('original-SPD',False,False),('original-general',False,True)]:
            records.append(run(f'confirm-{kind}-s{seed}',chosen['cfg'],seed,compatible,general))
    gates={'compatible_all_five_pass':gate(records,True),
           'original_general_all_five_pass':gate(records,False,True)}
    gates['stress_open']=chosen['development_qualified'] and all(gates.values())
    c.save(OUT/'stage_gates.json',gates)
    if gates['stress_open']:
        for a in (.5,1.,2.):
            for seed in CONFIRM:
                records.append(run(f'stress-a{a}-general-s{seed}',chosen['cfg'],seed,False,True,a))
    # Decisions above are frozen before any new confirmation test is evaluated.
    rows=[];checks=0
    for record in development+records:
        archived=record['name'].startswith('confirm-') and record['status']=='complete'
        row={k:record[k] for k in ('name','cfg','seed','compatible','general','a','status','train','validation')}
        assert record['source_sha256']==sha256_file(c.__file__)
        for key in ('train','validation'):
            assert sha256_file(ROOT/record[key]['path'])==record[key]['sha256']
        row['epochs_completed']=record['epochs_completed'];row['updates']=record['updates']
        row['archived_checkpoint']=None;row['validation_checkpoints']={}
        if record['status']=='complete':
            for epoch,entry in record['checkpoints'].items():
                evaluated=evaluate(record,int(epoch),'validation')
                assert evaluated['data']==record['validation']
                for k,v in evaluated['metrics'].items():
                    np.testing.assert_allclose(v,entry['metrics'][k],atol=1e-10,rtol=1e-8)
                row['validation_checkpoints'][epoch]=evaluated
                checks+=1
            final=record['checkpoints']['1000']
            row['final_checkpoint']=final
            if archived:row['archived_checkpoint']={k:final[k] for k in ('path','sha256')}
            if record['name'].startswith(('confirm-','stress-')):
                row['test']=evaluate(record,1000,'test')
        rows.append(row)
    expected=21+(15 if gates['stress_open'] else 0)
    expected_names={f'dev-lr{lr}-s{seed}' for lr in (.001,.01) for seed in DEV}
    expected_names|={f'confirm-{kind}-s{seed}' for kind in ('compatible','original-SPD','original-general') for seed in CONFIRM}
    if gates['stress_open']:expected_names|={f'stress-a{a}-general-s{seed}' for a in (.5,1.,2.) for seed in CONFIRM}
    assert len(rows)==expected and {r['name'] for r in rows}==expected_names
    assert all(r['cfg']==chosen['cfg'] for r in records)
    for seed in DEV:
        pair=[r for r in development if r['seed']==seed]
        assert pair[0]['train']==pair[1]['train'] and pair[0]['window_samples']==pair[1]['window_samples']
    assert selection(development)==chosen
    assert gate(records,True)==gates['compatible_all_five_pass']
    assert gate(records,False,True)==gates['original_general_all_five_pass']
    # Check matched data/windows/shuffle and coordinate correspondence.
    for seed in CONFIRM:
        pair=[r for r in records if r['seed']==seed and not r['compatible'] and r['a']==0.]
        assert pair[0]['train']==pair[1]['train'] and pair[0]['window_samples']==pair[1]['window_samples']
        original,_=c.generate(seed,'validation',False);compatible,_=c.generate(seed,'validation',True)
        error=max(float((x['x_seq']-y['x_seq']@torch.tensor(c.P.T,dtype=torch.float32)).abs().max()) for x,y in zip(original,compatible))
        assert error<1e-6
    summary={'selected':chosen,'gates':gates,'runs':rows,'run_count':len(rows),
             'updates':sum(r['updates'] for r in rows),'local_checkpoint_evaluations':checks,
             'failed_runs':[r['name'] for r in rows if r['status']!='complete'],
             'historical_gates_changed':False,'test_used_for_selection':False}
    c.save(OUT/'summary.json',summary)
    verify_saved()
    print('FULL LOCAL PROTOCOL VERIFIED',len(rows),'runs',summary['updates'],'updates',flush=True)


def verify_saved():
    snapshot()
    source=load(OUT/'source_snapshot.json')
    assert sha256_file(OUT/'protocol_before_run.md')==source['protocol_sha256']
    summary=load(OUT/'summary.json');count=0
    reconstructed=[{**r,'final_validation_or_tiny_train':r['validation_checkpoints'].get('1000',{}).get('metrics')}
                   for r in summary['runs']]
    development=[r for r in reconstructed if r['name'].startswith('dev-')]
    confirmation=[r for r in reconstructed if r['name'].startswith('confirm-')]
    assert selection(development)==summary['selected']
    gates={'compatible_all_five_pass':gate(confirmation,True),
           'original_general_all_five_pass':gate(confirmation,False,True)}
    gates['stress_open']=summary['selected']['development_qualified'] and all(gates.values())
    assert gates==summary['gates']
    assert summary['run_count']==21+(15 if gates['stress_open'] else 0)
    for row in summary['runs']:
        if not row['archived_checkpoint']:continue
        _,identity=c.generate(row['seed'],'train',row['compatible'],row['a'])
        assert identity==row['train']
        record={**row,'checkpoints':{'1000':row['archived_checkpoint']}}
        for split in ('validation','test'):
            actual=evaluate(record,1000,split)
            expected=row['validation_checkpoints']['1000'] if split=='validation' else row['test']
            assert actual['data']==expected['data']
            for group in ('metrics','shared_probe'):
                if group not in expected:continue
                for k,v in actual[group].items():np.testing.assert_allclose(v,expected[group][k],atol=1e-10,rtol=1e-8)
        count+=1
    expected=sum(r['name'].startswith('confirm-') and r['status']=='complete' for r in summary['runs'])
    assert count==expected
    c.save(OUT/'verification.json',{'archived_checkpoint_verdict':'PASS','archived_checkpoints':count,'selection_and_gate_metadata_checked':True,
        'scope':'source hashes, curated final checkpoints, regenerated validation/test data and metrics; full local checkpoint count is in summary.json'})
    print('CURATED CHECKPOINTS VERIFIED',count,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('all','verify-saved'),default='all')
    args=parser.parse_args();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True);c.OUT=OUT
    {'all':execute,'verify-saved':verify_saved}[args.stage]()


if __name__=='__main__':main()
