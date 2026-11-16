"""Fixed-presentation coverage controls and equal-budget per-head tuning."""
import argparse
import json
import statistics
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm

from research.baseline.common import ROOT, sha256_file, git_provenance
from research.diagnostics import p2_config_check as c

OUT=ROOT/'outputs/diagnosis/p2_drift_coverage_2026-11-16'
DOC=ROOT/'docs/superpowers/2026-11-16-p2-drift-coverage-tuning.md'
COVERAGE_SEEDS=(72,73,74,75,76)
DEV=(42,43,44)
CONFIRM=(82,83,84,85,86)
_generate=c.generate
_tensors=c.tensors


def load(path):return json.loads(path.read_text())


def balanced_tensors(data,seed):
    batch,samples=_tensors(data,seed)
    assert len(data) in (8,32) and len(samples)==20*len(data)
    repeats=32//len(data)
    batch=tuple(t.repeat((repeats,)+(1,)*(t.ndim-1)) for t in batch)
    return batch,samples*repeats


def run(name,seed,lr,ntrain,compatible=True,general=False):
    cfg={'lr':lr,'scheduler':'constant','epochs':300,'noenc':False}
    # Restore inputs even when the frozen runner skips an existing raw result.
    _generate(seed,'train',compatible,count=ntrain)
    _generate(seed,'validation',compatible)
    def sized(seed,split,*args,**kwargs):
        if split=='train':kwargs['count']=ntrain
        return _generate(seed,split,*args,**kwargs)
    previous=(c.generate,c.tensors)
    try:
        c.generate=sized;c.tensors=balanced_tensors
        record=c.train_run(name,cfg,seed,compatible=compatible,general=general)
    finally:c.generate,c.tensors=previous
    assert record['cfg']==cfg and record['seed']==seed
    assert record['source_sha256']==sha256_file(c.__file__)
    if record['status']=='complete':
        assert record['windows']==640 and record['updates']==3000 and record['epochs_completed']==300
        assert len(record['window_samples'])==640
    return {**record,'ntrain':ntrain,'unique_windows':20*ntrain,'presentations':record['updates']*64}


def choose(records,general):
    rankings=[]
    for lr in (.001,.01):
        rows=[r for r in records if r['general']==general and r['cfg']['lr']==lr]
        assert sorted(r['seed'] for r in rows)==list(DEV)
        metrics=[r['final_validation_or_tiny_train'] for r in rows if r['status']=='complete']
        passed=sum(m['passed'] for m in metrics)
        key=(3-passed,max((m['H15_ratio_to_ignore_input'] for m in metrics),default=float('inf')),
             max((m['response'] for m in metrics),default=float('inf')),lr)
        rankings.append((key,lr,passed))
    _,lr,passed=min(rankings)
    return {'lr':lr,'dev_pass_count':passed,'qualified':passed==3,
            'rule':'fewest dev failures, worst H15/ignore-input ratio, worst response, lower LR'}


def audit(model,seed,ntrain,compatible,split):
    data,identity=_generate(seed,split,compatible,count=ntrain if split=='train' else None)
    train,train_identity=_generate(seed,'train',compatible,count=ntrain)
    X=torch.cat([torch.cat([r['x0'],r['x_seq'][:-1]]) for r in data])
    U=torch.cat([r['u_seq'] for r in data]).double()
    T=torch.cat([torch.cat([r['x0'],r['x_seq'][:-1]]) for r in train]).double()
    with torch.no_grad():
        A,B=c.matrices(model);z=model.model.encoder(X).double()
        drift=(z@A.double().T)[:,:2]-X.double()@torch.tensor(c.F.T)
        G=torch.tensor([[0.],[1.]] if compatible else [[1.],[0.]],dtype=torch.float64)
        inputs=U@(B[:2].double()-G).T
        d=float(drift.square().mean());i=float(inputs.square().mean());cross=float((2*drift*inputs).mean())
        total=float((drift+inputs).square().mean());assert abs(total-d-i-cross)<1e-10
        coverage=torch.cdist(X.double()/T.std(0).clamp_min(1e-8),T/T.std(0).clamp_min(1e-8)).min(1).values
        x0=torch.cat([r['x0'] for r in data]);actions=torch.zeros(len(data),50,1)
        predicted=c.rollout(model,x0,actions)[:,:,:2].double().numpy()
        state=x0.double().numpy();truth=[];E=expm(.02*c.F)
        for _ in range(50):state=state@E.T;truth.append(state.copy())
        truth=np.stack(truth,axis=1)
        zero={str(h):float(np.mean((predicted[:,:h]-truth[:,:h])**2)) for h in (15,50)}
    return {'data':identity,'train_data':train_identity,'metrics':c.metric(model,data,compatible,0.),
            'drift_mse':d,'input_mse':i,'cross_term':cross,'vector_field_mse':total,
            'nearest_train_distance_p95':float(torch.quantile(coverage,.95)),'zero_input_mse':zero}


def evaluate(record,epoch,split):
    checkpoint=record['checkpoints'][str(epoch)]
    assert sha256_file(ROOT/checkpoint['path'])==checkpoint['sha256']
    model=c.make_model(record['seed'],record['general'])
    model.load_state_dict(torch.load(ROOT/checkpoint['path'],weights_only=True),strict=True)
    return audit(model,record['seed'],record['ntrain'],record['compatible'],split)


def snapshot():
    if (OUT/'source_snapshot.json').exists():
        existing=load(OUT/'source_snapshot.json')
        for name,digest in existing['files'].items():assert sha256_file(ROOT/name)==digest,name
        assert sha256_file(OUT/'protocol_before_run.md')==existing['protocol_sha256']
        return
    (OUT/'protocol_before_run.md').write_text(DOC.read_text())
    files=['research/diagnostics/p2_drift_coverage.py','research/diagnostics/p2_config_check.py',
           'research/diagnostics/p2_followup.py','research/diagnostics/run_p2_diagnostics.py',
           'research/baseline/common.py','research/benchmarks/rotation_family.py','models/encoder.py',
           'models/model_builder.py','models/koopman_operator.py','my_utils/dataset.py',
           'training/train.py','training/loss.py']
    c.save(OUT/'source_snapshot.json',{'files':{p:sha256_file(ROOT/p) for p in files},
        'protocol_sha256':sha256_file(OUT/'protocol_before_run.md'),'git':git_provenance(),
        'runtime':{'device':'cpu','threads':1,'dtype':'float32 training / float64 diagnostics',
                   'packages':{p:version(p) for p in ('torch','numpy','scipy','torchdiffeq','pytorch-lightning')}}})


def close(actual,expected):
    if isinstance(actual,dict):
        assert actual.keys()==expected.keys()
        for key in actual:close(actual[key],expected[key])
    elif isinstance(actual,(str,bool)) or actual is None:assert actual==expected
    else:np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-8)


def execute():
    snapshot();c.verify()
    coverage=[run(f'coverage-n{n}-s{s}',s,.001,n) for s in COVERAGE_SEEDS for n in (8,32)]
    development=[run(f'dev-{ "general" if g else "SPD"}-lr{lr}-s{s}',s,lr,32,False,g)
                 for g in (False,True) for lr in (.001,.01) for s in DEV]
    selected={kind:choose(development,g) for kind,g in [('SPD',False),('general',True)]}
    c.save(OUT/'selected_config.json',selected)
    confirmation=[run(f'confirm-{kind}-s{s}',s,selected[kind]['lr'],32,False,g)
                  for kind,g in [('SPD',False),('general',True)] for s in CONFIRM]
    # Selection is frozen before reading any held-out test split.
    raw=coverage+development+confirmation;rows=[];checks=0
    expected={f'coverage-n{n}-s{s}' for s in COVERAGE_SEEDS for n in (8,32)}
    expected|={f'dev-{k}-lr{lr}-s{s}' for k in ('SPD','general') for lr in (.001,.01) for s in DEV}
    expected|={f'confirm-{k}-s{s}' for k in ('SPD','general') for s in CONFIRM}
    assert {r['name'] for r in raw}==expected and len(raw)==32
    for seed in COVERAGE_SEEDS:
        small,_=_generate(seed,'train',True,count=8);large,_=_generate(seed,'train',True,count=32)
        for a,b in zip(small,large):
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        small_samples=_tensors(small,seed)[1];large_samples=_tensors(large,seed)[1]
        assert small_samples==large_samples[:160]
    for seed in DEV:
        pair=[r for r in development if r['seed']==seed]
        assert all(r['train']==pair[0]['train'] and r['window_samples']==pair[0]['window_samples'] for r in pair)
    for seed in CONFIRM:
        pair=[r for r in confirmation if r['seed']==seed]
        assert pair[0]['train']==pair[1]['train'] and pair[0]['window_samples']==pair[1]['window_samples']
        for A,B in zip(c.matrices(c.make_model(seed)),c.matrices(c.make_model(seed,True))):
            torch.testing.assert_close(A,B,atol=1e-7,rtol=1e-6)
    for r in raw:
        row={key:r[key] for key in ('name','seed','general','compatible','cfg','status','ntrain','train','validation','updates','presentations','unique_windows')}
        row['validation_checkpoints']={};row['archived_checkpoint']=None
        if r['status']=='complete':
            assert r['updates']==3000 and r['presentations']==192000
            for epoch,entry in r['checkpoints'].items():
                result=evaluate(r,int(epoch),'validation')
                assert result['data']==r['validation'] and result['train_data']==r['train']
                close(result['metrics'],entry['metrics']);checks+=1
                row['validation_checkpoints'][epoch]=result
            row['train_diagnostics']=evaluate(r,300,'train')
            if not r['name'].startswith('dev-'):
                row['test']=evaluate(r,300,'test')
                row['archived_checkpoint']={k:r['checkpoints']['300'][k] for k in ('path','sha256')}
        rows.append(row)
    assert {k:choose(development,g) for k,g in [('SPD',False),('general',True)]}==selected
    summary={'selected':selected,'rows':rows,'run_count':len(rows),'updates':sum(r['updates'] for r in rows),
             'presentations':sum(r['presentations'] for r in rows),'local_checkpoint_evaluations':checks,
             'failed_runs':[r['name'] for r in rows if r['status']!='complete'],
             'test_used_for_selection':False,'historical_gates_changed':False}
    c.save(OUT/'summary.json',summary)
    verify_saved();print('FULL LOCAL PROTOCOL VERIFIED',len(rows),'runs',summary['updates'],'updates',flush=True)


def verify_saved():
    snapshot();summary=load(OUT/'summary.json');count=0
    assert summary['run_count']==32 and len(summary['rows'])==32
    reconstructed=[{**r,'final_validation_or_tiny_train':r['validation_checkpoints'].get('300',{}).get('metrics')}
                   for r in summary['rows'] if r['name'].startswith('dev-')]
    assert {k:choose(reconstructed,g) for k,g in [('SPD',False),('general',True)]}==summary['selected']
    for row in summary['rows']:
        if not row['archived_checkpoint']:continue
        record={**row,'checkpoints':{'300':row['archived_checkpoint']}}
        for split in ('train','validation','test'):
            actual=evaluate(record,300,split)
            expected=row['train_diagnostics'] if split=='train' else row['validation_checkpoints']['300'] if split=='validation' else row['test']
            close(actual,expected)
        count+=1
    expected=sum(r['status']=='complete' and not r['name'].startswith('dev-') for r in summary['rows'])
    assert count==expected
    c.save(OUT/'verification.json',{'archived_checkpoint_verdict':'PASS','archived_checkpoints':count,
        'selection_metadata_checked':True,'scope':'curated final checkpoints + regenerated train/validation/test diagnostics; dev checkpoints require full local replay'})
    print('CURATED CHECKPOINTS VERIFIED',count,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('all','verify-saved'),default='all')
    args=parser.parse_args();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True);c.OUT=OUT
    {'all':execute,'verify-saved':verify_saved}[args.stage]()


if __name__=='__main__':main()
