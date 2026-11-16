"""Sequential, preregistered baseline configuration diagnosis (not paper reproduction).

Run from repo root: python -m research.diagnostics.p2_config_check --stage all
Saved final/checkpoint models require the geometry-aware loader in this file.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm

from research.baseline.common import ROOT, sha256_file, git_provenance
from research.diagnostics import p2_followup as old
from research.diagnostics.run_p2_diagnostics import build_training_dataset
from training.loss import prediction_loss, encoder_loss_from_true, encoder_loss_from_pred

OUT = ROOT/'outputs/diagnosis/p2_config_check_2026-11-16'
DOC = ROOT/'docs/superpowers/2026-11-16-p2-config-check.md'
F = np.array([[0., 1.], [-1., -.2]])
P = np.array([[.2, 1.], [-1., 0.]])
CHECKS = (100, 500, 1000)
SEEDS = tuple(range(42, 47))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def dynamics(x, u, compatible, a):
    G = np.array([0., 1.]) if compatible else np.array([np.cos(a*x[0]), np.sin(a*x[0])])
    return F @ x + G*float(u)


def generate(seed, split, compatible=True, a=0., solver='rk4', count=None, T=50):
    count = count if count is not None else {'train':8, 'validation':8, 'test':16}[split]
    key = f"{'compatible' if compatible else 'original'}-a{a}-{solver}-{split}-s{seed}-N{count}-T{T}"
    path = OUT/'data'/f'{key}.pt'
    if path.exists():
        return torch.load(path, weights_only=True), {'path': str(path.relative_to(ROOT)), 'sha256': sha256_file(path)}
    offset = {'train':10000, 'validation':20000, 'test':30000}[split]
    rng = np.random.default_rng(offset+seed)
    data = []
    for _ in range(count):
        y0 = rng.uniform(-1.5, 1.5, 2)
        x = y0 if compatible else P @ y0
        x0 = x.copy()
        u = rng.uniform(-2., 2., (T, 1))
        states = []
        for action in u:
            k1 = dynamics(x, action[0], compatible, a)
            if solver == 'euler':
                x = x+.02*k1
            else:
                k2 = dynamics(x+.01*k1, action[0], compatible, a)
                k3 = dynamics(x+.01*k2, action[0], compatible, a)
                k4 = dynamics(x+.02*k3, action[0], compatible, a)
                x = x+.02*(k1+2*k2+2*k3+k4)/6
            states.append(x.copy())
        data.append({'x0':torch.tensor(x0[None], dtype=torch.float32),
                     'u_seq':torch.tensor(u, dtype=torch.float32),
                     'x_seq':torch.tensor(np.stack(states), dtype=torch.float32)})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data, {'path':str(path.relative_to(ROOT)), 'sha256':sha256_file(path)}


def make_model(seed, general=False):
    return old.make_model(seed, general)


def matrices(model):
    return old.matrices(model)


def rollout(model, x, u, AB=None):
    A, B = matrices(model) if AB is None else AB
    z = model.model.encoder(x)
    values = []
    for action in u.unbind(1):
        z = z+model.dt*(z@A.T+action@B.T)
        values.append(z)
    return torch.stack(values, 1)


def components(model, batch):
    x, u, y = batch
    A, B = matrices(model)
    z = rollout(model, x, u, (A,B))
    pred = prediction_loss(z[:,:,:2], y)
    true = encoder_loss_from_true(y, z[:,:,2:], model.model.encoder)
    enc = .5*encoder_loss_from_pred(z[:,:,:2], z[:,:,2:], model.model.encoder)
    # The official coupled Euler Gramian, with the same simultaneous update order.
    Phi = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    W = torch.zeros(2, 2, dtype=A.dtype, device=A.device)
    for _ in range(15):
        term = (Phi@B)[:2]
        W = W+.02*(term@term.T)
        Phi = Phi+.02*(A@Phi)
    reg = .005*torch.nn.functional.softplus(.03-torch.linalg.eigvalsh(W).min())
    return {'prediction':pred, 'encoder_true':true, 'encoder_pred':enc, 'regularizer':reg}


def loss(model, batch, noenc=False):
    terms = components(model, batch)
    return terms['prediction']+terms['regularizer']+(0 if noenc else terms['encoder_true']+terms['encoder_pred'])


def tensors(data, seed):
    ds = build_training_dataset(data, seed=seed)
    return tuple(torch.stack([ds[i][k] for i in range(len(ds))]) for k in range(3)), ds.samples


def metric(model, data, compatible, a):
    x = torch.cat([r['x0'] for r in data])
    u = torch.stack([r['u_seq'][:15] for r in data])
    y = torch.stack([r['x_seq'][:15] for r in data])
    with torch.no_grad():
        z = rollout(model, x, u)
        mse = float((z[:,:,:2].double()-y.double()).square().mean())
        # Input-map response evaluated at ALL held-out predecessor states.
        X = torch.cat([torch.cat([r['x0'],r['x_seq'][:-1]]) for r in data]).double()
        if compatible:
            G = torch.tensor([0.,1.], dtype=torch.float64).expand(len(X),-1)
        else:
            G = torch.stack([torch.cos(a*X[:,0]),torch.sin(a*X[:,0])],1)
        B = matrices(model)[1][:2,0].detach().double()
        # Symmetric actions -1/+1 at the same states: unit-norm G -> normalized error.
        response = float((G-B).square().sum(1).mean())
        mean = G.mean(0)
        floor = float((G-mean).square().sum(1).mean())
        gap = float((B-mean).square().sum())
        assert abs(response-floor-gap)<1e-10
        drift = x.double().numpy()
        values = []
        E = expm(.02*F)
        for _ in range(15):
            drift = drift@E.T
            values.append(drift.copy())
        ignore = float(np.mean((np.stack(values,1)-y.double().numpy())**2))
    return {'response':response, 'H15_mse':mse, 'ignore_input_H15_mse':ignore,
            'H15_ratio_to_ignore_input':mse/ignore, 'CB':B.tolist(), 'fixed_map_floor':floor,
            'excess_fitting_error':gap, 'passed':response<=.05 and mse<=.25*ignore}


def gradient_audit(model, batch):
    terms = components(model, batch)
    groups = {'encoder':list(model.model.encoder.parameters()), 'operator':list(model.model.operator.parameters())}
    result = {}
    for name, term in terms.items():
        row = {'value':float(term.detach())}
        for group, params in groups.items():
            grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
            row[f'{group}_gradient_norm'] = math.sqrt(sum(float(g.detach().square().sum()) for g in grads if g is not None))
        result[name] = row
    return result


def verify():
    result = {}
    for seed in (42,43):
        model = make_model(seed)
        batch = (torch.randn(3,2),torch.randn(3,15,1),torch.randn(3,15,2))
        original = model(*batch[:2]); cached = rollout(model,*batch[:2])
        torch.testing.assert_close(original,cached,atol=1e-6,rtol=1e-5)
        model.log=lambda *args,**kwargs:None
        with contextlib.redirect_stdout(io.StringIO()): upstream=model.training_step(batch,0)
        fast=loss(model,batch)
        torch.testing.assert_close(upstream,fast,atol=1e-7,rtol=1e-5)
        params=list(model.parameters())
        gs=torch.autograd.grad(upstream,params); gf=torch.autograd.grad(fast,params)
        for x,y in zip(gs,gf):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-4)
        result[str(seed)]={'forward_error':float((original-cached).abs().max().detach()),
                           'gradient_error':max(float((x-y).abs().max()) for x,y in zip(gs,gf))}
    model=make_model(42)
    op=model.model.operator.core
    with torch.no_grad():
        for p in model.model.encoder.parameters():p.zero_()
        op.H_head.copy_(math.sqrt(.9)*torch.eye(2))
        op.A_free.zero_();op.A_free[0,:2]=torch.tensor([-1.,-.2])
        for j in range(4):op.A_free[j+1,j+2]=-.1
        op.B_free.zero_()
    A,B=matrices(model)
    torch.testing.assert_close(A[:2,:2],torch.tensor(F,dtype=torch.float32),atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(B[:2],torch.tensor([[0.],[1.]]),atol=1e-7,rtol=1e-6)
    data,_=generate(42,'train',solver='euler',count=2,T=30)
    m=metric(model,data,True,0.)
    assert m['H15_mse']<1e-12 and m['response']<1e-12,m
    result['known_answer']=m
    # Default dtype is scoped because upstream single-input builders create zeros
    # without an explicit dtype. This tests double precision without editing upstream.
    default=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        double=make_model(42)
        x=torch.randn(1,2);u=torch.randn(1,3,1);y=torch.randn(1,3,2)
        class LossWrapper(torch.nn.Module):
            def __init__(self,module):super().__init__();self.module=module
            def forward(self):return loss(self.module,(x,u,y))
        wrapper=LossWrapper(double)
        names=[n for n,p in wrapper.named_parameters() if '.operator.' in n]
        values=tuple(dict(wrapper.named_parameters())[n] for n in names)
        fn=lambda *args:torch.func.functional_call(wrapper,dict(zip(names,args)),())
        assert torch.autograd.gradcheck(fn,values,eps=1e-6,atol=2e-5,rtol=2e-4)
        result['float64_operator_loss_gradcheck']=True
    finally:
        torch.set_default_dtype(default)
    # Independent equivalence of a=0 coordinates, including held-out samples.
    original,_=generate(42,'validation',False)
    canonical,_=generate(42,'validation',True)
    error=max(float((a['x_seq']-b['x_seq']@torch.tensor(P.T,dtype=torch.float32)).abs().max()) for a,b in zip(original,canonical))
    assert error<1e-6,error
    result['coordinate_equivalence_max_error']=error
    rk4,_=generate(42,'test',True)
    result['known_answer_RK4_evaluation']=metric(model,rk4,True,0.)
    save(OUT/'harness_verification.json',result)
    print('Verification PASS',flush=True)


def train_run(name, cfg, seed, compatible=True, a=0., general=False, tiny=False, early=False):
    directory=OUT/'runs'/name
    if (directory/'result.json').exists():return json.loads((directory/'result.json').read_text())
    data,train_hash=generate(seed,'train',compatible,a,solver='euler' if tiny else 'rk4',count=2 if tiny else 8,T=30 if tiny else 50)
    validation,val_hash=generate(seed,'validation',compatible,a)
    batch,samples=tensors(data,seed)
    model=make_model(seed,general)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.1,patience=10)
    rng=torch.Generator().manual_seed(700000+seed)
    initial=gradient_audit(model,tuple(t[:64] for t in batch))
    history=[]; checkpoints={};status='complete';steps=0
    epochs=cfg['epochs']
    for epoch in range(1,epochs+1):
        totals=[]
        for ids in torch.randperm(len(batch[0]),generator=rng).split(64):
            optimizer.zero_grad(set_to_none=True)
            total=loss(model,tuple(t[ids] for t in batch),cfg.get('noenc',False))
            if not torch.isfinite(total):status='failed_nonfinite';break
            total.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):status='failed_nonfinite_gradient';break
            optimizer.step();steps+=1;totals.append((float(total.detach()),len(ids)))
        if status!='complete':break
        mean=sum(t*n for t,n in totals)/len(batch[0])
        if cfg['scheduler']=='epochmean':scheduler.step(mean)
        history.append({'epoch':epoch,'mean_loss':mean,'lr':optimizer.param_groups[0]['lr']})
        if epoch in CHECKS or epoch==epochs:
            # Tiny overfit gate is on its train set; tuning/confirmation use validation.
            measured=metric(model,data if tiny else validation,compatible,a)
            directory.mkdir(parents=True,exist_ok=True)
            ckpt=directory/f'epoch{epoch}.pt';torch.save(model.state_dict(),ckpt)
            checkpoints[str(epoch)]={'metrics':measured,'path':str(ckpt.relative_to(ROOT)),'sha256':sha256_file(ckpt)}
            print(name,epoch,'response',round(measured['response'],6),'ratio',round(measured['H15_ratio_to_ignore_input'],6),flush=True)
            if early and measured['passed']:break
    final_audit=gradient_audit(model,tuple(t[:64] for t in batch)) if status=='complete' else None
    final_metrics=metric(model,data if tiny else validation,compatible,a) if status=='complete' else None
    record={'name':name,'status':status,'cfg':cfg,'seed':seed,'compatible':compatible,'a':a,'general':general,'tiny':tiny,
            'epochs_completed':len(history),'updates':steps,'windows':len(batch[0]),'window_samples':samples,
            'shuffle_seed':700000+seed,'train':train_hash,'validation':val_hash,
            'history':history,'checkpoints':checkpoints,'initial_gradients':initial,'final_gradients':final_audit,
            'final_validation_or_tiny_train':final_metrics,'source_sha256':sha256_file(__file__)}
    save(directory/'result.json',record)
    return record


def select_config():
    candidates=[];tuning=[]
    for noenc in (False,True):
        for lr in (1e-4,1e-3,1e-2):
            for scheduler in ('constant','epochmean'):
                name=f'tune-lr{lr}-{scheduler}-noenc{int(noenc)}'
                cfg={'lr':lr,'scheduler':scheduler,'epochs':1000,'noenc':noenc}
                r=train_run(name,cfg,42,early=True);tuning.append(r['name'])
                if r['status']=='complete' and r['final_validation_or_tiny_train']['passed']:
                    candidates.append(r)
        if candidates:break
    if not candidates:
        selected={'status':'NO_CONFIG_PASSED','tuning_runs':tuning}
    else:
        best=min(candidates,key=lambda r:(r['updates'],r['final_validation_or_tiny_train']['response'],r['name']))
        selected={'status':'SELECTED','name':best['name'],'cfg':{**best['cfg'],'epochs':best['epochs_completed']},
                  'selection_metric':best['final_validation_or_tiny_train'],'tuning_runs':tuning,
                  'rule':'first passing checkpoint per run; minimum updates, then validation response, then name; test never consulted'}
    save(OUT/'selected_config.json',selected)
    return selected


def confirm_and_stress(selected):
    if selected['status']!='SELECTED':return {'status':'BLOCKED_NO_CONFIG'}
    cfg=selected['cfg'];confirm=[]
    for seed in SEEDS:
        confirm.append(train_run(f'confirm-compatible-s{seed}',cfg,seed))
    compatible_ok=all(r['status']=='complete' and r['final_validation_or_tiny_train']['passed'] for r in confirm)
    result={'compatible_all_five_pass':compatible_ok,'confirmation':[r['name'] for r in confirm]}
    # Record the matched-coordinate comparison even if confirmation fails, but
    # never run the nonlinear stress sweep until both prerequisite gates pass.
    original=[]
    for seed in SEEDS:
        for general in (False,True):
            original.append(train_run(f'original-a0-{ "general" if general else "SPD"}-s{seed}',cfg,seed,compatible=False,general=general))
    general_ok=all(r['status']=='complete' and r['final_validation_or_tiny_train']['passed'] for r in original if r['general'])
    result['original_comparison']=[r['name'] for r in original]
    result['original_general_all_five_pass']=general_ok
    result['stress_runs']=[]
    if compatible_ok and general_ok:
        for a in (.5,1.,2.):
            for seed in SEEDS:
                r=train_run(f'stress-a{a}-general-s{seed}',cfg,seed,compatible=False,a=a,general=True)
                result['stress_runs'].append(r['name'])
        result['status']='STRESS_COMPLETED'
    else:result['status']='STRESS_BLOCKED_PREREQUISITE_NOT_MET'
    save(OUT/'stage_gates.json',result)
    return result


def evaluate_test():
    # Called only after selection, confirmation and conditional stress are locked.
    selected=json.loads((OUT/'selected_config.json').read_text())
    results={}
    for path in sorted((OUT/'runs').glob('*/result.json')):
        r=json.loads(path.read_text())
        if r['name'].startswith('tune-') or r['tiny'] or r['status']!='complete':continue
        last=r['checkpoints'][str(r['epochs_completed'])]
        model=make_model(r['seed'],r['general'])
        assert sha256_file(ROOT/last['path'])==last['sha256']
        model.load_state_dict(torch.load(ROOT/last['path'],weights_only=True),strict=True)
        data,identity=generate(r['seed'],'test',r['compatible'],r['a'])
        results[r['name']]={'data':identity,'metrics':metric(model,data,r['compatible'],r['a'])}
    save(OUT/'test_evaluation.json',{'selected_config_sha256':sha256_file(OUT/'selected_config.json'),'results':results})
    return results


def recompute():
    checks=0;updates=0
    snapshot=json.loads((OUT/'source_snapshot.json').read_text())
    for name,expected in snapshot['files'].items():assert sha256_file(ROOT/name)==expected,name
    for path in sorted((OUT/'runs').glob('*/result.json')):
        r=json.loads(path.read_text())
        assert r['source_sha256']==sha256_file(__file__)
        for key in ('train','validation'):
            assert sha256_file(ROOT/r[key]['path'])==r[key]['sha256']
        updates+=r['updates']
        assert r['updates']==r['epochs_completed']*math.ceil(r['windows']/64) if r['status']=='complete' else True
        for epoch,entry in r['checkpoints'].items():
            assert sha256_file(ROOT/entry['path'])==entry['sha256']
            model=make_model(r['seed'],r['general'])
            model.load_state_dict(torch.load(ROOT/entry['path'],weights_only=True),strict=True)
            data=torch.load(ROOT/r['train' if r['tiny'] else 'validation']['path'],weights_only=True)
            actual=metric(model,data,r['compatible'],r['a'])
            for k,v in actual.items():
                if isinstance(v,bool):assert v==entry['metrics'][k]
                else:np.testing.assert_allclose(v,entry['metrics'][k],atol=1e-10,rtol=1e-8)
            checks+=1
    tests=json.loads((OUT/'test_evaluation.json').read_text())
    assert tests['selected_config_sha256']==sha256_file(OUT/'selected_config.json')
    for name,entry in tests['results'].items():
        r=json.loads((OUT/'runs'/name/'result.json').read_text())
        last=r['checkpoints'][str(r['epochs_completed'])]
        model=make_model(r['seed'],r['general'])
        model.load_state_dict(torch.load(ROOT/last['path'],weights_only=True),strict=True)
        assert sha256_file(ROOT/entry['data']['path'])==entry['data']['sha256']
        actual=metric(model,torch.load(ROOT/entry['data']['path'],weights_only=True),r['compatible'],r['a'])
        for k,v in actual.items():
            if isinstance(v,bool):assert v==entry['metrics'][k]
            else:np.testing.assert_allclose(v,entry['metrics'][k],atol=1e-10,rtol=1e-8)
    result={'verdict':'PASS','checkpoint_evaluations':checks,'test_evaluations':len(tests['results']),
            'runs':len(list((OUT/'runs').glob('*/result.json'))),'updates':updates,'source_files':len(snapshot['files'])}
    save(OUT/'recompute.json',result);print(result,flush=True)


def execute():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'protocol_before_run.md').exists():
        (OUT/'protocol_before_run.md').write_text(DOC.read_text())
    files=['research/diagnostics/p2_config_check.py','research/diagnostics/p2_followup.py',
           'research/diagnostics/run_p2_diagnostics.py','research/baseline/common.py','models/encoder.py',
           'models/model_builder.py','models/koopman_operator.py','my_utils/dataset.py','training/train.py','training/loss.py']
    if not (OUT/'source_snapshot.json').exists():
        save(OUT/'source_snapshot.json',{'files':{x:sha256_file(ROOT/x) for x in files},
            'protocol_sha256':sha256_file(OUT/'protocol_before_run.md'),'git':git_provenance(),
            'runtime':{'torch':torch.__version__,'device':'cpu','threads':1}})
    verify()
    tiny=train_run('tiny-lr0.001',{'lr':.001,'scheduler':'constant','epochs':1000,'noenc':False},42,tiny=True,early=True)
    if tiny['status']!='complete' or not tiny['final_validation_or_tiny_train']['passed']:
        train_run('tiny-lr0.01',{'lr':.01,'scheduler':'constant','epochs':1000,'noenc':False},42,tiny=True,early=True)
    selected=select_config()
    gates=confirm_and_stress(selected)
    if selected['status']!='SELECTED':save(OUT/'stage_gates.json',gates)
    evaluate_test()
    recompute()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('verify','all','recompute'),default='all')
    args=parser.parse_args();torch.set_num_threads(1)
    {'verify':verify,'all':execute,'recompute':recompute}[args.stage]()


if __name__=='__main__':main()
