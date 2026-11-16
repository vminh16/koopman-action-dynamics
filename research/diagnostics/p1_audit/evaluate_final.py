"""Final reporting for preregistered three-seed pairs only; no training/selection."""
import argparse
import json
import statistics
import numpy as np
import torch
import experiment as e
import geometry_experiment as g
from research.baseline.common import build_module, sha256_file
from research.diagnostics.metrics import compute_continuous_gramian_and_margin
from scipy.integrate import quad_vec
from scipy.linalg import expm


def evaluate(system):
    torch.set_num_threads(1)
    arms = ('baseline','noenc') if system=='pendulum' else ('baseline','negative_head')
    raw=torch.load(e.ROOT/f'data/{system}_test.pt',map_location='cpu',weights_only=True)
    n=2 if system=='pendulum' else 4
    x=torch.stack([r['x0'].reshape(n) for r in raw]);u=torch.stack([r['u_seq'] for r in raw]);y=torch.stack([r['x_seq'].reshape(250,n) for r in raw]).double()
    runs=[]
    for arm in arms:
        for seed in (42,43,44):
            directory=e.OUT/f'{system}-{arm}-s{seed}'
            result=json.loads((directory/'result.json').read_text());assert result['status']=='complete'
            assert sha256_file(directory/'checkpoint.pt')==result['checkpoint_sha256']
            torch.manual_seed(seed);model=build_module(e.config(system,seed));model.load_state_dict(torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True),strict=True);model.eval()
            with torch.no_grad():
                if arm=='negative_head':g.ARM=arm;A,B=g.geometry_matrices(model)
                else:A,B=e.matrices(model)
                pred=e.rollout(model,x,u,A,B)[:,:,:n]
                current=x.clone();chunks=[]
                for k in range(0,250,50):
                    chunk=e.rollout(model,current,u[:,k:k+50],A,B)[:,:,:n];chunks.append(chunk);current=chunk[:,-1]
                reenc=torch.cat(chunks,1)
                op=model.model.operator.core;P=op.build_P()
                if arm=='negative_head':P=torch.block_diag(-P[:n,:n],P[n:,n:])
                Ac=op.build_Ac();Bc=op.build_Bc()
                arr=[t.detach().double().numpy() for t in (A,B,P,Ac,Bc)]
            A64,B64,P64,Ac64,Bc64=arr
            assert np.allclose(A64,P64@Ac64@np.linalg.inv(P64),rtol=1e-4,atol=1e-5)
            assert np.allclose(B64,P64@Bc64,rtol=1e-4,atol=1e-5)
            C=np.eye(n,len(A64));reach=np.concatenate([C@np.linalg.matrix_power(A64,k)@B64 for k in range(n)],1);sv=np.linalg.svd(reach,compute_uv=False);rank=int(np.linalg.matrix_rank(reach));assert rank==n
            gram=compute_continuous_gramian_and_margin(A64,B64,C,T=.3)
            def integrand(t):
                v=C@expm(A64*t)@B64;return v@v.T
            W,error=quad_vec(integrand,0,.3,epsabs=1e-12,epsrel=1e-10);ev=np.linalg.eigvalsh((W+W.T)/2)
            assert np.allclose(ev[[0,-1]],[gram['lambda_min_Wy'],gram['lambda_max_Wy']],rtol=1e-5,atol=1e-10)
            metrics={}
            for name,pr,tr in [('open_loop_H15',pred[:,:15],y[:,:15]),('open_loop_H250',pred,y),('reencode_H250_K50',reenc,y)]:
                se=(pr.double()-tr).square();assert torch.isfinite(se).all()
                values=se.mean((1,2))
                metrics[name]=dict(mse=values.mean().item(),per_state_mse=se.mean((0,1)).tolist(),per_trajectory_mse=values.tolist())
            record=dict(system=system,arm=arm,seed=seed,validation_mse=result['final_validation_mse'],test=metrics,output_rank_first_n=rank,output_reachability_singular_values=sv.tolist(),P_min_singular_value=float(np.linalg.svd(P64,compute_uv=False).min()),CB=B64[:n,0].tolist(),gramian=gram,quadrature_error=float(error),quadrature_eigenvalues=ev.tolist(),epochs=result['epochs'],steps=result['steps'],final_lr=result['history'][-1]['lr'],checkpoint_sha256=result['checkpoint_sha256'])
            (directory/'final_evaluation.json').write_text(json.dumps(record,indent=2)+'\n');runs.append(record)
            print(system,arm,seed,'test_H15',metrics['open_loop_H15']['mse'],'rank',rank,flush=True)
    summary={}
    for arm in arms:
        chosen=[r for r in runs if r['arm']==arm];summary[arm]={}
        for metric in ('open_loop_H15','open_loop_H250','reencode_H250_K50'):
            vals=[r['test'][metric]['mse'] for r in chosen];mean=statistics.mean(vals);sd=statistics.stdev(vals);summary[arm][metric]=dict(mean=mean,sample_std=sd,values=vals,ci95_t_df2=[mean-4.302652729911275*sd/(3**.5),mean+4.302652729911275*sd/(3**.5)])
    pair=[]
    for seed in (42,43,44):
        b=next(r for r in runs if r['seed']==seed and r['arm']=='baseline');v=next(r for r in runs if r['seed']==seed and r['arm']!= 'baseline')
        pair.append(dict(seed=seed,baseline=b['test']['open_loop_H15']['mse'],variant=v['test']['open_loop_H15']['mse'],relative_reduction=1-v['test']['open_loop_H15']['mse']/b['test']['open_loop_H15']['mse']))
    output=dict(system=system,train_fraction=.05,test_trajectories=len(raw),selection='Preregistered validation-selected contrasts; final epochs, no test checkpoint selection.',protocol='Raw unweighted physical-state MSE; full 1000 test trajectories from x0. Not certified as identical to the unspecified Table 1 evaluation contract.',summary=summary,paired=pair,runs=runs)
    (e.OUT/f'{system}_final_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--system',choices=('pendulum','cartpole'),required=True);args=p.parse_args();evaluate(args.system)
