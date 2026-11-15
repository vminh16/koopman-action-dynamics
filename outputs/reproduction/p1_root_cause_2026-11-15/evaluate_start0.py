"""Final held-out reporting for a diagnostic sampling change, NOT paper replication."""
import json
import statistics
import numpy as np
import torch
import experiment as e
import geometry_experiment as g
from research.diagnostics.metrics import compute_continuous_gramian_and_margin

torch.set_num_threads(1)
raw=torch.load(e.ROOT/'data/cartpole_test.pt',map_location='cpu',weights_only=True)
x=torch.stack([r['x0'].reshape(4) for r in raw]);u=torch.stack([r['u_seq'][:15] for r in raw]);y=torch.stack([r['x_seq'][:15].reshape(15,4) for r in raw]).double()
results=[]
for arm,seeds in [('baseline_start0',(42,)),('negative_head_start0',(42,43,44))]:
    for seed in seeds:
        directory=e.OUT/f'cartpole-{arm}-s{seed}'
        r=json.loads((directory/'result.json').read_text());assert r['status']=='complete'
        torch.manual_seed(seed);model=e.build_module(e.config('cartpole',seed));model.load_state_dict(torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True));model.eval()
        with torch.no_grad():
            if arm=='baseline_start0':A,B=e.matrices(model)
            else:g.ARM='negative_head';A,B=g.geometry_matrices(model)
            pred=e.rollout(model,x,u,A,B)[:,:,:4]
            vals=(pred.double()-y).square().mean((1,2));assert torch.isfinite(vals).all()
            A64=A.double().numpy();B64=B.double().numpy();C=np.eye(4,25)
        K=np.concatenate([C@np.linalg.matrix_power(A64,k)@B64 for k in range(4)],1);rank=int(np.linalg.matrix_rank(K));assert rank==4
        result=dict(arm=arm,seed=seed,validation_mse=r['final_validation_mse'],test_first15_mse=vals.mean().item(),CB=B64[:4,0].tolist(),output_rank=rank,gramian=compute_continuous_gramian_and_margin(A64,B64,C,T=.3),test_per_trajectory_mse=vals.tolist())
        results.append(result);(directory/'diagnostic_final_evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
        print(arm,seed,result['test_first15_mse'],flush=True)
values=[r['test_first15_mse'] for r in results if r['arm']=='negative_head_start0']
summary=dict(verdict='DIAGNOSTIC_ONLY_NOT_FAITHFUL_PAPER_REPRODUCTION',reason='Training uses repeated first windows, intentionally departing from paper random-window sampling; no oracle gradients or test-based training.',negative_head_start0_mean=statistics.mean(values),negative_head_start0_sample_std=statistics.stdev(values),runs=results)
e.OUT.joinpath('cartpole_start0_diagnostic_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('SUMMARY',summary['negative_head_start0_mean'],summary['negative_head_start0_sample_std'])
