import json
from pathlib import Path
import torch
from dynamics.cartpole import CartPoleDynamics
from research.baseline.data_generator import rk4_step
from research.baseline.common import ROOT,load_config,select_training_trajectories

torch.set_num_threads(1)
rows=torch.load('data/cartpole_train.pt',map_location='cpu',weights_only=True)
f=CartPoleDynamics().dynamics
thresholds=(1e-4,1e-5,1e-6);counts={str(t):0 for t in thresholds};hits=[];sum_sq=0.;elements=0
for start in range(0,len(rows),128):
 b=rows[start:start+128];y=torch.stack([r['x_seq'] for r in b]).double();x=torch.cat((torch.stack([r['x0'] for r in b]).double().unsqueeze(1),y[:,:-1]),1);u=torch.stack([r['u_seq'] for r in b]).double()
 p=rk4_step(f,0.,x.reshape(-1,4),u.reshape(-1,1),.02).reshape_as(y);e=(p-y).abs();mx=e.amax(dim=(-1,-2))
 sum_sq+=e.square().sum().item();elements+=e.numel()
 for t in thresholds:counts[str(t)]+=int((mx>=t).sum())
 for i,j in (mx>=1e-4).nonzero().tolist():hits.append(dict(trajectory=start+i,step=j,max_error=mx[i,j].item()))
selected={}
for seed in (42,43,44):
 cfg=load_config(ROOT/f'configs/research/cartpole_controllable_win15_frac0.05_seed{seed}.yml');_,idx=select_training_trajectories(rows,cfg)
 selected[str(seed)]=[r for r in hits if r['trajectory'] in set(idx)]
r=dict(total_transitions=len(rows)*250,transition_counts_above_threshold=counts,local_state_rmse=(sum_sq/elements)**.5,failing_transitions=hits,failing_trajectories_in_C02_subsets=selected)
Path('/tmp/p1_dataset_scope_probe.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
