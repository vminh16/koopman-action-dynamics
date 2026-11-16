"""Evaluation-only analytic input-map statistics; no oracle training gradients."""
import json
import torch
import experiment as e

torch.set_num_threads(1)
result={}
for system in ('pendulum','cartpole'):
    cfg,training,validation,indices,val_indices=e.load_data(system,42)
    regions={}
    for region,(x,u,y) in [('sampled_training_windows',training),('first15_validation',validation)]:
        previous=torch.cat((x[:,None],y[:,:-1]),1).double()
        if system=='pendulum':
            G=torch.zeros_like(previous);G[...,1]=3.;variance_u=256/12
        else:
            c=previous[...,2].cos();den=1.1*.5*(4/3-.1*c.square()/1.1)
            G=torch.zeros_like(previous);G[...,3]=-c/den;G[...,1]=(1-.05*G[...,3]*c)/1.1;variance_u=400/12
        mean=G.mean((0,1));var=(G-mean).square().mean((0,1))
        regions[region]=dict(mean_G=mean.tolist(),variance_G_per_state=var.tolist(),unavoidable_mean_vectorfield_squared_error_for_any_constant_B=variance_u*var.sum().item(),description='Continuous-time vector-field lower bound, not a bound on the reported multi-step state MSE. Assumes independent zero-mean uniform controls, as in the generator.')
    result[system]=regions
e.OUT.joinpath('input_map_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
