"""Evaluation-only physical linearization control; no parameters are fitted."""
import json
import numpy as np
from scipy.linalg import expm
import torch
import experiment as e
from dynamics.cartpole import CartPoleDynamics
from dynamics.pendulum import PendulumDynamics

torch.set_num_threads(1)
results={}
for name,system in [('cartpole',CartPoleDynamics()),('pendulum',PendulumDynamics())]:
    n=system.Nstates
    _,_,validation,_,_=e.load_data(name,42)
    x,u,y=[t.double() for t in validation]
    zero=torch.zeros(1,n,dtype=torch.float64);force=torch.zeros(1,1,dtype=torch.float64)
    A=torch.autograd.functional.jacobian(lambda xx:system.dynamics(0,xx,force).reshape(-1),zero).reshape(n,n).numpy()
    B=torch.autograd.functional.jacobian(lambda uu:system.dynamics(0,zero,uu).reshape(-1),force).reshape(n,1).numpy()
    block=np.zeros((n+1,n+1));block[:n,:n]=A;block[:n,n:]=B
    E=expm(block*.02);F=torch.from_numpy(E[:n,:n]);G=torch.from_numpy(E[:n,n:]);pred=[]
    for k in range(15):x=x@F.T+u[:,k]@G.T;pred.append(x)
    pred=torch.stack(pred,1)
    results[name]=dict(validation_mse=(pred-y).square().mean().item(),description='Exact ZOH of analytic upright linearization, no fitting. Uses oracle knowledge for evaluation only; not a learned baseline result.')
e.OUT.joinpath('oracle_linearization_control.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
