import json
import numpy as np
import torch
from torchdiffeq import odeint
from scipy.integrate import solve_ivp
from dynamics.cartpole import CartPoleDynamics
from research.baseline.data_generator import rk4_step

torch.set_num_threads(1)
rows=torch.load('data/cartpole_train.pt',weights_only=True,map_location='cpu')
f=CartPoleDynamics().dynamics
best=(-1,None)
for start in range(0,len(rows),128):
 b=rows[start:start+128];y=torch.stack([r['x_seq'] for r in b]).double();x=torch.cat((torch.stack([r['x0'] for r in b]).double().unsqueeze(1),y[:,:-1]),1);u=torch.stack([r['u_seq'] for r in b]).double()
 pred=rk4_step(f,0.,x.reshape(-1,4),u.reshape(-1,1),.02).reshape_as(y)
 err=(pred-y).abs();maximum=err.max().item()
 if maximum>best[0]:
  idx=np.unravel_index(err.argmax().item(),err.shape)
  i,t,_,d=map(int,idx)
  best=(maximum,dict(trajectory=start+i,step=t,coordinate=d))
rec=best[1];row=rows[rec['trajectory']];step=rec['step'];x=(row['x0'] if step==0 else row['x_seq'][step-1]).clone();u=row['u_seq'][step].clone();target=row['x_seq'][step].clone()
results={'maximum_error':best[0],**rec,'x':x.tolist(),'u':u.tolist(),'target':target.tolist(),'replays':{}}
for dtype in (torch.float32,torch.float64):
 xx=x.to(dtype);uu=u.to(dtype)
 for origin in (0.,step*.02):
  times=torch.tensor([origin,origin+.02],dtype=dtype)
  yy=odeint(lambda t,z:f(t,z,uu.unsqueeze(0)),xx,times)[-1]
  results['replays'][f'dopri5_{dtype}_{origin}']=dict(max_error=(yy.double()-target.double()).abs().max().item(),predicted=yy.tolist(),time_interval=(times[-1]-times[0]).item())
 xx=x.double();uu=u.double()
 if dtype==torch.float64:
  for substeps in (1,2,4,16):
   z=xx.clone()
   for k in range(substeps):z=rk4_step(f,k*.02/substeps,z,uu,.02/substeps)
   results['replays'][f'rk4_{substeps}']=dict(max_error=(z-target.double()).abs().max().item(),predicted=z.tolist())
# High-accuracy independent SciPy ODE from the same predecessor.
def rhs(t,z):
 pos,vel,theta,omega=z;force=float(u[0]);c=np.cos(theta);s=np.sin(theta)
 temp=(force+.05*omega**2*s)/1.1
 acc=(9.8*s-c*temp)/(.5*(4/3-.1*c*c/1.1))
 return [vel,temp-.05*acc*c/1.1,omega,acc]
sol=solve_ivp(rhs,[0,.02],x.numpy().reshape(4).astype(np.float64),method='DOP853',rtol=1e-12,atol=1e-13)
results['replays']['scipy_dop853']=dict(max_error=float(np.max(np.abs(sol.y[:,-1]-target.numpy().reshape(4)))),predicted=sol.y[:,-1].tolist())
# Detect a control shift, if present.
for shift in (-1,1):
 if 0<=step+shift<250:
  z=rk4_step(f,0.,x.double(),row['u_seq'][step+shift].double(),.02)
  results['replays'][f'control_shift_{shift}']=dict(max_error=(z-target.double()).abs().max().item())
open('/tmp/p1_transition_probe.json','w').write(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
