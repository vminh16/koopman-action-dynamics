import contextlib
import io
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
import torch
from dynamics.cartpole import CartPoleDynamics
from research.baseline.common import ROOT,build_module,load_config,select_training_trajectories
from my_utils.dataset import SampledMultiTrajectoryDataset
from training.loss import prediction_loss,encoder_loss_from_true,encoder_loss_from_pred,controllability_loss,compute_gramian_coupled

torch.set_num_threads(1)
train=torch.load('data/cartpole_train.pt',map_location='cpu',weights_only=True)
test=torch.load('data/cartpole_test.pt',map_location='cpu',weights_only=True)
x0=torch.stack([r['x0'].reshape(4) for r in test]);u=torch.stack([r['u_seq'][:15] for r in test]);truth=torch.stack([r['x_seq'][:15].reshape(15,4) for r in test]).double()
result={}
for seed in (42,43,44):
 cfg=load_config(ROOT/f'configs/research/cartpole_controllable_win15_frac0.05_seed{seed}.yml')
 m=build_module(cfg);m.load_state_dict(torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True));m.eval()
 sys=CartPoleDynamics();zero=torch.zeros(1,4)
 oracle=(sys.dynamics(0,zero,torch.ones(1,1))-sys.dynamics(0,zero,torch.zeros(1,1))).reshape(-1)
 with torch.no_grad():
  A,B=m.model.operator.get_AB();CB=B[:4,0].clone()
  plus=m(zero,torch.ones(1,1,1))[0,0,:4];minus=m(zero,-torch.ones(1,1,1))[0,0,:4]
  orig=m(x0,u)[...,:4]
  z=m.model.encoder(x0).double();Ad=A.double();Bd=B.double();pred=[]
  for t in range(15):z=z+.02*(z@Ad.T+u[:,t].double()@Bd.T);pred.append(z[:,:4])
  cached=torch.stack(pred,1)
  assert torch.allclose(cached,orig.double(),atol=2e-5,rtol=2e-5)
  aug=np.zeros((26,26));aug[:25,:25]=A.double().numpy();aug[:25,25:]=B.double().numpy();E=expm(aug*.02)
  F=torch.from_numpy(E[:25,:25]);G=torch.from_numpy(E[:25,25:]);z=m.model.encoder(x0).double();pred=[]
  for t in range(15):z=z@F.T+u[:,t].double()@G.T;pred.append(z[:,:4])
  exact=torch.stack(pred,1)
  solver=dict(euler_mse=(orig.double()-truth).square().mean().item(),exact_zoh_mse=(exact-truth).square().mean().item(),euler_vs_zoh_prediction_rmse=(orig.double()-exact).square().mean().sqrt().item())
 selected,_=select_training_trajectories(train,cfg)
 ds=SampledMultiTrajectoryDataset(selected,window_len=15,max_windows_per_traj=20,seed=seed)
 # Fixed deterministic broad batch, spanning the full selected subset.
 batch=[ds[i] for i in np.linspace(0,len(ds)-1,64,dtype=int)]
 xx,uu,yy=[torch.stack([b[k] for b in batch]) for k in range(3)]
 zz=m(xx,uu);xp=zz[:,:,:4]
 A,B=m.model.operator.get_AB()
 with contextlib.redirect_stdout(io.StringIO()):reg=controllability_loss(A,B,state_dim=4)
 parts=dict(prediction=prediction_loss(xp,yy),encoder_true=encoder_loss_from_true(yy,zz[:,:,4:],m.model.encoder),encoder_pred_weighted=.5*encoder_loss_from_pred(xp,zz[:,:,4:],m.model.encoder),controllability_weighted=.005*reg)
 params=list(m.model.operator.parameters());grad={}
 for name,loss in parts.items():
  gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
  grad[name]=dict(value=loss.item(),operator_gradient_l2=float(torch.sqrt(sum(g.square().sum() for g in gs if g is not None))))
 W=compute_gramian_coupled(A,B,torch.eye(4,25));ev=torch.linalg.eigvalsh(W.detach())
 result[str(seed)]={'CB':CB.tolist(),'oracle_G_at_origin':oracle.tolist(),'one_step_central_input_slope':((plus-minus)/2/.02).tolist(),'sign_conflict':float(CB[3])>0 and float(oracle[3])<0,'solver':solver,'gradient_probe':grad,'training_euler_gramian_eigenvalues':ev.tolist(),'P_epsilon':m.model.operator.core.eps}
 print(seed,json.dumps(result[str(seed)]),flush=True)
Path('/tmp/p1_checkpoint_probe.json').write_text(json.dumps(result,indent=2))
