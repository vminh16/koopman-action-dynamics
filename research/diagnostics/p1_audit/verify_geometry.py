"""Mathematical checks for the diagnostic geometry, independent of training."""
import json
import torch
import experiment as e
import geometry_experiment as g
from research.baseline.common import build_module

torch.set_num_threads(1)
results={}
for system in ('pendulum','cartpole'):
    torch.manual_seed(42);negative=build_module(e.config(system,42))
    torch.manual_seed(42);general=build_module(e.config(system,42))
    g.ARM='negative_head';An,Bn=g.geometry_matrices(negative)
    g.ARM='general_head';Ag,Bg=g.geometry_matrices(general)
    assert torch.equal(An,Ag) and torch.equal(Bn,Bg)
    n=general.state_dim;op=general.model.operator.core
    base=general.diagnostic_head_base.detach().double();tail=op.build_P()[n:,n:].detach().double();Ac=op.build_Ac().detach().double();Bc=op.build_Bc().detach().double()
    def fn(H):
        P=torch.block_diag(base@torch.matrix_exp(H),tail)
        return P@Ac@torch.linalg.inv(P),P@Bc
    H=(.01*torch.randn(n,n,dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(fn,(H,),eps=1e-6,atol=1e-5,rtol=1e-4)
    A,B=fn(H)
    K=torch.cat([torch.linalg.matrix_power(A,k)@B for k in range(n)],1)[:n]
    rank=int(torch.linalg.matrix_rank(K).item());assert rank==n
    results[system]=dict(initial_dynamics_match_negative_or_positive_SPD=True,finite_difference_gradient_check=True,output_rank=rank)
e.OUT.joinpath('geometry_verification.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
