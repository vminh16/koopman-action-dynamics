"""Diagnostic domain control: first window only, repeated for matched update budget.

This intentionally differs from the paper's random-window description and must
not be counted as successful faithful paper reproduction.
"""
import argparse
import torch
import experiment as e
import geometry_experiment as g

_original_load_data=e.load_data


def first_windows(system,seed):
    cfg,training,validation,indices,v_indices=_original_load_data(system,seed)
    raw=torch.load(e.ROOT/cfg['data_train'],map_location='cpu',weights_only=True)
    selected=[raw[i] for i in indices]
    # Original dataset orders 20 windows for each trajectory consecutively.
    x=torch.stack([r['x0'].reshape(-1) for r in selected]).repeat_interleave(20,0)
    u=torch.stack([r['u_seq'][:15] for r in selected]).repeat_interleave(20,0)
    y=torch.stack([r['x_seq'][:15].reshape(15,-1) for r in selected]).repeat_interleave(20,0)
    return cfg,(x,u,y),validation,indices,v_indices


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('baseline_start0','negative_head_start0'),required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=100)
    args=p.parse_args();args.system='cartpole';e.load_data=first_windows
    if 'negative_head' in args.arm:g.ARM='negative_head';e.matrices=g.geometry_matrices
    e.train(args)
