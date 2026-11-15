"""Diagnostic-only paired ablations. Does not change the baseline implementation.

Run from repository root. Final test evaluation is a separate, explicit action.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from research.baseline.common import ROOT, build_module, load_config, select_training_trajectories, dataset_fingerprint, sha256_file
from my_utils.dataset import SampledMultiTrajectoryDataset
from training.loss import prediction_loss, encoder_loss_from_true, encoder_loss_from_pred, compute_gramian_coupled

OUT = ROOT / 'outputs/reproduction/p1_root_cause_2026-11-15'


def config(system, seed):
    suffix = 'yml' if system == 'cartpole' else 'yaml'
    return load_config(ROOT / f'configs/research/{system}_controllable_win15_frac0.05_seed{seed}.{suffix}')


def matrices(model, orientation=False):
    op = model.model.operator.core
    if not orientation:
        return op.get_AB()
    P = op.build_P()
    signs = torch.ones(P.shape[0], dtype=P.dtype, device=P.device)
    signs[model.state_dim - 1] = -1
    P = P * signs[None, :]
    return P @ op.build_Ac() @ torch.linalg.inv(P), P @ op.build_Bc()


def rollout(model, x0, u, A, B):
    z = model.model.encoder(x0)
    values = []
    for t in range(u.shape[1]):
        z = z + model.dt * (z @ A.T + u[:, t] @ B.T)
        values.append(z)
    return torch.stack(values, dim=1)


def losses(model, x0, u, target, orientation=False, no_encoder=False):
    A, B = matrices(model, orientation)
    z = rollout(model, x0, u, A, B)
    physical = z[:, :, :model.state_dim]
    pred = prediction_loss(physical, target)
    enc_true = encoder_loss_from_true(target, z[:, :, model.state_dim:], model.model.encoder)
    enc_pred = .5 * encoder_loss_from_pred(physical, z[:, :, model.state_dim:], model.model.encoder)
    C = torch.eye(model.state_dim, A.shape[0], dtype=A.dtype, device=A.device)
    W = compute_gramian_coupled(A, B, C, T=.3, dt=.02)
    reg = .005 * torch.nn.functional.softplus(.03 - torch.linalg.eigvalsh(W).min())
    total = pred + reg + (0 if no_encoder else enc_true + enc_pred)
    return total, pred, reg


def load_data(system, seed):
    cfg = config(system, seed)
    raw = torch.load(ROOT / cfg['data_train'], map_location='cpu', weights_only=True)
    selected, indices = select_training_trajectories(raw, cfg)
    # Validation excludes the union of the three preregistered training subsets.
    occupied = set()
    for s in (42, 43, 44):
        _, ids = select_training_trajectories(raw, config(system, s))
        occupied.update(ids)
    available = [i for i in range(len(raw)) if i not in occupied]
    rng = random.Random(20261115)
    validation_indices = rng.sample(available, 200)
    validation = [raw[i] for i in validation_indices]
    ds = SampledMultiTrajectoryDataset(selected, 15, 20, seed)
    samples = [ds[i] for i in range(len(ds))]
    x, u, y = [torch.stack([item[k] for item in samples]) for k in range(3)]
    vx = torch.stack([r['x0'].reshape(-1) for r in validation])
    vu = torch.stack([r['u_seq'][:15] for r in validation])
    vy = torch.stack([r['x_seq'][:15].reshape(15, -1) for r in validation])
    return cfg, (x, u, y), (vx, vu, vy), indices, validation_indices


def verify():
    torch.set_num_threads(1)
    result = {}
    for system in ('pendulum', 'cartpole'):
        torch.manual_seed(42)
        model = build_module(config(system, 42))
        n = model.state_dim
        x = torch.randn(4, n); u = torch.randn(4, 15, 1)
        original = model(x, u)
        A, B = matrices(model)
        cached = rollout(model, x, u, A, B)
        assert torch.allclose(original, cached, atol=1e-6, rtol=1e-5)
        p = list(model.parameters())
        go = torch.autograd.grad(original.square().mean(), p, retain_graph=True)
        gc = torch.autograd.grad(cached.square().mean(), p)
        error = max(float((a-b).abs().max()) for a,b in zip(go,gc))
        assert all(torch.allclose(a,b,atol=1e-5,rtol=1e-4) for a,b in zip(go,gc))
        with torch.no_grad():
            A,B = matrices(model, system == 'cartpole')
            blocks=[B]
            for k in range(n-1): blocks.append(A@blocks[-1])
            rank=int(torch.linalg.matrix_rank(torch.cat(blocks,1)[:n]).item())
            assert rank == n
            if system == 'cartpole': assert B[n-1,0] < 0
            else: assert B[n-1,0] > 0
        result[system] = dict(forward_max_difference=float((original-cached).abs().max().detach()), gradient_max_difference=error, oriented_output_rank=rank)
    (OUT / 'harness_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


def train(args):
    torch.set_num_threads(1)
    cfg, tensors, validation, indices, v_indices = load_data(args.system,args.seed)
    torch.manual_seed(args.seed)
    model = build_module(cfg)
    orientation = args.system == 'cartpole' and 'signed' in args.arm
    no_encoder = 'noenc' in args.arm
    epoch_monitor = 'epochmean' in args.arm
    constant_lr = 'constant' in args.arm
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=.1,patience=10)
    order_rng = torch.Generator().manual_seed(700000+args.seed)
    run_id = f'{args.system}-{args.arm}-s{args.seed}'
    directory = OUT / run_id
    directory.mkdir(exist_ok=True)
    if (directory/'result.json').exists():
        print(f'SKIP completed {run_id}',flush=True);return
    record = dict(run_id=run_id,system=args.system,seed=args.seed,arm=args.arm,
        training_trajectories=len(indices),training_windows=len(tensors[0]),validation_trajectories=len(v_indices),
        selection_seed=args.seed,shuffle_seed=700000+args.seed,validation_seed=20261115,
        train_indices=indices,validation_indices=v_indices,
        dataset=dataset_fingerprint(cfg),config=cfg,
        orientation=orientation,no_encoder=no_encoder,epoch_mean_monitor=epoch_monitor,constant_lr=constant_lr,
        status='running',harness_sha256=sha256_file(__file__),history=[])
    (directory/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    start=time.monotonic();steps=0
    for epoch in range(args.epochs):
        model.train();order=torch.randperm(len(tensors[0]),generator=order_rng);totals=[];preds=[]
        for ids in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            total,pred,reg=losses(model,*[t[ids] for t in tensors],orientation,no_encoder)
            if not torch.isfinite(total):
                record.update(status='failed_nonfinite',epoch=epoch,step=steps)
                (directory/'result.json').write_text(json.dumps(record,indent=2)+'\n');return
            total.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                record.update(status='failed_nonfinite_gradient',epoch=epoch,step=steps)
                (directory/'result.json').write_text(json.dumps(record,indent=2)+'\n');return
            optimizer.step();totals.append((float(total.detach()),len(ids)));preds.append(float(pred.detach()));steps+=1
        mean=sum(t*n for t,n in totals)/sum(n for _,n in totals)
        if not constant_lr:scheduler.step(mean if epoch_monitor else totals[-1][0])
        if epoch == 0 or (epoch+1)%10==0 or epoch==args.epochs-1:
            model.eval()
            with torch.no_grad():
                A,B=matrices(model,orientation);yp=rollout(model,*validation[:2],A,B)[:,:,:model.state_dim]
                val=(yp.double()-validation[2].double()).square().mean().item()
            item=dict(epoch=epoch+1,steps=steps,train_loss_epoch_mean=mean,train_prediction_batch_mean=float(np.mean(preds)),train_last_batch=totals[-1][0],lr=optimizer.param_groups[0]['lr'],validation_first15_mse=val,elapsed_seconds=time.monotonic()-start)
            record['history'].append(item)
            print(run_id,json.dumps(item),flush=True)
            (directory/'progress.json').write_text(json.dumps(record,indent=2)+'\n')
    torch.save(model.state_dict(),directory/'checkpoint.pt')
    record.update(status='complete',epochs=args.epochs,steps=steps,elapsed_seconds=time.monotonic()-start,final_validation_mse=record['history'][-1]['validation_first15_mse'],checkpoint_sha256=sha256_file(directory/'checkpoint.pt'))
    (directory/'result.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');parser.add_argument('--system',choices=('pendulum','cartpole'));parser.add_argument('--arm',default='baseline');parser.add_argument('--seed',type=int,default=42);parser.add_argument('--epochs',type=int,default=100)
    args=parser.parse_args()
    if args.verify:verify()
    else:train(args)
