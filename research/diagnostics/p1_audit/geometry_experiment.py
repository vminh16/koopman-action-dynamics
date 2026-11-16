"""Diagnostic geometry arms: negative-definite vs locally general invertible head.

No oracle gradients are used; train data and observational losses are unchanged.
"""
import argparse
import torch
import experiment as e

_original_matrices = e.matrices
ARM = ''


def prepare(model, general=False):
    if not general or hasattr(model, 'diagnostic_head_base'):
        return
    n = model.state_dim
    with torch.no_grad():
        base = model.model.operator.core.build_P()[:n, :n].detach().clone()
        if n == 4: base = -base
        model.register_buffer('diagnostic_head_base', base)
        model.model.operator.core.H_head.zero_()


def geometry_matrices(model, orientation=False):
    op = model.model.operator.core
    general = 'general' in ARM
    prepare(model, general)
    P = op.build_P()
    n = model.state_dim
    if general:
        head = model.diagnostic_head_base @ torch.matrix_exp(op.H_head)
    else:
        head = -P[:n, :n] if n == 4 else P[:n, :n]
    P = torch.block_diag(head, P[n:, n:])
    return P @ op.build_Ac() @ torch.linalg.inv(P), P @ op.build_Bc()


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--system',choices=('pendulum','cartpole'),required=True);parser.add_argument('--arm',choices=('negative_head','general_head','general_head_epochmean','general_head_constant','general_head_noenc'),required=True);parser.add_argument('--seed',type=int,default=42);parser.add_argument('--epochs',type=int,default=100)
    args=parser.parse_args();ARM=args.arm;e.matrices=geometry_matrices;e.train(args)
