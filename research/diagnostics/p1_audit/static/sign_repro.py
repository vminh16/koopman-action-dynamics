"""Diagnostic reproduction: intentionally fails on the current CartPole checkpoint."""
import torch
from dynamics.cartpole import CartPoleDynamics
from research.baseline.common import ROOT, build_module, load_config

cfg = load_config(ROOT / 'configs/research/cartpole_controllable_win15_frac0.05_seed42.yml')
model = build_module(cfg)
model.load_state_dict(torch.load(ROOT / cfg['checkpoint'], map_location='cpu', weights_only=True))
model.eval()
x = torch.zeros(1, 4)
with torch.no_grad():
    positive = model(x, torch.ones(1, 1, 1))[0, 0, 3]
    negative = model(x, -torch.ones(1, 1, 1))[0, 0, 3]
    model_slope = float((positive - negative) / (2 * model.dt))
    dynamics = CartPoleDynamics().dynamics
    oracle_slope = float((dynamics(0, x, torch.ones(1, 1)) - dynamics(0, x, torch.zeros(1, 1)))[0, 3])
print(f'model angular-input slope = {model_slope:+.9f}; oracle = {oracle_slope:+.9f}')
assert model_slope * oracle_slope > 0, 'SIGN MISMATCH: local angular acceleration responds in the opposite direction'
