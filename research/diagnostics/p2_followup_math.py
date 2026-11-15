"""Independent analytic controls for P2; no model training or oracle gradients."""
import json
import numpy as np
import torch
from scipy.linalg import block_diag, expm, logm
from research.benchmarks import load_benchmark
from research.diagnostics.p2_followup import OUT, old_report, checked_data, save
from research.diagnostics.run_p2_diagnostics import build_response_probe, _stack_transitions


def audit():
    torch.set_num_threads(1)
    report = old_report()
    data0 = checked_data(report['rotation_sweep']['a_0.0']['datasets']['eval'])
    probe = build_response_probe(data0, load_benchmark('rotation_family', a=0)['instance'])
    weights = probe['u'][:, 0].double().square()
    controls = {}
    F = np.array([[0., 1.], [-1., -.2]])
    for name, entry in report['rotation_sweep'].items():
        sys = load_benchmark('rotation_family', a=entry['a'])['instance']
        G = sys.analytic_Gx(probe['x']).double().numpy()[:, :, 0]
        w = weights.numpy()
        best = (w[:, None]*G).sum(0)/w.sum()
        floor = (w*np.sum((G-best)**2, axis=1)).sum()/w.sum()
        X, U, Y = _stack_transitions(checked_data(entry['datasets']['eval']))
        # Exact zero-input flow, evaluated against held-out transitions with inputs.
        drift_only = X.double().numpy() @ expm(F*.02).T
        drift_mse = float(np.mean((drift_only-Y.double().numpy())**2))
        controls[name] = {'weighted_best_constant_map': best.tolist(),
                         'fixed_B_normalized_floor': float(floor),
                         'ignore_input_normalized_response': 1.0,
                         'exact_drift_ignore_input_one_step_mse': drift_mse}
    # Exact general-head realization of the easiest system. Tail coordinates zero.
    Phead = np.array([[.2, 1.], [-1., 0.]])
    P = block_diag(Phead, np.eye(4))
    Ac = block_diag(F, -.1*np.eye(4))
    Bc = np.array([[0.], [1.], [0.], [0.], [0.], [0.]])
    A, B = P @ Ac @ np.linalg.inv(P), P @ Bc
    np.testing.assert_allclose(A[:2, :2], F, atol=1e-14)
    np.testing.assert_allclose(B[:2], [[1.], [0.]], atol=1e-14)
    M = logm(np.linalg.solve(.1*np.eye(2), Phead))
    assert np.max(np.abs(np.imag(M))) < 1e-12
    np.testing.assert_allclose(.1*expm(M.real), Phead, atol=1e-12)
    X, U, Y = _stack_transitions(data0)
    augmented = np.zeros((3, 3)); augmented[:2, :2] = F; augmented[0, 2] = 1
    E = expm(augmented*.02)
    pred = X.double().numpy() @ E[:2, :2].T + U.double().numpy() @ E[:2, 2:].T
    oracle_mse = float(np.mean((pred-Y.double().numpy())**2))
    # Verify bias/variance decomposition against every completed run.
    errors = []
    for path in OUT.glob('*/result.json'):
        r = json.loads(path.read_text()); m = r['metrics']
        reconstructed = m['best_constant_normalized_floor'] + m['learned_CB_sq_distance_to_best']
        errors.append(abs(reconstructed-m['response_normalized']))
    assert max(errors, default=0) < 2e-6, 'Response decomposition mismatch'
    result = {'rotation_controls': controls,
        'decomposition': 'E_response = weighted Var(G) + ||CB - weighted mean(G)||^2, since ||G(x)||=1',
        'max_float32_metric_decomposition_error': max(errors, default=0),
        'checked_runs': len(errors),
        'original_SPD_constraint': 'B_physical[1] = sum(H_head[1,:]^2)+eps >= .1; cannot equal true [1,0] at a=0',
        'original_a0_response_lower_bound': .01,
        'general_head_oracle': {'P_head': Phead.tolist(), 'determinant': float(np.linalg.det(Phead)),
            'AB_reconstruction_max_error': float(max(np.max(np.abs(A[:2,:2]-F)), np.max(np.abs(B[:2]-[[1.],[0.]])))),
            'real_exponential_realization': True, 'heldout_one_step_exact_ZOH_mse': oracle_mse,
            'meaning': 'analytic feasibility control only; not a trained result'},
        'scope': 'Physical continuous-time action response on a frozen probe; not a bound on rollout MSE or proof of H2 superiority'}
    save(OUT/'mathematical_controls.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    audit()
