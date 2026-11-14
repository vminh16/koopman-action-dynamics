"""Locked CartPole matrix and actual constructed hyperparameters."""
import unittest
from research.baseline.common import (
    ROOT, build_module, check_effective_hyperparameters, dataset_fingerprint, load_config,
)

CONFIGS = [f'configs/research/cartpole_controllable_win15_frac{frac}_seed{seed}.yml'
           for frac in ('0.01', '0.05', '0.1', '0.3') for seed in (42, 43, 44)]


class TestCartPoleConfigContract(unittest.TestCase):
    def test_all_configs_exist(self):
        for rel in CONFIGS:
            self.assertTrue((ROOT / rel).exists(), f'Missing config: {rel}')

    def test_config_contract_and_effective_hyperparameters(self):
        expected = dict(system='cartpole', latent_dim=25, num_layers=3, hidden_size=64,
                        window_len=15, batch_size=64, lr=.001, epochs=100,
                        max_windows_per_traj=20, window_start_max=235,
                        model_class='KoopmanLitModule', koopman_mode='controllable',
                        control_loss_flag=True, encoder_loss_flag=True, use_modulated_u=False,
                        u_encoder_layers=2, u_encoder_hidden=64, deterministic=False,
                        optimizer='AdamW', weight_decay=1e-5, scheduler='ReduceLROnPlateau',
                        scheduler_factor=.1, scheduler_patience=10, prediction_gamma=.8,
                        lambda_encoder_pred=.5, lambda_control=.005, gramian_horizon_steps=15,
                        data_train='data/cartpole_train.pt', data_test='data/cartpole_test.pt')
        for rel in CONFIGS:
            with self.subTest(config=rel):
                cfg = load_config(ROOT / rel)
                for key, value in expected.items():
                    self.assertEqual(cfg[key], value, key)
                fraction = rel.split('_frac')[1].split('_')[0]
                seed = int(rel.split('_seed')[1].split('.')[0])
                index = ('0.01', '0.05', '0.1', '0.3').index(fraction) + 1
                self.assertEqual(cfg['seed'], seed)
                self.assertEqual(cfg['train_fraction'], float(fraction))
                self.assertEqual(cfg['num_train_trajectories'], int(10000 * float(fraction)))
                self.assertEqual(cfg['run_id'], f'C{index:02d}-s{seed}')
                self.assertEqual(cfg['checkpoint'], f"checkpoints/{cfg['run_id']}.pt")
                self.assertEqual(set(dataset_fingerprint(cfg)), {'train', 'test'})
                module = build_module(cfg)
                effective = check_effective_hyperparameters(module, cfg)
                self.assertEqual(effective['latent_dim'], 25)
                self.assertEqual(effective['state_dim'], 4)
                self.assertEqual(effective['dt'], .02)

    def test_drift_is_rejected(self):
        cfg = load_config(ROOT / CONFIGS[0])
        module = build_module(cfg)
        for key, value in (('latent_dim', 16), ('lambda_control', .01), ('lr', .002)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_effective_hyperparameters(module, {**cfg, key: value})

    def test_consistently_wrong_cartpole_dimension_is_rejected(self):
        cfg = {**load_config(ROOT / CONFIGS[0]), 'latent_dim': 16}
        module = build_module(cfg)
        with self.assertRaisesRegex(ValueError, 'CartPole.*latent_dim=25'):
            check_effective_hyperparameters(module, cfg)
