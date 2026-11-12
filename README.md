# Koopman Action Dynamics

# Controllable Koopman

Code for learning controllability-aware Koopman representations for nonlinear control systems.

The learned model has the form

$\dot z = A z + B u$

where the lifted state $begin:math:text$z$end:math:text$ contains the original physical state $begin:math:text$x$end:math:text$ as its first components. This allows the learned Koopman model to be evaluated not only by prediction accuracy, but also by controllability properties related to the original state.

## Structure

```text
koopman-action-dynamics/
├── dynamics/       # Benchmark nonlinear systems
├── experiments/    # Training, evaluation, and analysis scripts
├── models/         # Encoders, Koopman operators, and baseline models
├── my_utils/       # Dataset, simulation, and visualization utilities
├── training/       # Losses and training modules
├── docs/            # Original paper and source documentation
├── research/        # Research implementation and extensions
├── configs/         # Experiment configurations
├── scripts/         # Reproducible command wrappers
├── tests/           # Tests
├── data/            # Local datasets
├── checkpoints/     # Trained checkpoints
├── outputs/         # Generated logs, figures, and reports
├── TRACKING.md      # Run registry and results
├── EXPERIMENT.md    # Experimental protocol
└── ROADMAP.md       # Research roadmap
```

## Installation

```bash
git clone https://github.com/vminh16/koopman-action-dynamics.git
cd koopman-action-dynamics
pip install -r requirements.txt
```

## Training

Train a free Koopman model:

```bash
PYTHONPATH=. python experiments/run_train.py \
    --system pendulum \
    --koopman_mode free \
    --latent_dim 16 \
    --window_len 50 \
    --batch_size 64 \
    --epochs 100 \
    --lr 5e-3
```

Train a controllability-structured Koopman model:

```bash
PYTHONPATH=. python experiments/run_train.py \
    --system pendulum \
    --koopman_mode controllable \
    --control_loss_flag \
    --latent_dim 16 \
    --window_len 50 \
    --batch_size 64 \
    --epochs 100 \
    --lr 5e-3
```

## Evaluation

Evaluate a trained model:

```bash
PYTHONPATH=. python experiments/run_eval.py \
    --config configs/path_to_config.yaml \
    --window_len 250 \
    --max_traj 100 \
    --num_plot 5
```

Evaluate with periodic re-encoding:

```bash
PYTHONPATH=. python experiments/run_eval.py \
    --config configs/path_to_config.yaml \
    --window_len 250 \
    --use_reencode \
    --reencode_step 50
```

## Controllability Analysis

```bash
PYTHONPATH=. python experiments/analyze_controllability.py \
    --config configs/path_to_config.yaml
```

## Citation

```bibtex
@inproceedings{miao2026learning,
  title={Learning Koopman Representations with Controllability Guarantees},
  author={Miao, Keyan and Wang, Han and Ding, Xuda and Gatsis, Konstantinos and Krause, Andreas and Papachristodoulou, Antonis},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```
