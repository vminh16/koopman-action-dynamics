# Diagnostic experiment protocol — 2026-11-15

User-authorized experimental spike; no production baseline code or old checkpoint is changed.

Scope: Pendulum and CartPole at 5%, using the existing locked data and Table 4/5 dimensions, encoder, batch 64, initial lr .001, 100 epochs. These are diagnostic variants, not pre-certified exact reproductions of the paper. Reference paper 5% MSE: Pendulum .0318; CartPole .0024. The exact Table 1 evaluation window/reduction is not fully specified, so report ratios without inventing a pass tolerance.

Paired arms, seed 42 initially:

- baseline: original SPD orientation, original losses, original last-batch scheduler semantics.
- signed (CartPole only): right-multiply the block-diagonal SPD transform by a fixed diagonal signature with -1 on the final physical coordinate. This preserves invertibility and the theorem's output rank while permitting negative angular input response. Pendulum uses positive orientation as its physical input response is positive.
- epochmean: only change the scheduler monitor to the sample-weighted epoch mean.
- constant: only disable LR reduction.
- noenc: only omit both encoder-consistency penalties.

Do not infer causal effects from original-vs-combined changes. If follow-up combinations are run, identify which paired single-factor results justify them and label all changes. A Gramian-formula change is evaluated separately; the literal reciprocal at eigenvalues near 1e-12 is numerically hazardous and cannot be silently substituted.

Same seed -> identical initialization, train subset, fixed 20 windows/trajectory and explicit minibatch order across arms. Shuffle generator 700000+seed is recorded and differs from the historical Lightning shuffle generator; the new paired baseline is the causal comparator. Each arm uses 250 (Pendulum) or 500 (CartPole) train trajectories and full batches plus the final partial batch. Cached Euler rollout must first match official forward values and parameter gradients.

Validation: 200 trajectories drawn with seed 20261115 from the complement of the union of training subsets for seeds 42/43/44. No overlap with any of those train subsets or the test split. Validation metric: unweighted first-15-step physical-state MSE. Use final epoch, not the best test checkpoint. Select follow-up variants using validation only; confirm promising candidates with seeds 43/44. Evaluate all 1000 test trajectories only in an explicit final reporting step.

Log full config, indices, hashes, updates, mean and last-batch training losses, LR trajectory, validation MSE, elapsed time, failures and checkpoint hash. Nonfinite values fail an arm rather than being silently patched. Do not overwrite original runs, declare exact replication from one successful seed, or open H2/H3 method tracks.

## Geometry audit extension, before test evaluation

A single-coordinate signature has negative determinant for CartPole; a local canonical realization of the actual CartPole linearization has positive determinant. Treat the initial `signed` arm as an incomplete-orientation diagnostic, not a proposed final repair. Add:

- negative_head: negate the entire CartPole physical SPD block (n=4), preserving positive determinant and allowing negative angular gain. Tail block remains unchanged. This is an isolated orientation change.
- general_head: allow the physical block to vary as `P_head_initial * exp(M)`, starting at the same negative physical block for CartPole and positive block for Pendulum, with M initialized to zero. The existing square H_head parameter stores M. This preserves invertibility for finite parameters while removing symmetry/definiteness restrictions locally. A saved `diagnostic_head_base` buffer makes the checkpoint interpretation explicit. This remains the author's canonical A/B architecture with the more general invertible transform permitted by Theorem 2; it is a separately labeled variant, not an H2/H3 model.

General-vs-negative_head compares flexibility at the same initial dynamics; general-vs-baseline on Pendulum does likewise. No oracle-gradient training is used. Geometry arms must still pass output-rank and inverse identities before their results are accepted.

## Gradient hypothesis control

`nogram` removes only the actually executed softplus Gramian penalty and retains the same train/init/shuffle seeds. A 100-epoch paired result, not only a small final gradient, is needed to establish whether that particular penalty materially affects these runs. Removing it also changes the monitored total loss by its scalar contribution; this is part of the stated single-factor objective ablation. It does not test the paper's different reciprocal/condition objective.

Confirmatory pairs: Pendulum baseline vs noenc, seeds 42/43/44; CartPole baseline vs negative_head, seeds 42/43/44. These pairs were chosen on training/validation evidence before final test evaluation. Other single-seed arms remain exploratory and are not promoted by test performance.

## Explicit domain-control extension (diagnostic, not faithful paper replication)

Training sampled windows cover the entire CartPole trajectory, where the physical angular input gain varies strongly with theta. The first-15-step validation gain is near constant negative. After seeing the paired geometry/optimization validation results, run `baseline_start0` and `negative_head_start0` with all 20 per-trajectory samples fixed at start=0, preserving trajectory count, initialization, batch ordering and optimizer-update budget. This intentionally changes the paper's stated random-window usage. Its purpose is to test domain mismatch and local capacity, not to pass Table 1 by changing data selection. It cannot promote the reproduction verdict even if it meets the numerical reference. Test evaluation remains restricted to the preselected three-seed baseline/negative_head pair.

The strong start=0 validation interaction is confirmed for negative_head_start0 at seeds 43/44. After those full 100-epoch runs finish, report held-out first-15-step MSE once, labeled DIAGNOSTIC_ONLY_NOT_FAITHFUL_PAPER_REPRODUCTION. The baseline_start0 comparison has seed 42 only; do not imply a three-seed paired baseline_start0 result. Each negative_head_start0 run has 500 unique first windows repeated 20 times to match the 15,700-update budget.
