# Experiments

## Action-chunk configuration ablation

**Question.** How does the jointly configured ACT prediction horizon and
replanning interval affect closed-loop transfer-cube success?

| label | `chunk_size` | `n_action_steps` | horizon / replanning interval at 50 Hz |
| --- | ---: | ---: | --- |
| short | 50 | 50 | 1.0 s |
| baseline | 100 | 100 | 2.0 s |
| long | 150 | 150 | 3.0 s |

`chunk_size` is the future sequence length predicted during training.
`n_action_steps` is how many actions are selected from that predicted sequence
before the next policy query. They are varied together here so that each
condition has a self-consistent horizon; the claim is therefore about an
**action-chunk configuration**, not about one field in isolation.

All other variables are controlled: dataset revision, single top camera,
14-D state/action representation, ACT architecture, seed 1000, batch size 8,
100k optimisation steps, optimizer defaults, 50-episode evaluation protocol,
and EGL rendering backend. The baseline's already-complete outcome is 70.0%
(35/50) success.

Run each new condition serially on **robot-cloud**:

```bash
bash scripts/train_chunk_ablation.sh short
bash scripts/train_chunk_ablation.sh long
```

The two added runs are expected to take about 5--6 GPU hours in total,
including four 50-episode milestones per run. At the observed 2 RMB/hour
voucher rate, this is about 10--12 RMB. Each run writes checkpoints, logs and
rollout videos under `/root/shared-nvme/act-robot-learning/experiments/chunk_ablation/`.

## Result

All three conditions completed 100k optimisation steps and the same final
50-episode closed-loop evaluation protocol.

| configuration | final successes | success rate | average return |
| --- | ---: | ---: | ---: |
| short `(50, 50)` | 43 / 50 | **86.0%** | **239.22** |
| baseline `(100, 100)` | 35 / 50 | 70.0% | 194.18 |
| long `(150, 150)` | 41 / 50 | 82.0% | 220.92 |

Under this fixed seed and simulator protocol, the short configuration produced
the strongest final outcome. It replans every 1.0 second rather than every 2.0
or 3.0 seconds, which is consistent with the hypothesis that more frequent
visual feedback helps correct manipulation drift. The long condition still
outperformed the baseline, so this single result does **not** establish a
monotonic relationship or prove that shorter is universally better. The
machine-readable table is `results/tables/chunk_ablation_metrics.csv`.

## Interpretation rules (pre-registered)

- Compare final 100k, 50-episode success rates first; loss is supporting
  evidence rather than a substitute for task completion.
- Inspect both successful and failed videos before assigning a mechanism.
- A longer horizon may smooth actions but can accumulate open-loop error before
  correction. A shorter horizon replans more frequently but may lose temporal
  consistency and increases policy-query frequency.
- One seed is an engineering ablation, not a statistical claim of universal
  superiority. A follow-up should add seeds if a result is close or important.

## Multi-seed confirmation protocol

The original comparison used seed 1000. To measure training-run variance for
the central result, repeat only the baseline `(100,100)` and short `(50,50)`
conditions with seeds **1001** and **1002**. This adds four full 100k-step
runs. Dataset revision, all model fields except the seed, batch size, steps,
evaluation frequency, 50-episode final evaluation, and EGL backend remain
fixed.

Run serially on **robot-cloud**:

```bash
bash scripts/train_multiseed.sh baseline 1001
bash scripts/train_multiseed.sh short 1001
bash scripts/train_multiseed.sh baseline 1002
bash scripts/train_multiseed.sh short 1002
```

The expected additional runtime is 10--12 RTX 4090 hours (about 20--24 RMB at
the observed 2 RMB/hour rate). The long condition is not repeated because the
immediate question is whether short improves on the baseline; this is a
targeted confirmation rather than a full three-condition statistical study.

### Multi-seed result

All six final evaluations use 50 fresh rollout episodes. The original seed
1000 result remains part of the comparison.

| condition | seed 1000 | seed 1001 | seed 1002 | mean +/- sample std |
| --- | ---: | ---: | ---: | ---: |
| baseline `(100,100)` | 70.0% | 84.0% | 64.0% | 72.67% +/- 10.26 |
| short `(50,50)` | 86.0% | 70.0% | 72.0% | 76.00% +/- 8.72 |

Short has a 3.33 percentage-point higher mean under these three seeds, but the
sample variation overlaps. It is therefore accurate to call short a promising
configuration under this benchmark, not to claim statistical superiority. The
per-run and aggregate records are in `results/tables/multiseed_final_runs.csv`
and `results/tables/multiseed_summary.csv`.
