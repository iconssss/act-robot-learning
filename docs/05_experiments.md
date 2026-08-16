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

## Interpretation rules (pre-registered)

- Compare final 100k, 50-episode success rates first; loss is supporting
  evidence rather than a substitute for task completion.
- Inspect both successful and failed videos before assigning a mechanism.
- A longer horizon may smooth actions but can accumulate open-loop error before
  correction. A shorter horizon replans more frequently but may lose temporal
  consistency and increases policy-query frequency.
- One seed is an engineering ablation, not a statistical claim of universal
  superiority. A follow-up should add seeds if a result is close or important.
