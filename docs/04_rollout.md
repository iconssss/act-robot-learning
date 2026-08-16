# Closed-loop rollout and baseline evaluation

## Result

The completed ACT baseline was evaluated in `AlohaTransferCube-v0` on
`robot-cloud`. Each complete milestone uses 50 fresh simulation episodes with
a 400-step cap. The final 100k-step policy achieves **70.0% success (35/50)**.

| checkpoint step | episodes | success rate | average return | evaluation time |
| ---: | ---: | ---: | ---: | ---: |
| 20,000 | — | not reported | — | interrupted before EGL fix |
| 40,000 | 50 | 64.0% | 187.42 | 22.1 min |
| 60,000 | 50 | 60.0% | 168.40 | 21.9 min |
| 80,000 | 50 | 58.0% | 179.56 | 23.0 min |
| 100,000 | 50 | **70.0%** | **194.18** | 10.7 min |

The 20k checkpoint remains valid, but its evaluation was interrupted by the
headless OpenGL error before any aggregate metric existed. It is intentionally
marked missing rather than presented as a zero-success result. The machine-
readable table is `results/tables/baseline_metrics.csv`.

## Actual closed loop

At each control cycle the environment provides a top RGB image and 14-D robot
state. ACT predicts an action chunk; rollout executes its selected action,
observes the changed environment, and queries the policy again. The controller
therefore acts on its own previous consequences instead of replaying a
demonstration.

```text
env.reset() -> RGB + proprioception -> ACT.select_action()
    -> action from predicted chunk -> env.step(action)
    -> new observation -> replan
```

## Artifacts and regeneration

Four MP4 rollout videos are retained for each complete checkpoint in cloud
shared storage, including the final directory:

```text
/root/shared-nvme/act-robot-learning/experiments/baseline/
  act_aloha_transfer_cube_seed1000/eval/videos_step_100000/aloha_0/
```

Videos and checkpoints are not committed to Git because they are large. To
regenerate complete metrics from the trainer log, run on **robot-cloud**:

```bash
python scripts/summarize_results.py \
  --log /root/shared-nvme/act-robot-learning/experiments/baseline/act_aloha_transfer_cube_seed1000.resume_020000.console.log \
  --output-csv results/tables/baseline_metrics.csv
```

The next stage selects specific successful and failed episodes for qualitative
analysis, then runs the controlled action-chunk ablation.
