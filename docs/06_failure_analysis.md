# Failure analysis

## Protocol

Failure analysis uses recorded closed-loop rollout videos, not training loss
alone. Each selected case must identify its configuration, checkpoint,
episode, success label, visible behavior, and the most plausible mechanism.
The analysis distinguishes observed facts from hypotheses.

The video index is generated from LeRobot's per-episode evaluation records by
`scripts/index_recorded_rollouts.py`. This avoids visually guessing whether a
video succeeded.

## Candidate case selection

Select 5--10 videos labelled `success=false` across the baseline, short, and
long conditions. Prefer a mixture of:

- failure before contact (object localization or approach);
- unstable or mistimed grasp;
- post-grasp drop or trajectory drift;
- late-stage placement error;
- long open-loop execution without timely correction.

## Case template

| case | configuration | checkpoint | episode | observed behavior | evidence | hypothesis |
| --- | --- | ---: | ---: | --- | --- |
| pending extraction | — | — | — | — | rollout MP4 + metric | — |

No causal conclusion should be written until the selected MP4 is reviewed.

## Interpretation boundaries

A 50-episode success rate quantifies task completion under the fixed simulator
protocol. It does not prove robustness to unseen object positions, camera
changes, real hardware dynamics, or another random seed. The current
one-seed chunk ablation is an engineering comparison; multi-seed repetitions
are the appropriate GPU-backed follow-up if a stronger statistical claim is
needed.
