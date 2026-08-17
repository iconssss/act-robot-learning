# Cloud artifact cleanup and reproducibility boundary

## Final Project 1 GPU status

The requested baseline/short multi-seed confirmation completed serially on the
RTX 4090 cloud container between 2026-08-16 22:52 and 2026-08-17 07:28
(about 8 hours 36 minutes; approximately RMB 17.2 at RMB 2/hour). The final
aggregate results are committed in `results/tables/` and explained in
`docs/08_multiseed_confirmation.md`.

After those results and supporting figures were committed, the cloud copies of
this project's raw runtime artifacts were intentionally removed to free the
shared storage for a later project:

- `/root/shared-nvme/act-robot-learning`
- `/root/shared-nvme/datasets/aloha_sim_transfer_cube_human`
- `/root/shared-nvme/conda-envs/lerobot-act`
- `/root/shared-nvme/hf-cache`
- `/root/shared-nvme/torch-hub`

The final check found the RTX 4090 idle and about 36 GB free on the 50 GB
shared volume. This repository therefore preserves the reproducibility record
(source, pinned configuration, commands, metrics CSVs, figures, and contact
sheets), but not raw checkpoints, downloaded dataset files, or rollout MP4s.

## Re-running later

To rerun a GPU experiment, recreate the isolated cloud environment and dataset
from the pinned versions in `environment/`, then run the committed training and
evaluation scripts. This cleanup does not affect the Windows working copy,
Git history, or public GitHub repository. No credential, private key, token,
or cloud password is stored in this repository.
