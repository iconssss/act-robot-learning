# Multi-seed confirmation and interview wording

## Completed protocol

The original chunk ablation used seed 1000. Baseline and short were then
retrained with seeds 1001 and 1002. Each of the six final measurements uses
100k training steps and 50 independent closed-loop evaluation episodes.

| condition | seed 1000 | seed 1001 | seed 1002 | mean +/- sample std |
| --- | ---: | ---: | ---: | ---: |
| baseline `(100,100)` | 70.0% | 84.0% | 64.0% | 72.67% +/- 10.26 |
| short `(50,50)` | 86.0% | 70.0% | 72.0% | 76.00% +/- 8.72 |

## Correct interpretation

The short configuration has a higher observed three-seed mean by 3.33
percentage points. However, the seed-to-seed variation overlaps, so the result
does not establish statistical superiority. The accurate claim is:

> Under this dataset, simulator, and three-seed training protocol, short
> `(50,50)` achieved a slightly higher mean closed-loop success rate than the
> baseline `(100,100)`. It is promising evidence for more frequent replanning,
> not proof that short chunks are universally superior.

## Interview correction

When using `docs/07_interview_prep.md`, replace any earlier single-seed claim
of “short is better” with the statement above. The original seed-1000 values
(86% vs 70%) remain valid individual runs, but they are not the final
cross-seed conclusion. Per-run and machine-readable aggregate values are in
`results/tables/multiseed_final_runs.csv` and
`results/tables/multiseed_summary.csv`.
