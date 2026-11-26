# Configurable MuJoCo Box-Pushing Environment

The box-pushing task is the single task definition. Curriculum, action latency,
action noise, and physics mismatch change settings or wrap this same environment;
they do not define separate tasks.

## Task

A four-wheel car must push a free box into a fixed target zone and keep it there
for consecutive control steps. The simulator uses `box_push.xml`, a 0.002 s
physics timestep, and six continuous car controls. Episodes end on success or
truncate at the configurable horizon.

The 13-value observation contains car planar position and heading, car-frame
forward/lateral/yaw velocity, box planar position and heading, and target planar
position. Actions are six finite values in `[-1, 1]`.

The shaped reward is the change in car-to-box distance and box-to-target
distance, plus a small `-0.01` step cost and a `+10` success reward. Success
requires the box center to remain within the configurable target radius for
five consecutive steps by default.

## Curriculum and settings

`BoxPushSettings` centralizes target location, target-zone radius, success hold
time, horizon, frame skip, physics mismatch scales, and curriculum stage. The
easy stage starts aligned and near the target. The randomized stage varies
car/box lateral offsets, headings, and box distance.
`reset(options={"start_pose": ...})` supports exact reproducible starts.

`ActionLatency` delays actions by a configurable number of control steps.
`ActionNoise` adds seeded Gaussian action noise and clips to the action bounds.
Both are Gymnasium wrappers around `BoxPushEnv`; they can be independently
enabled or composed.

Physics mismatch is evaluated on the same task by scaling box mass and ground,
tire, and box friction at model construction.

## Training

Install the existing project dependencies (`mujoco`, `gymnasium`,
`stable-baselines3[extra]`, and `tensorboard`), then run five independent seeds:

```powershell
0..4 | ForEach-Object {
    python train_box_push.py --seed $_ --output "models\box_push_seed_$_.zip"
}
```

By default, each run spends 250,000 steps in the easy stage and the remaining
steps in randomized starts. `--easy-steps`, `--timesteps`, `--num-envs`,
`--target-x`, `--target-y`, `--success-radius`, `--success-hold-steps`,
`--max-steps`, `--frame-skip`, `--latency-steps`, `--noise-std`,
`--box-mass-scale`, and `--friction-scale` are configurable. PPO collects full
rollout batches, so the actual count can exceed the requested budget; the config
records both counts and the actual steps in each stage.

## Evaluation protocol

Every variant must use at least five independently trained seeds, the same
20 fixed starts, and the same tuning runs and timesteps. Evaluate nominal,
latency, noise, and mismatch conditions separately against those exact starts.
Use equal tuning budgets for every condition and keep the seeds, PPO settings,
and training budgets fixed. The evaluator refuses fewer than five models and
requires the tuning budget to be recorded.

Train each variant with the same five seeds, PPO settings, and total step budget;
then evaluate all variants with the same five model files and tuning budget.
Example evaluation commands:

```powershell
python evaluate_box_push.py --variant nominal --models models\box_push_seed_0.zip models\box_push_seed_1.zip models\box_push_seed_2.zip models\box_push_seed_3.zip models\box_push_seed_4.zip --tuning-runs 5 --tuning-timesteps 1500000 --output results\nominal.json
python evaluate_box_push.py --variant latency --latency-steps 2 --models models\box_push_seed_0.zip models\box_push_seed_1.zip models\box_push_seed_2.zip models\box_push_seed_3.zip models\box_push_seed_4.zip --tuning-runs 5 --tuning-timesteps 1500000 --output results\latency.json
python evaluate_box_push.py --variant noise --noise-std 0.05 --models models\box_push_seed_0.zip models\box_push_seed_1.zip models\box_push_seed_2.zip models\box_push_seed_3.zip models\box_push_seed_4.zip --tuning-runs 5 --tuning-timesteps 1500000 --output results\noise.json
python evaluate_box_push.py --variant mismatch --box-mass-scale 1.5 --friction-scale 0.7 --models models\box_push_seed_0.zip models\box_push_seed_1.zip models\box_push_seed_2.zip models\box_push_seed_3.zip models\box_push_seed_4.zip --tuning-runs 5 --tuning-timesteps 1500000 --output results\mismatch.json
```

Each run reports the mean and standard deviation across training seeds for
success rate, return, and successful completion time; writes all successes and
failures to JSON and CSV; and draws a start-pose success heat map as SVG. Do not
filter failed episodes from the reported success rates or failure table.

## First experiment results (2026-11-26)

Five nominal policies were trained with seeds `0` through `4`, eight
environments, and the same PPO settings. Each requested 1,500,000 steps; PPO's
complete rollout batches produced 1,507,328 actual steps per seed (253,952 easy
stage and 1,253,376 randomized stage). The training configs beside each model
record the actual counts and PPO arguments.

Each condition below evaluates the five policies on the same 20 fixed starts
(100 episodes per condition). Values are mean ± sample standard deviation over
the five per-seed results; success-time statistics use successful episodes.

| Evaluation condition | Success rate | Episode return | Success time (s) | Failures |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 13.0% ± 5.7 pp | -5.070 ± 1.220 | 0.718 ± 0.107 | 87/100 |
| 2-step action latency | 16.0% ± 2.2 pp | -4.740 ± 0.577 | 1.343 ± 0.739 | 84/100 |
| Action noise, std 0.05 | 15.0% ± 7.9 pp | -4.754 ± 1.494 | 0.773 ± 0.195 | 85/100 |
| Mass ×1.5, friction ×0.7 | 11.0% ± 4.2 pp | -5.834 ± 0.812 | 1.380 ± 0.827 | 89/100 |

All 345 failures were timeouts. Successes in the nominal heat map were
concentrated at zero lateral start offset; inspect the per-start maps rather
than treating the aggregate rate as uniform performance.

### Results and failure data

- [Failure-only ledger (345 episodes)](results/failures.csv)
- Per-condition report, episode CSV, and heat map:
  [nominal](results/nominal.json), [episodes](results/nominal.csv),
  [heat map](results/nominal.svg);
  [latency](results/latency.json), [episodes](results/latency.csv),
  [heat map](results/latency.svg);
  [noise](results/noise.json), [episodes](results/noise.csv),
  [heat map](results/noise.svg);
  [mismatch](results/mismatch.json), [episodes](results/mismatch.csv),
  [heat map](results/mismatch.svg).

### Interpretation and limitations

These are first-pass results and do not solve the task reliably. The latency,
noise, and mismatch evaluations apply perturbations to the same nominal-trained
policies; they are robustness checks, not results from separately trained
variants. PPO settings, seeds, and training budgets were held constant for the
five nominal policies, but no hyperparameter search or variant-specific tuning
was performed. The `tuning_effort` values in the evaluation JSON are declared
protocol metadata, not evidence that a tuning search was run. A fair comparison
of separately tuned variants remains outstanding.
