# Configurable MuJoCo Box-Pushing Environment

The box-pushing task is the single task definition. Curriculum, action latency,
action noise, and physics mismatch change settings or wrap this same environment;
they do not define separate tasks.

## Task

A four-wheel car must push a free box into a fixed target zone and keep it there
for consecutive control steps. The simulator uses `box_push.xml`, a 0.002 s
physics timestep, and six continuous car controls. Episodes end on success or
truncate at the configurable horizon.

The observation contains car planar position and heading, car-frame
forward/lateral/yaw velocity, a 360-degree range-only LiDAR scan, and target
planar position. The default LiDAR has 32 rays, a 4 m maximum range, and 1 cm
Gaussian range noise; no return is reported as maximum range. The car's own
geometry is excluded from the scan. The box pose is not exposed directly to the
policy. Actions are six finite values in `[-1, 1]`.

The shaped reward is the change in car-to-box distance and box-to-target
distance, plus a small `-0.01` step cost and a `+10` success reward. Success
requires the box center to remain within the configurable target radius for
five consecutive steps by default. Reward and success also require the box to
remain near the ground and below the configurable speed limit, preventing
airborne/high-speed motion from receiving a false push reward. Additional
penalties discourage car speeds above 1 m/s, losing physical contact between
the box and the front face of the car chassis (the bumper), and pushing the box
through or out of the target zone. Defaults are a speed-penalty weight of 1.0,
a 1.0 contact-loss penalty, and a 2.0 target-overshoot penalty. The speed
penalty scales with excess speed and elapsed control time; contact-loss and
target-overshoot penalties are applied once per event.

## Curriculum and settings

`BoxPushSettings` centralizes target location, target-zone radius, success hold
time, horizon, frame skip, speed and event penalty weights, physics mismatch
scales, and curriculum stage. The easy stage starts aligned and near the
target. The randomized stage varies car/box lateral offsets, headings, and box
distance.
`reset(options={"start_pose": ...})` supports exact reproducible starts.

`ActionLatency` delays actions by a configurable number of control steps.
`ActionNoise` adds seeded Gaussian action noise and clips to the action bounds.
Both are Gymnasium wrappers around `BoxPushEnv`; they can be independently
enabled or composed.

Physics mismatch is evaluated on the same task by scaling box mass and ground,
tire, and box friction at model construction.
The box-floor sliding coefficient defaults to `0.35` (a light carton on a
smooth hard floor); tire friction remains higher to preserve car traction.
Change `BoxPushSettings.box_floor_friction` to explore a different surface.

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
`--box-floor-friction` adjusts the carton/floor sliding coefficient; keep it the
same for training and evaluation runs being compared.
The speed and event penalties are configurable with `--car-speed-limit`,
`--car-speed-penalty-per-second`, `--lost-bumper-contact-penalty`, and
`--target-overshoot-penalty`.
The sensor is configurable with `--lidar-num-rays`, `--lidar-max-range`, and
`--lidar-noise-std`. Policies trained with the previous direct box-pose
observation are incompatible and must not be evaluated in this sensor setup.

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

## Second experiment: LiDAR and carton friction (2026-11-26)

Five nominal policies were retrained with seeds `0` through `4`, eight
environments, and the same PPO settings and step budgets as the first
experiment. Each requested 1,500,000 steps and completed 1,507,328 actual steps
(253,952 easy-stage and 1,253,376 randomized-stage). The box-floor friction
coefficient was `0.35`. The policy observation used a 32-ray, 360-degree
range-only LiDAR with 4 m maximum range and 0.01 m seeded range noise; box pose
was not provided directly.

All five policies were evaluated on the same 20 fixed starts (100 episodes).
Values are mean ± sample standard deviation over the five seeds:

| Evaluation condition | Success rate | Episode return | Success time (s) | Failures |
| --- | ---: | ---: | ---: | ---: |
| Nominal, LiDAR and carton friction | 10.0% ± 0.0 pp | -35.983 ± 1.408 | 0.602 ± 0.006 | 90/100 |

All 90 failures were timeouts. Results include all episodes; the JSON report
also has a dedicated `failures` list.

- [Evaluation report](results/lidar_carton035.json)
- [All 100 episode records](results/lidar_carton035.csv)
- [Fixed-start heat map](results/lidar_carton035.svg)
- Each seed's policy and training config:
  `models/box_push_lidar_carton035_seed_0.zip` through
  `models/box_push_lidar_carton035_seed_4.zip` and matching `_config.json` files.

## Third experiment: LiDAR, carton friction, and event penalties (2026-11-27)

Five policies were trained with seeds `0` through `4` and the same PPO settings
and budgets. Each completed 1,507,328 steps. This run used the same 0.35
box-floor friction and LiDAR settings as the second experiment, plus a 1 m/s
car-speed limit, speed penalty weight `1.0`, contact-loss penalty `1.0`, and
target-overshoot penalty `2.0`.

The five policies were evaluated on the same 20 fixed starts (100 episodes):

| Evaluation condition | Success rate | Episode return | Success time (s) | Failures |
| --- | ---: | ---: | ---: | ---: |
| LiDAR, carton friction, and penalties | 12.0% ± 4.5 pp | -41.215 ± 6.267 | 0.998 ± 0.868 | 88/100 |

All 88 failures were timeouts.

- [Evaluation report](results/lidar_carton035_rewardv2.json)
- [All 100 episode records](results/lidar_carton035_rewardv2.csv)
- [Fixed-start heat map](results/lidar_carton035_rewardv2.svg)
- Each seed's policy and training config:
  `models/box_push_lidar_carton035_rewardv2_seed_0.zip` through
  `models/box_push_lidar_carton035_rewardv2_seed_4.zip` and matching
  `_config.json` files.

### Interpretation and limitations

The first-experiment results above were collected before ground-level/speed
success constraints, carton-floor friction, and LiDAR observations were added.
They describe the earlier center-only success rule with direct box-pose
observations and are not directly comparable with the later experiments.

None of the experiments solves the task reliably. The second experiment had
10% success and the third had 12%; this small difference is not evidence that
the new penalties improve performance. Returns are also affected by the changed
reward function. The first experiment's latency, noise, and mismatch
evaluations apply perturbations to the same nominal-trained policies; they are
robustness checks, not results from separately trained variants. PPO settings,
seeds, and training budgets were held constant, but no hyperparameter search or
variant-specific tuning was performed. The `tuning_effort` values in the
evaluation JSON are declared protocol metadata, not evidence that a tuning
search was run. A fair comparison of separately tuned variants remains
outstanding.
