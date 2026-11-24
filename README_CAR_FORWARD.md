# MuJoCo Car Forward Training

The environment, vehicle model, training script, and evaluator implement the
requirements in [DRIVE_FORWARD_SPEC.md](DRIVE_FORWARD_SPEC.md).

## Preconditions

Before training, run:

```powershell
python check_car.py
```

This verifies the six-actuator model, the 14-value observation, the green
non-colliding finish marker at `(15, 0, 0)`, the required drive distance, and a
full-throttle goal run. Also open `car.xml` in MuJoCo's `simulate.exe` and
visually confirm the finish marker before training.

## Train

```powershell
python train_car.py --seed 0 --timesteps 1500000 --num-envs 8
```

Training writes `ppo_car_forward_v1.zip`, the compatibility copy
`ppo_car_forward.zip`, `training_config.json`, per-environment Monitor logs,
and TensorBoard event files under `runs`.

Stable-Baselines3 accepts the training seed as `PPO(seed=...)`, not as a
`model.learn` argument. The vector environment receives that seed on reset.

## Evaluate

```powershell
python evaluate_car_forward.py --model ppo_car_forward_v1.zip --episodes 100 --seed 0
```

Evaluation runs deterministically, reports all six acceptance checks, and
writes `evaluation_metrics.json`. The stability check uses the final 200,000
training steps divided into ten equal windows. It requires positive final
reward, a nonnegative trend, at most five decreasing adjacent-window means,
and no more than 1% relative range.

To open a trained policy in the MuJoCo viewer:

```powershell
python watch_car.py
```
