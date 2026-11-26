# Drive Forward to Finish Line

> Legacy task specification. New experiments should use the shared box-pushing
> environment in [BOX_PUSH_SPEC.md](BOX_PUSH_SPEC.md); this document describes
> only the retained forward-driving baseline.

This specification is written before implementation. If later metrics conflict
with it, the specification takes precedence.

## 1. Problem statement

A four-wheeled vehicle, spawned at a known position and orientation in a flat
MuJoCo world, must drive forward along a straight corridor and cross a finish
line at a fixed distance ahead. The agent observes its own state and remaining
distance to the finish, and outputs continuous control signals for steering
and drive actuators. The episode terminates when the car crosses the finish
line, flips over, or exceeds the time limit. Success means reaching the finish
line before the time limit without flipping.

## 2. Formal MDP definition

### Environment and episode

- Environment: MuJoCo simulation of `car.xml`.
- Physics timestep: `dt = 0.002 s`.
- Control period: `frame_skip * dt`.
- Episode start: car at `(0, 0, z0)` with identity orientation; randomize yaw
  by up to `+/-0.02 rad` for robustness.
- Goal: cross the plane `x = D`, with `D = 15.0 m`, within a corridor of
  `y = +/-3 m`.
- Horizon: `H = 500` agent steps (approximately 5 seconds at 100 Hz control).

### State space

The observation is a 14-dimensional vector:

| Indices | Content | Unit |
| --- | --- | --- |
| 0-2 | Chassis position `(x, y, z)` | m |
| 3-6 | Chassis orientation quaternion `(w, x, y, z)` | — |
| 7-9 | Chassis linear velocity `(vx, vy, vz)` in world frame | m/s |
| 10-12 | Chassis angular velocity `(wx, wy, wz)` in world frame | rad/s |
| 13 | Remaining distance to finish `D - x` | m |

### Action space

Continuous raw actuator commands from the model, normalized to `[-1, 1]`.
Drive wheels and steering are mapped automatically via `mjModel.nu`. There is
no action preprocessing, scaling, or shaping.

### Transition

Apply the action to `data.ctrl`, then step physics `frame_skip` times using
standard MuJoCo integration.

### Reward

Compute `fwd`, `lat`, and `yaw` in the car's own frame:

```text
remaining = D - x
reached = remaining <= 0

r = +1.0 * fwd
  - 0.30 * |lat|
  - (0.50 if |fwd| < 0.2 else 0.02) * |yaw|
  - 0.01 * max(remaining, 0)
  + 50.0 * 1[reached]
```

Every term is required. Do not omit a term without re-validating that
`ep_rew_mean` remains positive and monotonically improving.

### Termination

`done = terminated OR truncated`, preserving the distinction for PPO
bootstrapping:

| Condition | Type | Meaning |
| --- | --- | --- |
| `remaining <= 0` (`x >= D`) | `terminated` | Success: chassis crosses the finish plane |
| `z < 0.10` | `terminated` | Flipped |
| `t >= H` | `truncated` | Timeout |

### Discount factor

`gamma = 0.99`. The specified rationale is that with `H = 500` and
`frame_skip = 5`, this gives an approximately 1.5-second effective planning
horizon: enough to see the finish line without making credit assignment
impractical.

## 3. Acceptance tests

Evaluate the trained policy deterministically on 100 episodes. All six
criteria must pass:

| # | Metric | Threshold | Measurement |
| --- | --- | --- | --- |
| 1 | Goal rate | At least 90% | `terminated AND reached`, `deterministic=True` |
| 2 | Mean time to goal | At most 4.0 s | `d.time` at goal (5.0 s timeout) |
| 3 | Flip rate | At most 5% | By termination reason |
| 4 | Lateral drift | `|y_goal| <= 1.0 m` | Position at finish |
| 5 | Trajectory smoothness | Mean `|yaw_rate| < 0.3 rad/s` | Logged episode states |
| 6 | Training stability | `ep_rew_mean` monotone-ish over final 200k steps | TensorBoard |

A run that fails any criterion is not complete. Do not select only successful
rollouts.

For reproducible pass/fail evaluation of "monotone-ish", the evaluator divides
the final 200,000 TensorBoard steps into ten equal windows and averages
`rollout/ep_rew_mean` in each. It passes this criterion only if the final
window is positive, the least-squares trend is nonnegative, at most five of the
nine adjacent-window transitions decrease, and the total range is at most 1%
of the mean reward. This tolerance treats small sampling noise on a stable
plateau as monotone-ish while rejecting material fluctuation or decline.

## 4. Non-goals

- Obstacle avoidance.
- Path following on non-straight routes.
- Sim-to-real transfer.
- Multi-agent interaction.
- Non-flat terrain.
- Steering beyond minor corrections.

Anything outside this list requires a separate specification.

## 5. Preconditions

Training must not begin until all four conditions hold:

1. `import mujoco; print(mujoco.__version__)` succeeds.
2. `pip install gymnasium stable-baselines3[extra] tensorboard` completed.
3. The actuator-mapping test passes: setting only drive-wheel actuators to
   `+0.5` for 1000 steps yields `delta x > 1.0 m`.
4. The finish line is visible: `simulate.exe` with the compiled model shows a
   green box at `(15, 0, 0)`.

If any precondition fails, fix it before training.

## 6. Registered failure modes

| Symptom | Likely cause | Fix to investigate |
| --- | --- | --- |
| `ep_rew_mean` approximately `-50`, `ep_len = H` | Yaw penalty dominates; car spins in place | Lower yaw coefficient 10x; verify actuator mapping |
| `ep_rew_mean` climbs to `+20` then collapses | Sparse goal bonus; PPO overfits to one lucky episode | Add distance shaping (already specified), or reduce `+50` to `+20` |
| `ep_len_mean` collapses to approximately `30` | Car flips early; upright invariant violated | Check spawn height; add a small upright penalty |
| Car drives backward to reach `x = D` | Goal-axis sign error | Verify `GOAL_AXIS` matches observed `delta x` direction |
| Goal rate 95% on seed 0, 30% on seed 1 | Insufficient training or high variance | Train 3 seeds; report mean and standard deviation; require mean at least 90% and standard deviation at most 5% |
| Policy drives and stops repeatedly | Distance shaping rewards closeness without finishing | Increase goal bonus, or remove shaping after first success |

## 7. Reproducibility contract

Every reported result must include:

- Seed for `model.learn(seed=...)` and `env.reset(seed=...)`.
- Exact `PPO(...)` keyword arguments.
- `total_timesteps`.
- Saved `ppo_car_forward.zip`.
- Evaluation script used to compute the metrics in section 3.

Without all five, a result is anecdotal.

## 8. Definition of done

`ppo_car_forward_v1.zip` exists, is loadable, and passes all six acceptance
tests on 100 fresh episodes with `deterministic=True`. The evaluation script
and metrics log are committed alongside it. The reward function and model XML
in the codebase match this specification exactly (diff-checked). Anything
less is a draft, not a solved task.
