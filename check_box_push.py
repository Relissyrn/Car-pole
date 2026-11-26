import numpy as np
import mujoco

from box_push import (
    ActionLatency,
    ActionNoise,
    BoxPushEnv,
    BoxPushSettings,
    build_model,
    make_environment,
)


def main():
    settings = BoxPushSettings(stage="easy", target_x=2.2)
    model = build_model(settings)
    if model.nu != 6:
        raise RuntimeError(f"Expected six car actuators, found {model.nu}.")
    nominal_model = build_model()
    mismatch_model = build_model(
        BoxPushSettings(box_mass_scale=1.5, friction_scale=0.7)
    )
    nominal_box = nominal_model.body_mass[
        mujoco.mj_name2id(nominal_model, mujoco.mjtObj.mjOBJ_BODY, "box")
    ]
    mismatch_box = mismatch_model.body_mass[
        mujoco.mj_name2id(mismatch_model, mujoco.mjtObj.mjOBJ_BODY, "box")
    ]
    if not np.isclose(mismatch_box / nominal_box, 1.5):
        raise RuntimeError("The box-mass mismatch setting was not applied.")
    ground_id = mujoco.mj_name2id(
        nominal_model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
    )
    mismatch_ground_id = mujoco.mj_name2id(
        mismatch_model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
    )
    if not np.isclose(
        mismatch_model.geom_friction[mismatch_ground_id, 0]
        / nominal_model.geom_friction[ground_id, 0],
        0.7,
    ):
        raise RuntimeError("The friction mismatch setting was not applied.")

    env = BoxPushEnv(settings)
    observation, info = env.reset(seed=7)
    if observation.shape != (13,) or env.observation_space.shape != (13,):
        raise RuntimeError("The box-pushing observation must have 13 values.")
    if info["stage"] != "easy":
        raise RuntimeError("The easy curriculum stage was not reported.")
    if not np.all(np.isfinite(observation)):
        raise RuntimeError("The observation contains non-finite values.")

    fixed_start = {
        "car_x": 0.2,
        "car_y": -0.1,
        "car_yaw": 0.05,
        "box_x": 0.74,
        "box_y": 0.0,
        "box_yaw": 0.0,
    }
    observation, info = env.reset(
        seed=7, options={"start_pose": fixed_start}
    )
    if info["start_pose"] != fixed_start:
        raise RuntimeError("The fixed evaluation start pose was not honored.")

    target_start = {
        "car_x": -1.0,
        "car_y": 0.0,
        "car_yaw": 0.0,
        "box_x": settings.target_x,
        "box_y": settings.target_y,
        "box_yaw": 0.0,
    }
    env.reset(seed=7, options={"start_pose": target_start})
    for step in range(settings.success_hold_steps):
        _, _, terminated, truncated, info = env.step(
            np.zeros(6, dtype=np.float32)
        )
        if terminated != (step + 1 == settings.success_hold_steps) or truncated:
            raise RuntimeError("Target success did not require the configured hold.")

    for _ in range(20):
        observation, reward, terminated, truncated, info = env.step(
            np.zeros(6, dtype=np.float32)
        )
        if not np.isfinite(reward) or not np.all(np.isfinite(observation)):
            raise RuntimeError("The environment emitted non-finite values.")
        if terminated or truncated:
            break
    env.close()

    push_env = BoxPushEnv(BoxPushSettings(stage="easy", target_x=2.2))
    push_env.reset(seed=1)
    full_throttle = np.array((1, 1, 1, 1, 0, 0), dtype=np.float32)
    for _ in range(push_env.settings.max_steps):
        _, _, terminated, truncated, info = push_env.step(full_throttle)
        if terminated or truncated:
            break
    push_env.close()
    if not info["success"]:
        raise RuntimeError("The easy-stage box could not be pushed into the target.")

    noisy_a = ActionNoise(BoxPushEnv(settings), 0.1, seed=3)
    noisy_b = ActionNoise(BoxPushEnv(settings), 0.1, seed=3)
    noisy_a.reset(seed=3)
    noisy_b.reset(seed=3)
    command = np.zeros(6, dtype=np.float32)
    if not np.array_equal(noisy_a.action(command), noisy_b.action(command)):
        raise RuntimeError("Action noise is not reproducible from its seed.")
    noisy_a.close()
    noisy_b.close()

    delayed = ActionLatency(BoxPushEnv(settings), latency_steps=1)
    delayed.reset(seed=0)
    full_throttle = np.array((1, 1, 1, 1, 0, 0), dtype=np.float32)
    delayed.step(full_throttle)
    if not np.allclose(delayed.unwrapped.data.ctrl, 0.0):
        raise RuntimeError("One-step latency did not apply the initial no-op.")
    delayed.step(full_throttle)
    if not np.allclose(delayed.unwrapped.data.ctrl, full_throttle):
        raise RuntimeError("One-step latency did not apply the delayed action.")
    delayed.close()
    composed = make_environment(
        settings, latency_steps=1, action_noise_std=0.05, noise_seed=5
    )
    composed.reset(seed=5)
    composed.step(np.zeros(6, dtype=np.float32))
    composed.close()
    print("Box-pushing environment, fixed starts, noise, and latency checks passed.")


if __name__ == "__main__":
    main()
