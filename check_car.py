import numpy as np
import mujoco

from train_car import (
    CarForward,
    CORRIDOR_HALF_WIDTH,
    DRIVE_ACTUATORS,
    GOAL_DISTANCE,
    MAX_STEPS,
    START_YAW_RANGE,
    STEERING_ACTUATORS,
    XML,
    build_model,
    get_chassis_state,
)


DRIVE_TEST_PHYSICS_STEPS = 1000
MIN_DRIVE_TEST_DISTANCE = 1.0
MAX_FULL_THROTTLE_TIME = 4.0


def actuator_ids(model):
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    ]
    expected_names = list(DRIVE_ACTUATORS + STEERING_ACTUATORS)
    if names != expected_names:
        raise RuntimeError(f"Unexpected actuator order: {names}")
    return names


def main():
    model = build_model(XML)
    names = actuator_ids(model)
    chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = (0.5, 0.5, 0.5, 0.5, 0.0, 0.0)
    for _ in range(DRIVE_TEST_PHYSICS_STEPS):
        mujoco.mj_step(model, data)
    position, _, _, _ = get_chassis_state(model, data, chassis_id)
    drive_distance = float(position[0])
    print(
        f"Actuators {names}: +0.5 drive moved {drive_distance:.3f} m "
        f"in {data.time:.2f} s."
    )
    if drive_distance <= MIN_DRIVE_TEST_DISTANCE:
        raise RuntimeError(
            "The actuator-mapping precondition failed: "
            f"delta x must exceed {MIN_DRIVE_TEST_DISTANCE:.1f} m."
        )

    finish_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "finish_line"
    )
    if not np.allclose(model.geom_pos[finish_id], (GOAL_DISTANCE, 0.0, 0.0)):
        raise RuntimeError("The finish marker is not at (15, 0, 0).")
    if not np.allclose(model.geom_rgba[finish_id][:3], (0.0, 1.0, 0.0)):
        raise RuntimeError("The finish marker is not green.")
    if model.geom_rgba[finish_id][3] < 1.0:
        raise RuntimeError("The finish marker must be fully opaque.")
    if not np.isclose(
        model.geom_size[finish_id][1], CORRIDOR_HALF_WIDTH
    ):
        raise RuntimeError("The finish marker does not span the specified corridor.")
    if model.geom_contype[finish_id] or model.geom_conaffinity[finish_id]:
        raise RuntimeError("The finish marker must not participate in collisions.")

    env = CarForward()
    observation, _ = env.reset(seed=0)
    if observation.shape != (14,) or env.action_space.shape != (6,):
        raise RuntimeError("The observation or action dimensions do not match the spec.")

    action = np.array((1, 1, 1, 1, 0, 0), dtype=np.float32)
    for _ in range(MAX_STEPS):
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print(
        f"Full-throttle check: reached={info['reached_goal']} "
        f"at x={info['x']:.3f} m, y={info['y']:.3f} m "
        f"after {info['time']:.3f} s."
    )
    if not info["reached_goal"]:
        raise RuntimeError(
            f"Full throttle did not cross the {GOAL_DISTANCE:.1f} m finish "
            f"line within {MAX_STEPS} agent steps."
        )
    if info["x"] < GOAL_DISTANCE:
        raise RuntimeError("The goal was reported before the car crossed the finish line.")
    if info["time"] > MAX_FULL_THROTTLE_TIME:
        raise RuntimeError(
            f"Full-throttle travel took {info['time']:.3f} s, exceeding "
            f"{MAX_FULL_THROTTLE_TIME:.1f} s."
        )

    for yaw in (-START_YAW_RANGE, START_YAW_RANGE):
        for steer in (-1.0, 1.0):
            env.reset(seed=0)
            root_qpos = env.model.jnt_qposadr[env.root_joint_id]
            env.data.qpos[root_qpos + 3 : root_qpos + 7] = (
                np.cos(yaw / 2.0),
                0.0,
                0.0,
                np.sin(yaw / 2.0),
            )
            mujoco.mj_forward(env.model, env.data)
            for _ in range(MAX_STEPS):
                _, _, terminated, truncated, info = env.step(
                    (1, 1, 1, 1, steer, steer)
                )
                if terminated or truncated:
                    break
            if (
                not info["reached_goal"]
                or info["time"] > MAX_FULL_THROTTLE_TIME
                or abs(info["y"]) > 1.0
            ):
                raise RuntimeError(
                    "The maximum steering command failed the yaw robustness "
                    f"check at initial yaw {yaw:+.3f} rad."
                )
    env.close()
    env.close()


if __name__ == "__main__":
    main()
