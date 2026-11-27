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


def check_reward_penalties():
    far_start = {
        "car_x": 0.0,
        "car_y": 0.0,
        "car_yaw": 0.0,
        "box_x": 20.0,
        "box_y": 0.0,
        "box_yaw": 0.0,
    }
    speed_penalty_env = BoxPushEnv(
        BoxPushSettings(
            car_speed_limit=0.01,
            car_speed_penalty_per_second=10.0,
        )
    )
    no_speed_penalty_env = BoxPushEnv(
        BoxPushSettings(
            car_speed_limit=0.01,
            car_speed_penalty_per_second=0.0,
        )
    )
    speed_penalty_env.reset(seed=11, options={"start_pose": far_start})
    no_speed_penalty_env.reset(seed=11, options={"start_pose": far_start})
    throttle = np.array((1, 1, 1, 1, 0, 0), dtype=np.float32)
    penalized_return = 0.0
    unpenalized_return = 0.0
    saw_speed_penalty = False
    for _ in range(25):
        _, penalized_reward, _, _, info = speed_penalty_env.step(throttle)
        _, unpenalized_reward, _, _, _ = no_speed_penalty_env.step(throttle)
        penalized_return += penalized_reward
        unpenalized_return += unpenalized_reward
        saw_speed_penalty |= info["car_speed_penalty"] > 0
    speed_penalty_env.close()
    no_speed_penalty_env.close()
    if not saw_speed_penalty or not penalized_return < unpenalized_return:
        raise RuntimeError("Excessive car speed did not reduce reward.")

    touching_start = {
        "car_x": 0.0,
        "car_y": 0.0,
        "car_yaw": 0.0,
        "box_x": 0.54,
        "box_y": 0.0,
        "box_yaw": 0.0,
    }
    contact_env = BoxPushEnv(
        BoxPushSettings(
            stage="easy",
            target_x=2.2,
            lost_bumper_contact_penalty=1.0,
        )
    )
    no_contact_penalty_env = BoxPushEnv(
        BoxPushSettings(
            stage="easy",
            target_x=2.2,
            lost_bumper_contact_penalty=0.0,
        )
    )
    _, contact_info = contact_env.reset(
        seed=13, options={"start_pose": touching_start}
    )
    no_contact_penalty_env.reset(
        seed=13, options={"start_pose": touching_start}
    )
    if not contact_info["bumper_contact"]:
        raise RuntimeError("The fixed start did not establish front bumper contact.")
    reverse = np.array((-1, -1, -1, -1, 0, 0), dtype=np.float32)
    lost_contact_seen = False
    for _ in range(20):
        _, penalized_reward, _, _, info = contact_env.step(reverse)
        _, unpenalized_reward, _, _, _ = no_contact_penalty_env.step(reverse)
        if info["lost_bumper_contact"]:
            lost_contact_seen = True
            if not np.isclose(unpenalized_reward - penalized_reward, 1.0):
                raise RuntimeError("Losing bumper contact had the wrong penalty.")
            break
    contact_env.close()
    no_contact_penalty_env.close()
    if not lost_contact_seen:
        raise RuntimeError("Reversing away did not register loss of bumper contact.")

    coasting_env = BoxPushEnv(BoxPushSettings())
    coasting_start = {
        "car_x": 0.0,
        "car_y": 0.0,
        "car_yaw": 0.0,
        "box_x": 2.0,
        "box_y": 0.0,
        "box_yaw": 0.0,
    }
    coasting_env.reset(seed=15, options={"start_pose": coasting_start})
    coasting_joint_id = mujoco.mj_name2id(
        coasting_env.model, mujoco.mjtObj.mjOBJ_JOINT, "box_root"
    )
    coasting_dof = coasting_env.model.jnt_dofadr[coasting_joint_id]
    coasting_env.data.qvel[coasting_dof] = 0.2
    mujoco.mj_forward(coasting_env.model, coasting_env.data)
    _, _, _, _, info = coasting_env.step(np.zeros(6, dtype=np.float32))
    coasting_env.close()
    if info["bumper_contact"] or info["box_target_progress_reward"] > 0:
        raise RuntimeError("Uncontacted box motion earned positive push reward.")

    target_start = {
        "car_x": 0.0,
        "car_y": 0.0,
        "car_yaw": 0.0,
        "box_x": 3.449,
        "box_y": 0.0,
        "box_yaw": 0.0,
    }
    overshoot_env = BoxPushEnv(
        BoxPushSettings(target_overshoot_penalty=2.0)
    )
    no_overshoot_penalty_env = BoxPushEnv(
        BoxPushSettings(target_overshoot_penalty=0.0)
    )
    if not overshoot_env._box_crossed_target_zone(
        np.array((2.3, 0.0)), np.array((3.7, 0.0))
    ):
        raise RuntimeError("A box crossing entirely through the target was missed.")
    overshoot_env.reset(seed=17, options={"start_pose": target_start})
    no_overshoot_penalty_env.reset(
        seed=17, options={"start_pose": target_start}
    )
    box_joint_id = mujoco.mj_name2id(
        overshoot_env.model, mujoco.mjtObj.mjOBJ_JOINT, "box_root"
    )
    box_dof = overshoot_env.model.jnt_dofadr[box_joint_id]
    overshoot_env.data.qvel[box_dof] = 1.0
    no_overshoot_penalty_env.data.qvel[box_dof] = 1.0
    mujoco.mj_forward(overshoot_env.model, overshoot_env.data)
    mujoco.mj_forward(
        no_overshoot_penalty_env.model, no_overshoot_penalty_env.data
    )
    no_action = np.zeros(6, dtype=np.float32)
    _, penalized_reward, _, _, info = overshoot_env.step(no_action)
    _, unpenalized_reward, _, _, _ = no_overshoot_penalty_env.step(no_action)
    overshoot_env.close()
    no_overshoot_penalty_env.close()
    if not info["target_overshoot"]:
        raise RuntimeError("Leaving the target zone did not register overshoot.")
    if not np.isclose(unpenalized_reward - penalized_reward, 2.0):
        raise RuntimeError("Crossing the target zone had the wrong penalty.")


def main():
    check_reward_penalties()
    settings = BoxPushSettings(stage="easy", target_x=2.2)
    model = build_model(settings)
    if model.nu != 6:
        raise RuntimeError(f"Expected six car actuators, found {model.nu}.")
    nominal_model = build_model()
    mismatch_model = build_model(
        BoxPushSettings(box_mass_scale=1.5, friction_scale=0.7)
    )
    box_geom_id = mujoco.mj_name2id(
        nominal_model, mujoco.mjtObj.mjOBJ_GEOM, "push_box"
    )
    mismatch_box_geom_id = mujoco.mj_name2id(
        mismatch_model, mujoco.mjtObj.mjOBJ_GEOM, "push_box"
    )
    tire_geom_id = mujoco.mj_name2id(
        nominal_model, mujoco.mjtObj.mjOBJ_GEOM, "tire_front_left"
    )
    if not np.isclose(nominal_model.geom_friction[box_geom_id, 0], 0.35):
        raise RuntimeError("The carton-box sliding friction was not applied.")
    if nominal_model.geom_friction[tire_geom_id, 0] <= nominal_model.geom_friction[
        box_geom_id, 0
    ]:
        raise RuntimeError("Tire traction must remain higher than box-floor friction.")
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
    if not np.isclose(
        mismatch_model.geom_friction[mismatch_box_geom_id, 0]
        / nominal_model.geom_friction[box_geom_id, 0],
        0.7,
    ):
        raise RuntimeError("The mismatch scale did not apply to carton friction.")

    env = BoxPushEnv(settings)
    observation, info = env.reset(seed=7)
    expected_observation_shape = (9 + settings.lidar_num_rays,)
    if (
        observation.shape != expected_observation_shape
        or env.observation_space.shape != expected_observation_shape
    ):
        raise RuntimeError("The LiDAR box-pushing observation has the wrong shape.")
    if info["stage"] != "easy":
        raise RuntimeError("The easy curriculum stage was not reported.")
    if not np.all(np.isfinite(observation)):
        raise RuntimeError("The observation contains non-finite values.")

    far_start = {
        "car_x": 0.0,
        "car_y": 0.0,
        "car_yaw": 0.0,
        "box_x": 20.0,
        "box_y": 0.0,
        "box_yaw": 0.0,
    }
    far_observation, _ = env.reset(
        seed=7, options={"start_pose": far_start}
    )
    if np.any(far_observation[7:-2] < settings.lidar_max_range):
        raise RuntimeError("The LiDAR is detecting the car or an out-of-range box.")

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
    lidar_ranges = observation[7:-2]
    if not np.any(lidar_ranges < settings.lidar_max_range):
        raise RuntimeError("The LiDAR failed to return a range for the box.")
    repeated_observation, _ = env.reset(
        seed=7, options={"start_pose": fixed_start}
    )
    if not np.allclose(observation, repeated_observation):
        raise RuntimeError("LiDAR observations are not reproducible from the seed.")

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

    partial_target_start = dict(target_start)
    partial_target_start["box_x"] += 0.3
    env.reset(seed=7, options={"start_pose": partial_target_start})
    for _ in range(settings.success_hold_steps):
        _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.float32))
        if terminated or info["box_fits_target_zone"]:
            raise RuntimeError("A partially outside box was accepted in the target.")

    env.reset(seed=7, options={"start_pose": target_start})
    box_joint_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, "box_root"
    )
    box_qpos = env.model.jnt_qposadr[box_joint_id]
    box_dof = env.model.jnt_dofadr[box_joint_id]
    env.data.qpos[box_qpos + 2] = 0.40
    env.data.qvel[box_dof + 2] = 2.0
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.float32))
    if terminated:
        raise RuntimeError("An airborne box was incorrectly accepted as success.")
    if info["physically_controlled"]:
        raise RuntimeError("Airborne/high-speed motion was not rejected.")

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
    initial_box_x = float(push_env.data.xpos[push_env.box_id, 0])
    for _ in range(35):
        _, _, _, _, info = push_env.step(full_throttle)
    final_box_x = float(push_env.data.xpos[push_env.box_id, 0])
    push_env.close()
    if final_box_x <= initial_box_x + 0.1:
        raise RuntimeError("The car did not move the box under controlled pushing.")
    if info["box_height"] > 0.28:
        raise RuntimeError("The short push launched the box.")

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
    print(
        "Box-pushing LiDAR, physics, fixed-start, noise, and latency checks passed."
    )


if __name__ == "__main__":
    main()
