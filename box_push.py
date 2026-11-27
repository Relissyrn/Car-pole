from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np


XML = Path(__file__).with_name("box_push.xml")
FRAME_SKIP = 5
MAX_STEPS = 700
SUCCESS_HOLD_STEPS = 5
LIDAR_NUM_RAYS = 32


@dataclass(frozen=True)
class BoxPushSettings:
    stage: str = "randomized"
    target_x: float = 3.0
    target_y: float = 0.0
    success_radius: float = 0.45
    success_hold_steps: int = SUCCESS_HOLD_STEPS
    max_steps: int = MAX_STEPS
    frame_skip: int = FRAME_SKIP
    max_box_height: float = 0.28
    max_box_speed: float = 0.5
    box_floor_friction: float = 0.35
    car_speed_limit: float = 1.0
    car_speed_penalty_per_second: float = 5.0
    lost_bumper_contact_penalty: float = 3.0
    target_overshoot_penalty: float = 2.0
    lidar_num_rays: int = LIDAR_NUM_RAYS
    lidar_max_range: float = 4.0
    lidar_noise_std: float = 0.01
    box_mass_scale: float = 1.0
    friction_scale: float = 1.0

    def __post_init__(self):
        if self.stage not in {"easy", "randomized"}:
            raise ValueError("stage must be 'easy' or 'randomized'.")
        if (
            not np.isfinite(self.success_radius)
            or self.success_radius <= 0
            or self.success_hold_steps <= 0
        ):
            raise ValueError("Success zone radius and hold steps must be positive.")
        if self.max_steps <= 0 or self.frame_skip <= 0:
            raise ValueError("max_steps and frame_skip must be positive.")
        if (
            not np.isfinite(self.box_mass_scale)
            or not np.isfinite(self.friction_scale)
            or self.box_mass_scale <= 0
            or self.friction_scale <= 0
        ):
            raise ValueError("Physics mismatch scales must be positive.")
        if (
            not np.isfinite(self.max_box_height)
            or not np.isfinite(self.max_box_speed)
            or self.max_box_height <= 0
            or self.max_box_speed <= 0
        ):
            raise ValueError("Box height and speed limits must be positive.")
        if not np.isfinite(self.box_floor_friction) or self.box_floor_friction <= 0:
            raise ValueError("Box-floor friction must be finite and positive.")
        if not np.isfinite(self.car_speed_limit) or self.car_speed_limit <= 0:
            raise ValueError("The car speed limit must be finite and positive.")
        reward_penalties = (
            self.car_speed_penalty_per_second,
            self.lost_bumper_contact_penalty,
            self.target_overshoot_penalty,
        )
        if not np.all(np.isfinite(reward_penalties)) or any(
            penalty < 0 for penalty in reward_penalties
        ):
            raise ValueError("Reward penalty settings must be finite and non-negative.")
        if self.lidar_num_rays < 8:
            raise ValueError("The LiDAR scan must contain at least eight rays.")
        if (
            not np.isfinite(self.lidar_max_range)
            or not np.isfinite(self.lidar_noise_std)
            or self.lidar_max_range <= 0
            or self.lidar_noise_std < 0
        ):
            raise ValueError("LiDAR range must be positive and noise non-negative.")
        if not np.all(np.isfinite((self.target_x, self.target_y))):
            raise ValueError("Target coordinates must be finite.")


def build_model(settings=BoxPushSettings(), xml_path=XML):
    if not xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo box-pushing model not found: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if not np.isclose(model.opt.timestep, 0.002):
        raise ValueError(
            f"Expected a 0.002 s physics timestep, got {model.opt.timestep}."
        )
    if model.nu != 6:
        raise ValueError(f"Expected six car actuators, found {model.nu}.")

    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    if box_id < 0:
        raise RuntimeError("The MuJoCo model is missing the pushable box.")
    model.body_mass[box_id] *= settings.box_mass_scale
    model.body_inertia[box_id] *= settings.box_mass_scale

    friction_geoms = (
        "ground",
        "tire_front_left",
        "tire_front_right",
        "tire_rear_left",
        "tire_rear_right",
        "push_box",
    )
    for name in ("ground", "push_box"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise RuntimeError(f"The MuJoCo model is missing geom {name!r}.")
        model.geom_friction[geom_id, 0] = settings.box_floor_friction
    for name in friction_geoms:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise RuntimeError(f"The MuJoCo model is missing geom {name!r}.")
        model.geom_friction[geom_id] *= settings.friction_scale

    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone")
    if target_id < 0:
        raise RuntimeError("The MuJoCo model is missing the target zone.")
    model.geom_pos[target_id, :2] = (settings.target_x, settings.target_y)
    model.geom_size[target_id, :2] = settings.success_radius
    return model


def _yaw_from_quaternion(quaternion):
    w, x, y, z = quaternion
    return float(
        np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def _set_free_pose(model, data, joint_name, x, y, z, yaw):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"The MuJoCo model is missing joint {joint_name!r}.")
    address = model.jnt_qposadr[joint_id]
    data.qpos[address : address + 7] = (
        x, y, z, np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)
    )


class BoxPushEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, settings=BoxPushSettings(), xml_path=XML):
        super().__init__()
        self.settings = settings
        self.model = build_model(settings, xml_path)
        self.data = mujoco.MjData(self.model)
        self.car_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis"
        )
        self.box_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "box"
        )
        self.bumper_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "chassis_body"
        )
        self.box_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "push_box"
        )
        if self.bumper_geom_id < 0 or self.box_geom_id < 0:
            raise RuntimeError("The MuJoCo model is missing the bumper or box geom.")
        self.target = np.array(
            (settings.target_x, settings.target_y), dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        observation_size = 9 + settings.lidar_num_rays
        observation_low = np.concatenate(
            (
                np.full(7, -np.inf, dtype=np.float32),
                np.zeros(settings.lidar_num_rays, dtype=np.float32),
                np.full(2, -np.inf, dtype=np.float32),
            )
        )
        observation_high = np.concatenate(
            (
                np.full(7, np.inf, dtype=np.float32),
                np.full(
                    settings.lidar_num_rays,
                    settings.lidar_max_range,
                    dtype=np.float32,
                ),
                np.full(2, np.inf, dtype=np.float32),
            )
        )
        self.observation_space = gym.spaces.Box(
            low=observation_low,
            high=observation_high,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self._lidar_angles = np.linspace(
            -np.pi,
            np.pi,
            settings.lidar_num_rays,
            endpoint=False,
            dtype=np.float64,
        )
        self.control_dt = settings.frame_skip * self.model.opt.timestep
        self.steps = 0
        self.success_steps = 0
        self._previous_car_box_distance = 0.0
        self._previous_box_target_distance = 0.0
        self._previous_box_position = np.zeros(2, dtype=np.float64)
        self._previous_bumper_contact = False

    def _box_crossed_target_zone(self, previous_position, current_position):
        segment = current_position - previous_position
        segment_length_squared = float(np.dot(segment, segment))
        if segment_length_squared == 0.0:
            return False
        closest_fraction = np.clip(
            np.dot(self.target - previous_position, segment)
            / segment_length_squared,
            0.0,
            1.0,
        )
        closest_position = previous_position + closest_fraction * segment
        closest_distance = float(np.linalg.norm(self.target - closest_position))
        current_distance = float(np.linalg.norm(self.target - current_position))
        return bool(
            closest_distance <= self.settings.success_radius
            and current_distance > self.settings.success_radius
        )

    def _box_fits_target_zone(self, box_position, box_yaw):
        half_box_x, half_box_y = self.model.geom_size[self.box_geom_id, :2]
        cos_yaw = abs(np.cos(box_yaw))
        sin_yaw = abs(np.sin(box_yaw))
        box_half_extent = np.array(
            (
                cos_yaw * half_box_x + sin_yaw * half_box_y,
                sin_yaw * half_box_x + cos_yaw * half_box_y,
            ),
            dtype=np.float64,
        )
        return bool(
            np.all(
                np.abs(box_position - self.target) + box_half_extent
                <= self.settings.success_radius
            )
        )

    def _has_front_bumper_contact(self):
        car_position = self.data.xpos[self.car_id, :2]
        car_yaw = _yaw_from_quaternion(self.data.xquat[self.car_id])
        cos_yaw = np.cos(car_yaw)
        sin_yaw = np.sin(car_yaw)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if {contact.geom1, contact.geom2} != {
                self.bumper_geom_id,
                self.box_geom_id,
            }:
                continue
            relative = contact.pos[:2] - car_position
            local_forward = cos_yaw * relative[0] + sin_yaw * relative[1]
            local_lateral = -sin_yaw * relative[0] + cos_yaw * relative[1]
            if local_forward >= 0.30 and abs(local_lateral) <= 0.19:
                return True
        return False

    def _body_velocity(self, body_id):
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            1,
        )
        return velocity

    def _lidar_scan(self, car_position, car_yaw):
        origin = np.array(
            (car_position[0], car_position[1], car_position[2] + 0.10),
            dtype=np.float64,
        )
        ranges = np.full(
            self.settings.lidar_num_rays,
            self.settings.lidar_max_range,
            dtype=np.float64,
        )
        geom_id = np.empty(1, dtype=np.int32)
        for index, angle in enumerate(self._lidar_angles):
            direction = np.array(
                (np.cos(car_yaw + angle), np.sin(car_yaw + angle), 0.0),
                dtype=np.float64,
            )
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                direction,
                None,
                True,
                self.car_id,
                geom_id,
            )
            if 0.0 <= distance < self.settings.lidar_max_range:
                ranges[index] = distance

        hit_mask = ranges < self.settings.lidar_max_range
        if self.settings.lidar_noise_std and np.any(hit_mask):
            ranges[hit_mask] += self.np_random.normal(
                0.0,
                self.settings.lidar_noise_std,
                size=int(np.count_nonzero(hit_mask)),
            )
        return np.clip(ranges, 0.0, self.settings.lidar_max_range)

    def _observation(self):
        car_position = self.data.xpos[self.car_id]
        car_yaw = _yaw_from_quaternion(self.data.xquat[self.car_id])
        car_velocity = self._body_velocity(self.car_id)
        lidar_ranges = self._lidar_scan(car_position, car_yaw)
        return np.asarray(
            np.concatenate(
                (
                    np.array(
                        (
                            car_position[0],
                            car_position[1],
                            np.sin(car_yaw),
                            np.cos(car_yaw),
                            car_velocity[3],
                            car_velocity[4],
                            car_velocity[2],
                        ),
                        dtype=np.float64,
                    ),
                    lidar_ranges,
                    self.target,
                )
            ),
            dtype=np.float32,
        )

    def _sample_start(self):
        if self.settings.stage == "easy":
            box_x = self.settings.target_x - 1.2
            box_y = self.settings.target_y
            return {
                "car_x": box_x - 0.54,
                "car_y": box_y,
                "car_yaw": 0.0,
                "box_x": box_x,
                "box_y": box_y,
                "box_yaw": 0.0,
            }

        box_x = float(self.np_random.uniform(0.75, 1.35))
        box_y = float(self.np_random.uniform(-0.45, 0.45))
        return {
            "car_x": box_x - 0.54,
            "car_y": box_y + float(self.np_random.uniform(-0.5, 0.5)),
            "car_yaw": float(self.np_random.uniform(-0.3, 0.3)),
            "box_x": box_x,
            "box_y": box_y,
            "box_yaw": float(self.np_random.uniform(-0.12, 0.12)),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        start = self._sample_start()
        if options and "start_pose" in options:
            start = dict(options["start_pose"])
            required = {
                "car_x", "car_y", "car_yaw", "box_x", "box_y", "box_yaw"
            }
            if set(start) != required:
                raise ValueError(
                    f"start_pose must contain exactly {sorted(required)}."
                )
            if not np.all(np.isfinite(tuple(start.values()))):
                raise ValueError("start_pose values must be finite.")

        _set_free_pose(
            self.model,
            self.data,
            "car_root",
            start["car_x"],
            start["car_y"],
            0.18,
            start["car_yaw"],
        )
        _set_free_pose(
            self.model,
            self.data,
            "box_root",
            start["box_x"],
            start["box_y"],
            0.16,
            start["box_yaw"],
        )
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.success_steps = 0
        car_position = self.data.xpos[self.car_id, :2]
        box_position = self.data.xpos[self.box_id, :2]
        self._previous_car_box_distance = float(
            np.linalg.norm(box_position - car_position)
        )
        self._previous_box_target_distance = float(
            np.linalg.norm(self.target - box_position)
        )
        self._previous_box_position = box_position.copy()
        self._previous_bumper_contact = self._has_front_bumper_contact()
        return self._observation(), {
            "start_pose": start.copy(),
            "stage": self.settings.stage,
            "bumper_contact": self._previous_bumper_contact,
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}."
            )
        if (
            not np.all(np.isfinite(action))
            or np.any(action < -1.0)
            or np.any(action > 1.0)
        ):
            raise ValueError("Actions must be finite values in [-1, 1].")

        self.data.ctrl[:] = action
        for _ in range(self.settings.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        car_position = self.data.xpos[self.car_id, :2].copy()
        box_position = self.data.xpos[self.box_id, :2].copy()
        box_yaw = _yaw_from_quaternion(self.data.xquat[self.box_id])
        car_velocity = self._body_velocity(self.car_id)
        car_speed = float(np.linalg.norm(car_velocity[3:5]))
        car_speed_penalty = (
            self.settings.car_speed_penalty_per_second
            * max(0.0, car_speed - self.settings.car_speed_limit)
            * self.control_dt
        )
        bumper_contact = self._has_front_bumper_contact()
        lost_bumper_contact = bool(
            self._previous_bumper_contact and not bumper_contact
        )
        box_velocity = self._body_velocity(self.box_id)
        box_height = float(self.data.xpos[self.box_id, 2])
        box_horizontal_speed = float(
            np.linalg.norm(box_velocity[3:5])
        )
        box_vertical_speed = float(box_velocity[5])
        physically_controlled = bool(
            box_height <= self.settings.max_box_height
            and box_horizontal_speed <= self.settings.max_box_speed
        )
        car_box_distance = float(np.linalg.norm(box_position - car_position))
        box_target_distance = float(np.linalg.norm(self.target - box_position))
        box_fits_target_zone = self._box_fits_target_zone(
            box_position, box_yaw
        )
        target_overshoot = self._box_crossed_target_zone(
            self._previous_box_position,
            box_position,
        )
        lost_bumper_contact_penalty = (
            self.settings.lost_bumper_contact_penalty
            * float(lost_bumper_contact)
        )
        target_overshoot_penalty = (
            self.settings.target_overshoot_penalty * float(target_overshoot)
        )
        if (
            box_fits_target_zone
            and physically_controlled
        ):
            self.success_steps += 1
        else:
            self.success_steps = 0
        succeeded = bool(
            physically_controlled
            and self.success_steps >= self.settings.success_hold_steps
        )

        if physically_controlled:
            box_target_progress = (
                self._previous_box_target_distance - box_target_distance
            )
            box_target_progress_reward = (
                box_target_progress
                if bumper_contact
                else min(0.0, box_target_progress)
            )
            shaped_reward = (
                0.15 * (self._previous_car_box_distance - car_box_distance)
                + box_target_progress_reward
            )
        else:
            shaped_reward = 0.0
            box_target_progress_reward = 0.0
        reward = (
            shaped_reward
            - 0.01
            - 0.5 * float(not physically_controlled)
            - 0.25 * max(0.0, box_vertical_speed)
            - car_speed_penalty
            - lost_bumper_contact_penalty
            - target_overshoot_penalty
            + 10.0 * float(succeeded)
        )
        self._previous_car_box_distance = car_box_distance
        self._previous_box_target_distance = box_target_distance
        self._previous_box_position = box_position.copy()
        self._previous_bumper_contact = bumper_contact

        terminated = bool(succeeded)
        truncated = bool(self.steps >= self.settings.max_steps and not terminated)
        reason = "success" if succeeded else ("timeout" if truncated else None)
        info = {
            "success": succeeded,
            "termination_reason": reason,
            "car_box_distance": car_box_distance,
            "box_target_distance": box_target_distance,
            "box_target_progress_reward": box_target_progress_reward,
            "box_fits_target_zone": box_fits_target_zone,
            "success_steps": self.success_steps,
            "car_speed": car_speed,
            "car_speed_penalty": car_speed_penalty,
            "bumper_contact": bumper_contact,
            "lost_bumper_contact": lost_bumper_contact,
            "lost_bumper_contact_penalty": lost_bumper_contact_penalty,
            "target_overshoot": target_overshoot,
            "target_overshoot_penalty": target_overshoot_penalty,
            "car_x": float(car_position[0]),
            "car_y": float(car_position[1]),
            "box_x": float(box_position[0]),
            "box_y": float(box_position[1]),
            "box_height": box_height,
            "box_horizontal_speed": box_horizontal_speed,
            "physically_controlled": physically_controlled,
            "time": float(self.data.time),
        }
        return self._observation(), float(reward), terminated, truncated, info


class ActionNoise(gym.ActionWrapper):
    def __init__(self, env, standard_deviation, seed=None):
        super().__init__(env)
        if standard_deviation < 0 or not np.isfinite(standard_deviation):
            raise ValueError("Action noise standard deviation must be finite and >= 0.")
        self.standard_deviation = float(standard_deviation)
        self.rng = np.random.default_rng(seed)

    def action(self, action):
        noisy = np.asarray(action, dtype=np.float32)
        if not np.all(np.isfinite(noisy)):
            raise ValueError("Actions must contain only finite values.")
        if self.standard_deviation:
            noisy = noisy + self.rng.normal(
                0.0, self.standard_deviation, size=noisy.shape
            )
        return np.clip(noisy, self.action_space.low, self.action_space.high)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return self.env.reset(seed=seed, options=options)


class ActionLatency(gym.Wrapper):
    def __init__(self, env, latency_steps):
        super().__init__(env)
        if latency_steps < 0:
            raise ValueError("Action latency must be a non-negative number of steps.")
        self.latency_steps = int(latency_steps)
        self.pending_actions = deque()

    def reset(self, *, seed=None, options=None):
        self.pending_actions.clear()
        self.pending_actions.extend(
            np.zeros(self.action_space.shape, dtype=np.float32)
            for _ in range(self.latency_steps)
        )
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}."
            )
        if (
            not np.all(np.isfinite(action))
            or np.any(action < self.action_space.low)
            or np.any(action > self.action_space.high)
        ):
            raise ValueError("Actions must be finite values within the action space.")
        if self.latency_steps == 0:
            return self.env.step(action)
        self.pending_actions.append(action.copy())
        delayed_action = self.pending_actions.popleft()
        return self.env.step(delayed_action)


def make_environment(
    settings=BoxPushSettings(),
    latency_steps=0,
    action_noise_std=0.0,
    noise_seed=None,
):
    env = BoxPushEnv(settings)
    if action_noise_std:
        env = ActionNoise(env, action_noise_std, seed=noise_seed)
    if latency_steps:
        env = ActionLatency(env, latency_steps)
    return env
