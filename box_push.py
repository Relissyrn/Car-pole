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


@dataclass(frozen=True)
class BoxPushSettings:
    stage: str = "randomized"
    target_x: float = 3.0
    target_y: float = 0.0
    success_radius: float = 0.45
    success_hold_steps: int = SUCCESS_HOLD_STEPS
    max_steps: int = MAX_STEPS
    frame_skip: int = FRAME_SKIP
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
        self.target = np.array(
            (settings.target_x, settings.target_y), dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        self.control_dt = settings.frame_skip * self.model.opt.timestep
        self.steps = 0
        self.success_steps = 0
        self._previous_car_box_distance = 0.0
        self._previous_box_target_distance = 0.0

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

    def _observation(self):
        car_position = self.data.xpos[self.car_id]
        car_yaw = _yaw_from_quaternion(self.data.xquat[self.car_id])
        box_position = self.data.xpos[self.box_id]
        box_yaw = _yaw_from_quaternion(self.data.xquat[self.box_id])
        car_velocity = self._body_velocity(self.car_id)
        return np.asarray(
            (
                car_position[0],
                car_position[1],
                np.sin(car_yaw),
                np.cos(car_yaw),
                car_velocity[3],
                car_velocity[4],
                car_velocity[2],
                box_position[0],
                box_position[1],
                np.sin(box_yaw),
                np.cos(box_yaw),
                self.target[0],
                self.target[1],
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
        return self._observation(), {
            "start_pose": start.copy(),
            "stage": self.settings.stage,
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
        car_box_distance = float(np.linalg.norm(box_position - car_position))
        box_target_distance = float(np.linalg.norm(self.target - box_position))
        if box_target_distance <= self.settings.success_radius:
            self.success_steps += 1
        else:
            self.success_steps = 0
        succeeded = self.success_steps >= self.settings.success_hold_steps

        reward = (
            0.15 * (self._previous_car_box_distance - car_box_distance)
            + (self._previous_box_target_distance - box_target_distance)
            - 0.01
            + 10.0 * float(succeeded)
        )
        self._previous_car_box_distance = car_box_distance
        self._previous_box_target_distance = box_target_distance

        terminated = bool(succeeded)
        truncated = bool(self.steps >= self.settings.max_steps and not terminated)
        reason = "success" if succeeded else ("timeout" if truncated else None)
        info = {
            "success": succeeded,
            "termination_reason": reason,
            "car_box_distance": car_box_distance,
            "box_target_distance": box_target_distance,
            "success_steps": self.success_steps,
            "car_x": float(car_position[0]),
            "car_y": float(car_position[1]),
            "box_x": float(box_position[0]),
            "box_y": float(box_position[1]),
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
