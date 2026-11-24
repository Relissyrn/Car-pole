import argparse
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


XML = Path(__file__).with_name("car.xml")
GOAL_DISTANCE = 15.0
CORRIDOR_HALF_WIDTH = 3.0
FRAME_SKIP = 5
MAX_STEPS = 500
START_YAW_RANGE = 0.02
FLIP_Z_THRESHOLD = 0.10
DRIVE_ACTUATORS = (
    "drive_front_left",
    "drive_front_right",
    "drive_rear_left",
    "drive_rear_right",
)
STEERING_ACTUATORS = ("steer_left", "steer_right")


def build_model(xml_path=XML):
    if not xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo car model not found: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if model.opt.timestep != 0.002:
        raise ValueError(
            f"Expected a 0.002 s physics timestep, got {model.opt.timestep}."
        )
    if model.nu != 6:
        raise ValueError(f"Expected six car actuators, found {model.nu}.")
    return model


def get_chassis_state(model, data, body_id):
    position = data.xpos[body_id].copy()
    quaternion = data.xquat[body_id].copy()
    spatial_velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        spatial_velocity,
        0,
    )
    return (
        position,
        quaternion,
        spatial_velocity[3:6].copy(),
        spatial_velocity[0:3].copy(),
    )


def rotate_world_to_car(vector, quaternion):
    inverse_quaternion = quaternion.copy()
    inverse_quaternion[1:] *= -1.0
    result = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(result, vector, inverse_quaternion)
    return result


def get_car_frame_velocity(data):
    inverse_quaternion = data.qpos[3:7].copy()
    inverse_quaternion[1:] *= -1.0
    velocity = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(velocity, data.qvel[:3], inverse_quaternion)
    return velocity


class CarForward(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, xml_path=XML):
        super().__init__()
        self.model = build_model(xml_path)
        self.data = mujoco.MjData(self.model)
        self.chassis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis"
        )
        self.root_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "root"
        )
        if self.chassis_id < 0 or self.root_joint_id < 0:
            raise RuntimeError("The car model is missing its chassis or root joint.")
        self.finish_line_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "finish_line"
        )
        if self.finish_line_id < 0:
            raise RuntimeError("The car model is missing the finish line.")

        self.control_dt = FRAME_SKIP * self.model.opt.timestep
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        self.steps = 0

    def _observation(self):
        position, quaternion, linear_velocity, angular_velocity = (
            get_chassis_state(self.model, self.data, self.chassis_id)
        )
        remaining = GOAL_DISTANCE - float(position[0])
        return np.concatenate(
            (position, quaternion, linear_velocity, angular_velocity, [remaining])
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        qpos_address = self.model.jnt_qposadr[self.root_joint_id]
        yaw = float(self.np_random.uniform(-START_YAW_RANGE, START_YAW_RANGE))
        self.data.qpos[qpos_address + 3 : qpos_address + 7] = (
            np.cos(yaw / 2.0),
            0.0,
            0.0,
            np.sin(yaw / 2.0),
        )
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self._observation(), {}

    def step(self, action):
        action = np.asarray(action)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}."
            )
        if not np.all(np.isfinite(action)) or np.any(action < -1.0) or np.any(action > 1.0):
            raise ValueError("Actions must be finite values in [-1, 1].")

        self.data.ctrl[:] = action
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        position, quaternion, linear_velocity, angular_velocity = get_chassis_state(
            self.model, self.data, self.chassis_id
        )
        car_velocity = rotate_world_to_car(linear_velocity, quaternion)
        car_angular_velocity = rotate_world_to_car(angular_velocity, quaternion)
        forward_speed = float(car_velocity[0])
        lateral_speed = float(car_velocity[1])
        yaw_rate = float(car_angular_velocity[2])
        remaining = GOAL_DISTANCE - float(position[0])
        reached = remaining <= 0.0
        flipped = position[2] < FLIP_Z_THRESHOLD

        yaw_coefficient = 0.50 if abs(forward_speed) < 0.2 else 0.02
        reward = (
            forward_speed
            - 0.30 * abs(lateral_speed)
            - yaw_coefficient * abs(yaw_rate)
            - 0.01 * max(remaining, 0.0)
            + 50.0 * float(reached)
        )

        terminated = bool(reached or flipped)
        truncated = bool(self.steps >= MAX_STEPS and not terminated)
        if reached:
            termination_reason = "reached"
        elif flipped:
            termination_reason = "flipped"
        elif truncated:
            termination_reason = "timeout"
        else:
            termination_reason = None

        info = {
            "reached_goal": bool(reached),
            "flipped": bool(flipped),
            "termination_reason": termination_reason,
            "forward_speed": forward_speed,
            "lateral_speed": lateral_speed,
            "yaw_rate": yaw_rate,
            "remaining": remaining,
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
            "time": float(self.data.time),
        }
        return self._observation(), float(reward), terminated, truncated, info


def make_env(rank, monitor_dir):
    def initialize():
        env = CarForward()
        return Monitor(env, filename=str(monitor_dir / f"worker_{rank}"))

    return initialize


def main():
    parser = argparse.ArgumentParser(
        description="Train a PPO policy to drive the MuJoCo car forward."
    )
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("ppo_car_forward_v1.zip"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).with_name("runs"),
    )
    args = parser.parse_args()
    if args.timesteps <= 0 or args.num_envs <= 0:
        parser.error("--timesteps and --num-envs must be positive.")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 2^32 - 1.")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = args.log_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    env = SubprocVecEnv(
        [make_env(rank, monitor_dir) for rank in range(args.num_envs)],
        start_method="spawn",
    )
    ppo_kwargs = {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_epochs": 10,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "seed": args.seed,
        "verbose": 1,
        "tensorboard_log": str(args.log_dir),
    }

    try:
        model = PPO(env=env, **ppo_kwargs)
        model.learn(total_timesteps=args.timesteps, tb_log_name="car_forward")
        model.save(str(args.output))
        compatibility_output = args.output.with_name("ppo_car_forward.zip")
        if compatibility_output != args.output:
            model.save(str(compatibility_output))
        config_path = args.output.with_name("training_config.json")
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(
                {
                    "seed": args.seed,
                    "total_timesteps": args.timesteps,
                    "num_envs": args.num_envs,
                    "control_dt": FRAME_SKIP * 0.002,
                    "ppo_kwargs": ppo_kwargs,
                    "tensorboard_run_dir": model.logger.dir,
                },
                config_file,
                indent=2,
            )
            config_file.write("\n")
        print(f"Saved policy -> {args.output}")
        print(f"Saved compatibility policy -> {compatibility_output}")
        print(f"Saved training config -> {config_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
