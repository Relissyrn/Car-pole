import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import PPO

from train_car import CarForward


MODEL = Path(__file__).with_name("ppo_car_forward_v1.zip")


def main():
    if not MODEL.is_file():
        raise FileNotFoundError(
            f"Trained policy not found: {MODEL}; run train_car.py first."
        )

    env = CarForward()
    observation, _ = env.reset()
    policy = PPO.load(str(MODEL))
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 6.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25
        while viewer.is_running():
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            viewer.cam.lookat[:] = env.data.xpos[env.chassis_id]
            viewer.sync()
            if terminated or truncated:
                break
            time.sleep(env.control_dt)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
    env.close()


if __name__ == "__main__":
    main()
