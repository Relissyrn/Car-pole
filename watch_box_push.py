import argparse
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import PPO

from box_push import BoxPushSettings, make_environment
from evaluate_box_push import fixed_evaluation_starts


DEFAULT_MODEL = (
    Path(__file__).with_name("models") / "box_push_lidar_carton035_seed_0.zip"
)


def main():
    parser = argparse.ArgumentParser(
        description="Watch a trained box-pushing policy in MuJoCo."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--start-id",
        type=int,
        default=2,
        help="Fixed evaluation start index (0-19).",
    )
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Policy not found: {args.model}")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 2^32 - 1.")
    starts = fixed_evaluation_starts(3.0, 0.0)
    if not 0 <= args.start_id < len(starts):
        parser.error(f"--start-id must be between 0 and {len(starts) - 1}.")

    env = make_environment(BoxPushSettings(stage="randomized"))
    policy = PPO.load(str(args.model), device="cpu")
    observation, _ = env.reset(
        seed=args.seed,
        options={"start_pose": starts[args.start_id]},
    )
    result = None
    try:
        with mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data) as viewer:
            viewer.cam.distance = 6.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -25
            viewer.cam.lookat[:] = (1.5, 0.0, 0.0)
            while viewer.is_running():
                action, _ = policy.predict(observation, deterministic=True)
                observation, _, terminated, truncated, result = env.step(action)
                viewer.sync()
                if terminated or truncated:
                    break
                time.sleep(env.unwrapped.control_dt)

            if result is not None:
                print(
                    f"Seed {args.seed}, start {args.start_id}: "
                    f"success={result['success']}, "
                    f"reason={result['termination_reason']}, "
                    f"box=({result['box_x']:.2f}, {result['box_y']:.2f}), "
                    f"distance={result['box_target_distance']:.2f} m"
                )
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
    finally:
        env.close()


if __name__ == "__main__":
    main()
