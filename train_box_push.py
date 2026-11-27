import argparse
import json
import math
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from box_push import BoxPushSettings, make_environment


def make_env(rank, settings, monitor_dir, latency_steps, action_noise_std):
    def initialize():
        env = make_environment(
            settings,
            latency_steps=latency_steps,
            action_noise_std=action_noise_std,
        )
        return Monitor(env, filename=str(monitor_dir / f"worker_{rank}"))

    return initialize


def make_vector_env(
    num_envs, settings, monitor_dir, latency_steps, action_noise_std
):
    monitor_dir.mkdir(parents=True, exist_ok=True)
    return SubprocVecEnv(
        [
            make_env(
                rank, settings, monitor_dir, latency_steps, action_noise_std
            )
            for rank in range(num_envs)
        ],
        start_method="spawn",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO on the configurable MuJoCo box-pushing environment."
    )
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--easy-steps", type=int, default=250_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("ppo_box_push.zip"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).with_name("runs") / "box_push",
    )
    parser.add_argument("--target-x", type=float, default=3.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--success-radius", type=float, default=0.45)
    parser.add_argument("--success-hold-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--latency-steps", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--box-mass-scale", type=float, default=1.0)
    parser.add_argument("--friction-scale", type=float, default=1.0)
    parser.add_argument("--box-floor-friction", type=float, default=0.35)
    parser.add_argument("--max-box-speed", type=float, default=0.5)
    parser.add_argument("--car-speed-limit", type=float, default=1.0)
    parser.add_argument("--car-speed-penalty-per-second", type=float, default=5.0)
    parser.add_argument("--lost-bumper-contact-penalty", type=float, default=3.0)
    parser.add_argument("--target-overshoot-penalty", type=float, default=2.0)
    parser.add_argument("--lidar-num-rays", type=int, default=32)
    parser.add_argument("--lidar-max-range", type=float, default=4.0)
    parser.add_argument("--lidar-noise-std", type=float, default=0.01)
    args = parser.parse_args()
    if args.timesteps <= 0 or args.num_envs <= 0:
        parser.error("--timesteps and --num-envs must be positive.")
    if not 0 <= args.easy_steps < args.timesteps:
        parser.error("--easy-steps must be non-negative and less than --timesteps.")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 2^32 - 1.")
    if args.latency_steps < 0 or args.noise_std < 0:
        parser.error("Latency and noise settings must be non-negative.")
    if not all(
        math.isfinite(value)
        for value in (args.target_x, args.target_y, args.noise_std)
    ):
        parser.error("Target coordinates and noise standard deviation must be finite.")
    try:
        BoxPushSettings(
            success_radius=args.success_radius,
            success_hold_steps=args.success_hold_steps,
            max_steps=args.max_steps,
            frame_skip=args.frame_skip,
            max_box_speed=args.max_box_speed,
            box_mass_scale=args.box_mass_scale,
            friction_scale=args.friction_scale,
            box_floor_friction=args.box_floor_friction,
            car_speed_limit=args.car_speed_limit,
            car_speed_penalty_per_second=args.car_speed_penalty_per_second,
            lost_bumper_contact_penalty=args.lost_bumper_contact_penalty,
            target_overshoot_penalty=args.target_overshoot_penalty,
            lidar_num_rays=args.lidar_num_rays,
            lidar_max_range=args.lidar_max_range,
            lidar_noise_std=args.lidar_noise_std,
        )
    except ValueError as error:
        parser.error(str(error))

    run_dir = args.log_dir / f"seed_{args.seed}"
    tensorboard_dir = run_dir / "tensorboard"
    seed_settings = {
        "target_x": args.target_x,
        "target_y": args.target_y,
        "success_radius": args.success_radius,
        "success_hold_steps": args.success_hold_steps,
        "max_steps": args.max_steps,
        "frame_skip": args.frame_skip,
        "max_box_speed": args.max_box_speed,
        "box_mass_scale": args.box_mass_scale,
        "friction_scale": args.friction_scale,
        "box_floor_friction": args.box_floor_friction,
        "car_speed_limit": args.car_speed_limit,
        "car_speed_penalty_per_second": args.car_speed_penalty_per_second,
        "lost_bumper_contact_penalty": args.lost_bumper_contact_penalty,
        "target_overshoot_penalty": args.target_overshoot_penalty,
        "lidar_num_rays": args.lidar_num_rays,
        "lidar_max_range": args.lidar_max_range,
        "lidar_noise_std": args.lidar_noise_std,
    }
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
        "tensorboard_log": str(tensorboard_dir),
    }

    env = None
    easy_actual_timesteps = 0
    try:
        if args.easy_steps:
            easy_settings = BoxPushSettings(stage="easy", **seed_settings)
            env = make_vector_env(
                args.num_envs,
                easy_settings,
                run_dir / "easy" / "monitor",
                args.latency_steps,
                args.noise_std,
            )
            model = PPO(env=env, **ppo_kwargs)
            model.learn(
                total_timesteps=args.easy_steps,
                tb_log_name="box_push",
            )
            easy_actual_timesteps = model.num_timesteps
            env.close()
            env = None
        else:
            model = None

        randomized_settings = BoxPushSettings(
            stage="randomized", **seed_settings
        )
        env = make_vector_env(
            args.num_envs,
            randomized_settings,
            run_dir / "randomized" / "monitor",
            args.latency_steps,
            args.noise_std,
        )
        if model is None:
            model = PPO(env=env, **ppo_kwargs)
        else:
            model.set_env(env)
        model.learn(
            total_timesteps=args.timesteps - args.easy_steps,
            tb_log_name="box_push",
            reset_num_timesteps=args.easy_steps == 0,
        )
        actual_timesteps = model.num_timesteps
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.output))

        config_path = args.output.with_name(f"{args.output.stem}_config.json")
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(
                {
                    "seed": args.seed,
                    "requested_total_timesteps": args.timesteps,
                    "actual_total_timesteps": actual_timesteps,
                    "requested_easy_steps": args.easy_steps,
                    "actual_easy_steps": easy_actual_timesteps,
                    "actual_randomized_steps": (
                        actual_timesteps - easy_actual_timesteps
                    ),
                    "num_envs": args.num_envs,
                    "target": [args.target_x, args.target_y],
                    "success_radius": args.success_radius,
                    "success_hold_steps": args.success_hold_steps,
                    "max_steps": args.max_steps,
                    "max_box_speed": args.max_box_speed,
                    "action_latency_steps": args.latency_steps,
                    "action_noise_std": args.noise_std,
                    "box_mass_scale": args.box_mass_scale,
                    "friction_scale": args.friction_scale,
                    "box_floor_friction": args.box_floor_friction,
                    "car_speed_limit": args.car_speed_limit,
                    "car_speed_penalty_per_second": (
                        args.car_speed_penalty_per_second
                    ),
                    "lost_bumper_contact_penalty": (
                        args.lost_bumper_contact_penalty
                    ),
                    "target_overshoot_penalty": args.target_overshoot_penalty,
                    "lidar_num_rays": args.lidar_num_rays,
                    "lidar_max_range": args.lidar_max_range,
                    "lidar_noise_std": args.lidar_noise_std,
                    "physics_timestep_s": 0.002,
                    "frame_skip": randomized_settings.frame_skip,
                    "ppo_kwargs": ppo_kwargs,
                    "tensorboard_run_dir": model.logger.dir,
                },
                config_file,
                indent=2,
            )
            config_file.write("\n")
        print(f"Saved policy -> {args.output}")
        print(f"Saved training config -> {config_path}")
        if actual_timesteps != args.timesteps:
            print(
                f"Note: PPO collected {actual_timesteps} steps for the "
                f"{args.timesteps}-step request (full rollout batches)."
            )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
