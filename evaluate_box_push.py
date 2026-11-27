import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from box_push import BoxPushSettings, make_environment


CAR_Y_VALUES = (-0.6, -0.3, 0.0, 0.3, 0.6)
CAR_YAW_VALUES = (-0.25, -0.08, 0.08, 0.25)


def fixed_evaluation_starts(target_x, target_y):
    box_x = target_x - 1.6
    return tuple(
        {
            "car_x": box_x - 0.54,
            "car_y": target_y + car_y,
            "car_yaw": car_yaw,
            "box_x": box_x,
            "box_y": target_y,
            "box_yaw": 0.0,
        }
        for car_yaw in CAR_YAW_VALUES
        for car_y in CAR_Y_VALUES
    )


def _spread(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def _heatmap_svg(success_rates):
    cell_width = 110
    cell_height = 72
    left = 105
    top = 64
    width = left + len(CAR_Y_VALUES) * cell_width + 16
    height = top + len(CAR_YAW_VALUES) * cell_height + 42
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="12" y="25" font-family="sans-serif" font-size="16">'
        "Success by fixed start (car dy from target)</text>",
    ]
    for column, car_y in enumerate(CAR_Y_VALUES):
        x = left + column * cell_width
        elements.append(
            f'<text x="{x + cell_width / 2}" y="50" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">dy={car_y:+.1f}m</text>'
        )
    for row, car_yaw in enumerate(CAR_YAW_VALUES):
        y = top + row * cell_height
        elements.append(
            f'<text x="{left - 10}" y="{y + cell_height / 2 + 4}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">yaw={car_yaw:+.2f}</text>'
        )
        for column in range(len(CAR_Y_VALUES)):
            rate = success_rates[row][column]
            red = int(210 * (1.0 - rate) + 35 * rate)
            green = int(55 * (1.0 - rate) + 155 * rate)
            blue = int(55 * (1.0 - rate) + 70 * rate)
            x = left + column * cell_width
            elements.extend(
                (
                    f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" '
                    f'fill="rgb({red},{green},{blue})" stroke="#ffffff"/>',
                    f'<text x="{x + cell_width / 2}" y="{y + 31}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="17">{rate:.0%}</text>',
                )
            )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def evaluate_model(
    model_path, seed, settings, latency_steps, noise_std, evaluation_starts
):
    model = PPO.load(str(model_path), device="cpu")
    env = make_environment(
        settings,
        latency_steps=latency_steps,
        action_noise_std=noise_std,
        noise_seed=seed,
    )
    episodes = []
    try:
        for start_id, start_pose in enumerate(evaluation_starts):
            observation, _ = env.reset(
                seed=seed * 1000 + start_id,
                options={"start_pose": start_pose},
            )
            total_reward = 0.0
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    episodes.append(
                        {
                            "seed": seed,
                            "model": str(model_path),
                            "start_id": start_id,
                            "start_pose": start_pose,
                            "success": bool(info["success"]),
                            "reason": info["termination_reason"],
                            "time_s": float(info["time"]),
                            "episode_return": float(total_reward),
                            "box_target_distance": float(
                                info["box_target_distance"]
                            ),
                            "box_x": float(info["box_x"]),
                            "box_y": float(info["box_y"]),
                        }
                    )
                    break
    finally:
        env.close()

    success_times = [episode["time_s"] for episode in episodes if episode["success"]]
    return {
        "seed": seed,
        "model": str(model_path),
        "success_rate": sum(episode["success"] for episode in episodes) / len(episodes),
        "mean_success_time_s": (
            float(np.mean(success_times)) if success_times else None
        ),
        "mean_episode_return": float(
            np.mean([episode["episode_return"] for episode in episodes])
        ),
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate >=5 trained seeds on identical, fixed start poses."
    )
    parser.add_argument(
        "--models",
        type=Path,
        nargs="+",
        required=True,
        help="At least five independently trained policy files.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Training seeds in model order; defaults to 0..N-1.",
    )
    parser.add_argument(
        "--variant",
        choices=("nominal", "latency", "noise", "mismatch"),
        required=True,
    )
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
    parser.add_argument("--target-x", type=float, default=3.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--success-radius", type=float, default=0.45)
    parser.add_argument("--success-hold-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--tuning-runs", type=int, required=True)
    parser.add_argument("--tuning-timesteps", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("box_push_evaluation.json"),
    )
    args = parser.parse_args()

    if len(args.models) < 5:
        parser.error("Provide at least five independently trained models.")
    seeds = args.seeds if args.seeds is not None else list(range(len(args.models)))
    if len(seeds) != len(args.models) or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain one unique seed per model.")
    if any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
        parser.error("Seeds must be between 0 and 2^32 - 1.")
    if args.tuning_runs <= 0 or args.tuning_timesteps <= 0:
        parser.error("--tuning-runs and --tuning-timesteps must be positive.")
    if args.latency_steps < 0 or args.noise_std < 0:
        parser.error("Latency and noise settings must be non-negative.")
    if not np.isfinite(args.noise_std):
        parser.error("--noise-std must be finite.")
    if not np.all(np.isfinite((args.target_x, args.target_y))):
        parser.error("Target coordinates must be finite.")
    if args.variant == "latency" and args.latency_steps == 0:
        parser.error("The latency variant requires --latency-steps > 0.")
    if args.variant == "noise" and args.noise_std == 0:
        parser.error("The noise variant requires --noise-std > 0.")
    if args.variant == "mismatch" and (
        args.box_mass_scale == 1.0
        and args.friction_scale == 1.0
        and np.isclose(
            args.box_floor_friction, BoxPushSettings().box_floor_friction
        )
    ):
        parser.error("Mismatch evaluation needs non-nominal mass or friction settings.")
    missing = [str(path) for path in args.models if not path.is_file()]
    if missing:
        parser.error(f"Model file(s) not found: {', '.join(missing)}")

    try:
        settings = BoxPushSettings(
            stage="randomized",
            target_x=args.target_x,
            target_y=args.target_y,
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
    evaluation_starts = fixed_evaluation_starts(
        settings.target_x, settings.target_y
    )
    seed_reports = [
        evaluate_model(
            model_path,
            seed,
            settings,
            args.latency_steps,
            args.noise_std,
            evaluation_starts,
        )
        for model_path, seed in zip(args.models, seeds)
    ]
    all_episodes = [
        episode
        for seed_report in seed_reports
        for episode in seed_report["episodes"]
    ]
    start_rates = []
    for start_id in range(len(evaluation_starts)):
        episodes = [item for item in all_episodes if item["start_id"] == start_id]
        start_rates.append(
            sum(item["success"] for item in episodes) / len(episodes)
        )
    success_rate_grid = [
        start_rates[row * len(CAR_Y_VALUES) : (row + 1) * len(CAR_Y_VALUES)]
        for row in range(len(CAR_YAW_VALUES))
    ]

    successful_times = [
        seed_report["mean_success_time_s"]
        for seed_report in seed_reports
        if seed_report["mean_success_time_s"] is not None
    ]
    report = {
        "variant": args.variant,
        "settings": {
            "latency_steps": args.latency_steps,
            "action_noise_std": args.noise_std,
            "box_mass_scale": args.box_mass_scale,
            "friction_scale": args.friction_scale,
            "box_floor_friction": args.box_floor_friction,
            "car_speed_limit": args.car_speed_limit,
            "car_speed_penalty_per_second": (
                args.car_speed_penalty_per_second
            ),
            "lost_bumper_contact_penalty": args.lost_bumper_contact_penalty,
            "target_overshoot_penalty": args.target_overshoot_penalty,
            "lidar_num_rays": args.lidar_num_rays,
            "lidar_max_range": args.lidar_max_range,
            "lidar_noise_std": args.lidar_noise_std,
            "stage": settings.stage,
            "target": [settings.target_x, settings.target_y],
            "success_radius": settings.success_radius,
            "success_hold_steps": settings.success_hold_steps,
            "max_steps": settings.max_steps,
            "max_box_speed": settings.max_box_speed,
            "frame_skip": settings.frame_skip,
        },
        "tuning_effort": {
            "runs_per_variant": args.tuning_runs,
            "timesteps_per_run": args.tuning_timesteps,
            "must_match_for_all_variants": True,
        },
        "evaluation": {
            "start_count": len(evaluation_starts),
            "identical_fixed_starts_for_every_seed_and_variant": True,
            "deterministic_policy": True,
            "seed_count": len(seed_reports),
        },
        "summary_across_seeds": {
            "success_rate": _spread(
                [item["success_rate"] for item in seed_reports]
            ),
            "mean_episode_return": _spread(
                [item["mean_episode_return"] for item in seed_reports]
            ),
            "mean_success_time_s": (
                _spread(successful_times) if successful_times else None
            ),
        },
        "per_seed": [
            {
                key: value
                for key, value in seed_report.items()
                if key != "episodes"
            }
            for seed_report in seed_reports
        ],
        "failures": [item for item in all_episodes if not item["success"]],
        "episode_results": all_episodes,
        "heatmap": {
            "columns_car_y_offset_m": list(CAR_Y_VALUES),
            "rows_car_yaw_rad": list(CAR_YAW_VALUES),
            "success_rate": success_rate_grid,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
        output_file.write("\n")

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "seed", "model", "start_id", "success", "reason",
                "time_s", "episode_return", "box_target_distance",
                "box_x", "box_y", "start_pose",
            ),
        )
        writer.writeheader()
        for episode in all_episodes:
            row = episode.copy()
            row["start_pose"] = json.dumps(row["start_pose"], sort_keys=True)
            writer.writerow(row)

    heatmap_path = args.output.with_suffix(".svg")
    heatmap_path.write_text(_heatmap_svg(success_rate_grid), encoding="utf-8")
    print(json.dumps(report["summary_across_seeds"], indent=2))
    print(f"Full results, including failures -> {args.output}")
    print(f"Episode table -> {csv_path}")
    print(f"Start-pose heat map -> {heatmap_path}")


if __name__ == "__main__":
    main()
