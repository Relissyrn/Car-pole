import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from train_car import CarForward


EVALUATION_EPISODES = 100
GOAL_RATE_THRESHOLD = 0.90
MEAN_GOAL_TIME_THRESHOLD = 4.0
FLIP_RATE_THRESHOLD = 0.05
MAX_LATERAL_DRIFT = 1.0
MEAN_ABSOLUTE_YAW_RATE_THRESHOLD = 0.3
STABILITY_WINDOW = 200_000
STABILITY_BINS = 10
MAX_STABILITY_DECREASES = 5
MAX_STABILITY_RELATIVE_RANGE = 0.01


def read_reward_history(log_dir):
    event_files = list(Path(log_dir).rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event files found under {log_dir}."
        )

    history = []
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        if "rollout/ep_rew_mean" not in accumulator.Tags().get("scalars", []):
            continue
        history.extend(
            (event.step, event.value)
            for event in accumulator.Scalars("rollout/ep_rew_mean")
        )
    if not history:
        raise ValueError(
            f"No rollout/ep_rew_mean values found under {log_dir}."
        )
    return sorted(history)


def evaluate_training_stability(log_dir):
    history = read_reward_history(log_dir)
    last_step = history[-1][0]
    start_step = last_step - STABILITY_WINDOW
    if start_step < 0:
        return {
            "passed": False,
            "reason": "Training log does not cover the final 200,000 steps.",
            "last_logged_step": last_step,
        }

    edges = np.linspace(start_step, last_step, STABILITY_BINS + 1)
    bin_means = []
    for index in range(STABILITY_BINS):
        lower = edges[index]
        upper = edges[index + 1]
        values = [
            value
            for step, value in history
            if lower <= step < upper or (index == STABILITY_BINS - 1 and step == upper)
        ]
        if not values:
            return {
                "passed": False,
                "reason": f"No reward samples in stability bin {index + 1}.",
                "last_logged_step": last_step,
            }
        bin_means.append(float(np.mean(values)))

    decreases = int(np.count_nonzero(np.diff(bin_means) < 0.0))
    trend_slope = float(np.polyfit(np.arange(STABILITY_BINS), bin_means, 1)[0])
    relative_range = float(
        (max(bin_means) - min(bin_means)) / abs(float(np.mean(bin_means)))
    )
    passed = (
        decreases <= MAX_STABILITY_DECREASES
        and trend_slope >= 0.0
        and bin_means[-1] > 0.0
        and relative_range <= MAX_STABILITY_RELATIVE_RANGE
    )
    return {
        "passed": bool(passed),
        "last_logged_step": last_step,
        "window_steps": STABILITY_WINDOW,
        "bin_means": bin_means,
        "decreasing_transitions": decreases,
        "maximum_decreasing_transitions": MAX_STABILITY_DECREASES,
        "trend_slope_per_bin": trend_slope,
        "relative_range": relative_range,
        "maximum_relative_range": MAX_STABILITY_RELATIVE_RANGE,
        "final_bin_positive": bool(bin_means[-1] > 0.0),
        "nonnegative_trend": bool(trend_slope >= 0.0),
    }


def evaluate_policy(model_path, episodes, seed):
    env = CarForward()
    episode_results = []
    yaw_samples = []
    try:
        policy = PPO.load(str(model_path), device="cpu")
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_yaw = []
            while True:
                action, _ = policy.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                episode_yaw.append(abs(info["yaw_rate"]))
                if terminated or truncated:
                    reached = bool(terminated and info["reached_goal"])
                    episode_results.append(
                        {
                            "reached": reached,
                            "flipped": bool(
                                terminated and info["termination_reason"] == "flipped"
                            ),
                            "time": float(info["time"]),
                            "y_goal": float(info["y"]) if reached else None,
                            "reason": info["termination_reason"],
                        }
                    )
                    yaw_samples.extend(episode_yaw)
                    break
    finally:
        env.close()

    successes = [result for result in episode_results if result["reached"]]
    goal_rate = len(successes) / episodes
    flip_rate = sum(result["flipped"] for result in episode_results) / episodes
    mean_goal_time = (
        float(np.mean([result["time"] for result in successes]))
        if successes
        else None
    )
    max_lateral_drift = (
        float(max(abs(result["y_goal"]) for result in successes))
        if successes
        else None
    )
    mean_absolute_yaw_rate = (
        float(np.mean(yaw_samples)) if yaw_samples else float("inf")
    )

    return {
        "episodes": episodes,
        "seed": seed,
        "deterministic": True,
        "goal_rate": goal_rate,
        "mean_time_to_goal_s": mean_goal_time,
        "flip_rate": flip_rate,
        "max_absolute_y_at_goal_m": max_lateral_drift,
        "mean_absolute_yaw_rate_rad_s": mean_absolute_yaw_rate,
        "episode_results": episode_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the deterministic car policy against the acceptance tests."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).with_name("ppo_car_forward_v1.zip"),
    )
    parser.add_argument("--episodes", type=int, default=EVALUATION_EPISODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tensorboard-log-dir",
        type=Path,
        help="Override the TensorBoard run directory recorded in training_config.json.",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        help="Override the training_config.json next to the model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("evaluation_metrics.json"),
    )
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Policy not found: {args.model}")
    if args.episodes != EVALUATION_EPISODES or args.seed < 0:
        parser.error(
            f"--episodes must be {EVALUATION_EPISODES} and --seed must be non-negative."
        )

    config_path = args.training_config or args.model.parent / "training_config.json"
    if not config_path.is_file():
        parser.error(f"Training config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        training_config = json.load(config_file)
    tensorboard_log_dir = args.tensorboard_log_dir or Path(
        training_config["tensorboard_run_dir"]
    )

    policy_metrics = evaluate_policy(args.model, args.episodes, args.seed)
    stability_metrics = evaluate_training_stability(tensorboard_log_dir)
    goal_rate_pass = policy_metrics["goal_rate"] >= GOAL_RATE_THRESHOLD
    time_pass = (
        policy_metrics["mean_time_to_goal_s"] is not None
        and policy_metrics["mean_time_to_goal_s"] <= MEAN_GOAL_TIME_THRESHOLD
    )
    flip_pass = policy_metrics["flip_rate"] <= FLIP_RATE_THRESHOLD
    lateral_pass = (
        policy_metrics["max_absolute_y_at_goal_m"] is not None
        and policy_metrics["max_absolute_y_at_goal_m"] <= MAX_LATERAL_DRIFT
    )
    yaw_pass = (
        policy_metrics["mean_absolute_yaw_rate_rad_s"]
        < MEAN_ABSOLUTE_YAW_RATE_THRESHOLD
    )

    report = {
        "model": str(args.model.resolve()),
        "training_config": str(config_path.resolve()),
        "training_seed": training_config["seed"],
        "training_total_timesteps": training_config["total_timesteps"],
        "ppo_kwargs": training_config["ppo_kwargs"],
        "tensorboard_run_dir": str(tensorboard_log_dir.resolve()),
        "checks": {
            "goal_rate_at_least_90_percent": {
                "passed": bool(goal_rate_pass),
                "value": policy_metrics["goal_rate"],
                "threshold": GOAL_RATE_THRESHOLD,
            },
            "mean_time_to_goal_at_most_4_seconds": {
                "passed": bool(time_pass),
                "value": policy_metrics["mean_time_to_goal_s"],
                "threshold_s": MEAN_GOAL_TIME_THRESHOLD,
            },
            "flip_rate_at_most_5_percent": {
                "passed": bool(flip_pass),
                "value": policy_metrics["flip_rate"],
                "threshold": FLIP_RATE_THRESHOLD,
            },
            "lateral_drift_at_most_1_meter": {
                "passed": bool(lateral_pass),
                "value": policy_metrics["max_absolute_y_at_goal_m"],
                "threshold_m": MAX_LATERAL_DRIFT,
            },
            "mean_absolute_yaw_rate_below_0_3": {
                "passed": bool(yaw_pass),
                "value": policy_metrics["mean_absolute_yaw_rate_rad_s"],
                "threshold_rad_s": MEAN_ABSOLUTE_YAW_RATE_THRESHOLD,
            },
            "training_reward_monotoneish": stability_metrics,
        },
        "policy_metrics": policy_metrics,
    }
    report["passed_all"] = all(
        check["passed"] for check in report["checks"].values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as metrics_file:
        json.dump(report, metrics_file, indent=2)
        metrics_file.write("\n")

    print(json.dumps(report, indent=2))
    print(f"Metrics saved -> {args.output}")
    if not report["passed_all"]:
        raise SystemExit("The policy did not pass all six acceptance tests.")


if __name__ == "__main__":
    main()
