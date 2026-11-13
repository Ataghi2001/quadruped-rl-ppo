
from pathlib import Path
import csv

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EVAL_LOG = ROOT / "runs" / "evaluations.npz"

BEST_MODEL = ROOT / "models" / "best_model.zip"
LAST_MODEL = ROOT / "models" / "last_model.zip"

N_EPISODES = 20
SEED = 2026


# 1. Learning Curve

def plot_learning_curve():
    if not EVAL_LOG.exists():
        print("No evaluation log found.")
        print("Training must reach its first evaluation.")
        return

    with np.load(EVAL_LOG) as data:
        timesteps = data["timesteps"]
        rewards = data["results"]

    means = rewards.mean(axis=1)
    stds = rewards.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(timesteps, means, linewidth=2)
    ax.fill_between(
        timesteps,
        means - stds,
        means + stds,
        alpha=0.2,
    )

    ax.set_title("PPO Learning Curve - Ant-v5")
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Mean Evaluation Reward")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "learning_curve.png",
        dpi=180,
    )
    plt.close(fig)

    print("Saved: results/learning_curve.png")


# 2. Evaluate an Agent

def evaluate_agent(agent_name, model=None):
    env = gym.make("Ant-v5")
    rows = []

    try:
        for episode in range(N_EPISODES):
            seed = SEED + episode

            obs, info = env.reset(seed=seed)
            env.action_space.seed(seed)

            x_start = float(env.unwrapped.data.qpos[0])
            total_reward = 0.0
            steps = 0

            while True:
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(
                        obs,
                        deterministic=True,
                    )

                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += float(reward)
                steps += 1

                if terminated or truncated:
                    break

            x_end = float(env.unwrapped.data.qpos[0])

            rows.append({
                "agent": agent_name,
                "episode": episode + 1,
                "reward": total_reward,
                "forward_distance_m": x_end - x_start,
                "episode_length": steps,
            })

    finally:
        env.close()

    return rows


# 3. Evaluation Comparison

def compare_agents():

    model_path = (
        BEST_MODEL if BEST_MODEL.exists()
        else LAST_MODEL
    )

    if not model_path.exists():
        print("No trained model found.")
        return

    model = PPO.load(str(model_path), device="cpu")

    print("Evaluating random agent...")
    random_rows = evaluate_agent("Random")

    print("Evaluating trained PPO agent...")
    ppo_rows = evaluate_agent("PPO", model)

    all_rows = random_rows + ppo_rows

    csv_path = RESULTS_DIR / "evaluation.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    metrics = [
        ("reward", "Episode Reward"),
        ("forward_distance_m", "Forward Distance (m)"),
        ("episode_length", "Episode Length"),
    ]

    markdown = [
        "# Evaluation Results",
        "",
        f"Model: `{model_path.name}`",
        f"Episodes per agent: {N_EPISODES}",
        "",
        "| Metric | Random Agent | PPO Agent |",
        "|---|---:|---:|",
    ]

    for key, label in metrics:
        values = []

        for rows in (random_rows, ppo_rows):
            numbers = np.array(
                [row[key] for row in rows],
                dtype=float,
            )
            mean = numbers.mean()
            std = numbers.std(ddof=1)
            values.append((mean, std))

        print(
            f"{label}: "
            f"Random={values[0][0]:.2f}, "
            f"PPO={values[1][0]:.2f}"
        )

        markdown.append(
            f"| {label} | "
            f"{values[0][0]:.2f} +/- {values[0][1]:.2f} | "
            f"{values[1][0]:.2f} +/- {values[1][1]:.2f} |"
        )

        if key in ("reward", "forward_distance_m"):
            fig, ax = plt.subplots(figsize=(7, 5))

            ax.bar(
                ["Random", "PPO"],
                [v[0] for v in values],
                yerr=[v[1] for v in values],
                capsize=6,
            )

            ax.set_title(f"Agent Comparison - {label}")
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.3)

            fig.tight_layout()

            filename = (
                "reward_comparison.png"
                if key == "reward"
                else "distance_comparison.png"
            )

            fig.savefig(
                RESULTS_DIR / filename,
                dpi=180,
            )
            plt.close(fig)

    markdown.extend([
        "",
        "Values are mean +/- sample standard deviation.",
        "Both agents use the same 20 reset seeds.",
    ])

    (RESULTS_DIR / "evaluation_summary.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )

    print("Evaluation files saved in results/")


if __name__ == "__main__":
    plot_learning_curve()
    compare_agents()
