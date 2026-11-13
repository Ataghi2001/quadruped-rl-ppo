
import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)


# ============================================
# 1. Configuration
# ============================================

ENV_ID = "Ant-v5"
SEED = 42

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "runs"

BEST_MODEL = MODEL_DIR / "best_model.zip"
LAST_MODEL = MODEL_DIR / "last_model.zip"


# ============================================
# 2. Create PPO Agent
# ============================================

def create_agent(env):

    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,

        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,

        policy_kwargs={
            "net_arch": {
                "pi": [256, 256],
                "vf": [256, 256],
            }
        },

        device="cpu",
        verbose=1,
        seed=SEED,
        tensorboard_log=str(LOG_DIR),
    )

    return model


# ============================================
# 3. Training
# ============================================

def train(total_steps, n_envs, resume=None):

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("\nCreating training environments...")

    # Single-process vectorized environments
    train_env = make_vec_env(
        ENV_ID,
        n_envs=n_envs,
        seed=SEED,
        vec_env_cls=DummyVecEnv,
    )

    # Separate environment for evaluation
    eval_env = make_vec_env(
        ENV_ID,
        n_envs=1,
        seed=SEED + 1000,
        vec_env_cls=DummyVecEnv,
    )

    model = None

    try:
        if resume is not None:

            print(f"Resuming from: {resume}")

            model = PPO.load(
                str(resume),
                env=train_env,
                device="cpu",
                tensorboard_log=str(LOG_DIR),
            )

        else:
            model = create_agent(train_env)

        # Save the best evaluation model
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(MODEL_DIR),
            log_path=str(LOG_DIR),
            eval_freq=max(50_000 // n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
        )

        # Save a checkpoint every 250k steps
        checkpoint_callback = CheckpointCallback(
            save_freq=max(250_000 // n_envs, 1),
            save_path=str(CHECKPOINT_DIR),
            name_prefix="ant_ppo",
            verbose=1,
        )

        callbacks = CallbackList([
            eval_callback,
            checkpoint_callback,
        ])

        print("\n==============================")
        print("Starting PPO Training")
        print("==============================")
        print(f"Environment: {ENV_ID}")
        print(f"Total steps: {total_steps:,}")
        print(f"Number of environments: {n_envs}")
        print(f"Device: CPU")
        print("==============================\n")

        try:
            model.learn(
                total_timesteps=total_steps,
                callback=callbacks,
                progress_bar=True,
                tb_log_name="ant_ppo",
                reset_num_timesteps=(resume is None),
            )

            print("\nTraining completed!")

        except KeyboardInterrupt:
            print("\nTraining interrupted.")

    finally:
        if model is not None:
            model.save(str(LAST_MODEL))
            print(f"Latest model saved to: {LAST_MODEL}")

        train_env.close()
        eval_env.close()


# ============================================
# 4. Watch the Robot
# ============================================

def watch(episodes=3, random_agent=False, model_path=None):

    model = None

    if not random_agent:

        if model_path is not None:
            selected_model = Path(model_path)

        elif BEST_MODEL.exists():
            selected_model = BEST_MODEL

        else:
            selected_model = LAST_MODEL

        if not selected_model.exists():
            print("No trained model found!")
            print("Please train the agent first.")
            return

        print(f"Loading model: {selected_model}")

        model = PPO.load(
            str(selected_model),
            device="cpu",
        )

    else:
        print("Running a RANDOM agent...")

    env = gym.make(
        ENV_ID,
        render_mode="human",
    )

    try:
        for episode in range(1, episodes + 1):

            obs, info = env.reset(
                seed=SEED + episode
            )

            initial_x = float(
                env.unwrapped.data.qpos[0]
            )

            total_reward = 0.0
            steps = 0

            while True:

                if random_agent:
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

            final_x = float(
                env.unwrapped.data.qpos[0]
            )

            distance = final_x - initial_x

            print(
                f"Episode {episode} | "
                f"Reward: {total_reward:.2f} | "
                f"Steps: {steps} | "
                f"Forward distance: {distance:.2f} m"
            )

    finally:
        env.close()


# ============================================
# 5. Command Line Interface
# ============================================

def main():

    parser = argparse.ArgumentParser(
        description="Quadruped Robot RL Project"
    )

    parser.add_argument(
        "mode",
        choices=["random", "train", "watch"],
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=3_000_000,
    )

    parser.add_argument(
        "--envs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.mode == "random":

        watch(
            episodes=args.episodes,
            random_agent=True,
        )

    elif args.mode == "train":

        if args.steps <= 0 or args.envs <= 0:
            parser.error(
                "--steps and --envs must be positive"
            )

        train(
            total_steps=args.steps,
            n_envs=args.envs,
            resume=args.resume,
        )

    elif args.mode == "watch":

        watch(
            episodes=args.episodes,
            model_path=args.model,
        )


if __name__ == "__main__":
    main()
