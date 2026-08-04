import os
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from .agent import create_agent, load_agent
from .env import DisassemblyEnv


def train(env, total_timesteps=200000, save_path="./models/"):
    """
    Train the RL agent and save the model.
    """
    os.makedirs(save_path, exist_ok=True)

    # Callback to save the model periodically during training
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=save_path, name_prefix="disassembly_agent"
    )

    agent = create_agent(env)

    print("\n--- Starting Training ---")
    agent.learn(
        total_timesteps=total_timesteps, callback=checkpoint_callback, progress_bar=True
    )

    model_path = os.path.join(save_path, "disassembly_agent_final.zip")
    agent.save(model_path)
    print(f"\n--- Training Complete. Model saved to {model_path} ---")
    return agent


def evaluate(env, model_path):
    """
    Evaluate a trained agent and print its actions.
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train an agent first.")
        return

    print(f"\n--- Loading model from {model_path} for evaluation ---")
    # We need to use the same VecEnv for evaluation
    agent = load_agent(model_path, env)

    obs = env.reset()

    total_reward = 0
    terminated = False

    print("\n--- Evaluation Run ---")
    while not terminated:
        # The action mask is automatically passed to the agent during prediction
        action, _states = agent.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        terminated = dones[0]
        info = infos[0]

        # Get action details from the unwrapped environment
        action_id = env.envs[0].idx_to_action[action.item()]
        action_details = env.envs[0].actions[action_id]
        print(
            f"Step: Took action '{action_id}' ({action_details['type']} {action_details['part']}), Reward: {rewards[0]:.2f}"
        )

        total_reward += rewards[0]

        if terminated:
            if info.get("is_goal"):
                print("Goal reached!")
            elif info.get("is_dead_end"):
                print("Reached a dead end.")
            else:
                print("Episode terminated.")

    print(f"--- Evaluation Complete ---")
    print(f"Total reward: {total_reward:.2f}")


def main():
    project_root = Path(__file__).resolve().parent.parent

    # Prefer a graph.pkl in the project root, then fall back to the bundled example data.
    graph_data_path = project_root / "graph.pkl"

    if not graph_data_path.exists():
        graph_data_path = project_root / "disassembly_graph" / "disassembly_angle_grinder" / "disassembly_angle_grinder" / "ui.html"
        if not graph_data_path.exists():
            print("Error: Data file not found.")
            print(
                "Please place 'graph.pkl' in the project root or keep the disassembly assets under 'disassembly_graph/'."
            )
            raise SystemExit(1)

    # Create the Gym environment, wrapped in a VecEnv for Stable Baselines3
    env = make_vec_env(
        DisassemblyEnv,
        n_envs=1,
        env_kwargs=dict(graph_path=str(graph_data_path)),
    )

    save_dir = project_root / "models"
    save_dir.mkdir(exist_ok=True)

    # --- 1. Train a new agent ---
    train(env, total_timesteps=200000, save_path=str(save_dir))

    # --- 2. Evaluate the final trained agent ---
    evaluate(env, str(save_dir / "disassembly_agent_final.zip"))


if __name__ == "__main__":
    main()
