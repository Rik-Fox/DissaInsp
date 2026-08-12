import os
from pathlib import Path

from .agent import create_agent, load_agent
from .env import AngleGrinderEnv


def train(env, episodes=2000, max_steps_per_episode=50, save_path="./models/"):
    """
    Train the Q-learning agent and save the learned table.
    """
    os.makedirs(save_path, exist_ok=True)

    agent = create_agent(env)

    print("\n--- Starting Training ---")
    agent.learn(env, episodes=episodes, max_steps_per_episode=max_steps_per_episode)

    model_path = os.path.join(save_path, "disassembly_agent_q_table.pkl")
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
    agent = load_agent(model_path, env)

    observation, info = env.reset()

    total_reward = 0.0
    terminated = False

    print("\n--- Evaluation Run ---")
    while not terminated:
        action, _ = agent.predict(
            observation,
            action_mask=info.get("action_mask"),
            deterministic=True,
        )
        observation, reward, terminated, truncated, info = env.step(action)

        action_id = env.idx_to_action[action]
        action_details = env.actions[action_id]
        print(
            f"Step: Took action '{action_id}' ({action_details['type']} {action_details['part']}), Reward: {reward:.2f}"
        )

        total_reward += reward

        if terminated or truncated:
            if info.get("is_goal"):
                print("Goal reached!")
            elif info.get("is_dead_end"):
                print("Reached a dead end.")
            else:
                print("Episode terminated.")
            break

    print("--- Evaluation Complete ---")
    print(f"Total reward: {total_reward:.2f}")


def main():
    project_root = Path(__file__).resolve().parent.parent

    # Prefer a graph.pkl in the project root, then fall back to the bundled example data.
    graph_data_path = project_root / "graph.pkl"

    if not graph_data_path.exists():
        print(
            "Please place 'graph.pkl' in the project root or keep the disassembly assets under 'disassembly_graph/'."
        )
        raise SystemExit(1)

    env = AngleGrinderEnv(graph_path=str(graph_data_path))

    save_dir = project_root / "models"
    save_dir.mkdir(exist_ok=True)

    # --- 1. Train a new agent ---
    train(env, episodes=2000, max_steps_per_episode=50, save_path=str(save_dir))

    # --- 2. Evaluate the final trained agent ---
    evaluate(env, str(save_dir / "disassembly_agent_q_table.pkl"))


if __name__ == "__main__":
    main()
