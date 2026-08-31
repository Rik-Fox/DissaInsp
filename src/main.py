import os
from pathlib import Path

from .agent import Model, belief_update, initial_belief, load_policy, save_policy, solve
from .env import AngleGrinderEnv
from .interface import best_action


def train(env, n_iterations=5, n_trajectories=20, horizon=10, save_path="./models/"):
    """
    Solve a PBVI policy (Gamma) and save it.
    """
    os.makedirs(save_path, exist_ok=True)

    model = Model(env)

    print("\n--- Starting Training ---")
    gamma = solve(model, n_iterations=n_iterations, n_trajectories=n_trajectories, horizon=horizon)

    policy_path = os.path.join(save_path, "disassembly_policy.pkl")
    save_policy(gamma, policy_path)
    print(f"\n--- Training Complete. Policy saved to {policy_path} ---")
    return gamma


def _observation_index(env, action_id, info):
    """Maps env.step()'s actual result to the observation index
    model.observation(action_id)'s labels use."""
    details = env.actions[action_id]
    if details["action_type"] == "Disassy":
        return 0  # o_null
    if details["type"] == "Inspect":
        return ["GOOD", "OK", "BAD"].index(info["condition_observation"])
    return env.state_to_idx[info["state_id"]]  # Verify: o = x' directly


def evaluate(env, policy_path):
    """
    Run a trained policy against the live env and print its actions.
    """
    if not os.path.exists(policy_path):
        print(f"Policy not found at {policy_path}. Please train a policy first.")
        return

    print(f"\n--- Loading policy from {policy_path} for evaluation ---")
    gamma = load_policy(policy_path)
    model = Model(env)

    env.reset()
    b = initial_belief(model)
    x_idx = env.state_to_idx[env.root_state]

    total_reward = 0.0
    print("\n--- Evaluation Run ---")
    while True:
        action_id = best_action(gamma, x_idx, b)
        if action_id is None:
            print("No action recorded for this belief - stopping.")
            break

        action_idx = env.action_to_idx[action_id]
        observation, reward, terminated, truncated, info = env.step(action_idx)
        total_reward += reward
        print(f"Step: Took action '{action_id}', Reward: {reward:.2f}")

        if env.actions[action_id]["action_type"] == "Triage":
            print(f"Final triage action: {action_id}")
            break

        o_idx = _observation_index(env, action_id, info)
        b = belief_update(model, b, action_id, o_idx)
        x_idx = env.state_to_idx[info["state_id"]]

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

    env = AngleGrinderEnv(graph_path=str(graph_data_path), config="repair_vs_reuse")

    save_dir = project_root / "models"
    save_dir.mkdir(exist_ok=True)

    # --- 1. Train a new policy ---
    train(env, n_iterations=15, n_trajectories=10, horizon=15, save_path=str(save_dir))

    # --- 2. Evaluate the final trained policy ---
    evaluate(env, str(save_dir / "disassembly_policy.pkl"))


if __name__ == "__main__":
    main()
