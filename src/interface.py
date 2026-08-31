"""Interactive REPL for testing a converged PBVI policy (Gamma) with a human
operator supplying real-world observations, in place of a live env.step().

Usage:
    python -m src.interface --policy policy.pkl --graph graph.pkl
"""
import argparse

import numpy as np

from .env import AngleGrinderEnv, CONDITIONS
from .agent import Model, belief_update, initial_belief, load_policy, most_likely_x


def best_action(gamma, x_idx, b):
    """Argmax over the alpha-vectors recorded for belief points nominally at
    physical node x_idx (falls back to the full policy if none were)."""
    candidates = gamma.get(x_idx) or [pair for points in gamma.values() for pair in points]
    best_value, chosen = -np.inf, None
    for alpha, action_id in candidates:
        if action_id is None:
            continue
        value = float(b @ alpha)
        if value > best_value:
            best_value, chosen = value, action_id
    return chosen


def prompt_observation(model, action_id):
    """Human-in-the-loop: Disassy actions bypass the prompt (o_null, full
    confidence); Verify/Inspect ask for both the observation and confidence."""
    details = model.env.actions[action_id]
    if details["action_type"] == "Disassy":
        return 0, 1.0

    _, obs_labels = model.observation(action_id)
    obs_labels = list(obs_labels)
    while True:
        raw = input(f"Observation for '{action_id}' {obs_labels}: ").strip()
        if raw in obs_labels:
            break
        print(f"Invalid observation - choose one of {obs_labels}")
    o_idx = obs_labels.index(raw)

    while True:
        raw_confidence = input("How confident are you (0-1)? ").strip()
        try:
            confidence = float(raw_confidence)
            if 0 <= confidence <= 1:
                return o_idx, confidence
        except ValueError:
            pass
        print("Enter a number between 0 and 1.")


def run(policy_path, graph_path):
    env = AngleGrinderEnv(graph_path=graph_path)
    model = Model(env)
    gamma = load_policy(policy_path)

    b = initial_belief(model)
    x_idx = model.env.state_to_idx[model.env.root_state]

    while True:
        action_id = best_action(gamma, x_idx, b)
        if action_id is None:
            print("No action recorded for this belief - stopping.")
            break
        print(f"AGENT DECISION: Execute action '{action_id}'")

        if model.env.actions[action_id]["action_type"] == "Triage":
            print(f"Episode complete - final triage action: {action_id}")
            break

        o_idx, confidence = prompt_observation(model, action_id)
        b = belief_update(model, b, action_id, o_idx, confidence)
        x_idx = most_likely_x(model, b)

        marginal_condition = b.reshape(model.space.n_y, model.space.n_x).sum(axis=1)
        summary = {c: round(float(p), 3) for c, p in zip(CONDITIONS, marginal_condition)}
        print(f"Belief over condition: {summary}")


def main():
    parser = argparse.ArgumentParser(description="Interactive PBVI policy tester")
    parser.add_argument("--policy", required=True, help="Path to a saved Gamma policy (pickle)")
    parser.add_argument("--graph", default="graph.pkl", help="Path to graph.pkl")
    args = parser.parse_args()
    run(args.policy, args.graph)


if __name__ == "__main__":
    main()
