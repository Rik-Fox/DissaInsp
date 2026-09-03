"""Factored PBVI solver using explicit T, Z, and R models."""
import pickle

import numpy as np
from tqdm import tqdm

from .configs import CONFIDENCE_SCALE
from .env import CONDITION_OBS, CONDITIONS, INSPECTION_ACTIONS
from .pomdp import TRIAGE_ACTIONS


class Model:
    """Adapts AngleGrinderEnv for factored PBVI over condition Y."""

    def __init__(self, env):
        self.env = env
        self.space = env.joint_space
        self.available_insp_actions = {"Inspect"}

    def valid_actions(self, x_idx):
        """Returns valid action set for state index x_idx."""
        state_id = self.env.state_list[x_idx]
        disassy = set(self.env.disassy_by_state[state_id].keys())
        return disassy | self.available_insp_actions | set(TRIAGE_ACTIONS)

    def transition(self, action_id, x_idx=None):
        """Returns transition operator T for action at node x_idx."""
        details = self.env.actions[action_id]
        if details["action_type"] == "Triage":
            return {"type": "terminal"}
        if details["action_type"] == "Disassy" and x_idx is not None:
            state_id = self.env.state_list[x_idx]
            edge = self.env.disassy_by_state[state_id][action_id]
            succ_idx = self.env.state_to_idx[edge["next_state"]]
            p_success = float(self.env.config.disassy_success_prob)
            return {"type": "disassy", "succ_idx": succ_idx, "fail_idx": x_idx, "p_success": p_success}
        return {"type": "identity", "x_idx": x_idx}

    def observation(self, action_id):
        """Returns observation matrix Z and labels for action."""
        details = self.env.actions[action_id]
        if details["type"] == "Inspect":
            return np.asarray(self.env.config.condition_obs_matrix), CONDITION_OBS
        if details["type"] == "Verify":
            return np.eye(2), ["YES", "NO"]
        return np.ones((len(CONDITIONS), 1)), ["null"]

    def reward(self, action_id, x_idx=None):
        """Returns reward/cost vector R for action across conditions."""
        details = self.env.actions[action_id]
        if details["action_type"] == "Triage":
            return np.array([self.env.config.triage_payoff[action_id][y] for y in CONDITIONS], dtype=float)
        if details["action_type"] == "Disassy" and x_idx is not None:
            state_id = self.env.state_list[x_idx]
            edge = self.env.disassy_by_state[state_id].get(action_id)
            cost = float(edge["time"]) if edge else 0.0
            return np.full(len(CONDITIONS), -cost, dtype=float)
        cost = float(details["time"]) if details["time"] is not None else 1.0
        return np.full(len(CONDITIONS), -cost, dtype=float)


def initial_belief(model=None):
    """Returns uniform prior condition belief vector over Y."""
    return np.full(len(CONDITIONS), 1.0 / len(CONDITIONS), dtype=float)


def most_likely_x(model, b=None):
    """Returns current physical state index."""
    return int(model.env.state_to_idx[model.env.current_state_id])


def belief_update(model, b, action_id, o_idx, confidence=1.0):
    """Performs Bayesian update on condition belief b for action."""
    details = model.env.actions[action_id]
    if details["type"] != "Inspect" or confidence == 0:
        return b
    Z, _ = model.observation(action_id)
    # Scales confidence weight using shared CONFIDENCE_SCALE.
    likelihood = Z[:, o_idx] ** (float(confidence) * CONFIDENCE_SCALE)
    weighted = likelihood * b
    total = weighted.sum()
    return weighted / total if total > 0 else weighted


def expand_beliefs(model, belief_sets, n_trajectories=20, horizon=10, epsilon=0.05, rng=None):
    """Phase 1: Monte Carlo rollouts over (x, b_y) with L1 pruning."""
    rng = rng or np.random.default_rng()

    for _ in range(n_trajectories):
        b = initial_belief(model)
        x_idx = model.env.state_to_idx[model.env.root_state]
        model.available_insp_actions = {"Inspect"}

        for _ in range(horizon):
            action_id = rng.choice(sorted(model.valid_actions(x_idx)))
            details = model.env.actions[action_id]
            if details["action_type"] == "Triage":
                break

            T = model.transition(action_id, x_idx)
            Z, obs_labels = model.observation(action_id)

            if details["action_type"] == "Disassy":
                # Stochastically transition according to T
                if rng.random() < T["p_success"]:
                    x_idx = T["succ_idx"]
                model.available_insp_actions = set(INSPECTION_ACTIONS)

            elif details["type"] == "Verify":
                model.available_insp_actions.discard("Verify")

            elif details["type"] == "Inspect":
                obs_dist = b @ Z
                o_idx = rng.choice(len(obs_labels), p=obs_dist / obs_dist.sum())
                b = belief_update(model, b, action_id, o_idx, confidence=rng.uniform(0, 1))
                model.available_insp_actions.discard("Inspect")

            insp_snapshot = frozenset(model.available_insp_actions)
            points = belief_sets.setdefault(x_idx, [])
            # Prune duplicate belief points on condition simplex
            if all(ex_insp != insp_snapshot or np.abs(b - ex_b).sum() >= epsilon for ex_b, ex_insp in points):
                points.append((b.copy(), insp_snapshot))

    return belief_sets



def _observation_cross_sum(model, gamma, T, Z, b, discount, x_idx, next_insp):
    """Computes discounted continuation value vector T @ (Z * alpha)."""
    t_type = T["type"]
    if t_type == "disassy":
        # Disassembly yields null observation; continuation projects through stochastic T
        # Disassembly refreshes inspection actions
        succ_pts = [a for a, act in gamma.get(T["succ_idx"], []) if act is not None]
        best_succ = max(succ_pts, key=lambda a: float(b @ a)) if succ_pts else np.zeros(len(CONDITIONS))
        curr_pts = [a for a, act in gamma.get(T["fail_idx"], []) if act is not None]
        best_fail = max(curr_pts, key=lambda a: float(b @ a)) if curr_pts else best_succ
        # Expected continuation value: T @ alpha
        expected_future = T["p_success"] * best_succ + (1.0 - T["p_success"]) * best_fail
        return discount * expected_future

    # Verify and Inspect: continuation must respect next_insp mask
    valid_next = (model.valid_actions(x_idx) - set(INSPECTION_ACTIONS)) | next_insp
    curr_pts = [a for a, act in gamma.get(x_idx, []) if act is not None and act in valid_next]

    if Z.shape == (2, 2):
        # Verify: identity transition on condition, evaluates current state alpha
        best_curr = max(curr_pts, key=lambda a: float(b @ a)) if curr_pts else np.zeros(len(CONDITIONS))
        return discount * best_curr

    # Inspect: Z is (3, 3) confusion matrix over diagnostic observations
    best_alphas_o = []
    for o in range(Z.shape[1]):
        likelihood = Z[:, o] ** CONFIDENCE_SCALE
        weighted = likelihood * b
        b_post = weighted / weighted.sum() if weighted.sum() > 0 else b
        best_a = max(curr_pts, key=lambda a: float(b_post @ a)) if curr_pts else np.zeros(len(CONDITIONS))
        best_alphas_o.append(best_a)
    # correction: expected continuation payoff at condition y across observations Z
    correction = np.array([sum(Z[y, o] * best_alphas_o[o][y] for o in range(Z.shape[1])) for y in range(len(CONDITIONS))])
    return discount * correction


def backup(model, belief_sets, gamma, discount=0.99):
    """Phase 2: One PBVI backup sweep using explicit T, Z, and R."""
    new_gamma = {x_idx: [] for x_idx in belief_sets}

    for x_idx, points in belief_sets.items():
        for b, insp_actions in points:
            model.available_insp_actions = insp_actions
            best_value, best_alpha, best_action = -np.inf, None, None

            for action_id in model.valid_actions(x_idx):
                details = model.env.actions[action_id]
                R = model.reward(action_id, x_idx)

                if details["action_type"] == "Triage":
                    alpha = R
                else:
                    T = model.transition(action_id, x_idx)
                    Z, obs_labels = model.observation(action_id)
                    # Next available inspection actions
                    if details["action_type"] == "Disassy":
                        next_insp = set(INSPECTION_ACTIONS)
                    else:
                        next_insp = insp_actions - {action_id}
                    alpha = R + _observation_cross_sum(model, gamma, T, Z, b, discount, x_idx, next_insp)

                value = float(b @ alpha)
                if value > best_value:
                    best_value, best_alpha, best_action = value, alpha, action_id

            new_gamma[x_idx].append((best_alpha, best_action))

    return new_gamma



def solve(model, n_iterations=5, n_trajectories=20, horizon=10, epsilon=0.05, discount=0.99, rng=None):
    """Alternates belief expansion and value backup for n_iterations."""
    belief_sets = {}
    root_x = model.env.state_to_idx[model.env.root_state]
    gamma = {root_x: [(np.zeros(len(CONDITIONS)), None)]}

    for i in tqdm(range(n_iterations), desc="Solving", unit="iter"):
        belief_sets = expand_beliefs(model, belief_sets, n_trajectories, horizon, epsilon, rng)
        tqdm.write(f"--- Iteration {i + 1}/{n_iterations}: {sum(len(v) for v in belief_sets.values())} belief points ---")

        gamma = backup(model, belief_sets, gamma, discount)
        tqdm.write(f"--- Iteration {i + 1}/{n_iterations}: {sum(len(v) for v in gamma.values())} alpha-vectors ---")

    return gamma


def save_policy(gamma, path):
    """Saves policy dictionary to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(gamma, f)


def load_policy(path):
    """Loads policy dictionary from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
