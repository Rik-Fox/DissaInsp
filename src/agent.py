"""Point-Based Value Iteration (PBVI) solver for the disassembly/triage
POMDP, built on top of a loaded AngleGrinderEnv's T/Z/R matrices.

Belief points are full joint (n_s,) distributions over (x, y) - dense numpy
arrays, since T/Z stay sparse and n_s is small enough (thousands) for dense
belief vectors to be cheap. Belief sets and alpha-vectors are partitioned by
the physical node x a point was reached while nominally "at" (an
organisational/pruning convenience - the belief itself can still place mass
on other x, it isn't restricted to that node).

This is a draft/reference implementation to demonstrate the algorithm end to
end, not a tuned/optimized solver - see README for what's still a
placeholder.
"""
import pickle

import numpy as np
import scipy.sparse as sp

from .env import ALWAYS_VALID_ACTIONS, CONDITION_OBS, CONDITIONS


class Model:
    """Adapts a loaded AngleGrinderEnv's per-action T/Z/R into the sparse
    matrices/vectors a POMDP solver needs, reusing env.py's own builders so
    the simulator and planner never disagree on semantics."""

    def __init__(self, env):
        self.env = env
        self.space = env.joint_space

    def valid_actions(self, x_idx):
        """Same masking rule as the live env, without needing a live episode."""
        state_id = self.env.state_list[x_idx]
        disassy = set(self.env.disassy_by_state[state_id].keys())
        return disassy | ALWAYS_VALID_ACTIONS

    def transition(self, action_id):
        """Sparse (n_s, n_s) T matrix, or None for Triage (terminal - no
        continuation value)."""
        details = self.env.actions[action_id]
        if details["action_type"] == "Triage":
            return None
        if details["action_type"] == "Disassy":
            return self.env._disassy_transition_matrix(action_id)
        return self.env.transition_model.identity()  # Insp: x', y' = x, y

    def observation(self, action_id):
        """(sparse (n_s, n_o) Z matrix, observation labels), or (None, None)
        for Triage (terminal - no observation)."""
        details = self.env.actions[action_id]
        if details["action_type"] == "Triage":
            return None, None
        if details["type"] == "Inspect":
            return self.env._condition_obs_matrix, CONDITION_OBS
        if details["type"] == "Verify":
            # o = x' directly (deterministic identity map) - see env.py.
            z_x = sp.identity(self.space.n_x, format="csr")
            return self.env.observation_model.verification(z_x), self.env.state_list
        return self.env.observation_model.null(1), ["null"]  # Disassy

    def reward(self, action_id):
        """(n_s,) reward vector for this action, from pomdp.RewardModel."""
        details = self.env.actions[action_id]
        if details["action_type"] == "Triage":
            return self.env.reward_model.triage(action_id, CONDITIONS)
        if details["action_type"] == "Disassy":
            # Cost varies by source state (each state's own edge time); 0
            # (harmless no-op) wherever this action isn't actually defined.
            cost_per_x = np.zeros(self.space.n_x)
            for state_id, edges in self.env.disassy_by_state.items():
                edge = edges.get(action_id)
                if edge is not None:
                    x_idx = self.env.state_to_idx[state_id]
                    cost_per_x[x_idx] = self.env.reward_model.flat_cost(edge["time"])
            cost_matrix = np.tile(cost_per_x.reshape(-1, 1), (1, self.space.n_y))
            return cost_matrix.reshape(-1, order="F")  # y-major, x-minor - see JointStateSpace
        cost = self.env.reward_model.flat_cost(details["time"])  # Insp: same everywhere
        return np.full(self.space.n_s, cost)


def initial_belief(model):
    """b0(x, y): certainty on the root physical node, uniform over y."""
    b0 = np.zeros(model.space.n_s)
    x0 = model.env.state_to_idx[model.env.root_state]
    for y_idx in range(model.space.n_y):
        b0[model.space.index(x0, y_idx)] = 1.0 / model.space.n_y
    return b0


def most_likely_x(model, b):
    """Marginalizes y out of a joint belief and returns the most likely x."""
    marginal_x = b.reshape(model.space.n_y, model.space.n_x).sum(axis=0)
    return int(np.argmax(marginal_x))


def belief_update(model, b, action_id, o_idx, confidence=1.0):
    """b' = normalize(Z[:, o]^confidence .* (T^T b)) - tempered Bayes update.
    confidence=1 is standard Bayes; confidence=0 ignores the observation."""
    T = model.transition(action_id)
    predicted = (T.T @ b) if T is not None else b
    Z, _ = model.observation(action_id)
    if Z is None or confidence == 0:
        return predicted
    likelihood = np.asarray(Z[:, o_idx].power(confidence).todense()).flatten()
    weighted = likelihood * predicted
    total = weighted.sum()
    return weighted / total if total > 0 else weighted


def expand_beliefs(model, belief_sets, n_trajectories=20, horizon=10, epsilon=0.05, rng=None):
    """Phase 1: forward Monte Carlo simulation with a uniform-random policy
    and randomized observation confidence, growing belief_sets =
    {x_idx: [belief_vector, ...]} with L1-distance pruning."""
    rng = rng or np.random.default_rng()

    for _ in range(n_trajectories):
        b = initial_belief(model)
        x_idx = model.env.state_to_idx[model.env.root_state]

        for _ in range(horizon):
            action_id = rng.choice(list(model.valid_actions(x_idx)))
            if model.env.actions[action_id]["action_type"] == "Triage":
                break  # terminal: nothing further to expand

            T = model.transition(action_id)
            predicted = (T.T @ b) if T is not None else b
            Z, obs_labels = model.observation(action_id)
            obs_dist = np.asarray((predicted @ Z)).flatten()
            obs_dist = obs_dist / obs_dist.sum()
            o_idx = rng.choice(len(obs_labels), p=obs_dist)

            b = belief_update(model, b, action_id, o_idx, confidence=rng.uniform(0, 1))
            x_idx = most_likely_x(model, b)

            points = belief_sets.setdefault(x_idx, [])
            if all(np.abs(b - existing).sum() >= epsilon for existing in points):
                points.append(b)

    return belief_sets


def _all_alphas(gamma):
    for points in gamma.values():
        for alpha, action_id in points:
            yield alpha, action_id


def backup(model, belief_sets, gamma, discount=0.95):
    """Phase 2: one point-based value backup sweep. For each belief point,
    picks the action (and, per observation, the best previous alpha-vector)
    maximizing value at that point - the standard PBVI simplification of
    "generate candidates then prune to those that are best somewhere": each
    point keeps exactly the one alpha-vector that's best for it, tagged with
    its action, rather than a separately-pruned candidate pool.
    """
    new_gamma = {x_idx: [] for x_idx in belief_sets}

    for x_idx, points in belief_sets.items():
        for b in points:
            best_value, best_alpha, best_action = -np.inf, None, None

            for action_id in model.valid_actions(x_idx):
                reward = model.reward(action_id)
                details = model.env.actions[action_id]

                if details["action_type"] == "Triage":
                    alpha = reward
                else:
                    T = model.transition(action_id)
                    Z, obs_labels = model.observation(action_id)
                    alpha = reward.copy()
                    for o_idx in range(len(obs_labels)):
                        z_col = np.asarray(Z[:, o_idx].todense()).flatten()
                        best_o_value, best_o_vec = -np.inf, np.zeros(model.space.n_s)
                        for prev_alpha, prev_action in _all_alphas(gamma):
                            if prev_action is None:
                                continue
                            projected = T @ (z_col * prev_alpha)  # sparse cross-sum
                            value = float(b @ projected)
                            if value > best_o_value:
                                best_o_value, best_o_vec = value, projected
                        alpha = alpha + discount * best_o_vec

                value = float(b @ alpha)
                if value > best_value:
                    best_value, best_alpha, best_action = value, alpha, action_id

            new_gamma[x_idx].append((best_alpha, best_action))

    return new_gamma


def solve(model, n_iterations=5, n_trajectories=20, horizon=10, epsilon=0.05, discount=0.95, rng=None):
    """Alternates belief expansion and value backup for n_iterations sweeps."""
    belief_sets = {}
    root_x = model.env.state_to_idx[model.env.root_state]
    gamma = {root_x: [(np.zeros(model.space.n_s), None)]}

    for _ in range(n_iterations):
        belief_sets = expand_beliefs(model, belief_sets, n_trajectories, horizon, epsilon, rng)
        gamma = backup(model, belief_sets, gamma, discount)

    return gamma


def save_policy(gamma, path):
    with open(path, "wb") as f:
        pickle.dump(gamma, f)


def load_policy(path):
    with open(path, "rb") as f:
        return pickle.load(f)
