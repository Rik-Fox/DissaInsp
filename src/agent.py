"""Point-Based Value Iteration (PBVI) solver for the disassembly/triage
POMDP, built on the sparse T/Z/R matrices from a loaded AngleGrinderEnv.
Belief points are (belief_vector, available_insp_actions) pairs - see
README for the model/algorithm description. Draft/reference
implementation, not tuned/optimized.
"""
import pickle

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from .env import CONDITION_OBS, CONDITIONS, INSPECTION_ACTIONS
from .pomdp import TRIAGE_ACTIONS


class Model:
    """Adapts an AngleGrinderEnv's per-action T/Z/R into the sparse
    matrices/vectors the PBVI solver needs, reusing env.py's builders.
    Caches per action_id, since none of them depend on x_idx or a belief
    point.
    """

    def __init__(self, env):
        self.env = env
        self.space = env.joint_space
        self._transition_cache = {}
        self._observation_cache = {}
        self._reward_cache = {}
        # Verify/Inspect currently available - mirrors AngleGrinderEnv's own field.
        self.available_insp_actions = set(INSPECTION_ACTIONS)

    def valid_actions(self, x_idx):
        """Disassy actions at x_idx, union available insp actions, union Triage."""
        state_id = self.env.state_list[x_idx]
        disassy = set(self.env.disassy_by_state[state_id].keys())
        return disassy | self.available_insp_actions | set(TRIAGE_ACTIONS)

    def transition(self, action_id):
        """Sparse (n_s, n_s) T matrix, or None for Triage (terminal - no
        continuation value)."""
        if action_id not in self._transition_cache:
            self._transition_cache[action_id] = self._build_transition(action_id)
        return self._transition_cache[action_id]

    def _build_transition(self, action_id):
        details = self.env.actions[action_id]
        if details["action_type"] == "Triage":
            return None
        if details["action_type"] == "Disassy":
            return self.env._disassy_transition_matrix(action_id)
        return self.env.transition_model.identity()  # Insp: x', y' = x, y

    def observation(self, action_id):
        """(sparse (n_s, n_o) Z matrix, observation labels), or (None, None)
        for Triage (terminal - no observation)."""
        if action_id not in self._observation_cache:
            self._observation_cache[action_id] = self._build_observation(action_id)
        return self._observation_cache[action_id]

    def _build_observation(self, action_id):
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
        if action_id not in self._reward_cache:
            self._reward_cache[action_id] = self._build_reward(action_id)
        return self._reward_cache[action_id]

    def _build_reward(self, action_id):
        details = self.env.actions[action_id]
        if details["action_type"] == "Triage":
            return self.env.reward_model.triage(action_id, CONDITIONS)
        if details["action_type"] == "Disassy":
            # Cost varies by source state; 0 where this action isn't defined there.
            cost_per_x = np.zeros(self.space.n_x)
            for state_id, edges in self.env.disassy_by_state.items():
                edge = edges.get(action_id)
                if edge is not None:
                    x_idx = self.env.state_to_idx[state_id]
                    cost_per_x[x_idx] = self.env.reward_model.flat_cost(edge["time"])
            cost_matrix = np.tile(cost_per_x.reshape(-1, 1), (1, self.space.n_y))
            return cost_matrix.reshape(-1, order="F")  # y-major, x-minor
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
    """Phase 1: random-policy Monte Carlo rollouts, growing belief_sets =
    {x_idx: [(belief_vector, available_insp_actions), ...]} with L1-distance
    pruning (points differing only in available_insp_actions are kept
    separate)."""
    rng = rng or np.random.default_rng()

    for _ in range(n_trajectories):
        b = initial_belief(model)
        x_idx = model.env.state_to_idx[model.env.root_state]
        model.available_insp_actions = set(INSPECTION_ACTIONS)  # fresh episode

        for _ in range(horizon):
            # sorted(), not list(): set order depends on (randomized) string hashing.
            action_id = rng.choice(sorted(model.valid_actions(x_idx)))
            details = model.env.actions[action_id]
            if details["action_type"] == "Triage":
                break  # terminal: nothing further to expand

            T = model.transition(action_id)
            predicted = (T.T @ b) if T is not None else b
            Z, obs_labels = model.observation(action_id)
            obs_dist = np.asarray((predicted @ Z)).flatten()
            obs_dist = obs_dist / obs_dist.sum()
            o_idx = rng.choice(len(obs_labels), p=obs_dist)

            b = belief_update(model, b, action_id, o_idx, confidence=rng.uniform(0, 1))
            x_idx = most_likely_x(model, b)

            if details["action_type"] == "Disassy":
                model.available_insp_actions = set(INSPECTION_ACTIONS)  # refreshed either way
            else:  # Insp: unavailable until the next Disassy attempt
                model.available_insp_actions.discard(action_id)

            # Snapshot: model.available_insp_actions keeps changing after this.
            insp_snapshot = frozenset(model.available_insp_actions)
            points = belief_sets.setdefault(x_idx, [])
            is_new = all(
                existing_insp != insp_snapshot or np.abs(b - existing_b).sum() >= epsilon
                for existing_b, existing_insp in points
            )
            if is_new:
                points.append((b, insp_snapshot))

    return belief_sets


def _all_alphas(gamma):
    for points in gamma.values():
        for alpha, action_id in points:
            yield alpha, action_id


def _stack_alphas(gamma, n_s):
    """Every alpha-vector with a real action, stacked into one (n_alphas,
    n_s) matrix - built once per backup() sweep rather than per point."""
    alphas = [alpha for alpha, action_id in _all_alphas(gamma) if action_id is not None]
    return np.array(alphas) if alphas else np.empty((0, n_s))


def _observation_cross_sum(prev_alphas, T, Z, b, discount):
    """Vectorized PBVI cross-sum: for each observation, picks whichever
    prior alpha-vector is best for the belief it would produce, then
    projects the result through T once. See README performance note."""
    if prev_alphas.shape[0] == 0:
        return 0.0

    predicted = T.T @ b
    weighted_alphas = prev_alphas * predicted  # (n_alphas, n_s), row-broadcast
    values = weighted_alphas @ Z  # (n_alphas, n_o) - one sparse matmul
    best_k = np.argmax(values, axis=0)  # best prior alpha per observation

    Z_csc = Z.tocsc()
    correction = np.zeros(prev_alphas.shape[1])
    for o_idx in range(Z_csc.shape[1]):
        start, end = Z_csc.indptr[o_idx], Z_csc.indptr[o_idx + 1]
        if start == end:
            continue
        rows = Z_csc.indices[start:end]
        alpha_row = prev_alphas[best_k[o_idx]]
        correction[rows] += Z_csc.data[start:end] * alpha_row[rows]

    return discount * (T @ correction)


def backup(model, belief_sets, gamma, discount=0.95):
    """Phase 2: one PBVI backup sweep - for each belief point, picks the
    action (and best previous alpha-vector per observation) maximizing
    value there, keeping just that one alpha-vector per point."""
    new_gamma = {x_idx: [] for x_idx in belief_sets}
    prev_alphas = _stack_alphas(gamma, model.space.n_s)

    for x_idx, points in belief_sets.items():
        for b, insp_actions in points:
            model.available_insp_actions = insp_actions  # this point's own mask
            best_value, best_alpha, best_action = -np.inf, None, None

            for action_id in model.valid_actions(x_idx):
                reward = model.reward(action_id)
                details = model.env.actions[action_id]

                if details["action_type"] == "Triage":
                    alpha = reward
                else:
                    T = model.transition(action_id)
                    Z, _ = model.observation(action_id)
                    alpha = reward + _observation_cross_sum(prev_alphas, T, Z, b, discount)

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

    for i in tqdm(range(n_iterations), desc="Solving", unit="iter"):
        belief_sets = expand_beliefs(model, belief_sets, n_trajectories, horizon, epsilon, rng)
        tqdm.write(f"--- Iteration {i + 1}/{n_iterations}: {sum(len(v) for v in belief_sets.values())} belief points ---")

        gamma = backup(model, belief_sets, gamma, discount)
        tqdm.write(f"--- Iteration {i + 1}/{n_iterations}: {sum(len(v) for v in gamma.values())} alpha-vectors ---")

    return gamma


def save_policy(gamma, path):
    with open(path, "wb") as f:
        pickle.dump(gamma, f)


def load_policy(path):
    with open(path, "rb") as f:
        return pickle.load(f)
