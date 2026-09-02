import pickle

import gymnasium as gym
import networkx as nx
import numpy as np
import scipy.sparse as sp
from gymnasium import spaces

from .configs import CONFIGS
from .pomdp import JointStateSpace, ObservationModel, RewardModel, TransitionModel, TRIAGE_ACTIONS

# Hidden condition state y.
CONDITIONS = ["Pristine", "Serviceable", "Degraded"]
CONDITION_OBS = ["GOOD", "OK", "BAD"]

# Raw graph "capability" edge attribute -> type. Only Disassy actions come
# from the graph; Verify/Inspect/Triage are added globally below.
CAPABILITY_TO_TYPE = {
    "ScrewingCapability": "Unscrew",
    "ManipulatingCapability": "Remove",
}

# Verify/Inspect: masked out once used until the next Disassy attempt.
INSPECTION_ACTIONS = {"Verify", "Inspect"}


class AngleGrinderEnv(gym.Env):
    """
    A custom Gymnasium environment for the disassembly/triage POMDP.

    States represent physical assembly configurations (parts removed/present).
    Actions are partitioned into "Disassy" (Unscrew/Remove), "Insp"
    (Verify/Inspect), and "Triage" (terminal) actions. Verify and 
    Inspect are each usable once per disassembly cycle.
    """

    action_space: spaces.Discrete
    observation_space: spaces.MultiBinary
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, graph_path, render_mode=None, config="no_inspection"):
        super().__init__()
        self.config = CONFIGS[config] if isinstance(config, str) else config

        graph_path = str(graph_path)
        print(f"Loading graph data from: {graph_path}")
        try:
            with open(graph_path, "rb") as f:
                loaded_graph = pickle.load(f)
        except (pickle.UnpicklingError, TypeError):
            print("Could not load as pickle.")
            raise SystemExit(1)

        if not isinstance(loaded_graph, nx.Graph):
            raise TypeError(f"Unsupported graph data type: {type(loaded_graph)!r}")
        self._load_from_networkx_graph(loaded_graph)

        self.current_state_id = self.root_state
        self.condition = None
        self.available_insp_actions = set(INSPECTION_ACTIONS)
        self.render_mode = render_mode
        # Lazily built & cached: action_id -> sparse (n_s, n_s) T matrix.
        self._disassy_transitions = {}

    def _load_from_networkx_graph(self, graph):
        self.states = {}
        self.parts = {}

        for node_id, node_attrs in graph.nodes(data=True):
            self.states[node_id] = {
                "parts_removed": list(node_attrs.get("parts_removed", [])),
                "parts_present": list(node_attrs.get("parts_present", [])),
            }
            for part_name in [*node_attrs.get("parts_present", []), *node_attrs.get("parts_removed", [])]:
                self.parts[part_name] = {}

        # Canonical edge per Disassy type per state: keep the
        # lexicographically smallest successor (see README for the tie-break).
        self.disassy_by_state = {state_id: {} for state_id in self.states}
        disassy_types = set()
        edge_items = graph.edges(keys=True, data=True) if graph.is_multigraph() else graph.edges(data=True)
        for edge_source, edge_target, *_, edge_data in edge_items:
            type_ = CAPABILITY_TO_TYPE.get(edge_data.get("capability"), edge_data.get("capability"))
            disassy_types.add(type_)
            candidate = {
                "next_state": edge_target,
                "time": edge_data.get("action_time") or edge_data.get("time") or 1.0,
                "part": edge_data.get("target_part"),
            }
            existing = self.disassy_by_state[edge_source].get(type_)
            if existing is None or edge_target < existing["next_state"]:
                self.disassy_by_state[edge_source][type_] = candidate

        self.root_state = next(iter(graph.nodes))

        self.actions = {
            type_: {"type": type_, "action_type": "Disassy", "part": None, "time": None, "tool": None}
            for type_ in disassy_types
        }
        self.actions["Verify"] = {
            "type": "Verify", "action_type": "Insp", "part": None,
            "time": self.config.verify_cost, "tool": None,
        }
        self.actions["Inspect"] = {
            "type": "Inspect", "action_type": "Insp", "part": None,
            "time": self.config.inspect_cost, "tool": None,
        }
        for action_id in TRIAGE_ACTIONS:
            self.actions[action_id] = {
                "type": action_id, "action_type": "Triage", "part": None,
                "time": 0.0, "tool": None,
            }

        self.part_list = sorted(self.parts.keys())
        self.part_to_idx = {name: i for i, name in enumerate(self.part_list)}

        self.state_list = sorted(self.states.keys())
        self.state_to_idx = {name: i for i, name in enumerate(self.state_list)}

        self.action_list = sorted(self.actions.keys())
        self.action_to_idx = {name: i for i, name in enumerate(self.action_list)}
        self.idx_to_action = {i: name for i, name in enumerate(self.action_list)}

        self.action_space = spaces.Discrete(len(self.action_list))
        self.observation_space = spaces.MultiBinary(len(self.part_list))

        self.joint_space = JointStateSpace(n_x=len(self.state_list), n_y=len(CONDITIONS))
        self.transition_model = TransitionModel(self.joint_space)
        self.reward_model = RewardModel(self.joint_space, triage_payoff=self.config.triage_payoff)
        self.observation_model = ObservationModel(self.joint_space)
        confusion = sp.csr_matrix(np.asarray(self.config.condition_obs_matrix))
        self._condition_obs_matrix = self.observation_model.condition(confusion)

    def _get_obs(self):
        """Binary vector of parts removed in the current physical state x."""
        obs = np.zeros(len(self.part_list), dtype=np.int8)
        for part_name in self.states[self.current_state_id]["parts_removed"]:
            obs[self.part_to_idx[part_name]] = 1
        return obs

    def _get_info(self, condition_observation=None):
        """Info dict with the current state id, action mask, and (if
        applicable) the condition observation from an Inspect action."""
        return {
            "state_id": self.current_state_id,
            "valid_action": self._get_valid_actions(),
            "condition_observation": condition_observation,
        }

    def _get_valid_actions(self):
        """Union of disassy actions at this node and Triage and
        whichever of Verify/Inspect are available."""
        disassy_actions = set(self.disassy_by_state[self.current_state_id].keys())
        return disassy_actions | self.available_insp_actions | set(TRIAGE_ACTIONS)

    def _normalize_action_details(self, action_id):
        """Copy of the action dict with 'type' lowercased for branching."""
        action_details = dict(self.actions[action_id])
        action_details["type"] = str(action_details["type"]).lower()
        return action_details

    def _disassy_transition_matrix(self, action_id):
        """Sparse (n_s, n_s) T matrix for a Disassy action, built once and
        cached across every state that defines it."""
        cached = self._disassy_transitions.get(action_id)
        if cached is not None:
            return cached

        n_x = self.joint_space.n_x
        x_succ = np.arange(n_x)
        p_success = np.zeros(n_x)  # default: action not defined here (masked out anyway)
        for state_id, edges in self.disassy_by_state.items():
            edge = edges.get(action_id)
            if edge is not None:
                x_idx = self.state_to_idx[state_id]
                x_succ[x_idx] = self.state_to_idx[edge["next_state"]]
                p_success[x_idx] = self.config.disassy_success_prob

        matrix = self.transition_model.disassembly(x_succ, p_success)
        self._disassy_transitions[action_id] = matrix
        return matrix

    def _sample_next_state(self, action_id):
        """Samples x' for a Disassy action from its (cached) T matrix."""
        matrix = self._disassy_transition_matrix(action_id)
        x_idx = self.state_to_idx[self.current_state_id]
        y_idx = CONDITIONS.index(self.condition)
        row = matrix.getrow(self.joint_space.index(x_idx, y_idx))
        next_s_idx = int(self.np_random.choice(row.indices, p=row.data))
        return self.state_list[next_s_idx % self.joint_space.n_x]

    def _sample_condition_observation(self):
        """Samples a GOOD/OK/BAD observation from P(o | y) via
        pomdp.ObservationModel, rather than a deterministic map of y."""
        x_idx = self.state_to_idx[self.current_state_id]
        y_idx = CONDITIONS.index(self.condition)
        row = self._condition_obs_matrix.getrow(self.joint_space.index(x_idx, y_idx))
        o_idx = int(self.np_random.choice(row.indices, p=row.data))
        return CONDITION_OBS[o_idx]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.condition = options.get("condition") or str(self.np_random.choice(CONDITIONS))
        self.current_state_id = self.root_state
        self.available_insp_actions = set(INSPECTION_ACTIONS)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        action_id = self.idx_to_action.get(action)

        if action_id not in self._get_valid_actions():
            raise ValueError(
                f"Invalid action {action_id!r} at state {self.current_state_id!r} - "
                f"valid actions are {sorted(self._get_valid_actions())}"
            )

        action_details = self._normalize_action_details(action_id)
        category = action_details["action_type"]
        terminated = False
        truncated = False
        condition_observation = None
        # reveals nothing about x only updated on verify action
        observation = np.zeros(len(self.part_list), dtype=np.int8)  

        if category == "Triage":
            # Only source of positive reward; invalid condition/action combos read 0.
            x_idx = self.state_to_idx[self.current_state_id]
            y_idx = CONDITIONS.index(self.condition)
            payoff = self.reward_model.triage(action_id, CONDITIONS)
            reward = float(payoff[self.joint_space.index(x_idx, y_idx)])
            observation = np.zeros(len(self.part_list), dtype=np.int8)
            terminated = True
        elif category == "Insp":
            # Verify/Inspect never change the physical state x.
            reward = self.reward_model.flat_cost(action_details["time"])
            if action_details["type"] == "verify":
                observation = self._get_obs()  # reveal the physical state x
            else:  # "inspect"
                condition_observation = self._sample_condition_observation()
            self.available_insp_actions.discard(action_id)  # unavailable until next Disassy attempt
        else:
            # Disassembly action: updates x but reveals no observation.
            edge = self.disassy_by_state[self.current_state_id][action_id]
            reward = self.reward_model.flat_cost(edge["time"])
            self.current_state_id = self._sample_next_state(action_id)
            observation = np.zeros(len(self.part_list), dtype=np.int8)
            self.available_insp_actions = set(INSPECTION_ACTIONS)  # refreshed either way

        info = self._get_info(condition_observation)

        if self.render_mode == "human":
            self.render(action_id, reward)

        return observation, reward, terminated, truncated, info

    def print_actions_and_components(self):
        print("Angle grinder actions:")
        for action_id, action_data in sorted(self.actions.items()):
            time_display = action_data["time"] if action_data["time"] is not None else "varies by state"
            print(
                f"  - {action_id}: type={action_data['type']}, action_type={action_data['action_type']}, "
                f"time={time_display}"
            )

        print("Angle grinder components:")
        for part_name in self.part_list:
            print(f"  - {part_name}")

    def render(self, last_action=None, last_reward=None):
        state_info = self.states[self.current_state_id]
        print("-" * 20)
        if last_action is not None:
            print(f"Action: {last_action} | Reward: {last_reward:.2f}")
        print(f"State: {self.current_state_id}")
        print(f"  Removed: {state_info['parts_removed']}")

    def close(self):
        pass
