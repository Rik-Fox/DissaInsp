import json
import pickle
import re

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces


class AngleGrinderEnv(gym.Env):
    action_space: spaces.Discrete
    observation_space: spaces.MultiBinary

    """
    A custom Gymnasium environment for learning a disassembly sequence.

    The environment is based on a pre-computed state graph where:
    - States represent configurations of the assembly (parts removed/present).
    - Actions are the disassembly operations (e.g., unscrew, manipulate).
    - The environment provides an action mask at each step to indicate valid actions.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, graph_path, render_mode=None):
        super().__init__()

        graph_path = str(graph_path)
        print(f"Loading graph data from: {graph_path}")
        try:
            with open(graph_path, "rb") as f:
                loaded_graph = pickle.load(f)
        except (pickle.UnpicklingError, TypeError):
            print("Could not load as pickle. Trying to parse as JS object...")
            self.graph_data = self._load_graph_from_js_object(graph_path)
            self._load_from_dict_format(self.graph_data)
            self.current_state_id = self.root_state
            self.render_mode = render_mode
            return

        if isinstance(loaded_graph, dict):
            self.graph_data = loaded_graph
            self._load_from_dict_format(self.graph_data)
        elif isinstance(loaded_graph, nx.Graph) or isinstance(loaded_graph, nx.DiGraph) or isinstance(loaded_graph, nx.MultiDiGraph):
            self.graph_data = self._load_from_networkx_graph(loaded_graph)
        else:
            raise TypeError(f"Unsupported graph data type: {type(loaded_graph)!r}")

        self.current_state_id = self.root_state
        self.render_mode = render_mode
        self.state_histories = {}

    def _load_from_dict_format(self, graph_data):
        self.states = graph_data["states"]
        self.actions = graph_data["actions"]
        self.parts = graph_data["parts"]
        self.root_state = graph_data["root"]
        self.transitions = {}
        self.state_histories = {}

        for state_id, state_info in self.states.items():
            children = []
            for child_id in state_info.get("children", []):
                action_id = self.states[child_id].get("action_id")
                if action_id:
                    children.append({"action_id": action_id, "next_state_id": child_id})
            self.transitions[state_id] = children

        self.part_list = sorted(self.parts.keys())
        self.part_to_idx = {name: i for i, name in enumerate(self.part_list)}

        self.action_list = sorted(self.actions.keys())
        self.action_to_idx = {name: i for i, name in enumerate(self.action_list)}
        self.idx_to_action = {i: name for i, name in enumerate(self.action_list)}

        self.action_space = spaces.Discrete(len(self.action_list))
        self.observation_space = spaces.MultiBinary(len(self.part_list))

    def _load_from_networkx_graph(self, graph):
        self.states = {}
        self.transitions = {}
        self.actions = {}
        self.parts = {}
        self.state_histories = {}

        for node_id, node_attrs in graph.nodes(data=True):
            self.states[node_id] = {
                "parts_removed": list(node_attrs.get("parts_removed", [])),
                "parts_present": list(node_attrs.get("parts_present", [])),
                "is_goal": False,
                "is_dead_end": False,
                "is_truncated": False,
                "action_history": [],
            }
            self.transitions[node_id] = []

            for part_name in node_attrs.get("parts_present", []):
                self.parts[part_name] = {}
            for part_name in node_attrs.get("parts_removed", []):
                self.parts[part_name] = {}

        edge_items = []
        if hasattr(graph, "is_multigraph") and graph.is_multigraph():
            edge_items = list(graph.edges(keys=True, data=True))
        else:
            edge_items = list(graph.edges(data=True))

        for edge in edge_items:
            if len(edge) == 3:
                edge_source, edge_target, edge_data = edge
            else:
                edge_source, edge_target, edge_key, edge_data = edge

            action_id = edge_data.get("label") or edge_data.get("target_part") or f"action_{len(self.actions)}"
            if action_id in self.actions:
                action_id = f"{action_id}_{len(self.actions)}"

            action_type = (
                edge_data.get("capability")
                or edge_data.get("type")
                or edge_data.get("kind")
                or edge_data.get("action_type")
                or "disassemble"
            )
            action_result = (
                edge_data.get("inspection_result")
                or edge_data.get("outcome")
                or edge_data.get("result")
            )
            self.actions[action_id] = {
                "type": action_type,
                "part": edge_data.get("target_part"),
                "time": edge_data.get("action_time") or edge_data.get("time") or 1.0,
                "tool": edge_data.get("tool"),
                "result": action_result,
                "is_inspection": str(action_type).lower() in {"inspect", "inspection"},
            }
            self.transitions[edge_source].append(
                {"action_id": action_id, "next_state_id": edge_target}
            )

        self.root_state = next(iter(graph.nodes))
        for state_id, state_info in self.states.items():
            if not self.transitions[state_id]:
                state_info["is_dead_end"] = True

        self.part_list = sorted(self.parts.keys())
        self.part_to_idx = {name: i for i, name in enumerate(self.part_list)}

        self.action_list = sorted(self.actions.keys())
        self.action_to_idx = {name: i for i, name in enumerate(self.action_list)}
        self.idx_to_action = {i: name for i, name in enumerate(self.action_list)}

        self.action_space = spaces.Discrete(len(self.action_list))
        self.observation_space = spaces.MultiBinary(len(self.part_list))

        return {
            "states": self.states,
            "actions": self.actions,
            "parts": self.parts,
            "root": self.root_state,
        }

    def _load_graph_from_js_object(self, file_path):
        """A helper to parse the JS object from the HTML file if graph.pkl is not a pickle file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        match = re.search(r"const DATA_PRELOADED = (\{.*?\});", content, re.DOTALL)
        if not match:
            # Fallback for just the object in the file
            match = re.search(r"(\{.*?\});?", content, re.DOTALL)

        if not match:
            raise ValueError("Could not find graph data object in the file.")

        js_object_str = match.group(1)

        # Make it valid JSON by quoting keys
        json_str = re.sub(
            r"([\{\s,])([a-zA-Z_][a-zA-Z0-9_]*):", r'\1"\2":', js_object_str
        )

        return json.loads(json_str)

    def _get_obs(self):
        """Creates the binary observation vector for the current state."""
        removed_parts = self.states[self.current_state_id].get("parts_removed", [])
        obs = np.zeros(len(self.part_list), dtype=np.int8)
        for part_name in removed_parts:
            if part_name in self.part_to_idx:
                obs[self.part_to_idx[part_name]] = 1
        return obs

    def _get_info(self):
        """Returns info dict, including the action mask and execution history for the current state."""
        state_info = self.states[self.current_state_id]
        history = list(self.state_histories.get(self.current_state_id, []))
        cumulative_time = sum(entry.get("time", 0.0) for entry in history)
        return {
            "state_id": self.current_state_id,
            "is_goal": state_info.get("is_goal", False),
            "is_dead_end": state_info.get("is_dead_end", False),
            "is_truncated_state": state_info.get("is_truncated", False),
            "action_mask": self._get_valid_actions_mask(),
            "state_action_history": history,
            "cumulative_time": cumulative_time,
        }

    def _get_valid_actions(self):
        """Gets the set of valid action IDs from the current state."""
        valid_actions = set()
        for transition in self.transitions[self.current_state_id]:
            if transition.get("action_id"):
                valid_actions.add(transition["action_id"])
        return valid_actions

    def _get_valid_actions_mask(self):
        """Creates a binary mask for valid actions."""
        valid_actions = self._get_valid_actions()
        mask = np.zeros(len(self.action_list), dtype=np.int8)
        for action_id in valid_actions:
            if action_id in self.action_to_idx:
                mask[self.action_to_idx[action_id]] = 1
        return mask

    def _normalize_action_details(self, action_id):
        action_details = dict(self.actions[action_id])
        action_type = action_details.get("type") or "disassemble"
        normalized_type = str(action_type).lower()
        action_details["type"] = normalized_type
        action_details["is_inspection"] = normalized_type in {"inspect", "inspection"}
        if action_details.get("result") is None:
            action_details["result"] = "ok"
        return action_details

    def _update_state_history(self, state_id, action_id, action_details, previous_state_id):
        history_entry = {
            "action_id": action_id,
            "action_type": action_details.get("type", "disassemble"),
            "part": action_details.get("part"),
            "time": float(action_details.get("time", 1.0)),
            "result": action_details.get("result"),
            "state_before": previous_state_id,
            "state_after": state_id,
        }
        history = list(self.state_histories.get(previous_state_id, []))
        history.append(history_entry)
        self.state_histories[state_id] = history
        self.states[state_id]["action_history"] = history
        return history_entry

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state_id = self.root_state
        self.state_histories = {state_id: [] for state_id in self.states}
        self.state_histories[self.root_state] = []
        for state_id in self.states:
            self.states[state_id]["action_history"] = []

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        action_id = self.idx_to_action.get(action)
        valid_actions = self._get_valid_actions()

        terminated = False
        truncated = False
        reward = 0

        if action_id in valid_actions:
            previous_state_id = self.current_state_id
            next_state_id = None
            for transition in self.transitions[self.current_state_id]:
                if transition.get("action_id") == action_id:
                    next_state_id = transition.get("next_state_id")
                    break

            action_details = self._normalize_action_details(action_id)
            if action_details.get("is_inspection"):
                next_state_id = previous_state_id
                self.states[previous_state_id]["last_action_result"] = action_details.get("result")
                self.states[previous_state_id]["last_action_id"] = action_id
                reward = -float(action_details.get("time", 1.0))
                if action_details.get("result") == "good":
                    reward += 1.0
                elif action_details.get("result") == "bad":
                    reward -= 5.0
            else:
                self.current_state_id = next_state_id
                self.states[next_state_id]["last_action_result"] = action_details.get("result")
                self.states[next_state_id]["last_action_id"] = action_id

                # Reward: negative of time taken to encourage efficiency
                reward = -float(action_details.get("time", 1.0))

                current_state_details = self.states[self.current_state_id]
                if current_state_details.get("is_goal", False):
                    reward += 1000  # Large positive reward for reaching the goal
                    terminated = True
                elif current_state_details.get("is_dead_end", False):
                    reward -= 500  # Large negative reward for a non-goal dead end
                    terminated = True
                elif current_state_details.get("is_truncated", False):
                    # This state was not fully explored, might not be a true dead end.
                    # We can treat it as episode truncation.
                    truncated = True

            self._update_state_history(next_state_id, action_id, action_details, previous_state_id)
            self.current_state_id = next_state_id
        else:
            # Agent took an invalid action
            reward = -100
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render(action_id, reward)

        return observation, reward, terminated, truncated, info

    def print_actions_and_components(self):
        print("Angle grinder actions:")
        for action_id, action_data in sorted(self.actions.items()):
            part = action_data.get("part") or "n/a"
            action_type = action_data.get("type", "disassemble")
            action_time = action_data.get("time", 1.0)
            tool = action_data.get("tool") or "n/a"
            print(
                f"  - {action_id}: type={action_type}, part={part}, time={action_time}, tool={tool}"
            )

        print("Angle grinder components:")
        for part_name in self.part_list:
            print(f"  - {part_name}")

    def render(self, last_action=None, last_reward=None):
        state_info = self.states[self.current_state_id]
        print("-" * 20)
        if last_action:
            action_info = self.actions[last_action]
            print(
                f"Action: {action_info['type']} {action_info['part']} | Reward: {last_reward:.2f}"
            )
        print(f"State: {self.current_state_id}")
        print(f"  Removed: {state_info.get('parts_removed', [])}")
        print(
            f"  Goal: {state_info.get('is_goal', False)}, Dead End: {state_info.get('is_dead_end', False)}"
        )

    def close(self):
        pass
