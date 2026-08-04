import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
import json
import re


class DisassemblyEnv(gym.Env):
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

        print(f"Loading graph data from: {graph_path}")
        try:
            with open(graph_path, "rb") as f:
                self.graph_data = pickle.load(f)
        except (pickle.UnpicklingError, TypeError):
            print("Could not load as pickle. Trying to parse as JS object...")
            self.graph_data = self._load_graph_from_js_object(graph_path)

        self.states = self.graph_data["states"]
        self.actions = self.graph_data["actions"]
        self.parts = self.graph_data["parts"]
        self.root_state = self.graph_data["root"]

        self.part_list = sorted(self.parts.keys())
        self.part_to_idx = {name: i for i, name in enumerate(self.part_list)}

        self.action_list = sorted(self.actions.keys())
        self.action_to_idx = {name: i for i, name in enumerate(self.action_list)}
        self.idx_to_action = {i: name for i, name in enumerate(self.action_list)}

        # Action space: Discrete space with size equal to the total number of unique actions.
        self.action_space = spaces.Discrete(len(self.action_list))

        # Observation space: A binary vector indicating which parts are removed.
        self.observation_space = spaces.MultiBinary(len(self.part_list))

        self.current_state_id = self.root_state
        self.render_mode = render_mode

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
        """Returns info dict, including the action mask for the current state."""
        state_info = self.states[self.current_state_id]
        return {
            "state_id": self.current_state_id,
            "is_goal": state_info.get("is_goal", False),
            "is_dead_end": state_info.get("is_dead_end", False),
            "is_truncated_state": state_info.get("is_truncated", False),
            "action_mask": self._get_valid_actions_mask(),
        }

    def _get_valid_actions(self):
        """Gets the set of valid action IDs from the current state."""
        children_ids = self.states[self.current_state_id].get("children", [])
        valid_actions = set()
        for child_id in children_ids:
            action_id = self.states[child_id].get("action_id")
            if action_id:
                valid_actions.add(action_id)
        return valid_actions

    def _get_valid_actions_mask(self):
        """Creates a binary mask for valid actions."""
        valid_actions = self._get_valid_actions()
        mask = np.zeros(len(self.action_list), dtype=np.int8)
        for action_id in valid_actions:
            if action_id in self.action_to_idx:
                mask[self.action_to_idx[action_id]] = 1
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state_id = self.root_state

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
            # Find the child state corresponding to this action
            next_state_id = None
            children_ids = self.states[self.current_state_id].get("children", [])
            for child_id in children_ids:
                if self.states[child_id].get("action_id") == action_id:
                    next_state_id = child_id
                    break

            self.current_state_id = next_state_id
            action_details = self.actions[action_id]

            # Reward: negative of time taken to encourage efficiency
            reward = -action_details.get("time", 1.0)

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
        else:
            # Agent took an invalid action
            reward = -100
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render(action_id, reward)

        return observation, reward, terminated, truncated, info

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
