import pickle
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np

from src.interface import best_action, prompt_observation
from src.env import AngleGrinderEnv
from src.agent import Model


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        graph = nx.DiGraph()
        graph.add_node("root", parts_removed=[], parts_present=["screw"])
        graph.add_node("done", parts_removed=["screw"], parts_present=[])
        graph.add_edge(
            "root", "done", capability="ScrewingCapability", target_part="screw", action_time=2.0,
        )
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_dir)
        graph_path = Path(temp_dir) / "graph.pkl"
        with graph_path.open("wb") as handle:
            pickle.dump(graph, handle)

        self.env = AngleGrinderEnv(graph_path=graph_path)
        self.model = Model(self.env)

    def test_best_action_picks_the_highest_value_alpha(self):
        b = np.array([1.0, 0.0])
        gamma = {
            "some_node": [
                (np.array([1.0, 0.0]), "Recycle"),
                (np.array([5.0, 0.0]), "Reuse"),
                (np.zeros(2), None),  # placeholder entries are ignored
            ]
        }
        self.assertEqual(best_action(gamma, self.env, b), "Reuse")

    def test_best_action_skips_actions_not_currently_valid(self):
        self.env.available_insp_actions = {"Inspect"}  # Verify just used
        b = np.array([1.0, 0.0])
        gamma = {
            "some_node": [
                (np.array([5.0, 0.0]), "Verify"),   # highest value, but masked out
                (np.array([1.0, 0.0]), "Inspect"),  # lower value, but still valid
            ]
        }
        self.assertEqual(best_action(gamma, self.env, b), "Inspect")

    def test_prompt_observation_bypasses_prompt_for_disassy_actions(self):
        with patch("builtins.input") as mock_input:
            o_idx, confidence = prompt_observation(self.model, "Unscrew")
        mock_input.assert_not_called()
        self.assertEqual((o_idx, confidence), (0, 1.0))

    def test_prompt_observation_rejects_invalid_input_then_accepts_valid(self):
        with patch("builtins.input", side_effect=["nonsense", "BAD", "0.8"]):
            o_idx, confidence = prompt_observation(self.model, "Inspect")
        self.assertEqual(o_idx, ["GOOD", "OK", "BAD"].index("BAD"))
        self.assertEqual(confidence, 0.8)

    def test_prompt_observation_only_accepts_verify_states_as_labels(self):
        with patch("builtins.input", side_effect=["done", "1.0"]):
            o_idx, confidence = prompt_observation(self.model, "Verify")
        self.assertEqual(self.env.state_list[o_idx], "done")
        self.assertEqual(confidence, 1.0)

    def test_prompt_observation_rejects_out_of_range_confidence(self):
        with patch("builtins.input", side_effect=["done", "1.5", "0.5"]):
            _, confidence = prompt_observation(self.model, "Verify")
        self.assertEqual(confidence, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
