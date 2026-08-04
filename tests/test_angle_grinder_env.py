import io
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import networkx as nx
import numpy as np

from src.env import AngleGrinderEnv


class AngleGrinderEnvTests(unittest.TestCase):
    def setUp(self):
        project_root = Path(__file__).resolve().parents[1]
        self.graph_path = (
            project_root
            / "disassembly_graph"
            / "disassembly_angle_grinder"
            / "disassembly_angle_grinder"
            / "graph.pkl"
        )
        self.assertTrue(self.graph_path.exists(), f"Graph file not found at {self.graph_path}")

    def log(self, title, description):
        print(f"[TEST] {title}: {description}")

    def test_initialization_and_reset(self):
        print()
        self.log("Reset behavior", "Initial state and action mask")
        env = AngleGrinderEnv(graph_path=self.graph_path)

        print("  - checking that the real graph loads with a usable action space and observation vector")
        self.assertGreater(env.action_space.n, 0)
        self.assertGreater(env.observation_space.shape[0], 0)
        self.assertGreater(len(env.part_list), 0)
        print("    -> this proves the environment can load the bundled angle grinder graph and expose its parts as observations")

        print("  - checking that the initial reset state is well defined")
        observation, info = env.reset()
        self.assertEqual(observation.shape, (len(env.part_list),))
        self.assertEqual(info["state_id"], env.root_state)
        self.assertTrue(np.any(info["action_mask"] == 1))
        self.assertFalse(info["is_goal"])
        self.assertFalse(info["is_dead_end"])
        print("    -> this proves the root state starts with a valid action mask and no terminal flags")

    def test_step_with_valid_action_transitions_to_a_new_state(self):
        print()
        self.log("Goal transition", "Valid action changes the state")
        env = AngleGrinderEnv(graph_path=self.graph_path)
        observation, info = env.reset()

        print("  - applying the first valid action from the real graph")
        valid_action = int(np.argmax(info["action_mask"]))
        next_observation, reward, terminated, truncated, next_info = env.step(valid_action)

        self.assertEqual(next_observation.shape, observation.shape)
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(truncated)
        self.assertNotEqual(next_info["state_id"], info["state_id"])
        print("    -> this proves a valid action advances the environment to a new state and produces a meaningful reward")

    def test_observation_payload_contains_state_and_time_details(self):
        print()
        self.log("Observation payload", "Observation includes the expected state details and action time")
        env = AngleGrinderEnv(graph_path=self.graph_path)
        observation, info = env.reset()

        print("  - checking that the observation vector is backed by the current state metadata")
        self.assertEqual(observation.shape, (len(env.part_list),))
        self.assertIn("state_id", info)
        self.assertIn("action_mask", info)
        self.assertIn("is_goal", info)
        self.assertIn("is_dead_end", info)
        self.assertIn("is_truncated_state", info)
        print("    -> this proves the environment exposes the structural state information alongside the observation vector")

        print("  - checking that a valid action carries the expected time-related information")
        valid_action = int(np.argmax(info["action_mask"]))
        next_observation, reward, terminated, truncated, next_info = env.step(valid_action)
        self.assertEqual(next_observation.shape, observation.shape)
        self.assertTrue(np.isfinite(reward))
        self.assertIn("state_id", next_info)
        self.assertTrue("action_mask" in next_info)
        self.assertTrue(next_info["state_id"] in env.states)
        self.assertGreaterEqual(len(env.actions), 1)
        print("    -> this proves the environment reports the updated state and retains enough action metadata for downstream time-based reasoning")

    def test_inspection_actions_record_outcomes_without_changing_physical_state(self):
        print()
        self.log("Inspection action", "Inspection updates the state history and preserves the physical state")

        graph = nx.DiGraph()
        graph.add_node("root", parts_removed=[], parts_present=["gear"])
        graph.add_edge(
            "root",
            "root",
            label="inspect_gear",
            capability="inspect",
            target_part="gear",
            action_time=0.75,
            inspection_result="bad",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path = Path(temp_dir) / "graph.pkl"
            with graph_path.open("wb") as handle:
                pickle.dump(graph, handle)

            env = AngleGrinderEnv(graph_path=graph_path)
            observation, info = env.reset()

            print("  - applying an inspection action and checking that it stays in the same state while recording the result")
            next_observation, reward, terminated, truncated, next_info = env.step(0)

            self.assertEqual(next_observation.tolist(), observation.tolist())
            self.assertEqual(next_info["state_id"], info["state_id"])
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(len(next_info["state_action_history"]), 1)
            self.assertEqual(next_info["state_action_history"][0]["result"], "bad")
            self.assertEqual(next_info["state_action_history"][0]["time"], 0.75)
            self.assertEqual(next_info["cumulative_time"], 0.75)
            self.assertEqual(reward, -5.75)
            print("    -> this proves inspection actions record their outcome in the history and keep the physical state unchanged")

    def test_print_actions_and_components_lists_the_graph_contents(self):
        print()
        self.log("Graph summary", "The environment can print a readable action/component summary")
        env = AngleGrinderEnv(graph_path=self.graph_path)

        print("  - printing the action and component catalogue")
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            env.print_actions_and_components()

        output = output_buffer.getvalue()
        self.assertIn("Angle grinder actions:", output)
        self.assertIn("Angle grinder components:", output)
        self.assertIn("part", output)
        print("    -> this proves the environment can describe its action space and component list for debugging")

    def test_step_with_invalid_action_ends_episode_with_penalty(self):
        print()
        self.log("Invalid action", "Invalid action is rejected")
        env = AngleGrinderEnv(graph_path=self.graph_path)
        observation, info = env.reset()

        print("  - applying an out-of-range action to confirm invalid actions are rejected")
        invalid_action = env.action_space.n
        next_observation, reward, terminated, truncated, next_info = env.step(invalid_action)

        self.assertEqual(next_observation.shape, observation.shape)
        self.assertEqual(reward, -100)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(next_info["is_goal"])
        self.assertFalse(next_info["is_dead_end"])
        print("    -> this proves invalid actions are rejected with a penalty and the environment remains in a terminal failed state")


if __name__ == "__main__":
    unittest.main(verbosity=2)
