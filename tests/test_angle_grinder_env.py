import io
import pickle
import shutil
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import networkx as nx
import numpy as np

from src.configs import DEFAULT
from src.env import (
    ALWAYS_VALID_ACTIONS,
    CAPABILITY_TO_TYPE,
    CONDITIONS,
    AngleGrinderEnv,
)


class AngleGrinderEnvTests(unittest.TestCase):
    def setUp(self):
        self.real_graph_path = Path(__file__).resolve().parents[1] / "graph.pkl"
        self.assertTrue(
            self.real_graph_path.exists(),
            f"Graph file not found at {self.real_graph_path}",
        )

    def _env_from_graph(self, graph):
        """Writes `graph` to a temp pickle and returns an env over it."""
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_dir)
        graph_path = Path(temp_dir) / "graph.pkl"
        with graph_path.open("wb") as handle:
            pickle.dump(graph, handle)
        return AngleGrinderEnv(graph_path=graph_path)

    def _branching_disassy_env(self):
        """
        root --Unscrew(screwA, t=2)--> mid_a --Unscrew(screwB, t=3)--> both_removed --Remove(gear, t=6)--> dead_end
        root --Unscrew(screwB, t=5)--> mid_b --Unscrew(screwA, t=4)--> both_removed

        "mid_a" < "mid_b" lexicographically, so the tie-break should always
        pick the root->mid_a edge (screwA) over root->mid_b (screwB).
        "Remove" (the gear) is only valid once both screws are gone.
        """
        graph = nx.DiGraph()
        graph.add_node("root", parts_removed=[], parts_present=["gear", "screwA", "screwB"])
        graph.add_node("mid_a", parts_removed=["screwA"], parts_present=["gear", "screwB"])
        graph.add_node("mid_b", parts_removed=["screwB"], parts_present=["gear", "screwA"])
        graph.add_node("both_removed", parts_removed=["screwA", "screwB"], parts_present=["gear"])
        graph.add_node("dead_end", parts_removed=["screwA", "screwB", "gear"], parts_present=[])
        graph.add_edge(
            "root", "mid_a", capability="ScrewingCapability", target_part="screwA", action_time=2.0,
        )
        graph.add_edge(
            "root", "mid_b", capability="ScrewingCapability", target_part="screwB", action_time=5.0,
        )
        graph.add_edge(
            "mid_a", "both_removed", capability="ScrewingCapability", target_part="screwB", action_time=3.0,
        )
        graph.add_edge(
            "mid_b", "both_removed", capability="ScrewingCapability", target_part="screwA", action_time=4.0,
        )
        graph.add_edge(
            "both_removed", "dead_end", capability="ManipulatingCapability", target_part="gear", action_time=6.0,
        )
        return self._env_from_graph(graph)

    def _step_until(self, env, action_idx, source_state, target_state, max_tries=50):
        """Repeatedly attempts a stochastic Disassy action from source_state
        until it lands in target_state (success or failure branch)."""
        for _ in range(max_tries):
            env.current_state_id = source_state
            result = env.step(action_idx)
            if result[-1]["state_id"] == target_state:
                return result
        self.fail(f"action did not reach {target_state!r} within {max_tries} tries")

    # -- graph loading / collapsed action space ------------------------------

    def test_real_graph_action_space_is_collapsed_to_seven_actions(self):
        env = AngleGrinderEnv(graph_path=self.real_graph_path)
        self.assertEqual(env.action_space.n, 7)
        self.assertEqual(
            set(env.action_list),
            {"Unscrew", "Remove", "Verify", "Inspect", "Reuse", "Refurbished", "Recycle"},
        )

    def test_capability_mapping_covers_known_and_unknown_capabilities(self):
        graph = nx.MultiDiGraph()
        graph.add_node("root", parts_removed=[], parts_present=["p"])
        graph.add_node("dead_end", parts_removed=["p"], parts_present=[])
        for capability in [*CAPABILITY_TO_TYPE, "SomeUnmappedCapability"]:
            graph.add_edge(
                "root", "dead_end", capability=capability, target_part="p", action_time=1.0,
            )
        env = self._env_from_graph(graph)

        for capability, type_ in CAPABILITY_TO_TYPE.items():
            self.assertIn(type_, env.action_list)
            self.assertEqual(env.actions[type_]["action_type"], "Disassy")

        self.assertIn("SomeUnmappedCapability", env.action_list)
        self.assertEqual(env.actions["SomeUnmappedCapability"]["action_type"], "Disassy")

    # -- deterministic tie-break among equivalent choices --------------------

    def test_disassy_tie_break_picks_lexicographically_smallest_successor(self):
        env = self._branching_disassy_env()
        env.reset()

        for _ in range(30):
            env.current_state_id = "root"
            _, reward, terminated, truncated, info = env.step(env.action_to_idx["Unscrew"])
            self.assertIn(info["state_id"], ("root", "mid_a"))  # never mid_b
            if info["state_id"] == "mid_a":
                self.assertEqual(reward, -2.0)  # root->mid_a's own time, not root->mid_b's

    def test_remove_masked_until_prerequisite_screws_unscrewed(self):
        env = self._branching_disassy_env()
        env.reset()

        for state_id in ("root", "mid_a", "mid_b"):
            env.current_state_id = state_id
            self.assertNotIn("Remove", env._get_valid_actions())

        env.current_state_id = "both_removed"
        self.assertIn("Remove", env._get_valid_actions())

    # -- action masking ------------------------------------------------------

    def test_verify_inspect_triage_are_always_valid_everywhere(self):
        env = self._branching_disassy_env()
        env.reset()

        for state_id in ("root", "mid_a", "mid_b", "both_removed", "dead_end"):
            with self.subTest(state_id=state_id):
                env.current_state_id = state_id
                self.assertTrue(ALWAYS_VALID_ACTIONS.issubset(env._get_valid_actions()))

    def test_no_further_disassy_actions_still_allows_only_always_valid_actions(self):
        env = self._branching_disassy_env()
        env.reset()
        env.current_state_id = "dead_end"

        self.assertEqual(env._get_valid_actions(), ALWAYS_VALID_ACTIONS)

    # -- Disassy transitions/reward (no goal/dead-end bonuses) --------------

    def test_disassy_action_only_ever_reaches_graph_successor_or_stays_at_x(self):
        env = self._branching_disassy_env()
        env.reset()
        action_idx = env.action_to_idx["Unscrew"]

        successes = 0
        n_trials = 300
        for _ in range(n_trials):
            env.current_state_id = "root"
            _, reward, terminated, truncated, info = env.step(action_idx)
            self.assertIn(info["state_id"], ("root", "mid_a"))
            self.assertEqual(reward, -2.0)  # flat cost only, no goal/dead-end bonus
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            successes += info["state_id"] == "mid_a"

        self.assertAlmostEqual(successes / n_trials, DEFAULT.disassy_success_prob, delta=0.1)

    def test_disassy_action_returns_null_observation_even_on_success(self):
        env = self._branching_disassy_env()
        env.reset()

        observation, _, _, _, info = self._step_until(
            env, env.action_to_idx["Unscrew"], "root", "mid_a"
        )
        self.assertEqual(observation.tolist(), [0, 0, 0])  # null, despite x really changing
        self.assertEqual(info["state_id"], "mid_a")

    def test_disassy_into_terminal_looking_state_gives_only_flat_cost(self):
        """Reaching a state with no further Disassy options (what used to be
        called a "dead end") is not special-cased or penalized any more -
        only the flat time cost applies."""
        env = self._branching_disassy_env()
        env.reset()

        _, reward, terminated, truncated, info = self._step_until(
            env, env.action_to_idx["Remove"], "both_removed", "dead_end"
        )

        self.assertEqual(reward, -6.0)  # just the edge's own time
        self.assertFalse(terminated)  # only Triage/invalid actions terminate now
        self.assertFalse(truncated)

    def test_info_no_longer_exposes_goal_or_dead_end_flags(self):
        env = self._branching_disassy_env()
        _, info = env.reset()
        self.assertNotIn("is_goal", info)
        self.assertNotIn("is_dead_end", info)
        self.assertNotIn("is_truncated_state", info)

    # -- Verify actions ----------------------------------------------------

    def test_verify_action_reveals_x_without_changing_physical_state(self):
        env = self._branching_disassy_env()
        env.reset()
        self._step_until(env, env.action_to_idx["Unscrew"], "root", "mid_a")
        self.assertEqual(env.part_list, ["gear", "screwA", "screwB"])  # alphabetical order

        observation, reward, terminated, truncated, info = env.step(
            env.action_to_idx["Verify"]
        )

        self.assertEqual(observation.tolist(), [0, 1, 0])  # only screwA removed
        self.assertEqual(info["state_id"], "mid_a")  # unchanged by Verify
        self.assertIsNone(info["condition_observation"])
        self.assertEqual(reward, -1.0)  # VERIFY_COST
        self.assertFalse(terminated)

    # -- Inspect actions (now probabilistic) ---------------------------------

    def test_inspect_action_is_null_on_x_and_probabilistic_on_condition(self):
        env = self._branching_disassy_env()
        condition_idx = CONDITIONS.index("Pristine")
        env.reset(options={"condition": "Pristine"})

        counts = Counter()
        n_trials = 3000
        for _ in range(n_trials):
            observation, reward, terminated, truncated, info = env.step(
                env.action_to_idx["Inspect"]
            )
            self.assertEqual(observation.tolist(), [0, 0, 0])  # null: no info about x
            self.assertEqual(reward, -1.0)  # INSPECT_COST
            self.assertFalse(terminated)
            counts[info["condition_observation"]] += 1

        for i, obs_label in enumerate(["GOOD", "OK", "BAD"]):
            expected_p = np.asarray(DEFAULT.condition_obs_matrix)[condition_idx, i]
            self.assertAlmostEqual(counts[obs_label] / n_trials, expected_p, delta=0.05)

    # -- Triage actions ------------------------------------------------------

    def test_triage_action_ends_episode_with_condition_dependent_payoff(self):
        env = self._branching_disassy_env()

        for condition, action_id, expected in [
            ("Pristine", "Reuse", 10.0),
            ("Serviceable", "Refurbished", 5.0),
            ("Degraded", "Recycle", 1.0),
            ("Degraded", "Reuse", 0.0),  # invalid combo -> 0 reward, no error
        ]:
            with self.subTest(condition=condition, action_id=action_id):
                env.reset(options={"condition": condition})
                _, reward, terminated, truncated, info = env.step(
                    env.action_to_idx[action_id]
                )
                self.assertEqual(reward, expected)
                self.assertTrue(terminated)
                self.assertFalse(truncated)

    # -- invalid actions -----------------------------------------------------

    def test_invalid_action_ends_episode_with_penalty(self):
        env = self._branching_disassy_env()
        observation, _ = env.reset()

        next_observation, reward, terminated, truncated, info = env.step(
            env.action_space.n
        )

        self.assertEqual(next_observation.shape, observation.shape)
        self.assertEqual(reward, -100)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    # -- misc ------------------------------------------------------------

    def test_print_actions_and_components_lists_the_graph_contents(self):
        env = AngleGrinderEnv(graph_path=self.real_graph_path)

        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            env.print_actions_and_components()

        output = output_buffer.getvalue()
        self.assertIn("Angle grinder actions:", output)
        self.assertIn("Angle grinder components:", output)
        self.assertIn("varies by state", output)  # Unscrew/Remove have no fixed time


if __name__ == "__main__":
    unittest.main(verbosity=2)
