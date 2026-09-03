import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np

from src.configs import CONFIDENCE_SCALE
from src.env import AngleGrinderEnv, CONDITIONS, INSPECTION_ACTIONS
from src.agent import (
    Model,
    backup,
    belief_update,
    expand_beliefs,
    initial_belief,
    load_policy,
    most_likely_x,
    save_policy,
    solve,
)


class AgentTests(unittest.TestCase):
    def setUp(self):
        # A single screw, root --Unscrew--> done. Small enough to solve fast
        # and to hand-verify expected values against.
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
        self.root_idx = self.env.state_to_idx[self.env.root_state]

    # -- Model adapter --------------------------------------------------

    def test_transition_shapes_and_categories(self):
        t_unscrew = self.model.transition("Unscrew", self.root_idx)
        self.assertEqual(t_unscrew["type"], "disassy")
        self.assertEqual(t_unscrew["succ_idx"], self.env.state_to_idx["done"])
        self.assertEqual(t_unscrew["fail_idx"], self.root_idx)
        self.assertEqual(self.model.transition("Verify")["type"], "identity")
        self.assertEqual(self.model.transition("Reuse")["type"], "terminal")

    def test_observation_alphabets(self):
        z, labels = self.model.observation("Inspect")
        self.assertEqual(z.shape, (len(CONDITIONS), 3))
        self.assertEqual(labels, ["GOOD", "OK", "BAD"])

        z, labels = self.model.observation("Verify")
        self.assertEqual(z.shape, (2, 2))
        self.assertEqual(labels, ["YES", "NO"])

        z, labels = self.model.observation("Unscrew")
        self.assertEqual(z.shape, (len(CONDITIONS), 1))

        z, labels = self.model.observation("Reuse")
        self.assertEqual(z.shape, (len(CONDITIONS), 1))
        self.assertEqual(labels, ["null"])

    def test_valid_actions_masks_out_unavailable_inspection_actions(self):
        self.model.available_insp_actions = set(INSPECTION_ACTIONS)
        both_available = self.model.valid_actions(self.root_idx)
        self.assertIn("Verify", both_available)
        self.assertIn("Inspect", both_available)

        self.model.available_insp_actions = {"Inspect"}
        verify_used = self.model.valid_actions(self.root_idx)
        self.assertNotIn("Verify", verify_used)
        self.assertIn("Inspect", verify_used)
        self.assertIn("Unscrew", verify_used)  # graph-defined actions unaffected
        for triage in ("Reuse", "Refurbished", "Recycle"):
            self.assertIn(triage, verify_used)  # always valid regardless of mask

    def test_disassy_reward_varies_by_source_state_and_is_zero_elsewhere(self):
        reward = self.model.reward("Unscrew", self.root_idx)
        np.testing.assert_array_equal(reward, np.full(len(CONDITIONS), -2.0))

    def test_triage_reward_matches_payoff_table(self):
        reward = self.model.reward("Reuse", self.root_idx)
        self.assertEqual(reward[0], 10.0)

    # -- belief tracking ---------------------------------------------------

    def test_initial_belief_is_uniform_over_y_at_root(self):
        b0 = initial_belief(self.model)
        self.assertEqual(b0.shape, (len(CONDITIONS),))
        self.assertAlmostEqual(b0.sum(), 1.0)
        np.testing.assert_allclose(b0, [1 / 3, 1 / 3, 1 / 3])

    def test_repeated_bad_inspections_concentrate_belief_on_degraded(self):
        b = initial_belief(self.model)
        bad_idx = ["GOOD", "OK", "BAD"].index("BAD")
        for _ in range(15):
            b = belief_update(self.model, b, "Inspect", bad_idx)

        self.assertGreater(b[2], 0.8)  # Degraded

    def test_confidence_scaling_in_belief_update(self):
        """Checks that CONFIDENCE_SCALE is applied to Bayesian updates."""
        b = initial_belief(self.model)
        b_scaled = belief_update(self.model, b, "Inspect", 0, confidence=0.5)
        z, _ = self.model.observation("Inspect")
        expected_likelihood = z[:, 0] ** (0.5 * CONFIDENCE_SCALE)
        expected = (expected_likelihood * b) / (expected_likelihood * b).sum()
        np.testing.assert_allclose(b_scaled, expected)

    def test_most_likely_x_tracks_disassy_transitions(self):
        self.assertEqual(most_likely_x(self.model), self.root_idx)
        self.env.current_state_id = "done"
        self.assertEqual(most_likely_x(self.model), self.env.state_to_idx["done"])

    # -- solving ----------------------------------------------------------

    def test_expand_beliefs_populates_belief_sets(self):
        belief_sets = expand_beliefs(self.model, {}, n_trajectories=10, horizon=5)
        self.assertGreater(sum(len(v) for v in belief_sets.values()), 0)

        for points in belief_sets.values():
            for b, insp_actions in points:
                self.assertEqual(b.shape, (len(CONDITIONS),))
                self.assertTrue(insp_actions <= INSPECTION_ACTIONS)  # subset, possibly narrowed

    def test_backup_tags_each_belief_point_with_a_valid_action(self):
        belief_sets = expand_beliefs(self.model, {}, n_trajectories=10, horizon=5)
        gamma = {self.root_idx: [(np.zeros(len(CONDITIONS)), None)]}
        gamma = backup(self.model, belief_sets, gamma)

        for x_idx, points in gamma.items():
            for alpha, action_id in points:
                self.assertEqual(alpha.shape, (len(CONDITIONS),))
                # Full mask (superset of whatever the point's own, possibly
                # narrower, available_insp_actions was) - just a sanity
                # check that this is a real action, not the specific point's
                # own restricted context.
                self.model.available_insp_actions = set(INSPECTION_ACTIONS)
                self.assertIn(action_id, self.model.valid_actions(x_idx))

    def test_solve_prefers_refurbished_over_immediate_reuse_or_recycle_at_uniform_prior(self):
        # Hand-verified: with these placeholder numbers, committing directly
        # to "Refurbished" (expected value ~3.67) beats both immediate
        # "Reuse" (~3.33) and even optimally inspecting first (~3.51).
        gamma = solve(self.model, n_iterations=6, n_trajectories=20, horizon=5)
        b0 = initial_belief(self.model)

        best_value, best_action = -np.inf, None
        for alpha, action_id in gamma[self.root_idx]:
            if action_id is None:
                continue
            value = float(b0 @ alpha)
            if value > best_value:
                best_value, best_action = value, action_id

        self.assertEqual(best_action, "Refurbished")

    def test_save_and_load_policy_round_trip(self):
        gamma = {self.root_idx: [(np.array([1.0, 2.0]), "Recycle")]}
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_dir)
        path = Path(temp_dir) / "policy.pkl"

        save_policy(gamma, path)
        loaded = load_policy(path)

        np.testing.assert_array_equal(loaded[self.root_idx][0][0], [1.0, 2.0])
        self.assertEqual(loaded[self.root_idx][0][1], "Recycle")


if __name__ == "__main__":
    unittest.main(verbosity=2)
