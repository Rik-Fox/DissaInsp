import unittest
from pathlib import Path

from src.agent import Model, belief_update, initial_belief
from src.configs import CONFIGS, DEFAULT
from src.env import CONDITIONS, AngleGrinderEnv


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.real_graph_path = Path(__file__).resolve().parents[1] / "graph.pkl"
        self.assertTrue(self.real_graph_path.exists())

    def test_default_config_is_used_when_none_given(self):
        env = AngleGrinderEnv(graph_path=self.real_graph_path)
        self.assertEqual(env.config, DEFAULT)

    def test_all_named_configs_load_and_expose_a_usable_action_space(self):
        for name in CONFIGS:
            with self.subTest(config=name):
                env = AngleGrinderEnv(graph_path=self.real_graph_path, config=name)
                self.assertEqual(env.action_space.n, 7)

    def _avg_confidence_after_n_inspections(self, model, true_condition, n, trials=200):
        """Repeats the same Bayes update every config uses - starting from a
        uniform belief, updating via belief_update(model, b, "Inspect", o)
        for `n` independently-sampled observations - and returns the average
        posterior mass on the true condition across `trials` runs."""
        y_idx = CONDITIONS.index(true_condition)
        totals = []
        for _ in range(trials):
            b = initial_belief(model)
            for _ in range(n):
                env = model.env
                env.condition = true_condition  # ground truth for sampling
                o_idx = ["GOOD", "OK", "BAD"].index(env._sample_condition_observation())
                b = belief_update(model, b, "Inspect", o_idx)
            marginal = b
            totals.append(marginal[y_idx])
        return sum(totals) / len(totals)

    def test_repeated_inspections_progressively_concentrate_belief_on_true_condition(self):
        """Every config uses the same mechanism: a fixed confusion matrix,
        Bayes-updated after each Inspect. Confidence in the true condition
        should climb as more (independent) inspections accumulate, for every
        config - just at different rates depending on matrix reliability."""
        for name in CONFIGS:
            with self.subTest(config=name):
                env = AngleGrinderEnv(graph_path=self.real_graph_path, config=name)
                env.reset(options={"condition": "Degraded"})
                model = Model(env)

                conf_1 = self._avg_confidence_after_n_inspections(model, "Degraded", 1)
                conf_20 = self._avg_confidence_after_n_inspections(model, "Degraded", 20)
                self.assertGreater(conf_20, conf_1)
                self.assertGreater(conf_20, 0.8)  # 20 readings should be decisive everywhere

    def test_progressive_confidence_via_weight_not_config(self):
        """The same config, same true condition: a low-confidence update
        should move belief toward the truth less than a full-confidence one,
        and confidence=0 should leave belief unchanged entirely."""
        env = AngleGrinderEnv(graph_path=self.real_graph_path, config="no_inspection")
        env.reset(options={"condition": "Degraded"})
        model = Model(env)
        y_idx = CONDITIONS.index("Degraded")

        b0 = initial_belief(model)
        o_idx = ["GOOD", "OK", "BAD"].index("BAD")  # correctly diagnostic reading

        b_unchanged = belief_update(model, b0, "Inspect", o_idx, confidence=0)
        self.assertTrue((b_unchanged == b0).all())

        b_low = belief_update(model, b0, "Inspect", o_idx, confidence=0.2)
        b_full = belief_update(model, b0, "Inspect", o_idx, confidence=1.0)
        self.assertLess(b_low[y_idx], b_full[y_idx])

    def test_reliable_repair_vs_reuse_maps_good_to_reuse_and_bad_to_refurbish(self):
        env = AngleGrinderEnv(graph_path=self.real_graph_path, config="repair_vs_reuse")
        model = Model(env)

        for observed, expected_action in [("GOOD", "Reuse"), ("OK", "Refurbished"), ("BAD", "Recycle")]:
            b0 = initial_belief(model)
            b = belief_update(model, b0, "Inspect", ["GOOD", "OK", "BAD"].index(observed))
            values = {a: float(b @ model.reward(a)) for a in ("Reuse", "Refurbished", "Recycle")}
            self.assertEqual(max(values, key=values.get), expected_action)


if __name__ == "__main__":
    unittest.main(verbosity=2)
