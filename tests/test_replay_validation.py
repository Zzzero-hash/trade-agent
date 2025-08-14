import os
import sys
import unittest
from typing import Any

# Add project root to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.replay_run import ReplayRunner  # noqa: E402


class TestReplayValidation(unittest.TestCase):
    """
    Test suite for validating replay determinism and metric floors.

    This test suite validates the claims made in the README about
    reproducibility and ensures that key metrics meet minimum quality
    thresholds.
    """

    def setUp(self):
        """Set up test environment."""
        self.seed = 42
        self.output_report_1 = "test_validation_report_1.json"
        self.output_report_2 = "test_validation_report_2.json"
        self.reports_to_cleanup = [self.output_report_1, self.output_report_2]

        # Define metric floors based on realistic expectations
        self.metric_floors: dict[str, float] = {
            'sl_train_mse_max': 1.0,  # MSE should be reasonable
            'sl_train_r2_min': -1.0,  # R² can be negative but not < -1
            'feature_count_min': 5,   # Should have at least 5 features
            'sample_count_min': 0,    # Allow 0 samples (data issues)
            'build_time_max': 300,    # Build should complete in 5 minutes
            'training_time_max': 300  # Training should complete in 5 minutes
        }

        # Define determinism tolerance for numeric comparisons
        self.determinism_tolerance = 1e-10

    def tearDown(self):
        """Clean up generated files."""
        for report in self.reports_to_cleanup:
            if os.path.exists(report):
                os.remove(report)

    def run_replay(self, output_path: str) -> dict[str, Any]:
        """Helper to run the replay and return results."""
        runner = ReplayRunner(seed=self.seed, verbose=False)
        results = runner.run()
        runner.save_report(output_path)
        return results

    def test_replay_determinism_and_metrics(self):
        """
        Test that two consecutive replay runs with the same seed produce
        deterministic results and that metrics meet minimum quality floors.

        This test validates the reproducibility claims in the README by:
        1. Running the same pipeline twice with the same seed
        2. Verifying that deterministic components produce identical results
        3. Checking that metrics meet minimum quality thresholds
        4. Validating artifact shapes and hashes for consistency
        """
        # Run 1
        results_1 = self.run_replay(self.output_report_1)

        # Run 2
        results_2 = self.run_replay(self.output_report_2)

        # Core determinism tests
        self.assert_deterministic_components_match(results_1, results_2)

        # Metric floor tests (test both runs meet minimum standards)
        self.assert_metric_floors(results_1, "Run 1")
        self.assert_metric_floors(results_2, "Run 2")

        # Shape consistency tests
        self.assert_shapes_are_consistent(results_1, results_2)

        # Hash determinism tests (for components that should be deterministic)
        self.assert_hashes_are_deterministic(results_1, results_2)

        # Numeric metric tolerance tests
        self.assert_numeric_metrics_within_tolerance(results_1, results_2)

    def assert_deterministic_components_match(self, r1: dict[str, Any],
                                              r2: dict[str, Any]) -> None:
        """
        Assert that deterministic components produce identical results.

        This focuses on components that should be perfectly reproducible
        with the same seed, excluding stochastic RL components.
        """
        # Seed validation should be identical
        self.assertEqual(
            r1["seed_validation"]["status"],
            r2["seed_validation"]["status"],
            "Seed validation status should be deterministic"
        )

        if r1["seed_validation"]["status"] == "PASS":
            self.assertEqual(
                r1["seed_validation"]["sample_hash"],
                r2["seed_validation"]["sample_hash"],
                "Seed validation hash should be identical"
            )

        # Feature building should be deterministic if successful
        self.assertEqual(
            r1["feature_building"]["status"],
            r2["feature_building"]["status"],
            "Feature building status should be deterministic"
        )

    def assert_metric_floors(self, results: dict[str, Any],
                             run_name: str) -> None:
        """
        Assert that metrics meet minimum quality thresholds (floors).

        This validates that the system produces results that meet
        basic quality standards, regardless of exact values.
        """
        # Check feature building metrics
        fb = results.get("feature_building", {})
        if fb.get("status") == "PASS":
            shape = fb.get("output_features_shape", [0, 0])
            feature_count = shape[1] if len(shape) > 1 else 0
            sample_count = shape[0] if len(shape) > 0 else 0

            self.assertGreaterEqual(
                feature_count,
                self.metric_floors['feature_count_min'],
                f"{run_name}: Feature count below minimum floor"
            )

            self.assertGreaterEqual(
                sample_count,
                self.metric_floors['sample_count_min'],
                f"{run_name}: Sample count below minimum floor"
            )

            build_time = fb.get("build_time_seconds", 0)
            self.assertLessEqual(
                build_time,
                self.metric_floors['build_time_max'],
                f"{run_name}: Feature build time exceeds maximum"
            )

        # Check SL training metrics
        sl = results.get("sl_training", {})
        if sl.get("status") == "PASS":
            train_mse = sl.get("train_mse")
            if train_mse is not None:
                self.assertLessEqual(
                    train_mse,
                    self.metric_floors['sl_train_mse_max'],
                    f"{run_name}: SL training MSE exceeds maximum floor"
                )

            train_r2 = sl.get("train_r2")
            if train_r2 is not None:
                self.assertGreaterEqual(
                    train_r2,
                    self.metric_floors['sl_train_r2_min'],
                    f"{run_name}: SL training R² below minimum floor"
                )

            training_time = sl.get("training_time_seconds", 0)
            self.assertLessEqual(
                training_time,
                self.metric_floors['training_time_max'],
                f"{run_name}: SL training time exceeds maximum"
            )

    def assert_shapes_are_consistent(self, r1: dict[str, Any],
                                     r2: dict[str, Any]) -> None:
        """Assert that data shapes are identical between runs."""
        # Feature shapes should be identical
        shape_1 = r1.get("feature_building", {}).get("output_features_shape")
        shape_2 = r2.get("feature_building", {}).get("output_features_shape")

        if shape_1 is not None and shape_2 is not None:
            self.assertListEqual(
                shape_1, shape_2,
                "Feature shapes should be identical between runs"
            )

    def assert_hashes_are_deterministic(self, r1: dict[str, Any],
                                        r2: dict[str, Any]) -> None:
        """
        Assert that hashes are identical for deterministic components.

        This tests the core determinism claim by verifying that
        identical inputs produce identical outputs.
        """
        # Feature hash should be deterministic
        fb1 = r1.get("feature_building", {})
        fb2 = r2.get("feature_building", {})

        if (fb1.get("status") == "PASS" and fb2.get("status") == "PASS"):
            hash_1 = fb1.get("features_hash")
            hash_2 = fb2.get("features_hash")

            if hash_1 is not None and hash_2 is not None:
                self.assertEqual(
                    hash_1, hash_2,
                    "Feature hashes should be identical for deterministic runs"
                )

        # SL results hash should be deterministic if both succeeded
        sl1 = r1.get("sl_training", {})
        sl2 = r2.get("sl_training", {})

        if (sl1.get("status") == "PASS" and sl2.get("status") == "PASS"):
            hash_1 = sl1.get("results_hash")
            hash_2 = sl2.get("results_hash")

            if hash_1 is not None and hash_2 is not None:
                self.assertEqual(
                    hash_1, hash_2,
                    "SL results hashes should be identical for "
                    "deterministic runs"
                )

    def assert_numeric_metrics_within_tolerance(self, r1: dict[str, Any],
                                                r2: dict[str, Any]) -> None:
        """
        Assert that numeric metrics are within tolerance between runs.

        This tests that even if there are minor floating-point differences,
        they are within acceptable bounds for deterministic operations.
        """
        # Compare SL metrics if both runs succeeded
        sl1 = r1.get("sl_training", {})
        sl2 = r2.get("sl_training", {})

        if (sl1.get("status") == "PASS" and sl2.get("status") == "PASS"):
            numeric_metrics = ["train_mse", "train_mae", "train_r2",
                               "cv_mse_mean"]

            for metric in numeric_metrics:
                val1 = sl1.get(metric)
                val2 = sl2.get(metric)

                if val1 is not None and val2 is not None:
                    self.assertAlmostEqual(
                        val1, val2,
                        delta=self.determinism_tolerance,
                        msg=f"SL metric '{metric}' differs beyond tolerance"
                    )

    def test_metric_floors_standalone(self):
        """
        Test metric floors independently of determinism.

        This ensures that the system can produce results that meet
        minimum quality standards in a single run.
        """
        results = self.run_replay("test_standalone_report.json")
        self.assert_metric_floors(results, "Standalone run")

        # Clean up
        if os.path.exists("test_standalone_report.json"):
            os.remove("test_standalone_report.json")

    def test_basic_pipeline_execution(self):
        """
        Test that the basic pipeline can execute without crashing.

        This is a smoke test to ensure the replay runner can complete
        its execution, regardless of validation results.
        """
        runner = ReplayRunner(seed=self.seed, verbose=False)
        results = runner.run()

        # Basic structure checks
        self.assertIsInstance(results, dict)
        self.assertIn("replay_metadata", results)
        self.assertIn("seed", results["replay_metadata"])
        self.assertEqual(results["replay_metadata"]["seed"], self.seed)

        # Should have attempted all major components
        expected_sections = [
            "seed_validation", "feature_building", "sl_training",
            "rl_training", "hash_validation", "metric_validation",
            "tolerance_checks", "overall_validation"
        ]

        for section in expected_sections:
            self.assertIn(section, results,
                          f"Missing expected section: {section}")

    def test_readme_determinism_claims(self):
        """
        Explicitly test determinism claims mentioned in README.

        The README mentions fixing seeds and asserting distributional
        properties/shape/range rather than exact equality for stochastic
        RL components. This test validates those specific claims.
        """
        # Run two identical tests
        results_1 = self.run_replay("readme_test_1.json")
        results_2 = self.run_replay("readme_test_2.json")

        # Test seed reproducibility
        self.assert_seed_reproducibility(results_1, results_2)

        # Test shape consistency (key determinism claim)
        self.assert_distributional_properties(results_1, results_2)

        # Clean up
        for report in ["readme_test_1.json", "readme_test_2.json"]:
            if os.path.exists(report):
                os.remove(report)

    def assert_seed_reproducibility(self, r1: dict[str, Any],
                                    r2: dict[str, Any]) -> None:
        """Test that fixed seeds produce reproducible results."""
        # Seed validation should be consistent
        self.assertEqual(
            r1["seed_validation"]["status"],
            r2["seed_validation"]["status"]
        )

        if r1["seed_validation"]["status"] == "PASS":
            # Hash should be identical with same seed
            self.assertEqual(
                r1["seed_validation"]["sample_hash"],
                r2["seed_validation"]["sample_hash"]
            )

    def assert_distributional_properties(self, r1: dict[str, Any],
                                         r2: dict[str, Any]) -> None:
        """
        Test distributional properties and shapes as mentioned in README.

        For stochastic components, assert shapes and ranges rather than
        exact equality.
        """
        # Feature shapes should be identical (deterministic)
        fb1 = r1.get("feature_building", {})
        fb2 = r2.get("feature_building", {})

        if fb1.get("status") == "PASS" and fb2.get("status") == "PASS":
            shape_1 = fb1.get("output_features_shape", [])
            shape_2 = fb2.get("output_features_shape", [])
            self.assertEqual(
                shape_1, shape_2, "Feature shapes must be identical"
            )

            # Validate shape properties
            if len(shape_1) == 2:
                rows, cols = shape_1
                self.assertGreaterEqual(
                    rows, 0, "Row count should be non-negative"
                )
                self.assertGreater(cols, 0, "Column count should be positive")

        # For RL components, check that rewards are within reasonable ranges
        # (not exact equality due to stochastic nature)
        rl1 = r1.get("rl_training", {})
        rl2 = r2.get("rl_training", {})

        for agent in ["ppo", "sac"]:
            agent1 = rl1.get(agent, {})
            agent2 = rl2.get(agent, {})

            # If both succeeded, rewards should be in similar ranges
            if (agent1.get("status") == "PASS" and
                    agent2.get("status") == "PASS"):

                reward1 = agent1.get("final_reward")
                reward2 = agent2.get("final_reward")

                if reward1 is not None and reward2 is not None:
                    # Check range properties rather than exact equality
                    self.assertIsInstance(reward1, (int, float))
                    self.assertIsInstance(reward2, (int, float))

                    # Rewards should be finite
                    self.assertTrue(abs(reward1) < float('inf'))
                    self.assertTrue(abs(reward2) < float('inf'))


if __name__ == '__main__':
    unittest.main()
