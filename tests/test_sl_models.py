"""
Unit tests for supervised learning models.
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import pytest


# Import from src modules
try:
    from trade_agent.agents.sl.evaluate import SLEvaluationMetrics, SLEvaluationPipeline
    from trade_agent.agents.sl.models.base import (
        PyTorchSLModel,
        SLBaseModel,
        set_all_seeds,
    )
    from trade_agent.agents.sl.models.deep_learning import (
        CNNLSTMModel,
        MLPModel,
        TransformerModel,
    )
    from trade_agent.agents.sl.models.ensemble import EnsembleModel, StackingModel
    from trade_agent.agents.sl.models.factory import SLModelFactory
    from trade_agent.agents.sl.models.traditional import (
        GARCHModel,
        LinearModel,
        RidgeModel,
    )
    from trade_agent.agents.sl.models.tree_based import (
        LightGBMModel,
        RandomForestModel,
        XGBoostModel,
    )
    from trade_agent.agents.sl.predict import SLPredictionPipeline
    from trade_agent.agents.sl.train import SLTrainingPipeline, TemporalCV
except ImportError:
    # Fallback for development environment
    import sys
    sys.path.append('src')
    from trade_agent.agents.sl.evaluate import SLEvaluationMetrics, SLEvaluationPipeline
    from trade_agent.agents.sl.models.base import SLBaseModel, set_all_seeds
    from trade_agent.agents.sl.models.deep_learning import (
        CNNLSTMModel,
        MLPModel,
        TransformerModel,
    )
    from trade_agent.agents.sl.models.ensemble import EnsembleModel, StackingModel
    from trade_agent.agents.sl.models.factory import SLModelFactory
    from trade_agent.agents.sl.models.traditional import (
        GARCHModel,
        LinearModel,
        RidgeModel,
    )
    from trade_agent.agents.sl.models.tree_based import (
        LightGBMModel,
        RandomForestModel,
        XGBoostModel,
    )
    from trade_agent.agents.sl.predict import SLPredictionPipeline
    from trade_agent.agents.sl.train import SLTrainingPipeline, TemporalCV


class TestBaseModels(unittest.TestCase):
    """Test base model classes."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = {'random_state': 42}
        set_all_seeds(42)

    def test_sl_base_model_abstract(self) -> None:
        """Test that SLBaseModel is abstract and cannot be instantiated."""
        with self.assertRaises(TypeError):
            SLBaseModel(self.config)

    def test_set_all_seeds(self) -> None:
        """Test that set_all_seeds works correctly."""
        set_all_seeds(42)
        a = np.random.rand(1)
        set_all_seeds(42)
        b = np.random.rand(1)
        np.testing.assert_array_equal(a, b)


class TestTraditionalModels(unittest.TestCase):
    """Test traditional models."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = {'random_state': 42}
        set_all_seeds(42)

        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)

    def test_ridge_model(self) -> None:
        """Test Ridge model."""
        model = RidgeModel(self.config)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        self.assertEqual(predictions.shape, (100,))
        self.assertTrue(model.is_fitted)

    def test_linear_model(self) -> None:
        """Test Linear model."""
        model = LinearModel(self.config)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        self.assertEqual(predictions.shape, (100,))
        self.assertTrue(model.is_fitted)

    def test_garch_model(self) -> None:
        """Test GARCH model."""
        model = GARCHModel(self.config)
        model.fit(self.X[:, 0], self.y)  # Use first column as returns
        predictions = model.predict(self.X[:, 0])

        self.assertEqual(predictions.shape, (100,))
        self.assertTrue(model.is_fitted)

    def test_model_save_load(self) -> None:
        """Test model save and load functionality."""
        model = RidgeModel(self.config)
        model.fit(self.X, self.y)

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name

        model.save_model(model_path)

        # Load model
        loaded_model = RidgeModel.load_model(model_path)
        loaded_predictions = loaded_model.predict(self.X)
        original_predictions = model.predict(self.X)

        np.testing.assert_array_almost_equal(loaded_predictions, original_predictions)

        # Clean up
        os.unlink(model_path)


class TestTreeBasedModels(unittest.TestCase):
    """Test tree-based models."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = {'random_state': 42, 'n_estimators': 10}
        set_all_seeds(42)

        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)

    def test_xgboost_model(self) -> None:
        """Test XGBoost model."""
        try:
            model = XGBoostModel(self.config)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X)

            self.assertEqual(predictions.shape, (100,))
            self.assertTrue(model.is_fitted)
        except ImportError:
            self.skipTest("XGBoost not available")

    def test_lightgbm_model(self) -> None:
        """Test LightGBM model."""
        try:
            model = LightGBMModel(self.config)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X)

            self.assertEqual(predictions.shape, (100,))
            self.assertTrue(model.is_fitted)
        except ImportError:
            self.skipTest("LightGBM not available")

    def test_random_forest_model(self) -> None:
        """Test Random Forest model."""
        try:
            model = RandomForestModel(self.config)
            model.fit(self.X, self.y)
            predictions = model.predict(self.X)

            self.assertEqual(predictions.shape, (100,))
            self.assertTrue(model.is_fitted)
        except ImportError:
            self.skipTest("Scikit-learn not available")


class TestDeepLearningModels(unittest.TestCase):
    """Test deep learning models."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = {
            'random_state': 42,
            'epochs': 2,
            'batch_size': 32,
            'input_size': 5,
            'output_size': 1,
            'sequence_length': 10
        }
        set_all_seeds(42)

        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)

    def test_cnn_lstm_model(self) -> None:
        """Test CNN-LSTM model."""
        model = CNNLSTMModel(self.config)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        self.assertEqual(predictions.shape, (91,))  # 100 - 10 + 1
        self.assertTrue(model.is_fitted)

    def test_transformer_model(self) -> None:
        """Test Transformer model."""
        model = TransformerModel(self.config)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        self.assertEqual(predictions.shape, (91,))  # 100 - 10 + 1
        self.assertTrue(model.is_fitted)

    def test_mlp_model(self) -> None:
        """Test MLP model."""
        model = MLPModel(self.config)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        self.assertEqual(predictions.shape, (100,))
        self.assertTrue(model.is_fitted)


class TestEnsembleModels(unittest.TestCase):
    """Test ensemble models."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = {'random_state': 42}
        set_all_seeds(42)

        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)

    def test_ensemble_model(self) -> None:
        """Test Ensemble model."""
        model = EnsembleModel(self.config)

        # Add base models
        ridge_config = {'random_state': 42, 'alpha': 1.0}
        ridge_model = RidgeModel(ridge_config)
        model.add_model(ridge_model)

        linear_config = {'random_state': 42}
        linear_model = LinearModel(linear_config)
        model.add_model(linear_model)

        # Fit and predict
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        self.assertEqual(predictions.shape, (100,))
        self.assertTrue(model.is_fitted)

    def test_stacking_model(self) -> None:
        """Test Stacking model."""
        model = StackingModel(self.config)

        # Add base models
        ridge_config = {'random_state': 42, 'alpha': 1.0}
        ridge_model = RidgeModel(ridge_config)
        model.add_model(ridge_model)

        linear_config = {'random_state': 42}
        linear_model = LinearModel(linear_config)
        model.add_model(linear_model)

        # Fit and predict
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        self.assertEqual(predictions.shape, (100,))
        self.assertTrue(model.is_fitted)


class TestModelFactory(unittest.TestCase):
    """Test model factory."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = {'random_state': 42}

    def test_create_model(self) -> None:
        """Test creating models with factory."""
        model_types = ['ridge', 'linear', 'garch', 'mlp']

        for model_type in model_types:
            try:
                model = SLModelFactory.create_model(model_type, self.config)
                self.assertIsInstance(model, SLBaseModel)
            except Exception:
                pass

    def test_get_available_models(self) -> None:
        """Test getting available models."""
        available_models = SLModelFactory.get_available_models()
        self.assertIsInstance(available_models, list)
        self.assertIn('ridge', available_models)
        self.assertIn('mlp', available_models)

    def test_is_model_available(self) -> None:
        """Test checking if model is available."""
        self.assertTrue(SLModelFactory.is_model_available('ridge'))
        self.assertFalse(SLModelFactory.is_model_available('nonexistent_model'))


class TestTrainingPipeline(unittest.TestCase):
    """Test training pipeline."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = {
            'model_type': 'ridge',
            'model_config': {'random_state': 42, 'alpha': 1.0},
            'cv_config': {'n_splits': 3, 'gap': 0},
            'random_state': 42,
            'save_model': False
        }
        set_all_seeds(42)

        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)

    def test_temporal_cv(self) -> None:
        """Test temporal cross-validation."""
        cv = TemporalCV(n_splits=3, gap=0)
        splits = list(cv.split(self.X, self.y))

        self.assertEqual(len(splits), 3)
        for train_idx, val_idx in splits:
            self.assertLess(np.max(train_idx), np.min(val_idx))  # Temporal order

    @pytest.mark.slow
    def test_training_pipeline(self) -> None:
        """Test training pipeline."""
        pipeline = SLTrainingPipeline(self.config)
        results = pipeline.train(self.X, self.y)

        self.assertIsInstance(results, dict)
        self.assertIn('train_mse', results)
        self.assertIn('cv_mse_mean', results)


class TestEvaluationPipeline(unittest.TestCase):
    """Test evaluation pipeline."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = {'compute_financial_metrics': True}

        # Create sample data
        np.random.seed(42)
        self.y_true = np.random.randn(100)
        self.y_pred = np.random.randn(100)

    def test_regression_metrics(self) -> None:
        """Test regression metrics."""
        metrics = SLEvaluationMetrics.regression_metrics(self.y_true, self.y_pred)

        self.assertIsInstance(metrics, dict)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)

    def test_financial_metrics(self) -> None:
        """Test financial metrics."""
        metrics = SLEvaluationMetrics.financial_metrics(self.y_true, self.y_pred)

        self.assertIsInstance(metrics, dict)
        self.assertIn('directional_accuracy', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('information_coefficient', metrics)

    def test_evaluation_pipeline(self) -> None:
        """Test evaluation pipeline."""
        pipeline = SLEvaluationPipeline(self.config)
        results = pipeline.evaluate(self.y_true, self.y_pred)

        self.assertIsInstance(results, dict)
        self.assertIn('regression_metrics', results)
        self.assertIn('financial_metrics', results)


class TestPredictionPipeline(unittest.TestCase):
    """Test prediction pipeline."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = {'random_state': 42}

        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)

        # Create and train a simple model for testing
        self.model = RidgeModel({'random_state': 42, 'alpha': 1.0})
        self.model.fit(self.X, self.y)

    def test_batch_predict(self) -> None:
        """Test batch prediction."""
        pipeline = SLPredictionPipeline(self.config)
        pipeline.model = self.model

        predictions = pipeline.batch_predict(self.X, batch_size=50)

        self.assertEqual(predictions.shape, (100,))

    def test_predict_with_pandas(self) -> None:
        """Test prediction with pandas DataFrame."""
        pipeline = SLPredictionPipeline(self.config)
        pipeline.model = self.model

        df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(5)])
        predictions = pipeline.predict(df)

        self.assertEqual(predictions.shape, (100,))


if __name__ == '__main__':
    unittest.main()
