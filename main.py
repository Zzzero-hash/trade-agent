"""
Trading RL Agent Main Entry Point

This script demonstrates the unified data pipeline orchestrator optimized
for CNN-LSTM training. It processes S&P 500 symbols through the complete
pipeline including validation, feature engineering, and CNN-LSTM tensor
preparation.
"""

import logging
import os
import tempfile

import ray

from src.data.unified_orchestrator import run_cnn_lstm_pipeline


def main():
    """Main entry point for the trading RL agent."""
    # Initialize Ray with secure temporary directory configuration
    temp_base_dir = tempfile.gettempdir()
    ray.init(
        object_spilling_directory=os.path.join(temp_base_dir, "ray_spill"),
        _temp_dir=os.path.join(temp_base_dir, "ray_temp")
    )

    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting Trading RL Agent Data Pipeline")

    try:
        # Run the unified CNN-LSTM optimized pipeline
        result = run_cnn_lstm_pipeline(
            symbols=None,  # Will use S&P 500 symbols
            enable_auto_correction=True,
            max_validation_iterations=3,
            target_assets=50,  # Start with 50 assets for demo
            sequence_length=30
        )

        if result['success']:
            logger.info("✅ Pipeline completed successfully!")
            logger.info(f"Processed data shape: {result['data'].shape}")

            cnn_lstm_tensor = result.get('cnn_lstm_tensor')
            if cnn_lstm_tensor is not None:
                logger.info(f"CNN-LSTM tensor shape: {cnn_lstm_tensor.shape}")
                logger.info("🧠 Data ready for CNN-LSTM training!")

            logger.info(f"Metrics: {result['metrics']}")
        else:
            logger.error(f"❌ Pipeline failed: {result['error']}")

    except Exception as e:
        logger.error(f"❌ Fatal error in main pipeline: {str(e)}")
        raise

    finally:
        # Cleanup Ray
        ray.shutdown()


if __name__ == "__main__":
    main()





def main():
    """Main entry point for the trading RL agent."""
    # Initialize Ray with secure temporary directory configuration
    temp_base_dir = tempfile.gettempdir()
    ray.init(
        object_spilling_directory=os.path.join(temp_base_dir, "ray_spill"),
        _temp_dir=os.path.join(temp_base_dir, "ray_temp")
    )

    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting Trading RL Agent Data Pipeline")

    try:
        # Run the unified CNN-LSTM optimized pipeline
        result = run_cnn_lstm_pipeline(
            symbols=None,  # Will use S&P 500 symbols
            enable_auto_correction=True,
            max_validation_iterations=3,
            target_assets=50,  # Start with 50 assets for demo
            sequence_length=30
        )

        if result['success']:
            logger.info("✅ Pipeline completed successfully!")
            logger.info(f"Processed data shape: {result['data'].shape}")

            cnn_lstm_tensor = result.get('cnn_lstm_tensor')
            if cnn_lstm_tensor is not None:
                logger.info(f"CNN-LSTM tensor shape: {cnn_lstm_tensor.shape}")
                logger.info("🧠 Data ready for CNN-LSTM training!")

            logger.info(f"Metrics: {result['metrics']}")
        else:
            logger.error(f"❌ Pipeline failed: {result['error']}")

    except Exception as e:
        logger.error(f"❌ Fatal error in main pipeline: {str(e)}")
        raise

    finally:
        # Cleanup Ray
        ray.shutdown()


if __name__ == "__main__":
    main()
    logger.info(f"Processed data shape: {processed_data.shape}")
    logger.info(f"Sample processed data:\n{processed_data.head()}")

    # Cleanup Ray resources
    ray.shutdown()


if __name__ == "__main__":
    print("Select mode: ")
    print("1. Training Mode")
    print("2. Backtesting Mode")
    print("3. Live Trading Mode")
    mode = input("Enter mode (1, 2, or 3): ")
    if mode == "1":
        main()
    elif mode == "2":
        # TODO: Implement Back Testing mode
        pass
    elif mode == "3":
        # TODO: Implement live trading mode
        pass
