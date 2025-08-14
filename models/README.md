# Models Directory

This directory contains trained machine learning models. All model files are excluded from Git due to their size and binary nature.

## Model Types

### Supervised Learning Models
- `sl_model_*.pkl` - Pickled scikit-learn models
- `sl_model_*_metadata.json` - Model metadata and hyperparameters

#### Model Types Available:
- **CNN-LSTM**: Deep learning models (~296KB each)
- **Transformer**: Attention-based models (~280KB each)  
- **MLP**: Multi-layer perceptron models (~20KB each)
- **Ridge**: Linear regression with L2 regularization (~1KB each)
- **Linear**: Basic linear models (~1KB each)
- **GARCH**: Volatility modeling (~1KB each)

### Reinforcement Learning Models
- `rl/` directory contains RL agent checkpoints
- `rl/best_model.zip` - Best performing RL model (~8.8MB)
- `rl/ppo_final.zip` - Final PPO agent (~955KB)
- `rl/sac.zip` - SAC agent checkpoint (~8.8MB)

## How to Generate Models

### Train Supervised Learning Models
```bash
# Train all SL models
python src/sl/train.py --config configs/

# Train specific model
python src/sl/train.py --model mlp --config configs/mlp_config.json
```

### Train Reinforcement Learning Models
```bash
# Train PPO agent
python src/rl/train.py --algorithm ppo --config configs/ppo_config.json

# Train SAC agent  
python src/rl/train.py --algorithm sac --config configs/sac_config.json
```

## Model Storage

### For Development
Models are automatically saved locally during training with timestamps.

### For Production
Consider using:
- **MLflow** for model registry and versioning
- **DVC** for large model version control
- **Git LFS** for models under 100MB
- **Cloud storage** (S3, GCS) for large models with metadata tracking

## Model Loading

```python
# Load supervised learning model
import pickle
with open('models/sl_model_mlp_latest.pkl', 'rb') as f:
    model = pickle.load(f)

# Load RL model using stable-baselines3
from stable_baselines3 import PPO
model = PPO.load('models/rl/best_model.zip')
```

## Model Evaluation

Models can be evaluated using:
```bash
python src/evaluation/backtest.py --model-path models/sl_model_*.pkl
python src/evaluation/rl_evaluate.py --model-path models/rl/best_model.zip
```

## File Size Guidelines

- Individual SL models: < 1MB (exclude from Git)
- RL models: 1-10MB (use Git LFS or external storage)
- Model ensembles: May be larger, store externally

## Security Notes

- Models may contain sensitive information about trading strategies
- Consider encryption for proprietary models
- Ensure model files follow your organization's IP policies
