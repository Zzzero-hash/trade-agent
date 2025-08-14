# Logs Directory

This directory contains training logs, evaluation metrics, and debugging information. All files are excluded from Git as they are runtime-generated.

## Log Structure

### Reinforcement Learning Logs
- `ppo/` - PPO (Proximal Policy Optimization) training logs
- `sac/` - SAC (Soft Actor-Critic) training logs
- Each subdirectory contains:
  - `PPO_*/` or `SAC_*/` - Individual training run logs
  - `events.out.tfevents.*` - TensorBoard event files
  - `evaluations.npz` - Evaluation metrics during training

### Log Contents

#### TensorBoard Files
- Training metrics over time
- Loss curves and rewards
- Policy gradients and actor/critic losses
- Episode rewards and lengths

#### Evaluation Files
- Periodic evaluation results during training
- Performance on validation environments
- Statistical summaries of agent performance

## How to Generate Logs

### Train RL Models
```bash
# PPO training (generates logs/ppo/)
python src/rl/train.py --algorithm ppo

# SAC training (generates logs/sac/)  
python src/rl/train.py --algorithm sac
```

### Monitor Training Progress
```bash
# View training in TensorBoard
tensorboard --logdir logs/

# Or specific algorithm
tensorboard --logdir logs/ppo/
tensorboard --logdir logs/sac/
```

## Viewing Logs

### TensorBoard Visualization
1. Install TensorBoard: `pip install tensorboard`
2. Run: `tensorboard --logdir logs/`
3. Open browser to `http://localhost:6006`

### Load Evaluation Data
```python
import numpy as np

# Load evaluation metrics
data = np.load('logs/ppo/evaluations.npz')
print(data.files)  # See available metrics
rewards = data['results']
timesteps = data['timesteps']
```

## Log Management

### Automatic Cleanup
Consider implementing log rotation:
```bash
# Remove logs older than 30 days
find logs/ -name "*.out.*" -mtime +30 -delete

# Keep only last 10 training runs
ls -dt logs/ppo/PPO_* | tail -n +11 | xargs rm -rf
```

### Storage Guidelines
- **Development**: Keep recent logs for debugging
- **Production**: Archive important training runs
- **CI/CD**: Clear logs between runs to save space

## Integration

### With MLflow
```python
# Log metrics to MLflow during training
import mlflow
mlflow.log_artifacts("logs/ppo/", artifact_path="training_logs")
```

### With Weights & Biases
```python
# Sync logs with W&B
import wandb
wandb.save("logs/ppo/**")
```

## Troubleshooting

### Large Log Files
- TensorBoard files can grow large during long training
- Use `--max_queue_size` and `--flush_secs` to control memory usage
- Consider downsampling frequency for very long runs

### Missing Logs
- Ensure logging is enabled in training configs
- Check file permissions in logs directory
- Verify TensorBoard writer initialization

## File Sizes

Typical sizes:
- TensorBoard event files: 1-100MB depending on training length
- Evaluation files: 1-10KB per run
- Complete training run: 10-200MB

## Security Notes

- Logs may contain model architecture information
- Training metrics could reveal proprietary strategies
- Consider log sanitization for shared environments
