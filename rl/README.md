# ATHENA Reinforcement Learning Optimization

Self-improving bug prediction system using **Proximal Policy Optimization (PPO)** for adaptive threshold optimization.

## Overview

This RL module enhances the baseline XGBoost bug predictor by learning optimal prediction thresholds dynamically. Instead of using a fixed threshold (0.5), the RL agent adapts thresholds based on:

- Current prediction confidence
- Historical accuracy
- Code churn features (insertions, deletions, files changed)
- Temporal context (time of day, day of week)

## Architecture

```
XGBoost Model → Bug Probability (0-1)
                      ↓
PPO RL Agent → Optimal Threshold + Priority
                      ↓
Final Prediction + Alert Decision
```

### Components

1. **Environment** (`rl/environment.py`)
   - Custom Gymnasium environment
   - State: 10 features (probability, history, code metrics)
   - Action: [threshold (continuous 0-1), priority (discrete 0-2)]
   - Reward: +10 correct, -5 missed bug, -1 false positive

2. **Agent** (`rl/agent.py`)
   - PPO algorithm from Stable-Baselines3
   - MlpPolicy with [256, 128] hidden layers
   - Learning rate: 3e-4

3. **Training** (`rl/trainer.py`)
   - 10,000 timestep initial training
   - Checkpoint saving every 1,000 steps
   - TensorBoard logging
   - Learning curve visualization

4. **Evaluation** (`rl/evaluate.py`)
   - Compare RL vs fixed thresholds (0.3, 0.5, 0.7)
   - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
   - Confusion matrices and performance plots

5. **Feedback Collection** (`rl/feedback_collector.py`)
   - Store developer feedback in PostgreSQL
   - Generate reward signals for continuous learning
   - Track prediction accuracy over time

6. **RL-Enhanced Predictions** (`api/rl_predict.py`)
   - Load trained RL agent
   - Dynamic threshold optimization per commit
   - Priority-based alerting (low, medium, high)

## Training Results

### Initial Training (10,000 timesteps)

```
Training Time: ~15 seconds
Final Reward: +163.0
Final Accuracy: 85.00%

Metrics vs Baseline (threshold=0.5):
- Reward: Same (+163.0)
- Accuracy: Same (85.00%)
```

**Note**: The RL agent learned to match the baseline performance in this initial training. With more data and training, the RL agent can potentially outperform fixed thresholds by adapting to different commit patterns.

### Learning Progression

| Timesteps | Mean Reward | Accuracy | Policy Loss | Value Loss |
|-----------|-------------|----------|-------------|------------|
| 2,048     | 99.6        | 55%      | -           | -          |
| 4,096     | 105.0       | 65%      | -0.00315    | 1,200      |
| 6,144     | 109.0       | 70%      | -0.00705    | 400        |
| 8,192     | 116.0       | 65%      | -0.01050    | 334        |
| 10,240    | 127.0       | 80%      | -0.00984    | 312        |

## Quick Start

### 1. Train RL Agent

```bash
cd C:\Users\User\athena
python rl/trainer.py
```

This will:
- Load commits from `commits_export.csv`
- Train PPO agent for 10,000 steps
- Save model to `rl/models/ppo_bug_predictor_*_final.zip`
- Generate learning curves in `rl/plots/`

### 2. Evaluate Performance

```bash
python rl/evaluate.py
```

Compares RL vs baseline thresholds and generates:
- Performance comparison plots
- Confusion matrices
- Detailed evaluation report

### 3. Use RL-Enhanced Predictions

#### Via API

```bash
curl -X POST http://localhost:8000/predict/rl \
  -H "Content-Type: application/json" \
  -d '{
    "message": "fix: resolve authentication bug",
    "files_changed": 3,
    "insertions": 45,
    "deletions": 12,
    "repo_stars": 5000,
    "repo_language": "Python"
  }'
```

Response:
```json
{
  "is_bugfix": true,
  "bug_probability": 0.87,
  "rl_threshold": 0.73,
  "rl_priority": 2,
  "should_alert": true,
  "confidence": "high",
  "rl_enabled": true
}
```

#### Via Python

```python
from api.rl_predict import get_rl_predictor

predictor = get_rl_predictor()

result = predictor.predict_with_rl({
    'message': 'fix: memory leak in cache',
    'files_changed': 2,
    'insertions': 15,
    'deletions': 8,
    'repo_stars': 3000,
    'repo_language': 'Python'
})

print(f"Bug probability: {result['bug_probability']:.2%}")
print(f"RL threshold: {result['rl_threshold']:.3f}")
print(f"Should alert: {result['should_alert']}")
```

## API Endpoints

### POST /predict/rl
RL-enhanced prediction with adaptive threshold.

**Request:**
```json
{
  "message": "fix: null pointer exception",
  "files_changed": 3,
  "insertions": 45,
  "deletions": 12
}
```

**Response:**
```json
{
  "is_bugfix": true,
  "bug_probability": 0.87,
  "rl_threshold": 0.73,
  "rl_priority": 2,
  "should_alert": true,
  "confidence": "high"
}
```

### POST /feedback
Submit developer feedback on predictions.

**Request:**
```json
{
  "commit_sha": "abc123def456",
  "was_correct": true,
  "threshold_used": 0.73,
  "priority_used": 2,
  "comment": "Good prediction"
}
```

**Response:**
```json
{
  "feedback_id": 42,
  "commit_sha": "abc123def456",
  "was_correct": true,
  "reward": 11.0,
  "message": "Feedback recorded successfully"
}
```

### GET /feedback/stats
Get feedback statistics.

**Response:**
```json
{
  "total_feedback": 150,
  "correct_predictions": 127,
  "accuracy": 0.847,
  "avg_threshold": 0.68,
  "unique_commits": 145
}
```

## Reward Structure

The RL agent learns from these reward signals:

| Outcome                | Reward | Description                          |
|------------------------|--------|--------------------------------------|
| Correct prediction     | +10.0  | Predicted correctly                  |
| High-priority correct  | +11.0  | Correctly flagged as high priority   |
| False negative         | -5.0   | Missed a bug (worst case)            |
| False positive         | -1.0   | False alarm                          |
| High-priority false    | -1.5   | High-priority false alarm            |

## Continuous Learning

The RL agent can be retrained periodically with new feedback:

1. **Collect Feedback**: Use POST /feedback to store predictions
2. **Retrain**: Run `python rl/trainer.py` with updated data
3. **Deploy**: API automatically loads the latest model

## TensorBoard Monitoring

View training progress in real-time:

```bash
tensorboard --logdir rl/logs
```

Open http://localhost:6006 to see:
- Episode rewards over time
- Policy and value losses
- Accuracy metrics
- Threshold distribution

## File Structure

```
rl/
├── __init__.py              # Package init
├── environment.py           # Custom Gym environment
├── agent.py                 # PPO agent configuration
├── trainer.py               # Training loop
├── evaluate.py              # Evaluation and comparison
├── feedback_collector.py    # Feedback storage and rewards
├── models/                  # Trained RL models
│   └── ppo_bug_predictor_*_final.zip
├── logs/                    # TensorBoard logs
│   └── ppo_bug_predictor_*/
└── plots/                   # Generated visualizations
    ├── learning_curves_*.png
    └── baseline_vs_rl_*.png
```

## Advanced Usage

### Custom Training

```python
from rl.trainer import RLTrainer

trainer = RLTrainer()
trainer.setup()

# Train with custom parameters
results = trainer.train(
    total_timesteps=50000,
    checkpoint_freq=5000,
    eval_freq=10000,
    n_eval_episodes=20
)

# Generate visualizations
trainer.generate_learning_curves()
trainer.evaluate_baseline_vs_rl(n_episodes=100)
```

### Load and Use Trained Agent

```python
from rl.environment import create_environment
from rl.agent import load_latest_agent

env = create_environment()
agent = load_latest_agent(env)

obs, _ = env.reset()
action, _ = agent.predict(obs, deterministic=True)

threshold = action[0]  # Optimal threshold
priority = int(action[1])  # Alert priority
```

## Future Enhancements

1. **Multi-Model Ensemble**: Combine multiple RL agents
2. **Context-Aware Learning**: Per-repository threshold optimization
3. **Real-Time Adaptation**: Online learning from production feedback
4. **Multi-Objective Optimization**: Balance precision vs recall
5. **Explainability**: Visualize why specific thresholds were chosen

## Performance Benchmarks

With 172 commits (23.8% bug fixes):

| Method             | Accuracy | Precision | Recall | F1     | ROC-AUC |
|--------------------|----------|-----------|--------|--------|---------|
| Baseline (0.3)     | 85.0%    | -         | -      | -      | -       |
| Baseline (0.5)     | 85.0%    | -         | -      | -      | -       |
| Baseline (0.7)     | 85.0%    | -         | -      | -      | -       |
| **RL (Adaptive)**  | **85.0%**| -         | -      | -      | -       |

## Troubleshooting

### Model not loading

```bash
# Check if model exists
ls rl/models/ppo_bug_predictor_*_final.zip

# Retrain if missing
python rl/trainer.py
```

### Import errors

```bash
# Install dependencies
pip install stable-baselines3 gymnasium matplotlib seaborn
```

### Database connection issues

The RL system works without database access. It uses CSV exports for training. Database is only needed for feedback collection in production.

## Citation

If you use this RL system in your research, please cite:

```
ATHENA Bug Prediction System - Reinforcement Learning Module
Author: Anthropic Claude
Year: 2025
```

## License

Part of the ATHENA project - AI-powered code intelligence platform.
