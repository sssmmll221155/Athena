"""
ATHENA RL Environment - Custom Gym Environment for Bug Prediction Optimization

State Space: [prediction_confidence, historical_accuracy, code_churn_features]
Action Space: [prediction_threshold (continuous 0-1), alert_priority (0-2)]
Reward: +10 correct prediction, -5 missed bug, -1 false positive
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional


class BugPredictionEnv(gym.Env):
    """
    Custom Gymnasium environment for learning optimal bug prediction thresholds.

    The agent learns to dynamically adjust prediction thresholds based on:
    - Current prediction confidence
    - Historical accuracy
    - Code churn features

    This creates a self-optimizing system that adapts to different commit patterns.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        commits_data: pd.DataFrame,
        model_predictions: pd.DataFrame,
        episode_length: int = 20,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the bug prediction environment.

        Args:
            commits_data: DataFrame with commit features
            model_predictions: DataFrame with XGBoost predictions and ground truth
            episode_length: Number of predictions per episode
            render_mode: Rendering mode (not used, kept for compatibility)
        """
        super().__init__()

        self.commits_data = commits_data.reset_index(drop=True)
        self.model_predictions = model_predictions.reset_index(drop=True)
        self.episode_length = episode_length
        self.render_mode = render_mode

        # Verify data alignment
        assert len(self.commits_data) == len(self.model_predictions), \
            "Commits and predictions must have same length"

        # State space: 10 features
        # [0] prediction_probability (0-1)
        # [1] historical_accuracy (0-1)
        # [2-9] code churn features (normalized)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32
        )

        # Action space: 2 dimensions
        # [0] prediction_threshold (continuous 0-1)
        # [1] alert_priority (discrete 0-2: low, medium, high)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 2.0]),
            dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.episode_rewards = []
        self.episode_accuracy = []
        self.historical_accuracy_window = []

        # Statistics
        self.total_correct = 0
        self.total_predictions = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset episode state
        self.current_step = 0
        self.episode_rewards = []
        self.episode_accuracy = []

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: [threshold, priority] where threshold ∈ [0,1], priority ∈ [0,2]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Parse action
        threshold = float(np.clip(action[0], 0.0, 1.0))
        priority = int(np.clip(action[1], 0, 2))

        # Get current prediction and ground truth
        idx = self.current_step % len(self.model_predictions)
        pred_prob = self.model_predictions.iloc[idx]['probability']
        ground_truth = self.model_predictions.iloc[idx]['is_bugfix']

        # Make prediction based on RL threshold
        prediction = 1 if pred_prob >= threshold else 0

        # Calculate reward
        reward = self._calculate_reward(prediction, ground_truth, priority)

        # Update statistics
        is_correct = (prediction == ground_truth)
        self.episode_accuracy.append(is_correct)
        self.historical_accuracy_window.append(is_correct)
        if len(self.historical_accuracy_window) > 100:
            self.historical_accuracy_window.pop(0)

        self.total_correct += is_correct
        self.total_predictions += 1

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= self.episode_length
        truncated = False

        # Get next observation
        obs = self._get_observation()
        info = self._get_info()
        info['prediction'] = prediction
        info['ground_truth'] = ground_truth
        info['threshold'] = threshold
        info['priority'] = priority

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation.

        Returns:
            State vector with 10 features
        """
        idx = self.current_step % len(self.model_predictions)

        # Get prediction probability
        pred_prob = self.model_predictions.iloc[idx]['probability']

        # Calculate historical accuracy (moving average)
        if len(self.historical_accuracy_window) > 0:
            historical_acc = np.mean(self.historical_accuracy_window)
        else:
            historical_acc = 0.5  # Neutral starting point

        # Get code churn features (normalized)
        commit = self.commits_data.iloc[idx]

        # Normalize features to [0, 1]
        insertions_norm = min(commit.get('insertions', 0) / 1000.0, 1.0)
        deletions_norm = min(commit.get('deletions', 0) / 1000.0, 1.0)
        files_changed_norm = min(commit.get('files_changed', 0) / 50.0, 1.0)
        total_changes_norm = min(commit.get('total_changes', 0) / 2000.0, 1.0)
        msg_length_norm = min(commit.get('msg_length', 0) / 500.0, 1.0)
        msg_word_count_norm = min(commit.get('msg_word_count', 0) / 100.0, 1.0)
        hour_norm = commit.get('hour', 12) / 24.0
        day_of_week_norm = commit.get('day_of_week', 3) / 7.0

        # Construct state vector
        state = np.array([
            pred_prob,
            historical_acc,
            insertions_norm,
            deletions_norm,
            files_changed_norm,
            total_changes_norm,
            msg_length_norm,
            msg_word_count_norm,
            hour_norm,
            day_of_week_norm
        ], dtype=np.float32)

        return state

    def _calculate_reward(self, prediction: int, ground_truth: int, priority: int) -> float:
        """
        Calculate reward based on prediction outcome.

        Reward structure:
        - Correct prediction: +10
        - Missed bug (false negative): -5
        - False positive: -1
        - Priority bonus: +1 for correct high-priority alerts

        Args:
            prediction: Agent's prediction (0 or 1)
            ground_truth: True label (0 or 1)
            priority: Alert priority (0=low, 1=medium, 2=high)

        Returns:
            reward (float)
        """
        if prediction == ground_truth:
            # Correct prediction
            reward = 10.0

            # Bonus for high-priority correct bug predictions
            if ground_truth == 1 and priority == 2:
                reward += 1.0

        elif prediction == 0 and ground_truth == 1:
            # Missed bug (false negative) - most costly
            reward = -5.0

        else:  # prediction == 1 and ground_truth == 0
            # False positive - less costly but still penalized
            reward = -1.0

            # Extra penalty for high-priority false alarms
            if priority == 2:
                reward -= 0.5

        return reward

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dictionary."""
        if len(self.episode_accuracy) > 0:
            episode_acc = np.mean(self.episode_accuracy)
        else:
            episode_acc = 0.0

        return {
            'step': self.current_step,
            'episode_accuracy': episode_acc,
            'total_accuracy': self.total_correct / max(self.total_predictions, 1)
        }

    def render(self):
        """Render environment (console output)."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, Episode Acc: {np.mean(self.episode_accuracy):.2%}")


def create_environment(commits_csv_path: str = "commits_export.csv") -> BugPredictionEnv:
    """
    Create bug prediction environment from exported commit data.

    Args:
        commits_csv_path: Path to CSV export from PostgreSQL

    Returns:
        BugPredictionEnv instance
    """
    print("=" * 80)
    print("CREATING RL ENVIRONMENT")
    print("=" * 80)

    # Load commits data
    print(f"\nLoading commits from: {commits_csv_path}")
    commits_df = pd.read_csv(commits_csv_path)
    print(f"Loaded {len(commits_df)} commits")

    # Load XGBoost model to generate predictions
    print("\nLoading XGBoost model for predictions...")
    model_files = list(Path("models").glob("bug_predictor_xgboost_*.pkl"))
    if not model_files:
        raise FileNotFoundError("No trained XGBoost model found. Train first with train_bug_predictor.py")

    model_path = max(model_files, key=lambda p: p.stat().st_ctime)
    model = joblib.load(model_path)
    print(f"Loaded model: {model_path.name}")

    # Load feature columns
    timestamp = model_path.stem.split('_')[-2] + '_' + model_path.stem.split('_')[-1]
    feature_cols_path = Path("models") / f"feature_columns_{timestamp}.pkl"
    feature_cols = joblib.load(feature_cols_path)

    # Extract features (same as training)
    print("\nExtracting features...")
    from train_bug_predictor import engineer_features, create_bug_labels

    commits_df = create_bug_labels(commits_df)
    features_df = engineer_features(commits_df)

    # Generate predictions
    print("Generating XGBoost predictions...")
    X = features_df[feature_cols].fillna(0)
    y_true = features_df['is_bugfix'].values
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'probability': y_pred_proba,
        'is_bugfix': y_true
    })

    print(f"\nPredictions generated:")
    print(f"  - Mean probability: {y_pred_proba.mean():.3f}")
    print(f"  - Bugfix rate: {y_true.mean():.1%}")

    # Create environment
    env = BugPredictionEnv(
        commits_data=features_df,
        model_predictions=predictions_df,
        episode_length=20
    )

    print("\nEnvironment created successfully!")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Episode length: {env.episode_length}")

    return env


if __name__ == "__main__":
    # Test environment
    env = create_environment()

    print("\n" + "=" * 80)
    print("TESTING ENVIRONMENT")
    print("=" * 80)

    # Run one episode
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")

    total_reward = 0
    for step in range(env.episode_length):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if step < 3:  # Show first 3 steps
            print(f"\nStep {step + 1}:")
            print(f"  Action (threshold, priority): [{action[0]:.3f}, {action[1]:.0f}]")
            print(f"  Reward: {reward:+.1f}")
            print(f"  Prediction: {info['prediction']}, Ground truth: {info['ground_truth']}")

    print(f"\nEpisode complete!")
    print(f"  Total reward: {total_reward:+.1f}")
    print(f"  Episode accuracy: {info['episode_accuracy']:.2%}")
