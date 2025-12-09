"""
ATHENA RL-Enhanced Predictions

Uses trained RL agent to dynamically optimize prediction thresholds.
Each commit gets a personalized threshold based on its features and context.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.agent import BugPredictionAgent, load_latest_agent
from rl.environment import BugPredictionEnv


class RLPredictor:
    """
    RL-enhanced bug predictor.

    Uses XGBoost for initial probability + RL agent for optimal threshold.
    """

    def __init__(self):
        """Initialize RL predictor."""
        self.xgboost_model = None
        self.feature_columns = None
        self.rl_agent = None
        self.rl_available = False
        self.model_version = None

        self._load_models()

    def _load_models(self):
        """Load XGBoost model and RL agent."""
        # Load XGBoost model
        try:
            model_files = list(Path("models").glob("bug_predictor_xgboost_*.pkl"))
            if not model_files:
                raise FileNotFoundError("No XGBoost model found")

            model_path = max(model_files, key=lambda p: p.stat().st_ctime)
            self.xgboost_model = joblib.load(model_path)
            self.model_version = model_path.stem

            # Load feature columns
            timestamp = '_'.join(model_path.stem.split('_')[-2:])
            feature_cols_path = Path("models") / f"feature_columns_{timestamp}.pkl"
            self.feature_columns = joblib.load(feature_cols_path)

            print(f"Loaded XGBoost model: {model_path.name}")

        except Exception as e:
            print(f"Error loading XGBoost model: {e}")
            raise

        # Try to load RL agent
        try:
            rl_model_path = Path("rl/models")
            if rl_model_path.exists():
                # Load latest RL model
                rl_files = list(rl_model_path.glob("ppo_bug_predictor_*_final.zip"))

                if rl_files:
                    latest_rl = max(rl_files, key=lambda p: p.stat().st_ctime)

                    # Create dummy environment for loading
                    # (we'll override observations manually)
                    from rl.environment import create_environment
                    dummy_env = create_environment()

                    self.rl_agent = BugPredictionAgent.load(str(latest_rl), dummy_env)
                    self.rl_available = True
                    print(f"Loaded RL agent: {latest_rl.name}")
                else:
                    print("No RL model found. Using fixed threshold (0.5).")
                    self.rl_available = False
            else:
                print("RL models directory not found. Using fixed threshold (0.5).")
                self.rl_available = False

        except Exception as e:
            print(f"Warning: Could not load RL agent: {e}")
            print("Falling back to fixed threshold (0.5).")
            self.rl_available = False

    def extract_features(self, commit_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract features from commit data.

        Args:
            commit_data: Dictionary with commit information

        Returns:
            DataFrame with features
        """
        # Code churn features
        insertions = commit_data.get('insertions', 0)
        deletions = commit_data.get('deletions', 0)
        files_changed = commit_data.get('files_changed', 0)
        total_changes = insertions + deletions

        change_ratio = insertions / deletions if deletions > 0 else insertions
        files_per_change = files_changed / total_changes if total_changes > 0 else 0

        # Message features
        message = commit_data.get('message', '')
        msg_length = len(message)
        msg_word_count = len(message.split())
        msg_has_question = 1 if '?' in message else 0
        msg_has_exclamation = 1 if '!' in message else 0
        msg_all_caps_ratio = sum(1 for c in message if c.isupper()) / max(len(message), 1)

        # Temporal features
        hour = commit_data.get('hour', datetime.now().hour)
        day_of_week = commit_data.get('day_of_week', datetime.now().weekday())
        is_weekend = 1 if day_of_week >= 5 else 0
        is_night = 1 if hour >= 22 or hour <= 6 else 0

        # Author features (use defaults for new authors)
        author_avg_changes = commit_data.get('author_avg_changes', total_changes)
        author_std_changes = commit_data.get('author_std_changes', 0)
        author_commit_count = commit_data.get('author_commit_count', 1)
        author_avg_files = commit_data.get('author_avg_files', files_changed)
        author_total_merges = commit_data.get('author_total_merges', 0)

        # Repository features
        repo_stars = commit_data.get('repo_stars', 0)
        is_merge = 1 if commit_data.get('is_merge', False) else 0

        # Language (one-hot encoding)
        repo_language = commit_data.get('repo_language', 'Python')
        lang_python = 1 if repo_language == 'Python' else 0

        # Create feature dictionary
        features = {
            'insertions': insertions,
            'deletions': deletions,
            'total_changes': total_changes,
            'files_changed': files_changed,
            'change_ratio': change_ratio,
            'files_per_change': files_per_change,
            'msg_length': msg_length,
            'msg_word_count': msg_word_count,
            'msg_has_question': msg_has_question,
            'msg_has_exclamation': msg_has_exclamation,
            'msg_all_caps_ratio': msg_all_caps_ratio,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_night': is_night,
            'author_avg_changes': author_avg_changes,
            'author_std_changes': author_std_changes,
            'author_commit_count': author_commit_count,
            'author_avg_files': author_avg_files,
            'author_total_merges': author_total_merges,
            'repo_stars': repo_stars,
            'is_merge': is_merge,
            'lang_Python': lang_python
        }

        # Match training feature columns
        features_df = pd.DataFrame([features])

        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0

        # Select only training columns in correct order
        features_df = features_df[self.feature_columns]

        return features_df

    def predict_with_rl(self, commit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction using RL-enhanced threshold.

        Args:
            commit_data: Commit information

        Returns:
            Prediction result with RL-optimized threshold
        """
        # Extract features
        features_df = self.extract_features(commit_data)

        # Get XGBoost probability
        bug_probability = float(self.xgboost_model.predict_proba(features_df)[0, 1])

        # Get RL-optimized threshold
        if self.rl_available:
            # Create observation for RL agent
            # State: [pred_prob, historical_acc, code_churn_features...]
            observation = self._create_rl_observation(bug_probability, features_df)

            # Get RL action
            action, _ = self.rl_agent.predict(observation, deterministic=True)
            rl_threshold = float(np.clip(action[0], 0.0, 1.0))
            rl_priority = int(np.clip(action[1], 0, 2))
        else:
            # Fallback to fixed threshold
            rl_threshold = 0.5
            rl_priority = 1  # Medium priority

        # Make prediction
        is_bugfix = bug_probability >= rl_threshold

        # Calculate confidence
        confidence = self._calculate_confidence(bug_probability, rl_threshold)

        # Determine if alert should be shown
        should_alert = is_bugfix and rl_priority >= 1

        return {
            'is_bugfix': bool(is_bugfix),
            'bug_probability': bug_probability,
            'rl_threshold': rl_threshold,
            'rl_priority': rl_priority,
            'should_alert': should_alert,
            'confidence': confidence,
            'features_used': features_df.iloc[0].to_dict(),
            'model_version': self.model_version,
            'rl_enabled': self.rl_available,
            'predicted_at': datetime.now().isoformat()
        }

    def _create_rl_observation(
        self,
        bug_probability: float,
        features_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Create observation vector for RL agent.

        Args:
            bug_probability: XGBoost prediction probability
            features_df: Extracted features

        Returns:
            Observation array
        """
        # Historical accuracy (use neutral 0.5 for API predictions)
        historical_acc = 0.5

        # Normalize code churn features
        insertions_norm = min(features_df['insertions'].iloc[0] / 1000.0, 1.0)
        deletions_norm = min(features_df['deletions'].iloc[0] / 1000.0, 1.0)
        files_changed_norm = min(features_df['files_changed'].iloc[0] / 50.0, 1.0)
        total_changes_norm = min(features_df['total_changes'].iloc[0] / 2000.0, 1.0)
        msg_length_norm = min(features_df['msg_length'].iloc[0] / 500.0, 1.0)
        msg_word_count_norm = min(features_df['msg_word_count'].iloc[0] / 100.0, 1.0)
        hour_norm = features_df['hour'].iloc[0] / 24.0
        day_of_week_norm = features_df['day_of_week'].iloc[0] / 7.0

        observation = np.array([
            bug_probability,
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

        return observation

    def _calculate_confidence(self, probability: float, threshold: float) -> str:
        """
        Calculate confidence level.

        Args:
            probability: Prediction probability
            threshold: Decision threshold

        Returns:
            Confidence level: high, medium, or low
        """
        distance_from_threshold = abs(probability - threshold)

        if distance_from_threshold >= 0.3:
            return "high"
        elif distance_from_threshold >= 0.15:
            return "medium"
        else:
            return "low"


# Singleton instance
_rl_predictor = None


def get_rl_predictor() -> RLPredictor:
    """Get singleton RL predictor instance."""
    global _rl_predictor
    if _rl_predictor is None:
        _rl_predictor = RLPredictor()
    return _rl_predictor


# Pydantic models for API

from pydantic import BaseModel, Field


class RLPredictionResponse(BaseModel):
    """Response model for RL-enhanced prediction."""
    is_bugfix: bool
    bug_probability: float = Field(..., ge=0.0, le=1.0)
    rl_threshold: float = Field(..., ge=0.0, le=1.0)
    rl_priority: int = Field(..., ge=0, le=2)
    should_alert: bool
    confidence: str
    features_used: Dict[str, Any]
    model_version: str
    rl_enabled: bool
    predicted_at: str


if __name__ == "__main__":
    # Test RL predictor
    print("=" * 80)
    print("TESTING RL-ENHANCED PREDICTOR")
    print("=" * 80)

    predictor = get_rl_predictor()

    # Test prediction
    test_commit = {
        'message': 'fix: resolve null pointer exception in authentication',
        'files_changed': 3,
        'insertions': 45,
        'deletions': 12,
        'repo_stars': 5000,
        'repo_language': 'Python',
        'hour': 14,
        'day_of_week': 2
    }

    print("\nTest Commit:")
    print(f"  Message: {test_commit['message']}")
    print(f"  Files: {test_commit['files_changed']}, +{test_commit['insertions']}/-{test_commit['deletions']}")

    print("\nMaking RL-enhanced prediction...")
    result = predictor.predict_with_rl(test_commit)

    print("\n" + "=" * 80)
    print("PREDICTION RESULT")
    print("=" * 80)
    print(f"Bug Fix: {result['is_bugfix']}")
    print(f"Bug Probability: {result['bug_probability']:.2%}")
    print(f"RL Threshold: {result['rl_threshold']:.3f}")
    print(f"RL Priority: {result['rl_priority']} ({'Low' if result['rl_priority']==0 else 'Medium' if result['rl_priority']==1 else 'High'})")
    print(f"Should Alert: {result['should_alert']}")
    print(f"Confidence: {result['confidence'].upper()}")
    print(f"RL Enabled: {result['rl_enabled']}")
    print()
