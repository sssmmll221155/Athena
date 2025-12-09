"""
ATHENA RL Feedback Collector

Collects developer feedback on predictions to generate rewards for RL training.
Stores feedback in PostgreSQL and provides reward signal generation.
"""
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


class FeedbackCollector:
    """
    Collects and manages developer feedback for RL training.

    Stores feedback in PostgreSQL feedback table and generates
    reward signals for the RL agent.
    """

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize feedback collector.

        Args:
            db_url: Database connection URL (uses env vars if None)
        """
        if db_url is None:
            db_url = (
                f"postgresql://{os.getenv('POSTGRES_USER', 'athena')}:"
                f"{os.getenv('POSTGRES_PASSWORD', 'athena_secure_password_change_me')}@"
                f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
                f"{os.getenv('POSTGRES_PORT', '5432')}/"
                f"{os.getenv('POSTGRES_DB', 'athena')}"
            )

        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def add_feedback(
        self,
        commit_sha: str,
        prediction_id: Optional[int],
        was_correct: bool,
        user_label: Optional[bool] = None,
        threshold_used: Optional[float] = None,
        priority_used: Optional[int] = None,
        comment: Optional[str] = None
    ) -> int:
        """
        Add developer feedback for a prediction.

        Args:
            commit_sha: SHA of the commit
            prediction_id: ID of the prediction (if available)
            was_correct: Whether prediction was correct
            user_label: User's corrected label (if prediction was wrong)
            threshold_used: Threshold used for prediction
            priority_used: Priority assigned (0-2)
            comment: Optional user comment

        Returns:
            feedback_id
        """
        with self.Session() as session:
            # Insert feedback using raw SQL
            query = text("""
                INSERT INTO feedback (
                    commit_sha,
                    prediction_id,
                    was_correct,
                    user_label,
                    threshold_used,
                    priority_used,
                    comment,
                    created_at
                ) VALUES (
                    :commit_sha,
                    :prediction_id,
                    :was_correct,
                    :user_label,
                    :threshold_used,
                    :priority_used,
                    :comment,
                    :created_at
                )
                RETURNING id
            """)

            result = session.execute(query, {
                'commit_sha': commit_sha,
                'prediction_id': prediction_id,
                'was_correct': was_correct,
                'user_label': user_label,
                'threshold_used': threshold_used,
                'priority_used': priority_used,
                'comment': comment,
                'created_at': datetime.now()
            })

            feedback_id = result.fetchone()[0]
            session.commit()

            return feedback_id

    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics on collected feedback.

        Returns:
            Dictionary with feedback statistics
        """
        with self.Session() as session:
            query = text("""
                SELECT
                    COUNT(*) as total_feedback,
                    SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct_predictions,
                    AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END) as accuracy,
                    AVG(threshold_used) as avg_threshold,
                    COUNT(DISTINCT commit_sha) as unique_commits
                FROM feedback
            """)

            result = session.execute(query).fetchone()

            return {
                'total_feedback': result[0] or 0,
                'correct_predictions': result[1] or 0,
                'accuracy': float(result[2] or 0.0),
                'avg_threshold': float(result[3] or 0.5),
                'unique_commits': result[4] or 0
            }

    def get_recent_feedback(self, limit: int = 100) -> pd.DataFrame:
        """
        Get recent feedback entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            DataFrame with feedback
        """
        query = text("""
            SELECT
                id,
                commit_sha,
                was_correct,
                user_label,
                threshold_used,
                priority_used,
                comment,
                created_at
            FROM feedback
            ORDER BY created_at DESC
            LIMIT :limit
        """)

        with self.Session() as session:
            df = pd.read_sql(query, session.connection(), params={'limit': limit})

        return df

    def calculate_reward(
        self,
        prediction: int,
        ground_truth: int,
        priority: int = 1,
        threshold: float = 0.5
    ) -> float:
        """
        Calculate reward for a prediction (same as environment).

        Args:
            prediction: Predicted label (0 or 1)
            ground_truth: True label (0 or 1)
            priority: Priority level (0-2)
            threshold: Threshold used

        Returns:
            Reward value
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
            # False positive
            reward = -1.0

            # Extra penalty for high-priority false alarms
            if priority == 2:
                reward -= 0.5

        return reward

    def simulate_feedback_from_labels(
        self,
        predictions_df: pd.DataFrame,
        commit_shas: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Simulate developer feedback from ground truth labels.

        Used for initial training when real feedback is not available.

        Args:
            predictions_df: DataFrame with columns [prediction, ground_truth, threshold, priority]
            commit_shas: List of commit SHAs corresponding to predictions

        Returns:
            List of feedback dictionaries
        """
        feedback_list = []

        for idx, row in predictions_df.iterrows():
            prediction = row['prediction']
            ground_truth = row['ground_truth']
            threshold = row.get('threshold', 0.5)
            priority = row.get('priority', 1)

            was_correct = (prediction == ground_truth)

            feedback = {
                'commit_sha': commit_shas[idx] if idx < len(commit_shas) else f"sim_{idx}",
                'prediction_id': None,
                'was_correct': was_correct,
                'user_label': ground_truth,
                'threshold_used': threshold,
                'priority_used': priority,
                'comment': 'Simulated feedback from ground truth',
                'reward': self.calculate_reward(prediction, ground_truth, priority, threshold)
            }

            feedback_list.append(feedback)

        return feedback_list

    def create_feedback_table(self):
        """
        Create feedback table if it doesn't exist.

        This table stores developer feedback on predictions.
        """
        create_table_sql = text("""
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                commit_sha VARCHAR(40),
                prediction_id INTEGER,
                was_correct BOOLEAN NOT NULL,
                user_label BOOLEAN,
                threshold_used FLOAT,
                priority_used INTEGER,
                comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_commit_sha ON feedback(commit_sha);
            CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);
        """)

        with self.engine.begin() as conn:
            conn.execute(create_table_sql)

        print("Feedback table created successfully!")


# API Pydantic models for FastAPI integration

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    commit_sha: str = Field(..., description="SHA of the commit")
    prediction_id: Optional[int] = Field(None, description="ID of the prediction")
    was_correct: bool = Field(..., description="Was the prediction correct?")
    user_label: Optional[bool] = Field(None, description="Corrected label if wrong")
    threshold_used: Optional[float] = Field(None, ge=0.0, le=1.0, description="Threshold used")
    priority_used: Optional[int] = Field(None, ge=0, le=2, description="Priority level")
    comment: Optional[str] = Field(None, max_length=1000, description="Optional comment")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    feedback_id: int
    commit_sha: str
    was_correct: bool
    reward: float
    message: str


class FeedbackStats(BaseModel):
    """Model for feedback statistics."""
    total_feedback: int
    correct_predictions: int
    accuracy: float
    avg_threshold: float
    unique_commits: int


def get_feedback_collector() -> FeedbackCollector:
    """Get singleton feedback collector instance."""
    return FeedbackCollector()


if __name__ == "__main__":
    # Test feedback collector
    print("=" * 80)
    print("TESTING FEEDBACK COLLECTOR")
    print("=" * 80)

    collector = FeedbackCollector()

    # Create table
    print("\nCreating feedback table...")
    collector.create_feedback_table()

    # Simulate some feedback
    print("\nSimulating feedback...")
    test_predictions = pd.DataFrame({
        'prediction': [1, 0, 1, 0, 1],
        'ground_truth': [1, 0, 0, 1, 1],
        'threshold': [0.7, 0.4, 0.6, 0.5, 0.8],
        'priority': [2, 1, 1, 2, 2]
    })

    test_shas = [f"abc123def456{i}" for i in range(5)]

    feedback_list = collector.simulate_feedback_from_labels(test_predictions, test_shas)

    print("\nGenerated feedback:")
    for i, fb in enumerate(feedback_list, 1):
        print(f"{i}. Correct: {fb['was_correct']}, Reward: {fb['reward']:+.1f}")

    # Get stats
    print("\nFeedback statistics:")
    stats = collector.get_feedback_stats()
    print(f"  Total feedback: {stats['total_feedback']}")
    print(f"  Accuracy: {stats['accuracy']:.2%}")
    print(f"  Avg threshold: {stats['avg_threshold']:.3f}")

    print("\nFeedback collector ready!")
