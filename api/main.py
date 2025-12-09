"""
ATHENA Bug Prediction API - Production FastAPI Server

Provides real-time bug prediction endpoints using the trained XGBoost model.
Enhanced with Reinforcement Learning for adaptive threshold optimization.
"""
import os
import sys
import logging
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Add parent directory to path for RL imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api/predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import RL modules (with graceful fallback)
try:
    from api.rl_predict import get_rl_predictor, RLPredictionResponse
    from rl.feedback_collector import (
        FeedbackCollector, FeedbackRequest,
        FeedbackResponse, FeedbackStats
    )
    RL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RL modules not available: {e}")
    RL_AVAILABLE = False

    # Create placeholder classes if RL not available
    class FeedbackRequest(BaseModel):
        pass

    class FeedbackResponse(BaseModel):
        pass

    class FeedbackStats(BaseModel):
        pass

    class RLPredictionResponse(BaseModel):
        pass

# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class CommitData(BaseModel):
    """Single commit data for prediction"""
    message: str = Field(..., description="Commit message", min_length=1)
    files_changed: int = Field(..., ge=0, description="Number of files changed")
    insertions: int = Field(..., ge=0, description="Lines inserted")
    deletions: int = Field(..., ge=0, description="Lines deleted")

    # Optional author context
    author_email: Optional[str] = Field(None, description="Author email for context")
    author_name: Optional[str] = Field(None, description="Author name")

    # Optional repository context
    repo_stars: Optional[int] = Field(0, ge=0, description="Repository stars")
    repo_language: Optional[str] = Field("Python", description="Primary language")

    # Optional temporal context
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of commit (0-23)")
    day_of_week: Optional[int] = Field(None, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")

    # Optional merge flag
    is_merge: Optional[bool] = Field(False, description="Is this a merge commit?")

    @validator('message')
    def message_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Commit message cannot be empty')
        return v

    class Config:
        schema_extra = {
            "example": {
                "message": "fix: resolve null pointer exception in user service",
                "files_changed": 3,
                "insertions": 45,
                "deletions": 12,
                "author_email": "dev@example.com",
                "repo_stars": 1200,
                "repo_language": "Python",
                "hour": 14,
                "day_of_week": 2
            }
        }

class BatchCommitData(BaseModel):
    """Batch of commits for prediction"""
    commits: List[CommitData] = Field(..., max_items=100, description="List of commits (max 100)")

class PredictionResponse(BaseModel):
    """Prediction response for a single commit"""
    is_bugfix: bool
    probability: float = Field(..., ge=0.0, le=1.0)
    confidence: str
    features_used: Dict[str, Any]
    model_version: str
    predicted_at: str

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_commits: int
    bugfix_count: int
    average_probability: float

class ModelInfo(BaseModel):
    """Model metadata"""
    model_path: str
    model_version: str
    training_date: str
    feature_count: int
    features: List[str]
    accuracy_metrics: Dict[str, float]
    model_params: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    uptime_seconds: float
    version: str

# ============================================================================
# Model Manager - Load and Manage XGBoost Model
# ============================================================================

class BugPredictorModel:
    """Manages the XGBoost bug prediction model"""

    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.model_path = None
        self.feature_columns_path = None
        self.feature_importance = None
        self.loaded_at = None
        self.accuracy_metrics = {}

    def load_latest_model(self):
        """Load the most recent trained model"""
        logger.info("Loading latest XGBoost model...")

        models_dir = Path("models")
        if not models_dir.exists():
            raise FileNotFoundError("Models directory not found. Train a model first.")

        # Find most recent model files
        model_files = list(models_dir.glob("bug_predictor_xgboost_*.pkl"))
        if not model_files:
            raise FileNotFoundError("No trained models found. Run train_bug_predictor.py first.")

        # Get most recent model
        self.model_path = max(model_files, key=os.path.getctime)
        timestamp = self.model_path.stem.split('_')[-2] + '_' + self.model_path.stem.split('_')[-1]

        # Load model
        self.model = joblib.load(self.model_path)
        logger.info(f"Loaded model: {self.model_path}")

        # Load feature columns
        feature_cols_file = models_dir / f"feature_columns_{timestamp}.pkl"
        if feature_cols_file.exists():
            self.feature_columns = joblib.load(feature_cols_file)
            logger.info(f"Loaded {len(self.feature_columns)} feature columns")
        else:
            logger.warning("Feature columns file not found, using default features")
            self.feature_columns = self._get_default_features()

        # Load feature importance if available
        importance_file = models_dir / f"feature_importance_{timestamp}.csv"
        if importance_file.exists():
            self.feature_importance = pd.read_csv(importance_file)
            logger.info("Loaded feature importance data")

        self.loaded_at = datetime.now()

        # Set accuracy metrics (these would come from training logs in production)
        self.accuracy_metrics = {
            "test_accuracy": 0.74,
            "test_auc_roc": 0.787,
            "test_f1_bugfix": 0.31,
            "train_accuracy": 0.99
        }

        logger.info("Model loaded successfully!")

    def _get_default_features(self):
        """Default feature columns if file not found"""
        return [
            'insertions', 'deletions', 'total_changes', 'files_changed',
            'change_ratio', 'files_per_change',
            'msg_length', 'msg_word_count', 'msg_has_question',
            'msg_has_exclamation', 'msg_all_caps_ratio',
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            'author_avg_changes', 'author_std_changes',
            'author_commit_count', 'author_avg_files', 'author_total_merges',
            'repo_stars', 'is_merge', 'lang_Python'
        ]

    def extract_features(self, commit: CommitData, author_stats: Optional[Dict] = None) -> pd.DataFrame:
        """
        Extract features from commit data matching training pipeline.

        Args:
            commit: Commit data
            author_stats: Optional precomputed author statistics

        Returns:
            DataFrame with extracted features
        """
        features = {}

        # 1. CODE CHURN FEATURES
        features['insertions'] = commit.insertions
        features['deletions'] = commit.deletions
        features['files_changed'] = commit.files_changed
        features['total_changes'] = commit.insertions + commit.deletions

        # Change ratio
        if commit.deletions > 0:
            features['change_ratio'] = commit.insertions / commit.deletions
        else:
            features['change_ratio'] = float(commit.insertions)

        # Files per change
        if features['total_changes'] > 0:
            features['files_per_change'] = commit.files_changed / features['total_changes']
        else:
            features['files_per_change'] = 0.0

        # 2. COMMIT MESSAGE FEATURES
        msg = commit.message
        features['msg_length'] = len(msg)
        features['msg_word_count'] = len(msg.split())
        features['msg_has_question'] = 1 if '?' in msg else 0
        features['msg_has_exclamation'] = 1 if '!' in msg else 0

        # All caps ratio
        if len(msg) > 0:
            features['msg_all_caps_ratio'] = sum(1 for c in msg if c.isupper()) / len(msg)
        else:
            features['msg_all_caps_ratio'] = 0.0

        # 3. TEMPORAL FEATURES
        if commit.hour is not None:
            features['hour'] = commit.hour
        else:
            features['hour'] = datetime.now().hour

        if commit.day_of_week is not None:
            features['day_of_week'] = commit.day_of_week
        else:
            features['day_of_week'] = datetime.now().weekday()

        features['is_weekend'] = 1 if features['day_of_week'] >= 5 else 0
        features['is_night'] = 1 if features['hour'] >= 22 or features['hour'] <= 6 else 0

        # 4. AUTHOR FEATURES (use defaults if not provided)
        if author_stats:
            features['author_avg_changes'] = author_stats.get('avg_changes', features['total_changes'])
            features['author_std_changes'] = author_stats.get('std_changes', 0.0)
            features['author_commit_count'] = author_stats.get('commit_count', 1)
            features['author_avg_files'] = author_stats.get('avg_files', commit.files_changed)
            features['author_total_merges'] = author_stats.get('total_merges', 0)
        else:
            # Use current commit as baseline
            features['author_avg_changes'] = float(features['total_changes'])
            features['author_std_changes'] = 0.0
            features['author_commit_count'] = 1
            features['author_avg_files'] = float(commit.files_changed)
            features['author_total_merges'] = 0

        # 5. REPOSITORY FEATURES
        features['repo_stars'] = commit.repo_stars or 0
        features['is_merge'] = 1 if commit.is_merge else 0

        # Language one-hot encoding
        features['lang_Python'] = 1 if commit.repo_language == 'Python' else 0

        # Create DataFrame with correct column order
        df = pd.DataFrame([features])

        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        # Select only the features used in training
        df = df[self.feature_columns]

        return df

    def predict(self, commit: CommitData) -> Dict[str, Any]:
        """
        Predict bug probability for a commit.

        Args:
            commit: Commit data

        Returns:
            Prediction result with probability and metadata
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_latest_model() first.")

        # Extract features
        features_df = self.extract_features(commit)

        # Make prediction
        probability = self.model.predict_proba(features_df)[0, 1]  # Probability of bug-fix class
        prediction = int(probability >= 0.5)  # Binary prediction

        # Determine confidence level
        if probability >= 0.8 or probability <= 0.2:
            confidence = "high"
        elif probability >= 0.65 or probability <= 0.35:
            confidence = "medium"
        else:
            confidence = "low"

        # Log prediction
        logger.info(f"Prediction - Bugfix: {prediction}, Prob: {probability:.3f}, "
                   f"Msg: '{commit.message[:50]}...', Files: {commit.files_changed}, "
                   f"+{commit.insertions}/-{commit.deletions}")

        return {
            "is_bugfix": bool(prediction),
            "probability": float(probability),
            "confidence": confidence,
            "features_used": features_df.iloc[0].to_dict(),
            "model_version": self.model_path.stem,
            "predicted_at": datetime.now().isoformat()
        }

# ============================================================================
# FastAPI Application
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="ATHENA Bug Prediction API",
    description="Real-time bug prediction for commits using XGBoost ML model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model_manager = BugPredictorModel()
app_start_time = datetime.now()

# Global RL components (optional)
rl_predictor = None
feedback_collector = None

# ============================================================================
# Startup Event - Load Model
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    global rl_predictor, feedback_collector

    try:
        model_manager.load_latest_model()
        logger.info("API startup complete - model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        logger.warning("API will start but predictions will fail until model is loaded")

    # Try to load RL components
    if RL_AVAILABLE:
        try:
            rl_predictor = get_rl_predictor()
            feedback_collector = FeedbackCollector()
            logger.info(f"RL predictor loaded - RL enabled: {rl_predictor.rl_available}")
        except Exception as e:
            logger.warning(f"Could not load RL components: {e}")
            logger.warning("RL-enhanced predictions will not be available")

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def demo_page():
    """Interactive HTML demo page for testing predictions"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ATHENA Bug Predictor - Demo</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header p { font-size: 1.1em; opacity: 0.9; }
            .content { padding: 40px; }
            .form-group {
                margin-bottom: 25px;
            }
            label {
                display: block;
                font-weight: 600;
                margin-bottom: 8px;
                color: #333;
                font-size: 0.95em;
            }
            input[type="text"], textarea, input[type="number"] {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 1em;
                transition: border-color 0.3s;
            }
            input:focus, textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            textarea {
                min-height: 100px;
                resize: vertical;
                font-family: monospace;
            }
            .row {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 15px;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                font-size: 1.1em;
                font-weight: 600;
                border-radius: 8px;
                cursor: pointer;
                width: 100%;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
            }
            button:active { transform: translateY(0); }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            .result {
                margin-top: 30px;
                padding: 25px;
                border-radius: 12px;
                display: none;
            }
            .result.show { display: block; }
            .result.bugfix {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
            }
            .result.normal {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
            }
            .result h2 { margin-bottom: 15px; font-size: 1.8em; }
            .result-details {
                background: rgba(255,255,255,0.2);
                padding: 15px;
                border-radius: 8px;
                margin-top: 15px;
            }
            .result-details p {
                margin: 8px 0;
                font-size: 1.05em;
            }
            .probability-bar {
                width: 100%;
                height: 30px;
                background: rgba(255,255,255,0.3);
                border-radius: 15px;
                overflow: hidden;
                margin: 15px 0;
            }
            .probability-fill {
                height: 100%;
                background: white;
                transition: width 0.5s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                color: #667eea;
            }
            .examples {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 12px;
            }
            .examples h3 { margin-bottom: 15px; color: #667eea; }
            .example {
                background: white;
                padding: 10px 15px;
                margin: 10px 0;
                border-radius: 8px;
                cursor: pointer;
                transition: transform 0.2s;
                border-left: 4px solid #667eea;
            }
            .example:hover {
                transform: translateX(5px);
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .spinner {
                border: 3px solid rgba(255,255,255,0.3);
                border-top: 3px solid white;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                animation: spin 1s linear infinite;
                display: inline-block;
                margin-left: 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 ATHENA Bug Predictor</h1>
                <p>Real-time ML-powered bug prediction for commits</p>
            </div>

            <div class="content">
                <form id="predictForm">
                    <div class="form-group">
                        <label for="message">Commit Message *</label>
                        <textarea id="message" name="message" placeholder="Enter commit message..." required></textarea>
                    </div>

                    <div class="row">
                        <div class="form-group">
                            <label for="files_changed">Files Changed *</label>
                            <input type="number" id="files_changed" name="files_changed" value="1" min="0" required>
                        </div>
                        <div class="form-group">
                            <label for="insertions">Insertions (+) *</label>
                            <input type="number" id="insertions" name="insertions" value="10" min="0" required>
                        </div>
                        <div class="form-group">
                            <label for="deletions">Deletions (-) *</label>
                            <input type="number" id="deletions" name="deletions" value="5" min="0" required>
                        </div>
                    </div>

                    <div class="row">
                        <div class="form-group">
                            <label for="repo_stars">Repo Stars</label>
                            <input type="number" id="repo_stars" name="repo_stars" value="1000" min="0">
                        </div>
                        <div class="form-group">
                            <label for="author_email">Author Email</label>
                            <input type="text" id="author_email" name="author_email" placeholder="dev@example.com">
                        </div>
                        <div class="form-group">
                            <label for="repo_language">Language</label>
                            <input type="text" id="repo_language" name="repo_language" value="Python">
                        </div>
                    </div>

                    <button type="submit" id="submitBtn">
                        Predict Bug Probability
                    </button>
                </form>

                <div id="result" class="result">
                    <h2 id="resultTitle"></h2>
                    <div class="probability-bar">
                        <div id="probabilityFill" class="probability-fill"></div>
                    </div>
                    <div class="result-details">
                        <p><strong>Confidence:</strong> <span id="confidence"></span></p>
                        <p><strong>Model Version:</strong> <span id="modelVersion"></span></p>
                        <p><strong>Prediction Time:</strong> <span id="predictionTime"></span></p>
                    </div>
                </div>

                <div class="examples">
                    <h3>📝 Example Commits (click to try)</h3>
                    <div class="example" onclick="fillExample('fix: resolve null pointer exception in user service', 3, 45, 12)">
                        <strong>Bug Fix:</strong> "fix: resolve null pointer exception in user service" (3 files, +45/-12)
                    </div>
                    <div class="example" onclick="fillExample('feat: add new dashboard component with charts', 5, 234, 8)">
                        <strong>Feature:</strong> "feat: add new dashboard component with charts" (5 files, +234/-8)
                    </div>
                    <div class="example" onclick="fillExample('chore: update dependencies to latest versions', 1, 2, 2)">
                        <strong>Chore:</strong> "chore: update dependencies to latest versions" (1 file, +2/-2)
                    </div>
                </div>
            </div>
        </div>

        <script>
            const form = document.getElementById('predictForm');
            const resultDiv = document.getElementById('result');
            const submitBtn = document.getElementById('submitBtn');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();

                // Get form data
                const formData = new FormData(form);
                const data = {
                    message: formData.get('message'),
                    files_changed: parseInt(formData.get('files_changed')),
                    insertions: parseInt(formData.get('insertions')),
                    deletions: parseInt(formData.get('deletions')),
                    repo_stars: parseInt(formData.get('repo_stars')) || 0,
                    author_email: formData.get('author_email') || null,
                    repo_language: formData.get('repo_language') || 'Python'
                };

                // Show loading state
                submitBtn.disabled = true;
                submitBtn.innerHTML = 'Predicting<span class="spinner"></span>';
                resultDiv.classList.remove('show');

                try {
                    // Call API
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });

                    if (!response.ok) {
                        throw new Error(`API error: ${response.statusText}`);
                    }

                    const result = await response.json();

                    // Display result
                    displayResult(result);

                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = 'Predict Bug Probability';
                }
            });

            function displayResult(result) {
                const isBugfix = result.is_bugfix;
                const probability = result.probability;

                // Set title and style
                resultDiv.className = 'result show ' + (isBugfix ? 'bugfix' : 'normal');
                document.getElementById('resultTitle').textContent =
                    isBugfix ? '🐛 Potential Bug Fix Detected!' : '✅ Normal Commit';

                // Set probability bar
                const probabilityFill = document.getElementById('probabilityFill');
                probabilityFill.style.width = (probability * 100) + '%';
                probabilityFill.textContent = (probability * 100).toFixed(1) + '%';

                // Set details
                document.getElementById('confidence').textContent = result.confidence.toUpperCase();
                document.getElementById('modelVersion').textContent = result.model_version;
                document.getElementById('predictionTime').textContent =
                    new Date(result.predicted_at).toLocaleString();

                // Scroll to result
                resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }

            function fillExample(message, files, insertions, deletions) {
                document.getElementById('message').value = message;
                document.getElementById('files_changed').value = files;
                document.getElementById('insertions').value = insertions;
                document.getElementById('deletions').value = deletions;

                // Scroll to form
                document.getElementById('message').scrollIntoView({ behavior: 'smooth', block: 'center' });
                document.getElementById('message').focus();
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict", response_model=PredictionResponse)
async def predict_commit(commit: CommitData):
    """
    Predict bug probability for a single commit.

    - **message**: Commit message (required)
    - **files_changed**: Number of files changed (required)
    - **insertions**: Lines inserted (required)
    - **deletions**: Lines deleted (required)
    - **author_email**: Author email for context (optional)
    - **repo_stars**: Repository stars (optional)
    - **repo_language**: Primary language (optional)
    - **hour**: Hour of commit 0-23 (optional)
    - **day_of_week**: Day of week 0-6 (optional)
    - **is_merge**: Is merge commit (optional)

    Returns prediction with probability and confidence level.
    """
    try:
        prediction = model_manager.predict(commit)
        return PredictionResponse(**prediction)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchCommitData):
    """
    Batch prediction for multiple commits (max 100).

    Returns predictions for all commits with summary statistics.
    """
    try:
        predictions = []
        bugfix_count = 0
        total_prob = 0.0

        for commit in batch.commits:
            pred = model_manager.predict(commit)
            predictions.append(PredictionResponse(**pred))
            if pred['is_bugfix']:
                bugfix_count += 1
            total_prob += pred['probability']

        avg_prob = total_prob / len(batch.commits) if batch.commits else 0.0

        logger.info(f"Batch prediction - {len(batch.commits)} commits, "
                   f"{bugfix_count} bugfixes, avg prob: {avg_prob:.3f}")

        return BatchPredictionResponse(
            predictions=predictions,
            total_commits=len(batch.commits),
            bugfix_count=bugfix_count,
            average_probability=avg_prob
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get model metadata and training information.

    Returns model path, version, features, and accuracy metrics.
    """
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get model parameters
    model_params = {
        "n_estimators": int(model_manager.model.n_estimators),
        "max_depth": int(model_manager.model.max_depth),
        "learning_rate": float(model_manager.model.learning_rate),
    }

    # Extract training date from model filename
    model_name = model_manager.model_path.stem
    date_parts = model_name.split('_')[-2:]
    training_date = f"{date_parts[0]}_{date_parts[1]}"

    return ModelInfo(
        model_path=str(model_manager.model_path),
        model_version=model_name,
        training_date=training_date,
        feature_count=len(model_manager.feature_columns),
        features=model_manager.feature_columns,
        accuracy_metrics=model_manager.accuracy_metrics,
        model_params=model_params
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns API status, model status, and uptime.
    """
    uptime = (datetime.now() - app_start_time).total_seconds()

    return HealthResponse(
        status="healthy" if model_manager.model is not None else "degraded",
        model_loaded=model_manager.model is not None,
        uptime_seconds=uptime,
        version="1.0.0"
    )

# ============================================================================
# RL-Enhanced Endpoints
# ============================================================================

@app.post("/predict/rl")
async def predict_commit_rl(commit: CommitData):
    """
    Make RL-enhanced prediction with adaptive threshold.

    Uses Reinforcement Learning agent to dynamically optimize the prediction
    threshold based on commit features and historical context.

    Returns:
        - bug_probability: XGBoost prediction probability
        - rl_threshold: RL agent's optimal threshold for this commit
        - rl_priority: Alert priority (0=low, 1=medium, 2=high)
        - should_alert: Whether to show alert to developer
        - confidence: Prediction confidence level
    """
    if not RL_AVAILABLE or rl_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="RL predictor not available. Using standard /predict endpoint."
        )

    try:
        # Convert CommitData to dict for RL predictor
        commit_dict = commit.dict()

        # Make RL-enhanced prediction
        result = rl_predictor.predict_with_rl(commit_dict)

        logger.info(f"RL Prediction - Bugfix: {result['is_bugfix']}, "
                   f"Prob: {result['bug_probability']:.3f}, "
                   f"Threshold: {result['rl_threshold']:.3f}, "
                   f"Priority: {result['rl_priority']}, "
                   f"Msg: '{commit.message[:50]}...'")

        return result

    except Exception as e:
        logger.error(f"RL prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RL prediction failed: {str(e)}")


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit developer feedback on a prediction.

    Feedback is stored in the database and can be used to:
    - Retrain the RL agent for better threshold optimization
    - Monitor model performance in production
    - Generate reward signals for continuous learning

    Args:
        commit_sha: SHA of the commit
        was_correct: Whether the prediction was correct
        user_label: Corrected label if prediction was wrong
        threshold_used: Threshold that was used
        priority_used: Priority level assigned
        comment: Optional developer comment
    """
    if not RL_AVAILABLE or feedback_collector is None:
        raise HTTPException(
            status_code=503,
            detail="Feedback collection not available"
        )

    try:
        # Store feedback
        feedback_id = feedback_collector.add_feedback(
            commit_sha=feedback.commit_sha,
            prediction_id=feedback.prediction_id,
            was_correct=feedback.was_correct,
            user_label=feedback.user_label,
            threshold_used=feedback.threshold_used,
            priority_used=feedback.priority_used,
            comment=feedback.comment
        )

        # Calculate reward for this feedback
        if feedback.user_label is not None:
            prediction = 1 if feedback.was_correct else (1 - feedback.user_label)
            reward = feedback_collector.calculate_reward(
                prediction=prediction,
                ground_truth=feedback.user_label,
                priority=feedback.priority_used or 1,
                threshold=feedback.threshold_used or 0.5
            )
        else:
            reward = 10.0 if feedback.was_correct else -1.0

        logger.info(f"Feedback submitted - ID: {feedback_id}, "
                   f"Commit: {feedback.commit_sha[:8]}, "
                   f"Correct: {feedback.was_correct}, "
                   f"Reward: {reward:+.1f}")

        return FeedbackResponse(
            feedback_id=feedback_id,
            commit_sha=feedback.commit_sha,
            was_correct=feedback.was_correct,
            reward=reward,
            message="Feedback recorded successfully"
        )

    except Exception as e:
        logger.error(f"Feedback submission error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")


@app.get("/feedback/stats", response_model=FeedbackStats)
async def get_feedback_stats():
    """
    Get statistics on collected feedback.

    Returns:
        - total_feedback: Total number of feedback entries
        - correct_predictions: Number of correct predictions
        - accuracy: Overall prediction accuracy from feedback
        - avg_threshold: Average threshold used
        - unique_commits: Number of unique commits with feedback
    """
    if not RL_AVAILABLE or feedback_collector is None:
        raise HTTPException(
            status_code=503,
            detail="Feedback collection not available"
        )

    try:
        stats = feedback_collector.get_feedback_stats()
        return FeedbackStats(**stats)

    except Exception as e:
        logger.error(f"Feedback stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get feedback stats: {str(e)}")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "path": str(request.url)
        }
    )

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting ATHENA Bug Prediction API...")
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("Demo Page: http://localhost:8000")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
