"""
ATHENA Bug Prediction Model - Baseline XGBoost

Trains a bug prediction model using commit features from the database.
Predicts whether a commit is a bug fix based on code patterns.
"""
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import joblib
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Database Connection
# ============================================================================

def get_db_connection():
    """Create database connection"""
    db_url = (
        f"postgresql://{os.getenv('POSTGRES_USER', 'athena')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'athena_secure_password_change_me')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'athena')}"
    )
    return create_engine(db_url)

# ============================================================================
# Feature Extraction
# ============================================================================

def extract_commits_from_db():
    """Extract commit data from CSV export"""
    print("=" * 80)
    print("EXTRACTING COMMITS FROM CSV")
    print("=" * 80)

    csv_path = 'commits_export.csv'

    if not os.path.exists(csv_path):
        print(f"\nERROR: {csv_path} not found!")
        print("Export data first with:")
        print('docker exec athena-postgres psql -U athena -d athena -c "\\COPY (SELECT c.id, c.sha, c.author_name, c.author_email, c.message, c.files_changed, c.insertions, c.deletions, c.is_merge, c.committed_at, r.full_name as repo_name, r.language as repo_language, r.stars as repo_stars FROM commits c JOIN repositories r ON c.repository_id = r.id ORDER BY c.committed_at DESC) TO STDOUT CSV HEADER" > commits_export.csv')
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df['committed_at'] = pd.to_datetime(df['committed_at'])

    print(f"\nExtracted {len(df)} commits from CSV")
    print(f"Repositories: {df['repo_name'].nunique()}")
    print(f"Authors: {df['author_name'].nunique()}")
    print(f"Date range: {df['committed_at'].min()} to {df['committed_at'].max()}")

    return df

def create_bug_labels(df):
    """
    Create labels by identifying bug-fix commits from commit messages.

    Bug-fix keywords: fix, bug, error, issue, crash, patch, repair, etc.
    """
    print("\n" + "=" * 80)
    print("CREATING BUG-FIX LABELS")
    print("=" * 80)

    # Bug-fix detection patterns
    bug_patterns = [
        r'\bfix\b', r'\bfixed\b', r'\bfixes\b', r'\bfixing\b',
        r'\bbug\b', r'\bbugs\b',
        r'\berror\b', r'\berrors\b',
        r'\bissue\b', r'\bissues\b',
        r'\bcrash\b', r'\bcrashes\b',
        r'\bpatch\b', r'\bpatched\b',
        r'\brepair\b', r'\brepaired\b',
        r'\bresolve\b', r'\bresolved\b',
        r'\bcorrect\b', r'\bcorrected\b',
    ]

    # Combine patterns
    combined_pattern = '|'.join(bug_patterns)

    # Apply to commit messages (case-insensitive)
    df['is_bugfix'] = df['message'].str.lower().str.contains(
        combined_pattern,
        regex=True,
        na=False
    ).astype(int)

    bugfix_count = df['is_bugfix'].sum()
    non_bugfix_count = len(df) - bugfix_count

    print(f"\nBug-fix commits: {bugfix_count} ({bugfix_count/len(df)*100:.1f}%)")
    print(f"Non-bug commits: {non_bugfix_count} ({non_bugfix_count/len(df)*100:.1f}%)")

    # Show sample bug-fix messages
    print("\nSample bug-fix commit messages:")
    print("-" * 80)
    bugfix_samples = df[df['is_bugfix'] == 1]['message'].head(5)
    for i, msg in enumerate(bugfix_samples, 1):
        print(f"{i}. {msg[:100]}...")

    return df

def engineer_features(df):
    """
    Engineer features for bug prediction model.

    Features:
    - Code churn metrics (insertions, deletions, total changes)
    - Files changed count
    - Commit message characteristics (length, complexity)
    - Temporal features (hour of day, day of week)
    - Repository context (stars, language)
    - Author statistics
    """
    print("\n" + "=" * 80)
    print("ENGINEERING FEATURES")
    print("=" * 80)

    features = df.copy()

    # 1. CODE CHURN FEATURES
    features['total_changes'] = features['insertions'] + features['deletions']
    features['change_ratio'] = np.where(
        features['deletions'] > 0,
        features['insertions'] / features['deletions'],
        features['insertions']
    )
    features['files_per_change'] = np.where(
        features['total_changes'] > 0,
        features['files_changed'] / features['total_changes'],
        0
    )

    # 2. COMMIT MESSAGE FEATURES
    features['msg_length'] = features['message'].str.len()
    features['msg_word_count'] = features['message'].str.split().str.len()
    features['msg_has_question'] = features['message'].str.contains(r'\?', regex=True).astype(int)
    features['msg_has_exclamation'] = features['message'].str.contains(r'!', regex=True).astype(int)
    features['msg_all_caps_ratio'] = features['message'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )

    # 3. TEMPORAL FEATURES
    features['hour'] = pd.to_datetime(features['committed_at']).dt.hour
    features['day_of_week'] = pd.to_datetime(features['committed_at']).dt.dayofweek
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
    features['is_night'] = ((features['hour'] >= 22) | (features['hour'] <= 6)).astype(int)

    # Convert PostgreSQL boolean columns ('t'/'f') to int FIRST
    features['is_merge'] = features['is_merge'].map({'t': 1, 'f': 0, True: 1, False: 0}).fillna(0).astype(int)

    # 4. AUTHOR FEATURES (aggregated)
    author_stats = features.groupby('author_email').agg({
        'total_changes': ['mean', 'std', 'count'],
        'files_changed': 'mean',
        'is_merge': 'sum'  # Now already converted to int
    }).reset_index()
    author_stats.columns = [
        'author_email', 'author_avg_changes', 'author_std_changes',
        'author_commit_count', 'author_avg_files', 'author_total_merges'
    ]
    features = features.merge(author_stats, on='author_email', how='left')

    # Fill NaN values for std (when author has only 1 commit)
    features['author_std_changes'] = features['author_std_changes'].fillna(0)

    # 5. REPOSITORY FEATURES
    # Already have: repo_stars, repo_language
    # One-hot encode language (if multiple languages exist)
    language_dummies = pd.get_dummies(features['repo_language'], prefix='lang')
    features = pd.concat([features, language_dummies], axis=1)

    print(f"\nEngineered {len(features.columns)} total columns")
    print("\nFeature categories:")
    print(f"  - Code churn: total_changes, change_ratio, files_per_change")
    print(f"  - Message: msg_length, msg_word_count, msg_has_question, etc.")
    print(f"  - Temporal: hour, day_of_week, is_weekend, is_night")
    print(f"  - Author: author_avg_changes, author_std_changes, author_commit_count")
    print(f"  - Repository: repo_stars, language (one-hot encoded)")

    return features

def select_feature_columns(df):
    """Select final feature columns for model training"""
    feature_cols = [
        # Code churn
        'insertions', 'deletions', 'total_changes', 'files_changed',
        'change_ratio', 'files_per_change',

        # Commit message
        'msg_length', 'msg_word_count', 'msg_has_question',
        'msg_has_exclamation', 'msg_all_caps_ratio',

        # Temporal
        'hour', 'day_of_week', 'is_weekend', 'is_night',

        # Author stats
        'author_avg_changes', 'author_std_changes',
        'author_commit_count', 'author_avg_files', 'author_total_merges',

        # Repository
        'repo_stars',

        # Merge flag
        'is_merge'
    ]

    # Add language dummy columns
    lang_cols = [col for col in df.columns if col.startswith('lang_')]
    feature_cols.extend(lang_cols)

    return feature_cols

# ============================================================================
# Model Training
# ============================================================================

def train_bug_predictor(df, feature_cols):
    """Train XGBoost bug prediction model"""
    print("\n" + "=" * 80)
    print("TRAINING BUG PREDICTION MODEL")
    print("=" * 80)

    # Prepare features and labels
    X = df[feature_cols].fillna(0)  # Fill any remaining NaNs
    y = df['is_bugfix']

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Label distribution:")
    print(f"  Bug-fixes: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"  Non-bugs:  {len(y) - y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")

    # Train XGBoost model
    print("\nTraining XGBoost classifier...")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print("Training complete!")

    return model, X_train, X_test, y_train, y_test, feature_cols

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_cols):
    """Evaluate model performance"""
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Training metrics
    print("\nTRAINING SET PERFORMANCE:")
    print("-" * 80)
    print(classification_report(y_train, y_train_pred, target_names=['Non-bug', 'Bug-fix']))
    train_auc = roc_auc_score(y_train, y_train_proba)
    print(f"AUC-ROC: {train_auc:.4f}")

    # Test metrics
    print("\nTEST SET PERFORMANCE:")
    print("-" * 80)
    print(classification_report(y_test, y_test_pred, target_names=['Non-bug', 'Bug-fix']))
    test_auc = roc_auc_score(y_test, y_test_proba)
    print(f"AUC-ROC: {test_auc:.4f}")

    # Confusion matrix
    print("\nCONFUSION MATRIX (Test Set):")
    print("-" * 80)
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"              Predicted")
    print(f"              Non-bug  Bug-fix")
    print(f"Actual Non-bug    {cm[0,0]:>4}     {cm[0,1]:>4}")
    print(f"       Bug-fix    {cm[1,0]:>4}     {cm[1,1]:>4}")

    # Feature importance
    print("\n" + "=" * 80)
    print("TOP 15 FEATURE IMPORTANCES")
    print("=" * 80)

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature                        Importance")
    print("-" * 80)
    for idx, row in feature_importance.head(15).iterrows():
        print(f"{row['feature']:30} {row['importance']:.4f}")

    return feature_importance

def save_model(model, feature_cols, feature_importance):
    """Save trained model and metadata"""
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(model_dir, f'bug_predictor_xgboost_{timestamp}.pkl')

    # Save model
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save feature columns
    feature_cols_path = os.path.join(model_dir, f'feature_columns_{timestamp}.pkl')
    joblib.dump(feature_cols, feature_cols_path)
    print(f"Feature columns saved to: {feature_cols_path}")

    # Save feature importance
    importance_path = os.path.join(model_dir, f'feature_importance_{timestamp}.csv')
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")

    print("\nModel artifacts saved successfully!")
    return model_path

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main training pipeline"""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + " " * 20 + "ATHENA BUG PREDICTOR TRAINING" + " " * 29 + "*")
    print("*" + " " * 25 + "Baseline XGBoost Model" + " " * 32 + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print("\n")

    # Step 1: Extract commits
    df = extract_commits_from_db()

    if len(df) == 0:
        print("\nERROR: No commits found in database!")
        print("Run the crawler first: python main_crawler.py")
        return

    # Step 2: Create labels
    df = create_bug_labels(df)

    # Check if we have both classes
    if df['is_bugfix'].sum() == 0:
        print("\nWARNING: No bug-fix commits detected!")
        print("The model will still train, but predictions may be biased.")

    # Step 3: Engineer features
    df = engineer_features(df)

    # Step 4: Select features
    feature_cols = select_feature_columns(df)
    print(f"\nUsing {len(feature_cols)} features for training")

    # Step 5: Train model
    model, X_train, X_test, y_train, y_test, feature_cols = train_bug_predictor(
        df, feature_cols
    )

    # Step 6: Evaluate
    feature_importance = evaluate_model(
        model, X_train, X_test, y_train, y_test, feature_cols
    )

    # Step 7: Save model
    model_path = save_model(model, feature_cols, feature_importance)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModel ready for inference!")
    print(f"Use the model at: {model_path}")
    print("\nNext steps:")
    print("  1. Collect more data with: python main_crawler.py")
    print("  2. Re-train with more commits for better accuracy")
    print("  3. Deploy the model for real-time bug prediction")
    print("\n")

if __name__ == "__main__":
    main()
