"""
Athena SQLAlchemy ORM Models
Production-grade database models with relationships and validation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, Float,
    DateTime, ForeignKey, JSON, Index, CheckConstraint,
    func, text
)
from sqlalchemy.orm import relationship, declarative_base, validates
from sqlalchemy.dialects.postgresql import JSONB, UUID
import re

Base = declarative_base()


# ============================================================================
# Core Models
# ============================================================================

class Repository(Base):
    """Repository metadata from GitHub/GitLab"""
    __tablename__ = 'repositories'

    id = Column(Integer, primary_key=True)
    owner = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    full_name = Column(String(511), nullable=False, unique=True, index=True)
    description = Column(Text)
    language = Column(String(100), index=True)
    stars = Column(Integer, default=0, index=True)
    forks = Column(Integer, default=0)
    watchers = Column(Integer, default=0)
    open_issues = Column(Integer, default=0)
    size_kb = Column(Integer, default=0)
    default_branch = Column(String(100), default='main')
    is_private = Column(Boolean, default=False)
    is_fork = Column(Boolean, default=False)
    is_archived = Column(Boolean, default=False, index=True)
    license = Column(String(100))
    topics = Column(JSONB, default=list)
    github_created_at = Column(DateTime, nullable=False)
    github_updated_at = Column(DateTime, nullable=False)
    github_pushed_at = Column(DateTime)
    last_crawled_at = Column(DateTime, index=True)
    crawl_status = Column(String(50), default='pending', index=True)
    crawl_error = Column(Text)
    extra_metadata = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    commits = relationship("Commit", back_populates="repository", cascade="all, delete-orphan")
    files = relationship("File", back_populates="repository", cascade="all, delete-orphan")
    issues = relationship("Issue", back_populates="repository", cascade="all, delete-orphan")
    pull_requests = relationship("PullRequest", back_populates="repository", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="repository", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_repos_owner_name', 'owner', 'name', unique=True),
    )

    @validates('full_name')
    def validate_full_name(self, key, full_name):
        """Ensure full_name format is owner/repo"""
        if not re.match(r'^[\w\-\.]+/[\w\-\.]+$', full_name):
            raise ValueError(f"Invalid repository full_name format: {full_name}")
        return full_name

    def __repr__(self):
        return f"<Repository(id={self.id}, full_name='{self.full_name}')>"


class Commit(Base):
    """Commit history with time-series optimization"""
    __tablename__ = 'commits'

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey('repositories.id', ondelete='CASCADE'), nullable=False, index=True)
    sha = Column(String(40), nullable=False, index=True)
    author_name = Column(String(255))
    author_email = Column(String(255))
    author_login = Column(String(255), index=True)
    committer_name = Column(String(255))
    committer_email = Column(String(255))
    message = Column(Text, nullable=False)
    message_subject = Column(String(500))
    files_changed = Column(Integer, default=0)
    insertions = Column(Integer, default=0)
    deletions = Column(Integer, default=0)
    parents = Column(JSONB, default=list)
    committed_at = Column(DateTime, nullable=False, index=True)
    authored_at = Column(DateTime)
    is_merge = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    raw_data = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    repository = relationship("Repository", back_populates="commits")
    commit_files = relationship("CommitFile", back_populates="commit", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_commits_repo_time', 'repository_id', 'committed_at'),
        Index('idx_commits_repo_sha', 'repository_id', 'sha', unique=True),
    )

    @property
    def total_changes(self):
        """Calculate total changes"""
        return (self.insertions or 0) + (self.deletions or 0)

    @validates('sha')
    def validate_sha(self, key, sha):
        """Ensure SHA is valid Git hash"""
        if not re.match(r'^[0-9a-f]{40}$', sha):
            raise ValueError(f"Invalid commit SHA: {sha}")
        return sha

    def __repr__(self):
        return f"<Commit(id={self.id}, sha='{self.sha[:7]}', repo_id={self.repository_id})>"


class File(Base):
    """Individual file tracking across commits"""
    __tablename__ = 'files'

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey('repositories.id', ondelete='CASCADE'), nullable=False, index=True)
    path = Column(Text, nullable=False)
    filename = Column(String(255), nullable=False)
    extension = Column(String(50), index=True)
    directory = Column(Text)
    language = Column(String(100), index=True)
    first_seen_at = Column(DateTime, default=datetime.utcnow)
    last_modified_at = Column(DateTime)
    total_commits = Column(Integer, default=0)
    total_authors = Column(Integer, default=0)
    lines_of_code = Column(Integer)
    complexity_score = Column(Float, index=True)
    is_test_file = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)
    extra_metadata = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    repository = relationship("Repository", back_populates="files")
    commit_files = relationship("CommitFile", back_populates="file", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="file", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_files_repo_path', 'repository_id', 'path', unique=True),
    )

    def __repr__(self):
        return f"<File(id={self.id}, path='{self.path}')>"


class CommitFile(Base):
    """Junction table tracking file changes in commits"""
    __tablename__ = 'commit_files'

    id = Column(Integer, primary_key=True)
    commit_id = Column(Integer, nullable=False, index=True)
    file_id = Column(Integer, ForeignKey('files.id', ondelete='CASCADE'), nullable=False, index=True)
    repository_id = Column(Integer, ForeignKey('repositories.id', ondelete='CASCADE'), nullable=False)
    status = Column(String(20), nullable=False)  # added, modified, deleted, renamed
    additions = Column(Integer, default=0)
    deletions = Column(Integer, default=0)
    changes = Column(Integer, default=0)
    patch = Column(Text)
    previous_filename = Column(Text)
    committed_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    commit = relationship("Commit", back_populates="commit_files")
    file = relationship("File", back_populates="commit_files")

    __table_args__ = (
        Index('idx_commit_files_repo_time', 'repository_id', 'committed_at'),
    )

    def __repr__(self):
        return f"<CommitFile(commit_id={self.commit_id}, file_id={self.file_id}, status='{self.status}')>"


class Issue(Base):
    """GitHub/GitLab issues"""
    __tablename__ = 'issues'

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey('repositories.id', ondelete='CASCADE'), nullable=False, index=True)
    issue_number = Column(Integer, nullable=False)
    title = Column(Text, nullable=False)
    body = Column(Text)
    state = Column(String(20), nullable=False, index=True)
    author_login = Column(String(255))
    assignees = Column(JSONB, default=list)
    labels = Column(JSONB, default=list)
    milestone = Column(String(255))
    comments_count = Column(Integer, default=0)
    is_pull_request = Column(Boolean, default=False)
    closed_by_login = Column(String(255))
    github_created_at = Column(DateTime, nullable=False)
    github_updated_at = Column(DateTime)
    github_closed_at = Column(DateTime)
    raw_data = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    repository = relationship("Repository", back_populates="issues")

    __table_args__ = (
        Index('idx_issues_repo_number', 'repository_id', 'issue_number', unique=True),
    )

    def __repr__(self):
        return f"<Issue(id={self.id}, number={self.issue_number}, state='{self.state}')>"


class PullRequest(Base):
    """GitHub/GitLab pull requests"""
    __tablename__ = 'pull_requests'

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey('repositories.id', ondelete='CASCADE'), nullable=False, index=True)
    pr_number = Column(Integer, nullable=False)
    title = Column(Text, nullable=False)
    body = Column(Text)
    state = Column(String(20), nullable=False, index=True)
    author_login = Column(String(255))
    merged_by_login = Column(String(255))
    base_branch = Column(String(255))
    head_branch = Column(String(255))
    commits_count = Column(Integer, default=0)
    changed_files = Column(Integer, default=0)
    additions = Column(Integer, default=0)
    deletions = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    review_comments_count = Column(Integer, default=0)
    is_draft = Column(Boolean, default=False)
    is_merged = Column(Boolean, default=False, index=True)
    github_created_at = Column(DateTime, nullable=False)
    github_updated_at = Column(DateTime)
    github_merged_at = Column(DateTime)
    github_closed_at = Column(DateTime)
    raw_data = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    repository = relationship("Repository", back_populates="pull_requests")

    __table_args__ = (
        Index('idx_prs_repo_number', 'repository_id', 'pr_number', unique=True),
    )

    def __repr__(self):
        return f"<PullRequest(id={self.id}, number={self.pr_number}, merged={self.is_merged})>"


# ============================================================================
# ML/Prediction Models
# ============================================================================

class Model(Base):
    """ML model registry"""
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, index=True)
    version = Column(String(100), nullable=False)
    model_type = Column(String(100), nullable=False, index=True)
    framework = Column(String(100))
    hyperparameters = Column(JSONB, default=dict)
    metrics = Column(JSONB, default=dict)
    artifact_path = Column(Text)
    mlflow_run_id = Column(String(255))
    status = Column(String(50), default='training', index=True)
    is_production = Column(Boolean, default=False, index=True)
    trained_on_samples = Column(Integer)
    validation_accuracy = Column(Float)
    deployment_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    predictions = relationship("Prediction", back_populates="model")

    __table_args__ = (
        Index('idx_models_name_version', 'name', 'version', unique=True),
    )

    def __repr__(self):
        return f"<Model(id={self.id}, name='{self.name}', version='{self.version}')>"


class Prediction(Base):
    """Model predictions with time-series optimization"""
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey('repositories.id', ondelete='CASCADE'), nullable=False, index=True)
    file_id = Column(Integer, ForeignKey('files.id', ondelete='CASCADE'), index=True)
    model_id = Column(Integer, ForeignKey('models.id'), nullable=False, index=True)
    prediction_type = Column(String(100), nullable=False)
    prediction_value = Column(Float, nullable=False)
    confidence = Column(Float)
    threshold = Column(Float)
    is_alert = Column(Boolean, default=False, index=True)
    features_used = Column(JSONB, default=dict)
    explanation = Column(JSONB, default=dict)
    predicted_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    repository = relationship("Repository", back_populates="predictions")
    file = relationship("File", back_populates="predictions")
    model = relationship("Model", back_populates="predictions")
    feedback = relationship("Feedback", back_populates="prediction", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_predictions_repo_time', 'repository_id', 'predicted_at'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_confidence_range'),
    )

    def __repr__(self):
        return f"<Prediction(id={self.id}, type='{self.prediction_type}', value={self.prediction_value:.3f})>"


class Feedback(Base):
    """Developer feedback for RL training"""
    __tablename__ = 'feedback'

    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey('predictions.id', ondelete='CASCADE'), nullable=False, index=True)
    repository_id = Column(Integer, ForeignKey('repositories.id', ondelete='CASCADE'), nullable=False, index=True)
    user_id = Column(String(255))
    action = Column(String(100), nullable=False, index=True)
    sentiment = Column(String(50))
    ground_truth = Column(Boolean)
    response_time_seconds = Column(Integer)
    comment = Column(Text)
    extra_metadata = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    prediction = relationship("Prediction", back_populates="feedback")

    def __repr__(self):
        return f"<Feedback(id={self.id}, action='{self.action}', prediction_id={self.prediction_id})>"


class RLEpisode(Base):
    """RL training episode history"""
    __tablename__ = 'rl_episodes'

    id = Column(Integer, primary_key=True)
    policy_version = Column(String(100), nullable=False, index=True)
    episode_number = Column(Integer, nullable=False)
    total_reward = Column(Float, nullable=False)
    steps = Column(Integer, nullable=False)
    state_samples = Column(JSONB, default=list)
    action_samples = Column(JSONB, default=list)
    reward_samples = Column(JSONB, default=list)
    metrics = Column(JSONB, default=dict)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<RLEpisode(id={self.id}, policy='{self.policy_version}', reward={self.total_reward:.3f})>"


# ============================================================================
# Feature & Embedding Models
# ============================================================================

class Feature(Base):
    """Extracted features for ML"""
    __tablename__ = 'features'

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey('repositories.id', ondelete='CASCADE'), nullable=False, index=True)
    file_id = Column(Integer, ForeignKey('files.id', ondelete='CASCADE'), index=True)
    feature_type = Column(String(100), nullable=False, index=True)
    feature_vector = Column(JSONB, nullable=False)
    computed_at = Column(DateTime, nullable=False, index=True)
    window_start = Column(DateTime)
    window_end = Column(DateTime)
    extra_metadata = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Feature(id={self.id}, type='{self.feature_type}')>"


class Embedding(Base):
    """Code embeddings metadata (vectors stored in Weaviate)"""
    __tablename__ = 'embeddings'

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey('repositories.id', ondelete='CASCADE'), nullable=False, index=True)
    file_id = Column(Integer, ForeignKey('files.id', ondelete='CASCADE'), index=True)
    commit_id = Column(Integer)
    embedding_type = Column(String(100), nullable=False, index=True)
    model_version = Column(String(100), nullable=False)
    weaviate_id = Column(UUID, index=True)
    embedding_metadata = Column(JSONB, default=dict)
    computed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Embedding(id={self.id}, type='{self.embedding_type}', weaviate_id={self.weaviate_id})>"


# ============================================================================
# Pattern Mining Models
# ============================================================================

class SequentialPattern(Base):
    """Discovered sequential patterns"""
    __tablename__ = 'sequential_patterns'

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey('repositories.id', ondelete='CASCADE'), index=True)
    pattern_type = Column(String(100), nullable=False, index=True)
    pattern = Column(JSONB, nullable=False)
    support = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False, index=True)
    lift = Column(Float)
    occurrences = Column(Integer, nullable=False)
    first_seen_at = Column(DateTime, nullable=False)
    last_seen_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<SequentialPattern(id={self.id}, confidence={self.confidence:.3f})>"


class AssociationRule(Base):
    """Association rules between code changes"""
    __tablename__ = 'association_rules'

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey('repositories.id', ondelete='CASCADE'), index=True)
    rule_type = Column(String(100), nullable=False, index=True)
    antecedent = Column(JSONB, nullable=False)
    consequent = Column(JSONB, nullable=False)
    support = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    lift = Column(Float, nullable=False, index=True)
    conviction = Column(Float)
    occurrences = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<AssociationRule(id={self.id}, lift={self.lift:.3f})>"


# ============================================================================
# Helper Functions
# ============================================================================

def init_db(engine):
    """Initialize database with all tables"""
    Base.metadata.create_all(engine)


def drop_db(engine):
    """Drop all tables (use with caution!)"""
    Base.metadata.drop_all(engine)
