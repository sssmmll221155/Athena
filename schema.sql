-- ============================================================================
-- Athena Database Schema v1.0
-- PostgreSQL 16 + TimescaleDB
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search
CREATE EXTENSION IF NOT EXISTS btree_gin;  -- For composite indexes

-- ============================================================================
-- Core Tables
-- ============================================================================

-- Repositories
CREATE TABLE IF NOT EXISTS repositories (
    id SERIAL PRIMARY KEY,
    owner VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    full_name VARCHAR(511) NOT NULL UNIQUE,
    description TEXT,
    language VARCHAR(100),
    stars INTEGER DEFAULT 0,
    forks INTEGER DEFAULT 0,
    watchers INTEGER DEFAULT 0,
    open_issues INTEGER DEFAULT 0,
    size_kb INTEGER DEFAULT 0,
    default_branch VARCHAR(100) DEFAULT 'main',
    is_private BOOLEAN DEFAULT FALSE,
    is_fork BOOLEAN DEFAULT FALSE,
    is_archived BOOLEAN DEFAULT FALSE,
    license VARCHAR(100),
    topics JSONB DEFAULT '[]'::jsonb,
    github_created_at TIMESTAMP NOT NULL,
    github_updated_at TIMESTAMP NOT NULL,
    github_pushed_at TIMESTAMP,
    last_crawled_at TIMESTAMP,
    crawl_status VARCHAR(50) DEFAULT 'pending',
    crawl_error TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_repo UNIQUE(owner, name)
);

CREATE INDEX idx_repos_full_name ON repositories(full_name);
CREATE INDEX idx_repos_language ON repositories(language);
CREATE INDEX idx_repos_stars ON repositories(stars DESC);
CREATE INDEX idx_repos_crawl_status ON repositories(crawl_status);
CREATE INDEX idx_repos_last_crawled ON repositories(last_crawled_at DESC);
CREATE INDEX idx_repos_topics ON repositories USING gin(topics);

-- Commits (Time-Series Optimized)
CREATE TABLE IF NOT EXISTS commits (
    id SERIAL,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    sha VARCHAR(40) NOT NULL,
    author_name VARCHAR(255),
    author_email VARCHAR(255),
    author_login VARCHAR(255),  -- GitHub username
    committer_name VARCHAR(255),
    committer_email VARCHAR(255),
    message TEXT NOT NULL,
    message_subject VARCHAR(500),
    files_changed INTEGER DEFAULT 0,
    insertions INTEGER DEFAULT 0,
    deletions INTEGER DEFAULT 0,
    total_changes INTEGER GENERATED ALWAYS AS (insertions + deletions) STORED,
    parents JSONB DEFAULT '[]'::jsonb,  -- Parent commit SHAs
    committed_at TIMESTAMP NOT NULL,
    authored_at TIMESTAMP,
    is_merge BOOLEAN DEFAULT FALSE,
    is_verified BOOLEAN DEFAULT FALSE,
    raw_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id, committed_at),
    CONSTRAINT unique_commit UNIQUE(repository_id, sha)
);

-- Convert to hypertable (TimescaleDB)
SELECT create_hypertable('commits', 'committed_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Indexes for commits
CREATE INDEX idx_commits_repo ON commits(repository_id, committed_at DESC);
CREATE INDEX idx_commits_sha ON commits(sha);
CREATE INDEX idx_commits_author ON commits(author_login);
CREATE INDEX idx_commits_message ON commits USING gin(to_tsvector('english', message));

-- Files (Track individual files across commits)
CREATE TABLE IF NOT EXISTS files (
    id SERIAL PRIMARY KEY,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    extension VARCHAR(50),
    directory TEXT,
    language VARCHAR(100),
    first_seen_at TIMESTAMP DEFAULT NOW(),
    last_modified_at TIMESTAMP,
    total_commits INTEGER DEFAULT 0,
    total_authors INTEGER DEFAULT 0,
    lines_of_code INTEGER,
    complexity_score FLOAT,
    is_test_file BOOLEAN DEFAULT FALSE,
    is_deleted BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_file UNIQUE(repository_id, path)
);

CREATE INDEX idx_files_repo ON files(repository_id);
CREATE INDEX idx_files_path ON files USING gin(path gin_trgm_ops);
CREATE INDEX idx_files_extension ON files(extension);
CREATE INDEX idx_files_language ON files(language);
CREATE INDEX idx_files_complexity ON files(complexity_score DESC NULLS LAST);

-- Commit Files (Junction table with changes)
CREATE TABLE IF NOT EXISTS commit_files (
    id SERIAL PRIMARY KEY,
    commit_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL,  -- added, modified, deleted, renamed
    additions INTEGER DEFAULT 0,
    deletions INTEGER DEFAULT 0,
    changes INTEGER DEFAULT 0,
    patch TEXT,
    previous_filename TEXT,
    committed_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_commit_files_commit ON commit_files(commit_id);
CREATE INDEX idx_commit_files_file ON commit_files(file_id);
CREATE INDEX idx_commit_files_repo_time ON commit_files(repository_id, committed_at DESC);

-- Issues
CREATE TABLE IF NOT EXISTS issues (
    id SERIAL PRIMARY KEY,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    issue_number INTEGER NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    state VARCHAR(20) NOT NULL,  -- open, closed
    author_login VARCHAR(255),
    assignees JSONB DEFAULT '[]'::jsonb,
    labels JSONB DEFAULT '[]'::jsonb,
    milestone VARCHAR(255),
    comments_count INTEGER DEFAULT 0,
    is_pull_request BOOLEAN DEFAULT FALSE,
    closed_by_login VARCHAR(255),
    github_created_at TIMESTAMP NOT NULL,
    github_updated_at TIMESTAMP,
    github_closed_at TIMESTAMP,
    raw_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_issue UNIQUE(repository_id, issue_number)
);

CREATE INDEX idx_issues_repo ON issues(repository_id);
CREATE INDEX idx_issues_state ON issues(state);
CREATE INDEX idx_issues_labels ON issues USING gin(labels);
CREATE INDEX idx_issues_created ON issues(github_created_at DESC);
CREATE INDEX idx_issues_title_body ON issues USING gin(to_tsvector('english', title || ' ' || COALESCE(body, '')));

-- Pull Requests
CREATE TABLE IF NOT EXISTS pull_requests (
    id SERIAL PRIMARY KEY,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    pr_number INTEGER NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    state VARCHAR(20) NOT NULL,  -- open, closed, merged
    author_login VARCHAR(255),
    merged_by_login VARCHAR(255),
    base_branch VARCHAR(255),
    head_branch VARCHAR(255),
    commits_count INTEGER DEFAULT 0,
    changed_files INTEGER DEFAULT 0,
    additions INTEGER DEFAULT 0,
    deletions INTEGER DEFAULT 0,
    comments_count INTEGER DEFAULT 0,
    review_comments_count INTEGER DEFAULT 0,
    is_draft BOOLEAN DEFAULT FALSE,
    is_merged BOOLEAN DEFAULT FALSE,
    github_created_at TIMESTAMP NOT NULL,
    github_updated_at TIMESTAMP,
    github_merged_at TIMESTAMP,
    github_closed_at TIMESTAMP,
    raw_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_pr UNIQUE(repository_id, pr_number)
);

CREATE INDEX idx_prs_repo ON pull_requests(repository_id);
CREATE INDEX idx_prs_state ON pull_requests(state);
CREATE INDEX idx_prs_merged ON pull_requests(is_merged);
CREATE INDEX idx_prs_created ON pull_requests(github_created_at DESC);

-- ============================================================================
-- Feature Tables
-- ============================================================================

-- Extracted Features (for ML)
CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    feature_type VARCHAR(100) NOT NULL,  -- temporal, code_metrics, graph, etc.
    feature_vector JSONB NOT NULL,
    computed_at TIMESTAMP NOT NULL,
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_features_repo ON features(repository_id);
CREATE INDEX idx_features_file ON features(file_id);
CREATE INDEX idx_features_type ON features(feature_type);
CREATE INDEX idx_features_computed ON features(computed_at DESC);

-- Code Embeddings (stored separately from vectors in Weaviate)
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    commit_id INTEGER,
    embedding_type VARCHAR(100) NOT NULL,  -- codebert, graphcodebert, custom
    model_version VARCHAR(100) NOT NULL,
    weaviate_id UUID,  -- Reference to vector in Weaviate
    embedding_metadata JSONB DEFAULT '{}'::jsonb,
    computed_at TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_embeddings_repo ON embeddings(repository_id);
CREATE INDEX idx_embeddings_file ON embeddings(file_id);
CREATE INDEX idx_embeddings_type ON embeddings(embedding_type);
CREATE INDEX idx_embeddings_weaviate ON embeddings(weaviate_id);

-- ============================================================================
-- ML/Prediction Tables
-- ============================================================================

-- Model Registry
CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(100) NOT NULL,
    model_type VARCHAR(100) NOT NULL,  -- xgboost, lightgbm, gnn, rl_policy
    framework VARCHAR(100),  -- pytorch, sklearn, etc.
    hyperparameters JSONB DEFAULT '{}'::jsonb,
    metrics JSONB DEFAULT '{}'::jsonb,
    artifact_path TEXT,
    mlflow_run_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'training',  -- training, deployed, archived
    is_production BOOLEAN DEFAULT FALSE,
    trained_on_samples INTEGER,
    validation_accuracy FLOAT,
    deployment_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_model_version UNIQUE(name, version)
);

CREATE INDEX idx_models_name ON models(name);
CREATE INDEX idx_models_type ON models(model_type);
CREATE INDEX idx_models_status ON models(status);
CREATE INDEX idx_models_production ON models(is_production);

-- Predictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    model_id INTEGER NOT NULL REFERENCES models(id),
    prediction_type VARCHAR(100) NOT NULL,  -- bug_probability, churn_risk, etc.
    prediction_value FLOAT NOT NULL,
    confidence FLOAT,
    threshold FLOAT,
    is_alert BOOLEAN DEFAULT FALSE,
    features_used JSONB DEFAULT '{}'::jsonb,
    explanation JSONB DEFAULT '{}'::jsonb,  -- SHAP values, etc.
    predicted_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Convert to hypertable for time-series predictions
SELECT create_hypertable('predictions', 'predicted_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_predictions_repo ON predictions(repository_id, predicted_at DESC);
CREATE INDEX idx_predictions_file ON predictions(file_id);
CREATE INDEX idx_predictions_model ON predictions(model_id);
CREATE INDEX idx_predictions_alert ON predictions(is_alert, predicted_at DESC);

-- Feedback (for RL)
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,  -- accepted, dismissed, fixed, false_positive
    sentiment VARCHAR(50),  -- positive, negative, neutral
    ground_truth BOOLEAN,  -- Actual outcome if known
    response_time_seconds INTEGER,
    comment TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_feedback_prediction ON feedback(prediction_id);
CREATE INDEX idx_feedback_repo ON feedback(repository_id);
CREATE INDEX idx_feedback_action ON feedback(action);
CREATE INDEX idx_feedback_created ON feedback(created_at DESC);

-- RL Training Episodes
CREATE TABLE IF NOT EXISTS rl_episodes (
    id SERIAL PRIMARY KEY,
    policy_version VARCHAR(100) NOT NULL,
    episode_number INTEGER NOT NULL,
    total_reward FLOAT NOT NULL,
    steps INTEGER NOT NULL,
    state_samples JSONB DEFAULT '[]'::jsonb,
    action_samples JSONB DEFAULT '[]'::jsonb,
    reward_samples JSONB DEFAULT '[]'::jsonb,
    metrics JSONB DEFAULT '{}'::jsonb,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_rl_episodes_policy ON rl_episodes(policy_version);
CREATE INDEX idx_rl_episodes_completed ON rl_episodes(completed_at DESC);

-- ============================================================================
-- Pattern Mining Tables
-- ============================================================================

-- Sequential Patterns
CREATE TABLE IF NOT EXISTS sequential_patterns (
    id SERIAL PRIMARY KEY,
    repository_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
    pattern_type VARCHAR(100) NOT NULL,  -- file_sequence, bug_sequence
    pattern JSONB NOT NULL,  -- Array of events
    support FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    lift FLOAT,
    occurrences INTEGER NOT NULL,
    first_seen_at TIMESTAMP NOT NULL,
    last_seen_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_seq_patterns_repo ON sequential_patterns(repository_id);
CREATE INDEX idx_seq_patterns_type ON sequential_patterns(pattern_type);
CREATE INDEX idx_seq_patterns_confidence ON sequential_patterns(confidence DESC);

-- Association Rules
CREATE TABLE IF NOT EXISTS association_rules (
    id SERIAL PRIMARY KEY,
    repository_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
    rule_type VARCHAR(100) NOT NULL,
    antecedent JSONB NOT NULL,
    consequent JSONB NOT NULL,
    support FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    lift FLOAT NOT NULL,
    conviction FLOAT,
    occurrences INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_assoc_rules_repo ON association_rules(repository_id);
CREATE INDEX idx_assoc_rules_type ON association_rules(rule_type);
CREATE INDEX idx_assoc_rules_lift ON association_rules(lift DESC);

-- ============================================================================
-- Analytics Views
-- ============================================================================

-- Repository Health Score
CREATE OR REPLACE VIEW repository_health AS
SELECT
    r.id,
    r.full_name,
    r.language,
    r.stars,
    COUNT(DISTINCT c.id) as total_commits,
    COUNT(DISTINCT c.author_login) as total_contributors,
    COUNT(DISTINCT CASE WHEN i.state = 'open' THEN i.id END) as open_issues,
    COUNT(DISTINCT CASE WHEN p.is_merged THEN p.id END) as merged_prs,
    AVG(CASE WHEN pred.prediction_type = 'bug_probability'
        THEN pred.prediction_value END) as avg_bug_risk,
    MAX(c.committed_at) as last_commit_at,
    r.last_crawled_at
FROM repositories r
LEFT JOIN commits c ON c.repository_id = r.id
LEFT JOIN issues i ON i.repository_id = r.id
LEFT JOIN pull_requests p ON p.repository_id = r.id
LEFT JOIN predictions pred ON pred.repository_id = r.id
    AND pred.predicted_at > NOW() - INTERVAL '30 days'
GROUP BY r.id, r.full_name, r.language, r.stars, r.last_crawled_at;

-- File Risk Analysis
CREATE OR REPLACE VIEW file_risk_analysis AS
SELECT
    f.id,
    f.repository_id,
    f.path,
    f.language,
    f.total_commits,
    f.total_authors,
    f.complexity_score,
    COUNT(DISTINCT cf.commit_id) as recent_changes,
    AVG(cf.changes) as avg_change_size,
    MAX(pred.prediction_value) as max_bug_probability,
    MAX(cf.committed_at) as last_modified
FROM files f
LEFT JOIN commit_files cf ON cf.file_id = f.id
    AND cf.committed_at > NOW() - INTERVAL '90 days'
LEFT JOIN predictions pred ON pred.file_id = f.id
    AND pred.predicted_at > NOW() - INTERVAL '7 days'
    AND pred.prediction_type = 'bug_probability'
WHERE f.is_deleted = FALSE
GROUP BY f.id, f.repository_id, f.path, f.language,
         f.total_commits, f.total_authors, f.complexity_score;

-- Recent Activity Summary
CREATE OR REPLACE VIEW recent_activity AS
SELECT
    r.full_name,
    r.language,
    COUNT(DISTINCT c.id) as commits_last_30d,
    COUNT(DISTINCT c.author_login) as active_contributors,
    COUNT(DISTINCT i.id) as issues_created,
    COUNT(DISTINCT p.id) as prs_created,
    SUM(c.insertions + c.deletions) as total_changes
FROM repositories r
LEFT JOIN commits c ON c.repository_id = r.id
    AND c.committed_at > NOW() - INTERVAL '30 days'
LEFT JOIN issues i ON i.repository_id = r.id
    AND i.github_created_at > NOW() - INTERVAL '30 days'
LEFT JOIN pull_requests p ON p.repository_id = r.id
    AND p.github_created_at > NOW() - INTERVAL '30 days'
GROUP BY r.id, r.full_name, r.language;

-- ============================================================================
-- Functions and Triggers
-- ============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$ LANGUAGE plpgsql;

-- Apply to relevant tables
CREATE TRIGGER update_repositories_updated_at BEFORE UPDATE ON repositories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_files_updated_at BEFORE UPDATE ON files
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_issues_updated_at BEFORE UPDATE ON issues
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_pull_requests_updated_at BEFORE UPDATE ON pull_requests
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Initial Data / Configuration
-- ============================================================================

-- Insert default model for tracking
INSERT INTO models (name, version, model_type, framework, status)
VALUES ('baseline', 'v0.1.0', 'xgboost', 'sklearn', 'training')
ON CONFLICT DO NOTHING;

-- ============================================================================
-- Maintenance
-- ============================================================================

-- Continuous aggregates for common queries (TimescaleDB)
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_commit_stats
WITH (timescaledb.continuous) AS
SELECT
    repository_id,
    time_bucket('1 day', committed_at) AS day,
    COUNT(*) as commit_count,
    COUNT(DISTINCT author_login) as unique_authors,
    SUM(insertions) as total_insertions,
    SUM(deletions) as total_deletions
FROM commits
GROUP BY repository_id, day
WITH NO DATA;

SELECT add_continuous_aggregate_policy('daily_commit_stats',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Refresh the materialized view
REFRESH MATERIALIZED VIEW daily_commit_stats;

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE repositories IS 'Core repository metadata from GitHub';
COMMENT ON TABLE commits IS 'Time-series commit history with change metrics';
COMMENT ON TABLE files IS 'Individual file tracking across commits';
COMMENT ON TABLE predictions IS 'ML model predictions with confidence scores';
COMMENT ON TABLE feedback IS 'Developer feedback for reinforcement learning';
COMMENT ON TABLE rl_episodes IS 'RL training episode history';
COMMENT ON TABLE sequential_patterns IS 'Discovered temporal patterns in commits';
COMMENT ON TABLE association_rules IS 'Association rules between code changes';

-- ============================================================================
-- Grants (adjust based on your user setup)
-- ============================================================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO athena;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO athena;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO athena;
