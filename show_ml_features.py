"""
Display ATHENA's collected data and ML features

This shows you exactly what data ATHENA extracts from GitHub
for machine learning and code intelligence.
"""
import subprocess
import json

def run_query(query):
    """Execute PostgreSQL query via Docker"""
    cmd = [
        'docker', 'exec', 'athena-postgres',
        'psql', '-U', 'athena', '-d', 'athena',
        '-t',  # Tuples only
        '-c', query
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

print("=" * 100)
print("ATHENA ML FEATURE EXTRACTION - What We Collect for Training")
print("=" * 100)

# Database counts
print("\nDATA COLLECTION SUMMARY:")
print("-" * 100)
counts = run_query("""
    SELECT
        (SELECT COUNT(*) FROM repositories) as repos,
        (SELECT COUNT(*) FROM commits) as commits,
        (SELECT COUNT(*) FROM files) as files,
        (SELECT COUNT(*) FROM issues) as issues,
        (SELECT COUNT(*) FROM commit_files) as commit_files
""")
if counts:
    print(f"  {counts}")

# Show commit features
print("\n" + "=" * 100)
print("COMMIT FEATURES (for ML training)")
print("=" * 100)
print("\nSample commits with extractable features:")
print("-" * 100)
commits = run_query("""
    SELECT
        SUBSTRING(c.sha, 1, 7) as sha,
        SUBSTRING(c.author_name, 1, 20) as author,
        c.files_changed,
        c.insertions,
        c.deletions,
        (c.insertions + c.deletions) as total_changes,
        CASE WHEN c.is_merge THEN 'Y' ELSE 'N' END as merge,
        LENGTH(c.message) as msg_len,
        r.full_name as repo
    FROM commits c
    JOIN repositories r ON c.repository_id = r.id
    ORDER BY c.committed_at DESC
    LIMIT 10
""")
print(commits if commits else "No data yet")

#  Show file change patterns
print("\n" + "=" * 100)
print("FILE CHANGE PATTERNS (for sequential pattern mining)")
print("=" * 100)
print("\nFiles frequently modified together:")
print("-" * 100)
file_patterns = run_query("""
    SELECT
        r.full_name as repo,
        COUNT(DISTINCT cf.commit_id) as commits,
        COUNT(*) as total_changes,
        SUBSTRING(f.path, 1, 60) as file_path,
        f.extension
    FROM commit_files cf
    JOIN files f ON cf.file_id = f.id
    JOIN repositories r ON f.repository_id = r.id
    WHERE f.extension IS NOT NULL
    GROUP BY r.full_name, f.path, f.extension
    ORDER BY commits DESC, total_changes DESC
    LIMIT 15
""")
print(file_patterns if file_patterns else "No data yet")

# Show repository metadata
print("\n" + "=" * 100)
print("REPOSITORY METADATA (for context features)")
print("=" * 100)
print("\nRepository activity patterns:")
print("-" * 100)
repo_meta = run_query("""
    SELECT
        SUBSTRING(full_name, 1, 40) as repository,
        stars,
        forks,
        open_issues,
        (SELECT COUNT(*) FROM commits WHERE repository_id = r.id) as total_commits,
        language
    FROM repositories r
    ORDER BY stars DESC
""")
print(repo_meta if repo_meta else "No data yet")

# Show author activity
print("\n" + "=" * 100)
print("AUTHOR ACTIVITY (for developer behavior patterns)")
print("=" * 100)
print("\nTop contributors and their patterns:")
print("-" * 100)
authors = run_query("""
    SELECT
        SUBSTRING(author_name, 1, 25) as author,
        COUNT(*) as commits,
        SUM(files_changed) as total_files,
        SUM(insertions) as total_insertions,
        SUM(deletions) as total_deletions,
        ROUND(AVG(files_changed)::numeric, 1) as avg_files_per_commit,
        SUM(CASE WHEN is_merge THEN 1 ELSE 0 END) as merges
    FROM commits
    WHERE author_name IS NOT NULL
    GROUP BY author_name
    HAVING COUNT(*) > 1
    ORDER BY commits DESC
    LIMIT 10
""")
print(authors if authors else "No data yet")

# ML Features Summary
print("\n" + "=" * 100)
print("ML FEATURE CATEGORIES")
print("=" * 100)
print("""
These are the feature types ATHENA extracts for machine learning:

1. COMMIT FEATURES:
   - Code churn (insertions, deletions, total changes)
   - Files changed per commit
   - Commit message length and patterns
   - Merge vs regular commits
   - Temporal patterns (time of day, day of week)

2. FILE FEATURES:
   - File extension/language
   - Change frequency
   - Number of authors who touched the file
   - File path patterns
   - Complexity metrics (when available)

3. AUTHOR FEATURES:
   - Commit frequency
   - Average code churn
   - Files touched per commit
   - Merge ratio
   - Activity patterns over time

4. REPOSITORY CONTEXT:
   - Project popularity (stars, forks)
   - Project activity (open issues, commits)
   - Primary language
   - Topics and metadata

5. SEQUENTIAL PATTERNS:
   - Files frequently changed together
   - Commit sequences
   - Issue-commit relationships

6. ISSUE/PR FEATURES:
   - Issue labels and sentiment
   - Time to close
   - Comment activity
   - PR merge patterns

These features feed into:
- Bug prediction models
- Code quality assessment
- Developer productivity analysis
- Automated code review
- Pattern mining and anomaly detection
""")

print("=" * 100)
print("Feature extraction complete!")
print("=" * 100)
