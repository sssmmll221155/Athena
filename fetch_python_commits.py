"""Fetch files for Python repository commits"""
import asyncio
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Import the existing fetcher
from fetch_commit_files import GitHubFileFetcher, get_database_session, process_file_changes
from agents.crawler.models import Repository, Commit

load_dotenv()


async def fetch_python_repo_files(repo_id: int = 14, limit: int = 10):
    """Fetch files for a specific Python repository"""
    session = get_database_session()

    try:
        # Query commits from the Python repository without files
        query = text("""
        SELECT c.id, c.sha, c.repository_id, r.owner, r.name, c.committed_at
        FROM commits c
        JOIN repositories r ON c.repository_id = r.id
        WHERE c.repository_id = :repo_id
        AND NOT EXISTS (
            SELECT 1 FROM commit_files cf WHERE cf.commit_id = c.id
        )
        ORDER BY c.committed_at DESC
        LIMIT :limit
        """)

        result = session.execute(query, {"repo_id": repo_id, "limit": limit})
        commits_to_process = result.fetchall()

        if not commits_to_process:
            print(f"No commits to process for repository ID {repo_id}")
            return

        print(f"Found {len(commits_to_process)} commits from Python repository")

        # Fetch GitHub token
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            print("ERROR: GITHUB_TOKEN not found")
            return

        async with GitHubFileFetcher(github_token) as fetcher:
            total_files = 0

            for commit_row in commits_to_process:
                commit_id, sha, repo_id, owner, repo_name, committed_at = commit_row

                print(f"Fetching files for commit {sha[:7]} from {owner}/{repo_name}")

                # Fetch file changes from GitHub
                files_data = await fetcher.fetch_commit_files(owner, repo_name, sha)

                if files_data:
                    # Get repository and commit objects
                    repository = session.get(Repository, repo_id)
                    commit = session.get(Commit, commit_id)

                    # Process files
                    files_created = process_file_changes(session, repository, commit, files_data)
                    total_files += files_created

                    # Show Python files
                    python_files = [f['filename'] for f in files_data if f.get('filename', '').endswith('.py')]
                    if python_files:
                        print(f"  [OK] Found {len(python_files)} Python files: {', '.join(python_files[:3])}")
                    else:
                        print(f"  [OK] Processed {len(files_data)} files (no Python files)")

                # Commit batch
                session.commit()

                # Rate limiting
                await asyncio.sleep(0.5)

            print(f"\n[SUCCESS] Total Python files created: {total_files}")

    except Exception as e:
        print(f"Error: {e}")
        session.rollback()
    finally:
        session.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  Fetching Python Repository Files")
    print("=" * 60)
    asyncio.run(fetch_python_repo_files(repo_id=14, limit=10))
