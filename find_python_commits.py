"""Find commits that likely have Python files based on file changes"""
import asyncio
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from fetch_commit_files import GitHubFileFetcher, get_database_session, process_file_changes
from agents.crawler.models import Repository, Commit

load_dotenv()


async def find_and_fetch_python_commits(limit: int = 20):
    """Find and fetch commits that actually have Python code"""
    session = get_database_session()

    try:
        # Get all commits from Python repositories without files
        query = text("""
        SELECT c.id, c.sha, c.repository_id, r.owner, r.name, c.committed_at, c.files_changed
        FROM commits c
        JOIN repositories r ON c.repository_id = r.id
        WHERE r.language = 'Python'
        AND NOT EXISTS (
            SELECT 1 FROM commit_files cf WHERE cf.commit_id = c.id
        )
        AND c.files_changed > 0
        ORDER BY c.committed_at ASC
        LIMIT :limit
        """)

        result = session.execute(query, {"limit": limit})
        commits_to_process = result.fetchall()

        if not commits_to_process:
            print("No commits found")
            return

        print(f"Found {len(commits_to_process)} commits to check")

        github_token = os.getenv('GITHUB_TOKEN')
        async with GitHubFileFetcher(github_token) as fetcher:
            python_files_found = 0

            for commit_row in commits_to_process:
                commit_id, sha, repo_id, owner, repo_name, committed_at, files_changed = commit_row

                # Fetch file changes
                files_data = await fetcher.fetch_commit_files(owner, repo_name, sha)

                if files_data:
                    # Check for Python files
                    python_files = [f for f in files_data if f.get('filename', '').endswith('.py')]

                    if python_files:
                        print(f"\n[FOUND] Commit {sha[:7]} has {len(python_files)} Python files:")
                        for pf in python_files[:5]:
                            print(f"  - {pf['filename']}")

                        # Process this commit
                        repository = session.get(Repository, repo_id)
                        commit = session.get(Commit, commit_id)
                        files_created = process_file_changes(session, repository, commit, files_data)
                        session.commit()

                        python_files_found += len(python_files)

                        if python_files_found >= 10:
                            break

                await asyncio.sleep(0.5)

            print(f"\n[SUCCESS] Found and processed {python_files_found} Python files")

    finally:
        session.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  Finding Commits with Python Files")
    print("=" * 60)
    asyncio.run(find_and_fetch_python_commits(limit=50))
