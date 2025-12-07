"""
Bridge Script: Fetch File Contents for Commits
Fetches file information from GitHub for commits already in the database
"""

import asyncio
import aiohttp
import argparse
import logging
import os
import sys
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from agents.crawler.models import Repository, Commit, File, CommitFile

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('file_fetcher.log')
    ]
)
logger = logging.getLogger(__name__)


class GitHubFileFetcher:
    """Fetches file information from GitHub API for commits"""

    def __init__(self, github_token: str):
        self.token = github_token
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Athena-File-Fetcher/1.0"
        }
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(headers=self.headers, timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_commit_files(self, owner: str, repo: str, sha: str) -> Optional[List[Dict]]:
        """Fetch file changes for a specific commit"""
        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('files', [])
                elif response.status == 404:
                    logger.warning(f"Commit not found: {owner}/{repo}@{sha}")
                    return None
                elif response.status == 403:
                    logger.error(f"Rate limit exceeded for {owner}/{repo}")
                    await asyncio.sleep(60)  # Wait 1 minute
                    return None
                else:
                    logger.error(f"Request failed: {url} - Status {response.status}")
                    return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching commit: {owner}/{repo}@{sha}")
            return None
        except Exception as e:
            logger.error(f"Error fetching commit files: {e}")
            return None


def get_database_session():
    """Create database session"""
    db_host = os.getenv('DB_HOST') or os.getenv('POSTGRES_HOST', 'localhost')
    db_port = os.getenv('DB_PORT') or os.getenv('POSTGRES_PORT', '5432')
    db_name = os.getenv('DB_NAME') or os.getenv('POSTGRES_DB', 'athena')
    db_user = os.getenv('DB_USER') or os.getenv('POSTGRES_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD') or os.getenv('POSTGRES_PASSWORD', '')

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(db_url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    return Session()


def detect_language(filename: str) -> Optional[str]:
    """Detect language from file extension"""
    ext_to_lang = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.ts': 'TypeScript',
        '.jsx': 'JavaScript',
        '.tsx': 'TypeScript',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c': 'C',
        '.h': 'C',
        '.hpp': 'C++',
        '.go': 'Go',
        '.rs': 'Rust',
        '.rb': 'Ruby',
        '.php': 'PHP',
        '.swift': 'Swift',
        '.kt': 'Kotlin',
        '.scala': 'Scala',
        '.sh': 'Shell',
        '.sql': 'SQL',
        '.html': 'HTML',
        '.css': 'CSS',
        '.json': 'JSON',
        '.xml': 'XML',
        '.yaml': 'YAML',
        '.yml': 'YAML',
    }

    for ext, lang in ext_to_lang.items():
        if filename.endswith(ext):
            return lang
    return None


def process_file_changes(session, repository: Repository, commit: Commit, files_data: List[Dict]) -> int:
    """Process file changes and create File and CommitFile records"""
    files_created = 0

    for file_data in files_data:
        filename = file_data.get('filename')
        if not filename:
            continue

        # Get or create File record
        file_record = session.query(File).filter(
            File.repository_id == repository.id,
            File.path == filename
        ).first()

        if not file_record:
            # Create new file record
            path_parts = filename.rsplit('/', 1)
            directory = path_parts[0] if len(path_parts) > 1 else ''
            file_name = path_parts[-1]
            extension = f".{file_name.split('.')[-1]}" if '.' in file_name else ''

            file_record = File(
                repository_id=repository.id,
                path=filename,
                filename=file_name,
                extension=extension,
                directory=directory,
                language=detect_language(filename),
                first_seen_at=commit.committed_at,
                last_modified_at=commit.committed_at,
                total_commits=1,
                is_deleted=(file_data.get('status') == 'removed')
            )
            session.add(file_record)
            session.flush()  # Get the file_record.id
            files_created += 1
        else:
            # Update existing file record
            file_record.last_modified_at = commit.committed_at
            file_record.total_commits += 1
            if file_data.get('status') == 'removed':
                file_record.is_deleted = True

        # Check if CommitFile already exists
        existing_cf = session.query(CommitFile).filter(
            CommitFile.commit_id == commit.id,
            CommitFile.file_id == file_record.id
        ).first()

        if not existing_cf:
            # Create CommitFile junction record
            commit_file = CommitFile(
                commit_id=commit.id,
                file_id=file_record.id,
                repository_id=repository.id,
                status=file_data.get('status', 'modified'),
                additions=file_data.get('additions', 0),
                deletions=file_data.get('deletions', 0),
                changes=file_data.get('changes', 0),
                patch=file_data.get('patch'),
                previous_filename=file_data.get('previous_filename'),
                committed_at=commit.committed_at
            )
            session.add(commit_file)

    return files_created


async def process_commits(limit: Optional[int] = None, batch_size: int = 10):
    """Process commits and fetch their file information"""
    session = get_database_session()

    try:
        # Query commits without files
        query = """
        SELECT c.id, c.sha, c.repository_id, r.owner, r.name, c.committed_at
        FROM commits c
        JOIN repositories r ON c.repository_id = r.id
        WHERE NOT EXISTS (
            SELECT 1 FROM commit_files cf WHERE cf.commit_id = c.id
        )
        ORDER BY c.committed_at DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        result = session.execute(text(query))
        commits_to_process = result.fetchall()

        if not commits_to_process:
            logger.info("No commits to process! All commits already have files.")
            return

        logger.info(f"Found {len(commits_to_process)} commits to process")

        # Fetch GitHub token
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            logger.error("GITHUB_TOKEN not found in environment variables")
            return

        async with GitHubFileFetcher(github_token) as fetcher:
            total_files = 0
            processed_commits = 0

            # Process in batches
            for i in range(0, len(commits_to_process), batch_size):
                batch = commits_to_process[i:i + batch_size]

                for commit_row in batch:
                    commit_id, sha, repo_id, owner, repo_name, committed_at = commit_row

                    logger.info(f"Processing commit {sha[:7]} from {owner}/{repo_name}")

                    # Fetch file changes from GitHub
                    files_data = await fetcher.fetch_commit_files(owner, repo_name, sha)

                    if files_data:
                        # Get repository and commit objects
                        repository = session.query(Repository).get(repo_id)
                        commit = session.query(Commit).get(commit_id)

                        # Process files
                        files_created = process_file_changes(session, repository, commit, files_data)
                        total_files += files_created
                        processed_commits += 1

                        logger.info(f"  [OK] Processed {len(files_data)} files ({files_created} new)")

                    # Commit batch
                    session.commit()

                    # Rate limiting
                    await asyncio.sleep(0.5)

            logger.info(f"\n[SUCCESS] Processed {processed_commits} commits, created {total_files} file records")

    except Exception as e:
        logger.error(f"Error processing commits: {e}")
        session.rollback()
    finally:
        session.close()


def print_banner():
    """Print CLI banner"""
    banner = """
===============================================
  Athena File Fetcher - Bridge Script
  Fetches file data for existing commits
===============================================
"""
    print(banner)


def main():
    parser = argparse.ArgumentParser(description='Fetch file information for commits')
    parser.add_argument('--limit', type=int, help='Limit number of commits to process')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    args = parser.parse_args()

    print_banner()

    try:
        asyncio.run(process_commits(limit=args.limit, batch_size=args.batch_size))
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
