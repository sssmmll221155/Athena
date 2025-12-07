"""
Athena Main Crawler - Production GitHub Data Collection
Integrates crawler, database, and Kafka for continuous data ingestion.

Usage:
    python main_crawler.py --mode trending --limit 10
    python main_crawler.py --mode language --language python --limit 20
    python main_crawler.py --mode custom --repos "microsoft/vscode,python/cpython"
    python main_crawler.py --mode trending --continuous --interval 3600
"""

import asyncio
import argparse
import logging
import signal
import sys
import os
from datetime import datetime
from typing import List, Optional, Tuple
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv

# Import our existing modules
from agents.crawler.github_crawler import CrawlerConfig, RepositoryCrawler, CrawlResult
from agents.crawler.kafka_producer import (
    AthenaKafkaProducer, KafkaConfig, CommitMessage,
    IssueMessage, CrawlerEventMessage
)
from agents.crawler.models import Base, Repository, Commit, Issue


# ============================================================================
# Configuration
# ============================================================================

load_dotenv()

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"crawler_{datetime.now():%Y%m%d}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Database Manager
# ============================================================================

class DatabaseManager:
    """Manages database operations with upsert logic"""

    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"Database connection established")

    def get_session(self):
        """Get database session"""
        return self.SessionLocal()

    def upsert_repository(self, crawl_result: CrawlResult) -> Optional[int]:
        """
        Insert or update repository
        Returns repository ID
        """
        if not crawl_result.success or not crawl_result.metadata:
            logger.warning(f"Skipping repository {crawl_result.owner}/{crawl_result.repo}: No metadata")
            return None

        with self.get_session() as session:
            try:
                # Check if repository exists
                full_name = crawl_result.metadata['full_name']
                stmt = select(Repository).where(Repository.full_name == full_name)
                repo = session.execute(stmt).scalar_one_or_none()

                if repo:
                    # UPDATE existing
                    repo.description = crawl_result.metadata.get('description')
                    repo.language = crawl_result.metadata.get('language')
                    repo.stars = crawl_result.metadata.get('stargazers_count', 0)
                    repo.forks = crawl_result.metadata.get('forks_count', 0)
                    repo.watchers = crawl_result.metadata.get('watchers_count', 0)
                    repo.open_issues = crawl_result.metadata.get('open_issues_count', 0)
                    repo.github_updated_at = datetime.fromisoformat(
                        crawl_result.metadata['updated_at'].replace('Z', '+00:00')
                    )
                    repo.last_crawled_at = datetime.utcnow()
                    repo.crawl_status = 'success'

                    logger.info(f"✓ Updated repository: {full_name} (id={repo.id})")
                else:
                    # INSERT new
                    repo = Repository(
                        owner=crawl_result.owner,
                        name=crawl_result.repo,
                        full_name=full_name,
                        description=crawl_result.metadata.get('description'),
                        language=crawl_result.metadata.get('language'),
                        stars=crawl_result.metadata.get('stargazers_count', 0),
                        forks=crawl_result.metadata.get('forks_count', 0),
                        watchers=crawl_result.metadata.get('watchers_count', 0),
                        open_issues=crawl_result.metadata.get('open_issues_count', 0),
                        size_kb=crawl_result.metadata.get('size', 0),
                        default_branch=crawl_result.metadata.get('default_branch', 'main'),
                        is_private=crawl_result.metadata.get('private', False),
                        is_fork=crawl_result.metadata.get('fork', False),
                        is_archived=crawl_result.metadata.get('archived', False),
                        license=crawl_result.metadata.get('license', {}).get('name') if crawl_result.metadata.get('license') else None,
                        topics=crawl_result.metadata.get('topics', []),
                        github_created_at=datetime.fromisoformat(
                            crawl_result.metadata['created_at'].replace('Z', '+00:00')
                        ),
                        github_updated_at=datetime.fromisoformat(
                            crawl_result.metadata['updated_at'].replace('Z', '+00:00')
                        ),
                        last_crawled_at=datetime.utcnow(),
                        crawl_status='success',
                        metadata=crawl_result.metadata
                    )
                    session.add(repo)
                    logger.info(f"✓ Inserted new repository: {full_name}")

                session.commit()
                return repo.id

            except Exception as e:
                session.rollback()
                logger.error(f"✗ Failed to upsert repository {crawl_result.owner}/{crawl_result.repo}: {e}")
                return None

    def insert_commits_batch(self, repository_id: int, commits_data: List[dict]) -> int:
        """
        Insert commits in batch (skip duplicates)
        Returns number of commits inserted
        """
        if not commits_data:
            return 0

        with self.get_session() as session:
            inserted = 0
            for commit_data in commits_data:
                try:
                    # Check if commit already exists
                    sha = commit_data['sha']
                    stmt = select(Commit).where(
                        Commit.repository_id == repository_id,
                        Commit.sha == sha
                    )
                    existing = session.execute(stmt).scalar_one_or_none()

                    if existing:
                        continue  # Skip duplicate

                    # Parse commit data
                    commit = Commit(
                        repository_id=repository_id,
                        sha=sha,
                        author_name=commit_data.get('commit', {}).get('author', {}).get('name'),
                        author_email=commit_data.get('commit', {}).get('author', {}).get('email'),
                        author_login=commit_data.get('author', {}).get('login') if commit_data.get('author') else None,
                        committer_name=commit_data.get('commit', {}).get('committer', {}).get('name'),
                        committer_email=commit_data.get('commit', {}).get('committer', {}).get('email'),
                        message=commit_data.get('commit', {}).get('message', ''),
                        message_subject=commit_data.get('commit', {}).get('message', '').split('\n')[0][:500],
                        files_changed=len(commit_data.get('files', [])),
                        insertions=commit_data.get('stats', {}).get('additions', 0),
                        deletions=commit_data.get('stats', {}).get('deletions', 0),
                        parents=[p['sha'] for p in commit_data.get('parents', [])],
                        committed_at=datetime.fromisoformat(
                            commit_data['commit']['author']['date'].replace('Z', '+00:00')
                        ),
                        is_merge=len(commit_data.get('parents', [])) > 1,
                        raw_data=commit_data
                    )
                    session.add(commit)
                    inserted += 1

                except Exception as e:
                    logger.warning(f"Failed to insert commit {commit_data.get('sha', 'unknown')}: {e}")
                    continue

            if inserted > 0:
                session.commit()
                logger.info(f"✓ Inserted {inserted} commits for repository_id={repository_id}")

            return inserted

    def insert_issues_batch(self, repository_id: int, issues_data: List[dict]) -> int:
        """
        Insert issues in batch (skip duplicates)
        Returns number of issues inserted
        """
        if not issues_data:
            return 0

        with self.get_session() as session:
            inserted = 0
            for issue_data in issues_data:
                try:
                    # Skip pull requests (they're tracked separately)
                    if 'pull_request' in issue_data:
                        continue

                    # Check if issue already exists
                    issue_number = issue_data['number']
                    stmt = select(Issue).where(
                        Issue.repository_id == repository_id,
                        Issue.issue_number == issue_number
                    )
                    existing = session.execute(stmt).scalar_one_or_none()

                    if existing:
                        # Update if state changed
                        if existing.state != issue_data['state']:
                            existing.state = issue_data['state']
                            existing.github_updated_at = datetime.fromisoformat(
                                issue_data['updated_at'].replace('Z', '+00:00')
                            )
                            if issue_data.get('closed_at'):
                                existing.github_closed_at = datetime.fromisoformat(
                                    issue_data['closed_at'].replace('Z', '+00:00')
                                )
                        continue

                    # Insert new issue
                    issue = Issue(
                        repository_id=repository_id,
                        issue_number=issue_number,
                        title=issue_data['title'],
                        body=issue_data.get('body'),
                        state=issue_data['state'],
                        author_login=issue_data.get('user', {}).get('login'),
                        assignees=[a['login'] for a in issue_data.get('assignees', [])],
                        labels=[l['name'] for l in issue_data.get('labels', [])],
                        comments_count=issue_data.get('comments', 0),
                        github_created_at=datetime.fromisoformat(
                            issue_data['created_at'].replace('Z', '+00:00')
                        ),
                        github_updated_at=datetime.fromisoformat(
                            issue_data['updated_at'].replace('Z', '+00:00')
                        ),
                        github_closed_at=datetime.fromisoformat(
                            issue_data['closed_at'].replace('Z', '+00:00')
                        ) if issue_data.get('closed_at') else None,
                        raw_data=issue_data
                    )
                    session.add(issue)
                    inserted += 1

                except Exception as e:
                    logger.warning(f"Failed to insert issue {issue_data.get('number', 'unknown')}: {e}")
                    continue

            if inserted > 0:
                session.commit()
                logger.info(f"✓ Inserted {inserted} issues for repository_id={repository_id}")

            return inserted


# ============================================================================
# Main Crawler Orchestrator
# ============================================================================

class AthenaCrawler:
    """Main crawler orchestrator - integrates all components"""

    def __init__(
        self,
        github_token: str,
        db_connection: str,
        kafka_servers: str
    ):
        self.crawler_config = CrawlerConfig(
            github_token=github_token,
            max_concurrent_requests=10
        )
        self.kafka_config = KafkaConfig(bootstrap_servers=kafka_servers)
        self.db = DatabaseManager(db_connection)
        self.should_stop = False

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("🚀 Athena Crawler initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("\n⚠️  Shutdown signal received. Finishing current operations...")
        self.should_stop = True

    async def crawl_trending_repos(self, language: str = "python", limit: int = 10):
        """Crawl trending repositories"""
        logger.info(f"📊 Fetching trending {language} repositories (limit={limit})")

        # Build search query
        query = f"language:{language} stars:>100"

        async with RepositoryCrawler(self.crawler_config) as crawler:
            # Discover repositories
            results = await crawler.discover_and_crawl(
                query=query,
                max_repos=limit
            )

            await self._process_results(results)

    async def crawl_language_repos(self, language: str, limit: int = 10):
        """Crawl repositories by language"""
        logger.info(f"📊 Fetching {language} repositories (limit={limit})")

        query = f"language:{language} stars:>500"

        async with RepositoryCrawler(self.crawler_config) as crawler:
            results = await crawler.discover_and_crawl(
                query=query,
                max_repos=limit
            )

            await self._process_results(results)

    async def crawl_custom_repos(self, repo_list: List[Tuple[str, str]]):
        """Crawl specific repositories"""
        logger.info(f"📊 Fetching {len(repo_list)} custom repositories")

        async with RepositoryCrawler(self.crawler_config) as crawler:
            results = await crawler.crawl_repositories(
                repo_list,
                fetch_commits=True,
                fetch_issues=True
            )

            await self._process_results(results)

    async def _process_results(self, results: List[CrawlResult]):
        """Process crawl results: save to DB and send to Kafka"""
        if not results:
            logger.warning("No results to process")
            return

        successful_results = [r for r in results if r.success]
        logger.info(f"Processing {len(successful_results)} successful crawls")

        # Initialize Kafka producer
        async with AthenaKafkaProducer(self.kafka_config) as kafka:

            for result in successful_results:
                if self.should_stop:
                    logger.info("Stopping due to shutdown signal")
                    break

                try:
                    # 1. Upsert repository to database
                    repo_id = self.db.upsert_repository(result)

                    if not repo_id:
                        logger.error(f"Failed to save repository {result.owner}/{result.repo}")
                        continue

                    # 2. Fetch and save DETAILED commit data
                    logger.info(f"Fetching detailed commits for {result.owner}/{result.repo}...")
                    async with RepositoryCrawler(self.crawler_config) as temp_crawler:
                        commits = await temp_crawler.fetch_commits(
                            result.owner,
                            result.repo,
                            max_commits=100
                        )

                        if commits:
                            # Save commits to database
                            commits_saved = self.db.insert_commits_batch(repo_id, commits)
                            logger.info(f"✓ Saved {commits_saved} commits")

                            # Send commit messages to Kafka (sample first 10)
                            for commit in commits[:10]:
                                commit_msg = CommitMessage(
                                    repository_id=repo_id,
                                    repository_full_name=f"{result.owner}/{result.repo}",
                                    sha=commit['sha'],
                                    author_name=commit.get('commit', {}).get('author', {}).get('name'),
                                    author_email=commit.get('commit', {}).get('author', {}).get('email'),
                                    author_login=commit.get('author', {}).get('login') if commit.get('author') else None,
                                    message=commit.get('commit', {}).get('message', ''),
                                    files_changed=len(commit.get('files', [])),
                                    insertions=commit.get('stats', {}).get('additions', 0),
                                    deletions=commit.get('stats', {}).get('deletions', 0),
                                    committed_at=commit['commit']['author']['date'],
                                    raw_data=commit
                                )
                                await kafka.send_commit(commit_msg)

                    # 3. Fetch and save issues
                    logger.info(f"Fetching issues for {result.owner}/{result.repo}...")
                    async with RepositoryCrawler(self.crawler_config) as temp_crawler:
                        issues = await temp_crawler.fetch_issues(
                            result.owner,
                            result.repo,
                            max_issues=100
                        )

                        if issues:
                            issues_saved = self.db.insert_issues_batch(repo_id, issues)
                            logger.info(f"✓ Saved {issues_saved} issues")

                    # 4. Send crawler event to Kafka
                    event = CrawlerEventMessage(
                        event_type="completed",
                        repository_full_name=f"{result.owner}/{result.repo}",
                        status="success",
                        commits_count=commits_saved if commits else 0,
                        issues_count=issues_saved if issues else 0,
                        files_count=0,
                        duration_seconds=result.crawl_time
                    )
                    await kafka.send_crawler_event(event)

                    logger.info(
                        f"✅ Processed {result.owner}/{result.repo}: "
                        f"{commits_saved if commits else 0} commits, "
                        f"{issues_saved if issues else 0} issues"
                    )

                except Exception as e:
                    logger.error(f"Error processing {result.owner}/{result.repo}: {e}")

                    # Send error event
                    try:
                        error_event = CrawlerEventMessage(
                            event_type="failed",
                            repository_full_name=f"{result.owner}/{result.repo}",
                            status="error",
                            commits_count=0,
                            issues_count=0,
                            files_count=0,
                            duration_seconds=result.crawl_time,
                            error=str(e)
                        )
                        await kafka.send_crawler_event(error_event)
                    except:
                        pass

        # Summary
        logger.info(f"✅ Completed processing {len(successful_results)} repositories")

    async def run_continuous(self, mode: str, interval: int, **kwargs):
        """Run crawler continuously with specified interval"""
        logger.info(f"🔄 Starting continuous mode (interval={interval}s)")

        while not self.should_stop:
            try:
                # Run crawl based on mode
                if mode == "trending":
                    await self.crawl_trending_repos(
                        language=kwargs.get('language', 'python'),
                        limit=kwargs.get('limit', 10)
                    )
                elif mode == "language":
                    await self.crawl_language_repos(
                        language=kwargs.get('language', 'python'),
                        limit=kwargs.get('limit', 10)
                    )
                elif mode == "custom":
                    await self.crawl_custom_repos(kwargs.get('repo_list', []))

                # Wait for interval
                logger.info(f"⏰ Waiting {interval} seconds until next run...")
                for _ in range(interval):
                    if self.should_stop:
                        break
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in continuous mode: {e}")
                logger.info("Waiting 60 seconds before retry...")
                await asyncio.sleep(60)

        logger.info("🛑 Continuous mode stopped")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Athena GitHub Crawler - Fetch and analyze repository data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch 10 trending Python repos
  python main_crawler.py --mode trending --language python --limit 10

  # Fetch JavaScript repos
  python main_crawler.py --mode language --language javascript --limit 20

  # Fetch specific repos
  python main_crawler.py --mode custom --repos "microsoft/vscode,python/cpython"

  # Run continuously (every hour)
  python main_crawler.py --mode trending --limit 20 --continuous --interval 3600
        """
    )

    parser.add_argument(
        '--mode',
        choices=['trending', 'language', 'custom'],
        default='trending',
        help='Crawl mode (default: trending)'
    )

    parser.add_argument(
        '--language',
        default='python',
        help='Programming language filter (default: python)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Number of repositories to crawl (default: 10)'
    )

    parser.add_argument(
        '--repos',
        help='Comma-separated list of repos (owner/repo format) for custom mode'
    )

    parser.add_argument(
        '--continuous',
        action='store_true',
        help='Run continuously with interval'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=3600,
        help='Interval in seconds for continuous mode (default: 3600 = 1 hour)'
    )

    return parser.parse_args()


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main entry point"""
    args = parse_args()

    # Load environment variables
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        logger.error("❌ GITHUB_TOKEN not found in environment")
        sys.exit(1)

    db_connection = (
        f"postgresql://{os.getenv('POSTGRES_USER', 'athena')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'athena_dev_pass')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'athena')}"
    )

    kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:29092')

    # Initialize crawler
    crawler = AthenaCrawler(
        github_token=github_token,
        db_connection=db_connection,
        kafka_servers=kafka_servers
    )

    logger.info("="*70)
    logger.info("🚀 ATHENA GITHUB CRAWLER")
    logger.info("="*70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Limit: {args.limit}")
    logger.info(f"Continuous: {args.continuous}")
    if args.continuous:
        logger.info(f"Interval: {args.interval}s")
    logger.info("="*70)

    try:
        if args.continuous:
            # Continuous mode
            kwargs = {
                'language': args.language,
                'limit': args.limit
            }

            if args.mode == 'custom' and args.repos:
                repo_list = [
                    tuple(repo.split('/'))
                    for repo in args.repos.split(',')
                ]
                kwargs['repo_list'] = repo_list

            await crawler.run_continuous(args.mode, args.interval, **kwargs)

        else:
            # One-shot mode
            if args.mode == 'trending':
                await crawler.crawl_trending_repos(args.language, args.limit)

            elif args.mode == 'language':
                await crawler.crawl_language_repos(args.language, args.limit)

            elif args.mode == 'custom':
                if not args.repos:
                    logger.error("❌ --repos required for custom mode")
                    sys.exit(1)

                repo_list = [
                    tuple(repo.split('/'))
                    for repo in args.repos.split(',')
                ]
                await crawler.crawl_custom_repos(repo_list)

        logger.info("✅ Crawler completed successfully")

    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
