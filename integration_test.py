"""
Athena End-to-End Integration Test
Tests complete data flow: GitHub → Crawler → Kafka → Database

This validates that all components work together correctly.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Import our components
sys.path.insert(0, os.path.dirname(__file__))
from agents.crawler.kafka_producer import (
    AthenaKafkaProducer, KafkaConfig, CommitMessage,
    IssueMessage, CrawlerEventMessage
)
from agents.crawler.models import Base, Repository, Commit, Issue


# ============================================================================
# Configuration
# ============================================================================

load_dotenv()

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
POSTGRES_URL = f"postgresql://{os.getenv('POSTGRES_USER', 'athena')}:" \
               f"{os.getenv('POSTGRES_PASSWORD', 'athena_password')}@" \
               f"{os.getenv('POSTGRES_HOST', 'localhost')}:" \
               f"{os.getenv('POSTGRES_PORT', '5432')}/" \
               f"{os.getenv('POSTGRES_DB', 'athena')}"
KAFKA_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')


# ============================================================================
# Test Fixtures
# ============================================================================

TEST_REPOS = [
    ("octocat", "Hello-World"),  # Small, stable repo
    ("github", "gitignore"),     # Another small repo
]


# ============================================================================
# Database Helper
# ============================================================================

class DatabaseManager:
    """Manages database connections and operations"""

    def __init__(self, url: str):
        self.engine = create_engine(url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)
        logging.info("✓ Database tables created/verified")

    def get_session(self):
        """Get database session"""
        return self.SessionLocal()

    def verify_repository(self, full_name: str) -> Optional[Repository]:
        """Check if repository exists in database"""
        with self.get_session() as session:
            stmt = select(Repository).where(Repository.full_name == full_name)
            return session.execute(stmt).scalar_one_or_none()

    def count_commits(self, repository_id: int) -> int:
        """Count commits for a repository"""
        with self.get_session() as session:
            stmt = select(Commit).where(Commit.repository_id == repository_id)
            return len(session.execute(stmt).scalars().all())

    def count_issues(self, repository_id: int) -> int:
        """Count issues for a repository"""
        with self.get_session() as session:
            stmt = select(Issue).where(Issue.repository_id == repository_id)
            return len(session.execute(stmt).scalars().all())

    def create_test_repository(self, owner: str, repo: str):
        """Create a test repository"""
        with self.get_session() as session:
            test_repo = Repository(
                owner=owner,
                name=repo,
                full_name=f"{owner}/{repo}",
                description="Test repository",
                language="Python",
                stars=100,
                forks=10,
                watchers=50,
                github_created_at=datetime.utcnow(),
                github_updated_at=datetime.utcnow(),
                last_crawled_at=datetime.utcnow(),
                crawl_status='success',
                metadata={}
            )
            session.add(test_repo)
            session.commit()
            logging.info(f"✓ Created test repository: {owner}/{repo}")
            return test_repo.id


# ============================================================================
# Integration Tests
# ============================================================================

class IntegrationTest:
    """End-to-end integration test"""

    def __init__(self):
        self.db = DatabaseManager(POSTGRES_URL)
        self.results = {
            'infrastructure': False,
            'kafka': False,
            'database': False,
            'end_to_end': False
        }

    async def test_infrastructure(self):
        """Test 1: Verify infrastructure is running"""
        logging.info("\n" + "="*70)
        logging.info("TEST 1: Infrastructure Check")
        logging.info("="*70)

        # Test database
        try:
            self.db.create_tables()
            logging.info("✓ PostgreSQL connected and tables created")
        except Exception as e:
            logging.error(f"✗ PostgreSQL connection failed: {e}")
            return False

        # Test Kafka
        try:
            kafka_config = KafkaConfig(bootstrap_servers=KAFKA_SERVERS)
            async with AthenaKafkaProducer(kafka_config) as producer:
                await producer.send_metric(
                    "test_metric",
                    1.0,
                    {"test": "true"}
                )
            logging.info("✓ Kafka connected and message sent")
        except Exception as e:
            logging.error(f"✗ Kafka connection failed: {e}")
            return False

        self.results['infrastructure'] = True
        logging.info("✓ Infrastructure check passed\n")
        return True

    async def test_kafka_integration(self):
        """Test 2: Send data to Kafka"""
        logging.info("\n" + "="*70)
        logging.info("TEST 2: Kafka Integration")
        logging.info("="*70)

        kafka_config = KafkaConfig(bootstrap_servers=KAFKA_SERVERS)

        try:
            async with AthenaKafkaProducer(kafka_config) as producer:

                # Send crawler event
                event = CrawlerEventMessage(
                    event_type="completed",
                    repository_full_name="octocat/Hello-World",
                    status="success",
                    commits_count=10,
                    issues_count=5,
                    files_count=20,
                    duration_seconds=15.5
                )
                await producer.send_crawler_event(event)
                logging.info("✓ Sent crawler event")

                # Send commit message
                commit_msg = CommitMessage(
                    repository_id=1,
                    repository_full_name="octocat/Hello-World",
                    sha="abc1234567890123456789012345678901234567",
                    author_name="Test Author",
                    author_email="test@example.com",
                    author_login="testuser",
                    message="Test commit message",
                    files_changed=3,
                    insertions=50,
                    deletions=10,
                    committed_at=datetime.utcnow().isoformat(),
                    raw_data={}
                )
                await producer.send_commit(commit_msg)
                logging.info("✓ Sent commit message")

                # Send issue message
                issue_msg = IssueMessage(
                    repository_id=1,
                    repository_full_name="octocat/Hello-World",
                    issue_number=1,
                    title="Test Issue",
                    body="This is a test issue",
                    state="open",
                    author_login="testuser",
                    labels=["bug", "test"],
                    created_at=datetime.utcnow().isoformat(),
                    raw_data={}
                )
                await producer.send_issue(issue_msg)
                logging.info("✓ Sent issue message")

                logging.info(f"✓ Kafka stats: {producer.stats}")
                self.results['kafka'] = True
                logging.info("✓ Kafka integration test passed\n")
                return True

        except Exception as e:
            logging.error(f"✗ Kafka integration test failed: {e}\n")
            return False

    async def test_database_integration(self):
        """Test 3: Save data to PostgreSQL"""
        logging.info("\n" + "="*70)
        logging.info("TEST 3: Database Integration")
        logging.info("="*70)

        try:
            # Create test repository
            repo_id = self.db.create_test_repository("octocat", "Hello-World")

            # Verify repository was saved
            repo = self.db.verify_repository("octocat/Hello-World")

            if repo:
                logging.info(
                    f"✓ Verified repository in DB: {repo.full_name} "
                    f"(id={repo.id}, stars={repo.stars})"
                )
                self.results['database'] = True
                logging.info("✓ Database integration test passed\n")
                return True
            else:
                logging.error("✗ Repository not found in DB")
                return False

        except Exception as e:
            logging.error(f"✗ Database integration test failed: {e}\n")
            return False

    async def test_end_to_end(self):
        """Test 4: Complete flow verification"""
        logging.info("\n" + "="*70)
        logging.info("TEST 4: End-to-End Flow")
        logging.info("="*70)

        # This test verifies all previous tests passed
        if all([
            self.results['infrastructure'],
            self.results['kafka'],
            self.results['database']
        ]):
            self.results['end_to_end'] = True
            logging.info("✓ End-to-end test passed\n")
            return True
        else:
            logging.error("✗ End-to-end test failed (some components failed)\n")
            return False

    async def run_all_tests(self):
        """Run complete test suite"""
        logging.info("\n" + "🚀 " + "="*68)
        logging.info("🚀 ATHENA INTEGRATION TEST SUITE")
        logging.info("🚀 " + "="*68 + "\n")

        # Test 1: Infrastructure
        if not await self.test_infrastructure():
            logging.error("❌ Infrastructure test failed. Cannot proceed.")
            return False

        # Test 2: Kafka
        if not await self.test_kafka_integration():
            logging.error("❌ Kafka integration test failed.")

        # Test 3: Database
        if not await self.test_database_integration():
            logging.error("❌ Database integration test failed.")

        # Test 4: End-to-End
        await self.test_end_to_end()

        # Print summary
        self.print_summary()

        return self.results['end_to_end']

    def print_summary(self):
        """Print test summary"""
        logging.info("\n" + "="*70)
        logging.info("TEST SUMMARY")
        logging.info("="*70)

        for test_name, passed in self.results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logging.info(f"{status} - {test_name.replace('_', ' ').title()}")

        all_passed = all(self.results.values())
        logging.info("="*70)

        if all_passed:
            logging.info("✅ ALL TESTS PASSED!")
            logging.info("\n🎉 Athena is ready to go!")
            logging.info("\nNext steps:")
            logging.info("1. Start building your crawler application")
            logging.info("2. Monitor Kafka topics in Kafka UI: http://localhost:8080")
            logging.info("3. Check Grafana dashboards: http://localhost:3000")
            logging.info("4. View data in PostgreSQL")
        else:
            logging.error("❌ SOME TESTS FAILED")
            logging.error("\nPlease check:")
            logging.error("1. Docker containers are running: docker-compose ps")
            logging.error("2. Kafka topics created: bash create_topics.sh")
            logging.error("3. .env file configured correctly")

        logging.info("="*70 + "\n")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run integration tests"""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Run tests
    test = IntegrationTest()
    success = await test.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
