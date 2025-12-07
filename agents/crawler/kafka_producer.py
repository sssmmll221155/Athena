"""
Athena Kafka Producer
High-performance async Kafka producer with batching and error handling.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
import backoff


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class KafkaConfig:
    """Kafka producer configuration"""
    bootstrap_servers: str = "localhost:9092"
    client_id: str = "athena-producer"
    compression_type = None
    acks: int = 1  # 0=no ack, 1=leader ack, all=full replication
    max_batch_size: int = 16384  # 16KB
    linger_ms: int = 10  # Wait up to 10ms for batching
    max_request_size: int = 1048576  # 1MB
    request_timeout_ms: int = 30000
    retry_backoff_ms: int = 100


class TopicType(Enum):
    """Kafka topic names"""
    RAW_COMMITS = "raw_commits"
    RAW_ISSUES = "raw_issues"
    RAW_PRS = "raw_prs"
    RAW_FILES = "raw_files"
    PARSED_AST = "parsed_ast"
    EXTRACTED_FEATURES = "extracted_features"
    CODE_EMBEDDINGS = "code_embeddings"
    PREDICTIONS = "predictions"
    FEEDBACK_EVENTS = "feedback_events"
    CRAWLER_EVENTS = "crawler_events"
    ERRORS = "errors"
    METRICS = "metrics"


# ============================================================================
# Message Schemas
# ============================================================================

@dataclass
class CommitMessage:
    """Schema for commit messages"""
    repository_id: int
    repository_full_name: str
    sha: str
    author_name: Optional[str]
    author_email: Optional[str]
    author_login: Optional[str]
    message: str
    files_changed: int
    insertions: int
    deletions: int
    committed_at: str  # ISO format
    raw_data: Dict[str, Any]
    timestamp: str = None  # Processing timestamp

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class IssueMessage:
    """Schema for issue messages"""
    repository_id: int
    repository_full_name: str
    issue_number: int
    title: str
    body: Optional[str]
    state: str
    author_login: Optional[str]
    labels: List[str]
    created_at: str
    raw_data: Dict[str, Any]
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class FileMessage:
    """Schema for file change messages"""
    repository_id: int
    commit_sha: str
    path: str
    status: str
    additions: int
    deletions: int
    patch: Optional[str]
    committed_at: str
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class CrawlerEventMessage:
    """Schema for crawler events"""
    event_type: str  # started, completed, failed, rate_limited
    repository_full_name: str
    status: str
    commits_count: int
    issues_count: int
    files_count: int
    duration_seconds: float
    error: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


# ============================================================================
# Kafka Producer
# ============================================================================

class AthenaKafkaProducer:
    """
    High-performance async Kafka producer with batching and retry logic
    """

    def __init__(self, config: KafkaConfig):
        self.config = config
        self.producer: Optional[AIOKafkaProducer] = None
        self._closed = False
        self.stats = {
            'messages_sent': 0,
            'messages_failed': 0,
            'bytes_sent': 0
        }

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def start(self):
        """Initialize Kafka producer"""
        if self.producer is None:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                client_id=self.config.client_id,
                compression_type=self.config.compression_type,
                acks=self.config.acks,
                max_batch_size=self.config.max_batch_size,
                linger_ms=self.config.linger_ms,
                max_request_size=self.config.max_request_size,
                request_timeout_ms=self.config.request_timeout_ms,
                retry_backoff_ms=self.config.retry_backoff_ms,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            await self.producer.start()
            logging.info("Kafka producer started")

    async def close(self):
        """Close Kafka producer gracefully"""
        if self.producer and not self._closed:
            await self.producer.flush()  # Send remaining messages
            await self.producer.stop()
            self._closed = True
            logging.info(f"Kafka producer closed. Stats: {self.stats}")

    @backoff.on_exception(
        backoff.expo,
        KafkaError,
        max_tries=3,
        jitter=backoff.full_jitter
    )
    async def send_message(
        self,
        topic: TopicType,
        message: Dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Send a message to Kafka

        Args:
            topic: Topic to send to
            message: Message payload (dict)
            key: Optional message key for partitioning
            headers: Optional message headers

        Returns:
            True if sent successfully
        """
        try:
            # Convert headers to bytes
            kafka_headers = None
            if headers:
                kafka_headers = [(k, v.encode('utf-8')) for k, v in headers.items()]

            # Send message
            future = await self.producer.send(
                topic.value,
                value=message,
                key=key,
                headers=kafka_headers
            )

            # Wait for confirmation
            record_metadata = await future

            # Update stats
            self.stats['messages_sent'] += 1
            message_size = len(json.dumps(message).encode('utf-8'))
            self.stats['bytes_sent'] += message_size

            logging.debug(
                f"Message sent to {topic.value} "
                f"(partition={record_metadata.partition}, "
                f"offset={record_metadata.offset}, "
                f"size={message_size} bytes)"
            )

            return True

        except KafkaError as e:
            self.stats['messages_failed'] += 1
            logging.error(f"Failed to send message to {topic.value}: {e}")
            raise

    async def send_batch(
        self,
        topic: TopicType,
        messages: List[Dict[str, Any]],
        key_func: Optional[callable] = None
    ) -> int:
        """
        Send multiple messages efficiently

        Args:
            topic: Topic to send to
            messages: List of message payloads
            key_func: Optional function to extract key from message

        Returns:
            Number of messages sent successfully
        """
        tasks = []
        for msg in messages:
            key = key_func(msg) if key_func else None
            task = self.send_message(topic, msg, key)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = sum(1 for r in results if r is True)

        logging.info(
            f"Batch sent to {topic.value}: "
            f"{successful}/{len(messages)} successful"
        )

        return successful

    # ========================================================================
    # Convenience Methods for Specific Message Types
    # ========================================================================

    async def send_commit(
        self,
        commit: CommitMessage,
        partition_key: Optional[str] = None
    ) -> bool:
        """Send commit message"""
        key = partition_key or commit.repository_full_name
        return await self.send_message(
            TopicType.RAW_COMMITS,
            asdict(commit),
            key=key,
            headers={
                'type': 'commit',
                'repo': commit.repository_full_name,
                'timestamp': commit.timestamp
            }
        )

    async def send_commits(self, commits: List[CommitMessage]) -> int:
        """Send multiple commits"""
        return await self.send_batch(
            TopicType.RAW_COMMITS,
            [asdict(c) for c in commits],
            key_func=lambda m: m['repository_full_name']
        )

    async def send_issue(self, issue: IssueMessage) -> bool:
        """Send issue message"""
        return await self.send_message(
            TopicType.RAW_ISSUES,
            asdict(issue),
            key=issue.repository_full_name,
            headers={
                'type': 'issue',
                'repo': issue.repository_full_name,
                'timestamp': issue.timestamp
            }
        )

    async def send_issues(self, issues: List[IssueMessage]) -> int:
        """Send multiple issues"""
        return await self.send_batch(
            TopicType.RAW_ISSUES,
            [asdict(i) for i in issues],
            key_func=lambda m: m['repository_full_name']
        )

    async def send_file_change(self, file: FileMessage) -> bool:
        """Send file change message"""
        return await self.send_message(
            TopicType.RAW_FILES,
            asdict(file),
            key=file.path,
            headers={
                'type': 'file_change',
                'timestamp': file.timestamp
            }
        )

    async def send_crawler_event(self, event: CrawlerEventMessage) -> bool:
        """Send crawler event"""
        return await self.send_message(
            TopicType.CRAWLER_EVENTS,
            asdict(event),
            key=event.repository_full_name,
            headers={
                'type': 'crawler_event',
                'event_type': event.event_type,
                'timestamp': event.timestamp
            }
        )

    async def send_error(
        self,
        error_type: str,
        error_message: str,
        context: Dict[str, Any]
    ) -> bool:
        """Send error message"""
        message = {
            'error_type': error_type,
            'error_message': error_message,
            'context': context,
            'timestamp': datetime.utcnow().isoformat()
        }
        return await self.send_message(
            TopicType.ERRORS,
            message,
            key=error_type
        )

    async def send_metric(
        self,
        metric_name: str,
        metric_value: float,
        tags: Dict[str, str]
    ) -> bool:
        """Send metric message"""
        message = {
            'metric_name': metric_name,
            'metric_value': metric_value,
            'tags': tags,
            'timestamp': datetime.utcnow().isoformat()
        }
        return await self.send_message(
            TopicType.METRICS,
            message,
            key=metric_name
        )


# ============================================================================
# Example Usage
# ============================================================================

async def example_usage():
    """Example of using the Kafka producer"""

    # Configure
    config = KafkaConfig(
        bootstrap_servers="localhost:9092",
        client_id="athena-crawler"
    )

    # Create producer
    async with AthenaKafkaProducer(config) as producer:

        # Send single commit
        commit = CommitMessage(
            repository_id=1,
            repository_full_name="octocat/Hello-World",
            sha="abc123",
            author_name="Test Author",
            author_email="test@example.com",
            author_login="testuser",
            message="Fix bug in authentication",
            files_changed=3,
            insertions=45,
            deletions=12,
            committed_at="2024-01-01T12:00:00Z",
            raw_data={}
        )
        await producer.send_commit(commit)

        # Send multiple commits
        commits = [commit for _ in range(10)]
        sent_count = await producer.send_commits(commits)
        print(f"Sent {sent_count} commits")

        # Send crawler event
        event = CrawlerEventMessage(
            event_type="completed",
            repository_full_name="octocat/Hello-World",
            status="success",
            commits_count=100,
            issues_count=25,
            files_count=50,
            duration_seconds=45.2
        )
        await producer.send_crawler_event(event)

        # Send error
        await producer.send_error(
            error_type="RateLimitExceeded",
            error_message="GitHub API rate limit exceeded",
            context={"repo": "octocat/Hello-World"}
        )

        print(f"Producer stats: {producer.stats}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
