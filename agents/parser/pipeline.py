"""
Parsing Pipeline Orchestrator
Coordinates fetching, parsing, and writing of code files.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from sqlalchemy import select, and_, text
from sqlalchemy.orm import Session
from tqdm import tqdm

from agents.crawler.models import File, Repository, Commit
from agents.parser.fetcher import FileContentFetcher, FileContent
from agents.parser.python_parser import PythonASTParser, ParsedPythonFile
from agents.parser.writer import DatabaseWriter, WriteResult
from agents.parser.config import ParserConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PipelineStats:
    """Statistics from pipeline execution"""
    total_files: int = 0
    fetched_files: int = 0
    parsed_files: int = 0
    written_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0

    total_functions: int = 0
    total_classes: int = 0
    total_imports: int = 0

    fetch_duration_ms: int = 0
    parse_duration_ms: int = 0
    write_duration_ms: int = 0
    total_duration_ms: int = 0

    errors: List[str] = field(default_factory=list)


@dataclass
class FileToProcess:
    """File information for processing"""
    file_id: int
    repository_id: int
    owner: str
    repo: str
    file_path: str
    commit_sha: str


# ============================================================================
# Parsing Pipeline
# ============================================================================

class ParsingPipeline:
    """
    Orchestrates the complete parsing pipeline:
    1. Fetch file contents from GitHub
    2. Parse files with AST parser
    3. Write parsed data to database
    """

    def __init__(
        self,
        session: Session,
        config: ParserConfig,
        github_token: str
    ):
        """
        Initialize parsing pipeline.

        Args:
            session: SQLAlchemy database session
            config: Parser configuration
            github_token: GitHub API token
        """
        self.session = session
        self.config = config
        self.github_token = github_token

        self.parser = PythonASTParser()
        self.writer = DatabaseWriter(session, batch_size=config.BATCH_INSERT_SIZE)

    def get_files_for_commit(
        self,
        repository_id: int,
        commit_sha: str,
        limit: Optional[int] = None
    ) -> List[FileToProcess]:
        """
        Get files to parse for a specific commit.

        Args:
            repository_id: Repository database ID
            commit_sha: Commit SHA
            limit: Maximum number of files to return

        Returns:
            List of FileToProcess objects
        """
        # Get repository info
        repo = self.session.get(Repository, repository_id)
        if not repo:
            logger.error(f"Repository {repository_id} not found")
            return []

        # Get files from commit_files that are Python files
        query = """
            SELECT DISTINCT
                f.id as file_id,
                f.repository_id,
                f.path
            FROM files f
            JOIN commit_files cf ON cf.file_id = f.id
            WHERE f.repository_id = :repo_id
            AND cf.committed_at = (
                SELECT committed_at FROM commits
                WHERE repository_id = :repo_id AND sha = :commit_sha
            )
            AND f.extension = '.py'
            AND f.is_deleted = FALSE
        """

        if limit:
            query += f" LIMIT {limit}"

        result = self.session.execute(
            query,
            {'repo_id': repository_id, 'commit_sha': commit_sha}
        )

        files_to_process = []
        for row in result:
            files_to_process.append(FileToProcess(
                file_id=row.file_id,
                repository_id=row.repository_id,
                owner=repo.owner,
                repo=repo.name,
                file_path=row.path,
                commit_sha=commit_sha
            ))

        logger.info(f"Found {len(files_to_process)} Python files to process for commit {commit_sha[:7]}")
        return files_to_process

    def get_unparsed_files(
        self,
        repository_id: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[FileToProcess]:
        """
        Get files that haven't been parsed yet (backfill mode).

        Args:
            repository_id: Optional repository ID to filter
            limit: Maximum number of files to return

        Returns:
            List of FileToProcess objects
        """
        # Get Python files that don't have parsed records
        query = """
            SELECT DISTINCT
                f.id as file_id,
                f.repository_id,
                r.owner,
                r.name as repo_name,
                f.path,
                c.sha as commit_sha
            FROM files f
            JOIN repositories r ON r.id = f.repository_id
            JOIN commit_files cf ON cf.file_id = f.id
            JOIN commits c ON c.id = cf.commit_id
            LEFT JOIN parsed_files pf ON pf.file_id = f.id AND pf.commit_sha = c.sha
            WHERE f.extension = '.py'
            AND f.is_deleted = FALSE
            AND pf.id IS NULL
        """

        if repository_id:
            query += f" AND f.repository_id = {repository_id}"

        if limit:
            query += f" LIMIT {limit}"

        result = self.session.execute(text(query))

        files_to_process = []
        for row in result:
            files_to_process.append(FileToProcess(
                file_id=row.file_id,
                repository_id=row.repository_id,
                owner=row.owner,
                repo=row.repo_name,
                file_path=row.path,
                commit_sha=row.commit_sha
            ))

        logger.info(f"Found {len(files_to_process)} unparsed Python files")
        return files_to_process

    async def fetch_files(
        self,
        files: List[FileToProcess]
    ) -> List[FileContent]:
        """
        Fetch file contents from GitHub.

        Args:
            files: List of files to fetch

        Returns:
            List of FileContent objects
        """
        if not files:
            return []

        # Group files by repository
        files_by_repo: Dict[Tuple[str, str], List[FileToProcess]] = {}
        for file in files:
            key = (file.owner, file.repo)
            if key not in files_by_repo:
                files_by_repo[key] = []
            files_by_repo[key].append(file)

        all_file_contents = []

        # Fetch files for each repository
        async with FileContentFetcher(
            github_token=self.github_token,
            max_file_size_mb=self.config.MAX_FILE_SIZE_MB,
            max_concurrent=self.config.CONCURRENT_FILES
        ) as fetcher:
            for (owner, repo), repo_files in files_by_repo.items():
                logger.info(f"Fetching {len(repo_files)} files from {owner}/{repo}")

                # Convert to tuple format for fetcher
                file_tuples = [
                    (f.file_id, f.repository_id, f.file_path, f.commit_sha)
                    for f in repo_files
                ]

                fetch_result = await fetcher.fetch_files_batch(owner, repo, file_tuples)
                all_file_contents.extend(fetch_result.files)

        return all_file_contents

    def parse_files(
        self,
        file_contents: List[FileContent],
        show_progress: bool = True
    ) -> List[Tuple[ParsedPythonFile, int, int, str]]:
        """
        Parse fetched file contents.

        Args:
            file_contents: List of FileContent objects
            show_progress: Whether to show progress bar

        Returns:
            List of tuples (ParsedPythonFile, file_id, repository_id, commit_sha)
        """
        parsed_files = []

        # Filter only successfully fetched files
        files_to_parse = [
            fc for fc in file_contents
            if fc.fetch_status == 'success' and fc.content
        ]

        logger.info(f"Parsing {len(files_to_parse)} files")

        iterator = tqdm(files_to_parse, desc="Parsing files") if show_progress else files_to_parse

        for file_content in iterator:
            try:
                parsed_file = self.parser.parse_file(
                    file_path=file_content.file_path,
                    content=file_content.content
                )

                parsed_files.append((
                    parsed_file,
                    file_content.file_id,
                    file_content.repository_id,
                    file_content.commit_sha
                ))

            except Exception as e:
                logger.error(f"Error parsing {file_content.file_path}: {e}")
                # Continue with next file

        logger.info(f"Parsed {len(parsed_files)} files successfully")
        return parsed_files

    def write_parsed_files(
        self,
        parsed_files: List[Tuple[ParsedPythonFile, int, int, str]]
    ) -> WriteResult:
        """
        Write parsed files to database.

        Args:
            parsed_files: List of tuples (ParsedPythonFile, file_id, repository_id, commit_sha)

        Returns:
            WriteResult with statistics
        """
        logger.info(f"Writing {len(parsed_files)} parsed files to database")

        result = self.writer.write_batch(parsed_files)

        logger.info(
            f"Write complete: {result.successful} successful, "
            f"{result.skipped} skipped, {result.failed} failed"
        )

        return result

    async def run_for_commit(
        self,
        repository_id: int,
        commit_sha: str,
        limit: Optional[int] = None,
        show_progress: bool = True
    ) -> PipelineStats:
        """
        Run pipeline for a specific commit.

        Args:
            repository_id: Repository database ID
            commit_sha: Commit SHA
            limit: Maximum number of files to process
            show_progress: Whether to show progress bars

        Returns:
            PipelineStats with execution statistics
        """
        start_time = datetime.utcnow()
        stats = PipelineStats()

        logger.info(f"Starting pipeline for commit {commit_sha[:7]} in repository {repository_id}")

        try:
            # 1. Get files to process
            files_to_process = self.get_files_for_commit(
                repository_id=repository_id,
                commit_sha=commit_sha,
                limit=limit
            )
            stats.total_files = len(files_to_process)

            if not files_to_process:
                logger.warning("No files to process")
                return stats

            # 2. Fetch file contents
            fetch_start = datetime.utcnow()
            file_contents = await self.fetch_files(files_to_process)
            fetch_end = datetime.utcnow()
            stats.fetch_duration_ms = int((fetch_end - fetch_start).total_seconds() * 1000)
            stats.fetched_files = sum(1 for fc in file_contents if fc.fetch_status == 'success')

            # 3. Parse files
            parse_start = datetime.utcnow()
            parsed_files = self.parse_files(file_contents, show_progress=show_progress)
            parse_end = datetime.utcnow()
            stats.parse_duration_ms = int((parse_end - parse_start).total_seconds() * 1000)
            stats.parsed_files = len(parsed_files)

            # 4. Write to database
            write_start = datetime.utcnow()
            write_result = self.write_parsed_files(parsed_files)
            write_end = datetime.utcnow()
            stats.write_duration_ms = int((write_end - write_start).total_seconds() * 1000)
            stats.written_files = write_result.successful
            stats.skipped_files = write_result.skipped
            stats.failed_files = write_result.failed
            stats.total_functions = write_result.total_functions
            stats.total_classes = write_result.total_classes
            stats.total_imports = write_result.total_imports
            stats.errors = write_result.errors

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            stats.errors.append(str(e))
            raise

        finally:
            end_time = datetime.utcnow()
            stats.total_duration_ms = int((end_time - start_time).total_seconds() * 1000)

        logger.info(f"Pipeline complete in {stats.total_duration_ms}ms")
        return stats

    async def run_backfill(
        self,
        repository_id: Optional[int] = None,
        limit: Optional[int] = None,
        show_progress: bool = True
    ) -> PipelineStats:
        """
        Run pipeline to backfill unparsed files.

        Args:
            repository_id: Optional repository ID to filter
            limit: Maximum number of files to process
            show_progress: Whether to show progress bars

        Returns:
            PipelineStats with execution statistics
        """
        start_time = datetime.utcnow()
        stats = PipelineStats()

        logger.info("Starting backfill pipeline")

        try:
            # 1. Get unparsed files
            files_to_process = self.get_unparsed_files(
                repository_id=repository_id,
                limit=limit
            )
            stats.total_files = len(files_to_process)

            if not files_to_process:
                logger.info("No unparsed files found")
                return stats

            # 2. Fetch file contents
            fetch_start = datetime.utcnow()
            file_contents = await self.fetch_files(files_to_process)
            fetch_end = datetime.utcnow()
            stats.fetch_duration_ms = int((fetch_end - fetch_start).total_seconds() * 1000)
            stats.fetched_files = sum(1 for fc in file_contents if fc.fetch_status == 'success')

            # 3. Parse files
            parse_start = datetime.utcnow()
            parsed_files = self.parse_files(file_contents, show_progress=show_progress)
            parse_end = datetime.utcnow()
            stats.parse_duration_ms = int((parse_end - parse_start).total_seconds() * 1000)
            stats.parsed_files = len(parsed_files)

            # 4. Write to database
            write_start = datetime.utcnow()
            write_result = self.write_parsed_files(parsed_files)
            write_end = datetime.utcnow()
            stats.write_duration_ms = int((write_end - write_start).total_seconds() * 1000)
            stats.written_files = write_result.successful
            stats.skipped_files = write_result.skipped
            stats.failed_files = write_result.failed
            stats.total_functions = write_result.total_functions
            stats.total_classes = write_result.total_classes
            stats.total_imports = write_result.total_imports
            stats.errors = write_result.errors

        except Exception as e:
            logger.error(f"Backfill pipeline error: {e}")
            stats.errors.append(str(e))
            raise

        finally:
            end_time = datetime.utcnow()
            stats.total_duration_ms = int((end_time - start_time).total_seconds() * 1000)

        logger.info(f"Backfill pipeline complete in {stats.total_duration_ms}ms")
        return stats


# ============================================================================
# Helper Functions
# ============================================================================

async def run_parser_pipeline(
    session: Session,
    github_token: str,
    repository_id: Optional[int] = None,
    commit_sha: Optional[str] = None,
    backfill: bool = False,
    limit: Optional[int] = None,
    config: Optional[ParserConfig] = None
) -> PipelineStats:
    """
    Convenience function to run the parsing pipeline.

    Args:
        session: SQLAlchemy database session
        github_token: GitHub API token
        repository_id: Optional repository ID
        commit_sha: Optional commit SHA
        backfill: Run in backfill mode
        limit: Maximum files to process
        config: Parser configuration (uses default if not provided)

    Returns:
        PipelineStats
    """
    if config is None:
        config = ParserConfig()

    pipeline = ParsingPipeline(
        session=session,
        config=config,
        github_token=github_token
    )

    if backfill:
        return await pipeline.run_backfill(
            repository_id=repository_id,
            limit=limit
        )
    elif commit_sha and repository_id:
        return await pipeline.run_for_commit(
            repository_id=repository_id,
            commit_sha=commit_sha,
            limit=limit
        )
    else:
        raise ValueError("Must specify either backfill=True or both repository_id and commit_sha")
