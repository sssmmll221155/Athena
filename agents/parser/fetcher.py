"""
File Content Fetcher
Fetches file contents from GitHub API for parsing.
"""

import asyncio
import aiohttp
import logging
import base64
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FileContent:
    """Container for fetched file content"""
    file_id: int
    repository_id: int
    commit_sha: str
    file_path: str
    content: Optional[str] = None
    encoding: str = 'utf-8'
    size_bytes: int = 0
    is_binary: bool = False
    is_too_large: bool = False
    fetch_status: str = 'pending'  # pending, success, failed, skipped
    error: Optional[str] = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FetchResult:
    """Result of fetching multiple files"""
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    files: List[FileContent] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ============================================================================
# File Content Fetcher
# ============================================================================

class FileContentFetcher:
    """
    Fetches file contents from GitHub API.
    Handles batching, rate limiting, and error recovery.
    """

    def __init__(
        self,
        github_token: str,
        max_file_size_mb: int = 1,
        max_concurrent: int = 10,
        timeout: int = 30,
        base_url: str = "https://api.github.com"
    ):
        """
        Initialize file content fetcher.

        Args:
            github_token: GitHub API token
            max_file_size_mb: Maximum file size to fetch in MB
            max_concurrent: Maximum concurrent requests
            timeout: Request timeout in seconds
            base_url: GitHub API base URL
        """
        self.github_token = github_token
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.base_url = base_url

        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Athena-Parser/1.0"
        }

        self.session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Binary file extensions to skip
        self.binary_extensions = {
            '.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe', '.bin',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
            '.mp3', '.mp4', '.wav', '.avi', '.mov', '.pdf', '.zip',
            '.tar', '.gz', '.7z', '.rar', '.whl', '.egg'
        }

    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def is_binary_file(self, file_path: str) -> bool:
        """Check if file is likely binary based on extension"""
        import os
        _, ext = os.path.splitext(file_path.lower())
        return ext in self.binary_extensions

    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension"""
        import os
        _, ext = os.path.splitext(file_path.lower())

        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.jl': 'julia',
            '.m': 'matlab',
            '.sh': 'bash',
            '.sql': 'sql',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css',
        }

        return language_map.get(ext)

    async def fetch_file_content(
        self,
        owner: str,
        repo: str,
        file_path: str,
        ref: str,
        file_id: int,
        repository_id: int
    ) -> FileContent:
        """
        Fetch content of a single file from GitHub.

        Args:
            owner: Repository owner
            repo: Repository name
            file_path: Path to file in repository
            ref: Git reference (commit SHA or branch)
            file_id: Database file ID
            repository_id: Database repository ID

        Returns:
            FileContent object
        """
        result = FileContent(
            file_id=file_id,
            repository_id=repository_id,
            commit_sha=ref,
            file_path=file_path
        )

        # Check if binary
        if self.is_binary_file(file_path):
            result.is_binary = True
            result.fetch_status = 'skipped'
            result.error = 'Binary file'
            logger.debug(f"Skipping binary file: {file_path}")
            return result

        # Check language support (currently only Python)
        language = self.detect_language(file_path)
        if language != 'python':
            result.fetch_status = 'skipped'
            result.error = f'Unsupported language: {language}'
            logger.debug(f"Skipping non-Python file: {file_path}")
            return result

        try:
            async with self._semaphore:
                url = f"{self.base_url}/repos/{owner}/{repo}/contents/{file_path}?ref={ref}"

                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Check file size
                        size = data.get('size', 0)
                        result.size_bytes = size

                        if size > self.max_file_size_bytes:
                            result.is_too_large = True
                            result.fetch_status = 'skipped'
                            result.error = f'File too large: {size} bytes'
                            logger.warning(f"Skipping large file {file_path}: {size} bytes")
                            return result

                        # Decode content
                        encoding = data.get('encoding', 'base64')
                        content_encoded = data.get('content', '')

                        if encoding == 'base64':
                            try:
                                content_bytes = base64.b64decode(content_encoded)
                                result.content = content_bytes.decode('utf-8')
                                result.encoding = 'utf-8'
                                result.fetch_status = 'success'
                                logger.debug(f"Fetched {file_path}: {len(result.content)} chars")
                            except UnicodeDecodeError:
                                result.fetch_status = 'failed'
                                result.error = 'Unicode decode error (likely binary)'
                                logger.warning(f"Unicode decode failed for {file_path}")
                        else:
                            result.fetch_status = 'failed'
                            result.error = f'Unknown encoding: {encoding}'
                            logger.warning(f"Unknown encoding for {file_path}: {encoding}")

                    elif response.status == 404:
                        result.fetch_status = 'failed'
                        result.error = 'File not found'
                        logger.warning(f"File not found: {file_path}")

                    elif response.status == 403:
                        result.fetch_status = 'failed'
                        result.error = 'Rate limit exceeded'
                        logger.error(f"Rate limit exceeded fetching {file_path}")

                    else:
                        result.fetch_status = 'failed'
                        result.error = f'HTTP {response.status}'
                        logger.error(f"Failed to fetch {file_path}: HTTP {response.status}")

        except asyncio.TimeoutError:
            result.fetch_status = 'failed'
            result.error = 'Request timeout'
            logger.error(f"Timeout fetching {file_path}")

        except Exception as e:
            result.fetch_status = 'failed'
            result.error = str(e)
            logger.error(f"Error fetching {file_path}: {e}")

        return result

    async def fetch_files_batch(
        self,
        owner: str,
        repo: str,
        files: List[Tuple[int, int, str, str]],  # (file_id, repo_id, path, commit_sha)
    ) -> FetchResult:
        """
        Fetch multiple files in batch.

        Args:
            owner: Repository owner
            repo: Repository name
            files: List of tuples (file_id, repository_id, file_path, commit_sha)

        Returns:
            FetchResult with statistics
        """
        logger.info(f"Fetching {len(files)} files from {owner}/{repo}")

        # Create tasks
        tasks = [
            self.fetch_file_content(
                owner=owner,
                repo=repo,
                file_path=file_path,
                ref=commit_sha,
                file_id=file_id,
                repository_id=repo_id
            )
            for file_id, repo_id, file_path, commit_sha in files
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        fetch_result = FetchResult(total_files=len(files))

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                file_id, repo_id, file_path, commit_sha = files[i]
                logger.error(f"Exception fetching {file_path}: {result}")
                fetch_result.failed += 1
                fetch_result.errors.append(f"{file_path}: {str(result)}")

                # Create failed FileContent
                failed_content = FileContent(
                    file_id=file_id,
                    repository_id=repo_id,
                    commit_sha=commit_sha,
                    file_path=file_path,
                    fetch_status='failed',
                    error=str(result)
                )
                fetch_result.files.append(failed_content)
            else:
                fetch_result.files.append(result)
                if result.fetch_status == 'success':
                    fetch_result.successful += 1
                elif result.fetch_status == 'skipped':
                    fetch_result.skipped += 1
                else:
                    fetch_result.failed += 1

        logger.info(
            f"Fetch complete: {fetch_result.successful} successful, "
            f"{fetch_result.skipped} skipped, {fetch_result.failed} failed"
        )

        return fetch_result


# ============================================================================
# Helper Functions
# ============================================================================

async def fetch_repository_files(
    github_token: str,
    owner: str,
    repo: str,
    files: List[Tuple[int, int, str, str]],
    max_file_size_mb: int = 1,
    max_concurrent: int = 10
) -> FetchResult:
    """
    Convenience function to fetch files from a repository.

    Args:
        github_token: GitHub API token
        owner: Repository owner
        repo: Repository name
        files: List of tuples (file_id, repository_id, file_path, commit_sha)
        max_file_size_mb: Maximum file size in MB
        max_concurrent: Maximum concurrent requests

    Returns:
        FetchResult
    """
    async with FileContentFetcher(
        github_token=github_token,
        max_file_size_mb=max_file_size_mb,
        max_concurrent=max_concurrent
    ) as fetcher:
        return await fetcher.fetch_files_batch(owner, repo, files)
