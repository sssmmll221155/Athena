"""
GitHub Repository Crawler
Fetches repository metadata, commits, and issues from GitHub API
"""

import asyncio
import aiohttp
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time

logger = logging.getLogger(__name__)


@dataclass
class CrawlerConfig:
    """Configuration for GitHub crawler"""
    github_token: str
    max_concurrent_requests: int = 10
    rate_limit_delay: float = 1.0  # seconds between requests
    timeout: int = 30  # request timeout in seconds
    base_url: str = "https://api.github.com"


@dataclass
class CrawlResult:
    """Result of crawling a single repository"""
    owner: str
    repo: str
    success: bool
    metadata: Optional[Dict[str, Any]] = None
    commits: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    commits_count: int = 0
    issues_count: int = 0
    error: Optional[str] = None
    crawl_time: float = 0.0  # duration in seconds


class RepositoryCrawler:
    """
    Asynchronous GitHub repository crawler
    Fetches repository data, commits, and issues
    """

    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers = {
            "Authorization": f"token {config.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Athena-Crawler/1.0"
        }
        self._rate_limit_lock = asyncio.Lock()
        self._last_request_time = 0.0

    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _rate_limit(self):
        """Enforce rate limiting between requests"""
        async with self._rate_limit_lock:
            now = time.time()
            time_since_last = now - self._last_request_time
            if time_since_last < self.config.rate_limit_delay:
                await asyncio.sleep(self.config.rate_limit_delay - time_since_last)
            self._last_request_time = time.time()

    async def _make_request(self, url: str) -> Optional[Dict[str, Any]]:
        """Make a rate-limited GET request to GitHub API"""
        await self._rate_limit()

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    logger.warning(f"Resource not found: {url}")
                    return None
                elif response.status == 403:
                    # Rate limit hit
                    reset_time = response.headers.get('X-RateLimit-Reset')
                    logger.error(f"Rate limit exceeded. Reset at: {reset_time}")
                    return None
                else:
                    logger.error(f"Request failed: {url} - Status {response.status}")
                    return None
        except asyncio.TimeoutError:
            logger.error(f"Request timeout: {url}")
            return None
        except Exception as e:
            logger.error(f"Request error: {url} - {e}")
            return None

    async def _make_paginated_request(
        self,
        url: str,
        max_pages: int = 10,
        per_page: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch paginated results from GitHub API"""
        results = []

        for page in range(1, max_pages + 1):
            # Use & if URL already has parameters, otherwise use ?
            separator = '&' if '?' in url else '?'
            page_url = f"{url}{separator}per_page={per_page}&page={page}"
            data = await self._make_request(page_url)

            if not data:
                break

            if isinstance(data, list):
                results.extend(data)
                if len(data) < per_page:
                    # Last page
                    break
            else:
                # Single object response
                results.append(data)
                break

        return results

    async def fetch_repository_metadata(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """Fetch repository metadata"""
        url = f"{self.config.base_url}/repos/{owner}/{repo}"
        logger.info(f"Fetching metadata for {owner}/{repo}")
        return await self._make_request(url)

    async def fetch_commits(
        self,
        owner: str,
        repo: str,
        max_commits: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch recent commits from repository"""
        url = f"{self.config.base_url}/repos/{owner}/{repo}/commits"
        logger.info(f"Fetching commits for {owner}/{repo}")

        commits = await self._make_paginated_request(
            url,
            max_pages=max_commits // 100,
            per_page=min(100, max_commits)
        )

        # Fetch detailed commit info (includes file changes and stats)
        detailed_commits = []
        for commit in commits[:max_commits]:
            sha = commit.get('sha')
            if sha:
                detail_url = f"{self.config.base_url}/repos/{owner}/{repo}/commits/{sha}"
                detailed = await self._make_request(detail_url)
                if detailed:
                    detailed_commits.append(detailed)

        logger.info(f"Fetched {len(detailed_commits)} commits for {owner}/{repo}")
        return detailed_commits

    async def fetch_issues(
        self,
        owner: str,
        repo: str,
        state: str = "all",
        max_issues: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch issues from repository"""
        url = f"{self.config.base_url}/repos/{owner}/{repo}/issues"
        url += f"?state={state}"

        logger.info(f"Fetching issues for {owner}/{repo}")

        issues = await self._make_paginated_request(
            url,
            max_pages=max_issues // 100,
            per_page=min(100, max_issues)
        )

        logger.info(f"Fetched {len(issues)} issues for {owner}/{repo}")
        return issues[:max_issues]

    async def crawl_repository(
        self,
        owner: str,
        repo: str,
        fetch_commits: bool = True,
        fetch_issues: bool = True,
        max_commits: int = 100,
        max_issues: int = 100
    ) -> CrawlResult:
        """
        Crawl a single repository and fetch all requested data
        """
        start_time = time.time()

        try:
            # Fetch metadata
            metadata = await self.fetch_repository_metadata(owner, repo)

            if not metadata:
                return CrawlResult(
                    owner=owner,
                    repo=repo,
                    success=False,
                    error="Failed to fetch repository metadata"
                )

            commits = []
            issues = []

            # Fetch commits if requested
            if fetch_commits:
                commits = await self.fetch_commits(owner, repo, max_commits)

            # Fetch issues if requested
            if fetch_issues:
                issues = await self.fetch_issues(owner, repo, max_issues=max_issues)

            crawl_time = time.time() - start_time

            return CrawlResult(
                owner=owner,
                repo=repo,
                success=True,
                metadata=metadata,
                commits=commits,
                issues=issues,
                commits_count=len(commits),
                issues_count=len(issues),
                crawl_time=crawl_time
            )

        except Exception as e:
            logger.error(f"Error crawling {owner}/{repo}: {e}")
            return CrawlResult(
                owner=owner,
                repo=repo,
                success=False,
                error=str(e),
                crawl_time=time.time() - start_time
            )

    async def crawl_repositories(
        self,
        repositories: List[Tuple[str, str]],
        fetch_commits: bool = True,
        fetch_issues: bool = True,
        max_commits: int = 100,
        max_issues: int = 100
    ) -> List[CrawlResult]:
        """
        Crawl multiple repositories concurrently

        Args:
            repositories: List of (owner, repo) tuples
            fetch_commits: Whether to fetch commits
            fetch_issues: Whether to fetch issues
            max_commits: Maximum commits to fetch per repo
            max_issues: Maximum issues to fetch per repo

        Returns:
            List of CrawlResult objects
        """
        logger.info(f"Starting crawl of {len(repositories)} repositories")

        # Create tasks with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        async def crawl_with_semaphore(owner: str, repo: str):
            async with semaphore:
                return await self.crawl_repository(
                    owner, repo, fetch_commits, fetch_issues, max_commits, max_issues
                )

        tasks = [
            crawl_with_semaphore(owner, repo)
            for owner, repo in repositories
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and convert to CrawlResult
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                owner, repo = repositories[i]
                logger.error(f"Exception crawling {owner}/{repo}: {result}")
                valid_results.append(CrawlResult(
                    owner=owner,
                    repo=repo,
                    success=False,
                    error=str(result)
                ))
            else:
                valid_results.append(result)

        successful = sum(1 for r in valid_results if r.success)
        logger.info(f"Crawl complete: {successful}/{len(repositories)} successful")

        return valid_results

    async def search_repositories(
        self,
        query: str,
        max_results: int = 100,
        sort: str = "stars",
        order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """
        Search for repositories using GitHub search API

        Args:
            query: Search query (e.g., "language:python stars:>1000")
            max_results: Maximum number of results
            sort: Sort field (stars, forks, updated)
            order: Sort order (asc, desc)

        Returns:
            List of repository metadata
        """
        url = f"{self.config.base_url}/search/repositories"
        url += f"?q={query}&sort={sort}&order={order}"

        logger.info(f"Searching repositories: {query}")

        results = await self._make_paginated_request(
            url,
            max_pages=max(1, max_results // 100),
            per_page=min(100, max_results)
        )

        # Extract items from search results
        repositories = []
        for result in results:
            if isinstance(result, dict) and 'items' in result:
                repositories.extend(result['items'])
            elif isinstance(result, dict):
                repositories.append(result)

        logger.info(f"Found {len(repositories)} repositories")
        return repositories[:max_results]

    async def discover_and_crawl(
        self,
        query: str,
        max_repos: int = 10,
        fetch_commits: bool = True,
        fetch_issues: bool = True,
        max_commits: int = 100,
        max_issues: int = 100
    ) -> List[CrawlResult]:
        """
        Discover repositories via search and crawl them

        Args:
            query: Search query
            max_repos: Maximum repositories to crawl
            fetch_commits: Whether to fetch commits
            fetch_issues: Whether to fetch issues
            max_commits: Maximum commits per repo
            max_issues: Maximum issues per repo

        Returns:
            List of CrawlResult objects
        """
        # Search for repositories
        repos = await self.search_repositories(query, max_results=max_repos)

        if not repos:
            logger.warning("No repositories found")
            return []

        # Extract owner/repo tuples
        repo_list = [
            (repo['owner']['login'], repo['name'])
            for repo in repos[:max_repos]
        ]

        # Crawl discovered repositories
        return await self.crawl_repositories(
            repo_list,
            fetch_commits=fetch_commits,
            fetch_issues=fetch_issues,
            max_commits=max_commits,
            max_issues=max_issues
        )
