"""
Athena GitHub API Crawler
Production-grade async crawler with rate limiting and retry logic.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List, Any

import aiohttp
import backoff
from prometheus_client import Counter, Histogram, Gauge


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CrawlerConfig:
    """Configuration for GitHub crawler"""
    github_token: str
    max_concurrent_requests: int = 50
    rate_limit_buffer: int = 100
    base_url: str = "https://api.github.com"
    timeout_seconds: int = 30
    max_retries: int = 3
    backoff_factor: float = 2.0
    user_agent: str = "Athena-Code-Intelligence/1.0"


@dataclass  
class CrawlResult:
    """Result of crawling a single repository"""
    owner: str
    repo: str
    success: bool
    commits_count: int = 0
    issues_count: int = 0
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    crawl_time: float = 0.0


# ============================================================================
# Metrics
# ============================================================================

requests_total = Counter(
    'athena_crawler_requests_total',
    'Total GitHub API requests',
    ['status', 'endpoint']
)

requests_duration = Histogram(
    'athena_crawler_request_duration_seconds',
    'Request duration',
    ['endpoint']
)

rate_limit_remaining = Gauge(
    'athena_crawler_rate_limit_remaining',
    'GitHub API rate limit remaining'
)

repos_crawled = Counter(
    'athena_crawler_repos_total',
    'Total repositories crawled'
)


# ============================================================================
# Rate Limiter
# ============================================================================

class RateLimiter:
    """Token bucket rate limiter with GitHub API awareness"""
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.requests_remaining: int = 5000
        self.reset_time: Optional[datetime] = None
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        async with self.lock:
            if self.requests_remaining < self.config.rate_limit_buffer:
                if self.reset_time and datetime.now() < self.reset_time:
                    wait_seconds = (self.reset_time - datetime.now()).total_seconds()
                    logging.warning(
                        f"Rate limit low ({self.requests_remaining}). "
                        f"Waiting {wait_seconds:.1f}s"
                    )
                    await asyncio.sleep(wait_seconds + 1)
                    self.requests_remaining = 5000
    
    def update_from_headers(self, headers: Dict[str, str]):
        """Update rate limit state from API response"""
        try:
            self.requests_remaining = int(headers.get('X-RateLimit-Remaining', 5000))
            reset_timestamp = int(headers.get('X-RateLimit-Reset', 0))
            if reset_timestamp:
                self.reset_time = datetime.fromtimestamp(reset_timestamp)
            rate_limit_remaining.set(self.requests_remaining)
        except (ValueError, TypeError) as e:
            logging.error(f"Error parsing rate limit headers: {e}")
    
    def get_request_rate(self) -> float:
        """Calculate current request rate"""
        return 0.0


# ============================================================================
# GitHub API Client
# ============================================================================

class GitHubAPIClient:
    """Async GitHub API client"""
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self._closed = False
        
    async def __aenter__(self):
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def start(self):
        """Initialize HTTP session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'Authorization': f'token {self.config.github_token}',
                    'User-Agent': self.config.user_agent,
                    'Accept': 'application/vnd.github.v3+json'
                }
            )
            logging.info("GitHub API client started")
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self._closed:
            await self.session.close()
            self._closed = True
            logging.info("GitHub API client closed")
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        jitter=backoff.full_jitter
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> tuple:
        """Make HTTP request with retry logic"""
        await self.rate_limiter.acquire()
        
        url = f"{self.config.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                duration = time.time() - start_time
                
                requests_total.labels(
                    status=response.status,
                    endpoint=endpoint.split('?')[0]
                ).inc()
                
                requests_duration.labels(
                    endpoint=endpoint.split('?')[0]
                ).observe(duration)
                
                self.rate_limiter.update_from_headers(dict(response.headers))
                
                if response.status == 200:
                    data = await response.json()
                    return response.status, data, dict(response.headers)
                elif response.status == 404:
                    return response.status, {}, dict(response.headers)
                else:
                    text = await response.text()
                    logging.error(f"API error {response.status}: {text}")
                    response.raise_for_status()
                    
        except Exception as e:
            logging.error(f"Request failed for {endpoint}: {e}")
            raise
    
    async def get_repo(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """Fetch repository metadata"""
        endpoint = f"/repos/{owner}/{repo}"
        try:
            status, data, _ = await self._request('GET', endpoint)
            if status == 200:
                return data
            return None
        except Exception as e:
            logging.error(f"Failed to fetch repo {owner}/{repo}: {e}")
            return None
    
    async def get_commits(
        self,
        owner: str,
        repo: str,
        since: Optional[datetime] = None,
        page: int = 1,
        per_page: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch commits"""
        endpoint = f"/repos/{owner}/{repo}/commits"
        params = {'page': page, 'per_page': per_page}
        if since:
            params['since'] = since.isoformat()
        
        try:
            status, data, _ = await self._request('GET', endpoint, params=params)
            if status == 200 and isinstance(data, list):
                return data
            return []
        except Exception as e:
            logging.error(f"Failed to fetch commits for {owner}/{repo}: {e}")
            return []
    
    async def get_issues(
        self,
        owner: str,
        repo: str,
        state: str = 'all',
        page: int = 1,
        per_page: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch issues"""
        endpoint = f"/repos/{owner}/{repo}/issues"
        params = {'state': state, 'page': page, 'per_page': per_page}
        
        try:
            status, data, _ = await self._request('GET', endpoint, params=params)
            if status == 200 and isinstance(data, list):
                return data
            return []
        except Exception as e:
            logging.error(f"Failed to fetch issues for {owner}/{repo}: {e}")
            return []
    
    async def search_repositories(
        self,
        query: str,
        sort: str = 'stars',
        order: str = 'desc',
        per_page: int = 100
    ) -> List[Dict[str, Any]]:
        """Search repositories"""
        endpoint = "/search/repositories"
        params = {
            'q': query,
            'sort': sort,
            'order': order,
            'per_page': per_page
        }
        
        try:
            status, data, _ = await self._request('GET', endpoint, params=params)
            if status == 200 and 'items' in data:
                return data['items']
            return []
        except Exception as e:
            logging.error(f"Failed to search repositories: {e}")
            return []


# ============================================================================
# Repository Crawler
# ============================================================================

class RepositoryCrawler:
    """Orchestrates crawling of repositories"""
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.client = GitHubAPIClient(config)
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
    async def __aenter__(self):
        await self.client.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()
    
    async def crawl_repository(
        self,
        owner: str,
        repo: str,
        fetch_commits: bool = True,
        fetch_issues: bool = True,
        since: Optional[datetime] = None
    ) -> CrawlResult:
        """Crawl a single repository"""
        start_time = time.time()
        
        async with self.semaphore:
            try:
                metadata = await self.client.get_repo(owner, repo)
                if not metadata:
                    return CrawlResult(
                        owner=owner,
                        repo=repo,
                        success=False,
                        error="Repository not found",
                        crawl_time=time.time() - start_time
                    )
                
                commits_count = 0
                issues_count = 0
                
                if fetch_commits:
                    commits = await self.client.get_commits(owner, repo, since=since)
                    commits_count = len(commits)
                
                if fetch_issues:
                    issues = await self.client.get_issues(owner, repo)
                    issues_count = len(issues)
                
                repos_crawled.inc()
                
                return CrawlResult(
                    owner=owner,
                    repo=repo,
                    success=True,
                    commits_count=commits_count,
                    issues_count=issues_count,
                    metadata=metadata,
                    crawl_time=time.time() - start_time
                )
                
            except Exception as e:
                logging.error(f"Failed to crawl {owner}/{repo}: {e}")
                return CrawlResult(
                    owner=owner,
                    repo=repo,
                    success=False,
                    error=str(e),
                    crawl_time=time.time() - start_time
                )
    
    async def crawl_repositories(
        self,
        repos: List[tuple],
        **kwargs
    ) -> List[CrawlResult]:
        """Crawl multiple repositories"""
        tasks = [
            self.crawl_repository(owner, repo, **kwargs)
            for owner, repo in repos
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = [r for r in results if isinstance(r, CrawlResult)]
        
        successful = sum(1 for r in valid_results if r.success)
        logging.info(
            f"Crawled {len(valid_results)} repositories. "
            f"Successful: {successful}"
        )
        
        return valid_results
    
    async def discover_and_crawl(
        self,
        query: str,
        max_repos: int = 100
    ) -> List[CrawlResult]:
        """Discover and crawl repositories via search"""
        logging.info(f"Discovering repositories: {query}")
        
        repos_data = await self.client.search_repositories(
            query=query,
            per_page=min(max_repos, 100)
        )
        
        repos = [
            (repo['owner']['login'], repo['name'])
            for repo in repos_data[:max_repos]
        ]
        
        logging.info(f"Found {len(repos)} repositories")
        
        results = await self.crawl_repositories(repos)
        return results