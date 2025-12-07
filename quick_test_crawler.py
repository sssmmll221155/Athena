"""Quick test - fetch repos with minimal commits"""
import asyncio
import os
from dotenv import load_dotenv
from agents.crawler.github_crawler import CrawlerConfig, RepositoryCrawler

load_dotenv()

async def quick_test():
    github_token = os.getenv('GITHUB_TOKEN')

    config = CrawlerConfig(
        github_token=github_token,
        max_concurrent_requests=5,
        rate_limit_delay=0.5  # Faster for testing
    )

    async with RepositoryCrawler(config) as crawler:
        print("=== Quick Test: 2 repos, 5 commits each ===\n")

        # Fetch just 2 specific repos with minimal data
        repos = [
            ("python", "cpython"),
            ("django", "django")
        ]

        results = await crawler.crawl_repositories(
            repos,
            fetch_commits=True,
            fetch_issues=False,  # Skip issues for speed
            max_commits=5,  # Only 5 commits
            max_issues=0
        )

        print(f"\nResults:")
        for result in results:
            if result.success:
                print(f"  [OK] {result.owner}/{result.repo}")
                print(f"       Stars: {result.metadata.get('stargazers_count', 0)}")
                print(f"       Commits fetched: {result.commits_count}")
                print(f"       Time: {result.crawl_time:.2f}s")
            else:
                print(f"  [FAIL] {result.owner}/{result.repo}: {result.error}")

if __name__ == "__main__":
    asyncio.run(quick_test())
