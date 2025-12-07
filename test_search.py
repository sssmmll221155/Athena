"""Test the search_repositories method"""
import asyncio
import os
from dotenv import load_dotenv
from agents.crawler.github_crawler import CrawlerConfig, RepositoryCrawler

load_dotenv()

async def test_search():
    github_token = os.getenv('GITHUB_TOKEN')

    config = CrawlerConfig(
        github_token=github_token,
        max_concurrent_requests=5
    )

    async with RepositoryCrawler(config) as crawler:
        print("=== Testing search_repositories method ===")
        query = "language:python stars:>100"

        repos = await crawler.search_repositories(query, max_results=5)

        print(f"Found {len(repos)} repositories")
        for repo in repos:
            if isinstance(repo, dict):
                print(f"  - {repo.get('full_name', 'N/A')} ({repo.get('stargazers_count', 0)} stars)")

if __name__ == "__main__":
    asyncio.run(test_search())
