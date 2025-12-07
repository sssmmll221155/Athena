"""Quick test of GitHub API connectivity"""
import asyncio
import os
from dotenv import load_dotenv
from agents.crawler.github_crawler import CrawlerConfig, RepositoryCrawler

load_dotenv()

async def test_api():
    github_token = os.getenv('GITHUB_TOKEN')
    print(f"Token length: {len(github_token)}")
    print(f"Token starts with: {github_token[:7]}...")

    config = CrawlerConfig(
        github_token=github_token,
        max_concurrent_requests=5
    )

    async with RepositoryCrawler(config) as crawler:
        # Test 1: Fetch a known repository
        print("\n=== Test 1: Fetching python/cpython ===")
        metadata = await crawler.fetch_repository_metadata("python", "cpython")
        if metadata:
            print(f"[OK] Success: {metadata.get('full_name')}, Stars: {metadata.get('stargazers_count')}")
        else:
            print("[FAIL] Failed to fetch repository")

        # Test 2: Search for repositories
        print("\n=== Test 2: Searching for Python repos ===")
        query = "language:python stars:>100"

        # Make a direct request to see what we get
        search_url = f"{config.base_url}/search/repositories?q={query}&per_page=5"
        print(f"URL: {search_url}")

        result = await crawler._make_request(search_url)
        print(f"Response type: {type(result)}")
        print(f"Response keys: {result.keys() if isinstance(result, dict) else 'N/A'}")

        if isinstance(result, dict):
            print(f"Total count: {result.get('total_count', 'N/A')}")
            print(f"Items count: {len(result.get('items', []))}")

            if result.get('items'):
                print("\nFirst 3 repos:")
                for repo in result.get('items', [])[:3]:
                    print(f"  - {repo['full_name']} ({repo['stargazers_count']} stars)")

if __name__ == "__main__":
    asyncio.run(test_api())
