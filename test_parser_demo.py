"""
Demo: Create a test Python file record and parse it
This demonstrates Agent 2 functionality even without actual GitHub data
"""

import asyncio
from agents.parser.python_parser import PythonASTParser, ParsedPythonFile
from agents.parser.writer import DatabaseWriter
from agents.parser.models import ParsedFile, CodeFunction, CodeImport
from agents.crawler.models import Repository, File, Commit
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Sample Python code to parse
SAMPLE_PYTHON_CODE = '''
"""
Sample Python module for testing AST parser
"""

import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class User:
    """User data class"""
    name: str
    email: str
    age: int

def calculate_fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number

    Args:
        n: The position in the Fibonacci sequence

    Returns:
        The Fibonacci number at position n
    """
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def process_user_data(users: List[User]) -> Dict[str, int]:
    """Process a list of users and return statistics"""
    stats = {}
    for user in users:
        if user.age > 18:
            stats[user.name] = user.age
    return stats

async def fetch_data(url: str) -> Optional[Dict]:
    """Async function to fetch data from URL"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
    return None

class DataProcessor:
    """Main data processing class"""

    def __init__(self, config: Dict):
        self.config = config
        self.cache = {}

    def process(self, data: List) -> List:
        """Process data with caching"""
        result = []
        for item in data:
            if item not in self.cache:
                self.cache[item] = self._transform(item)
            result.append(self.cache[item])
        return result

    def _transform(self, item):
        """Private method to transform item"""
        return str(item).upper()

if __name__ == "__main__":
    print("Running demo...")
'''

def get_database_session():
    """Create database session"""
    db_host = os.getenv('DB_HOST') or os.getenv('POSTGRES_HOST', 'localhost')
    db_port = os.getenv('DB_PORT') or os.getenv('POSTGRES_PORT', '5432')
    db_name = os.getenv('DB_NAME') or os.getenv('POSTGRES_DB', 'athena')
    db_user = os.getenv('DB_USER') or os.getenv('POSTGRES_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD') or os.getenv('POSTGRES_PASSWORD', '')

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    return Session()


def demo_parser():
    """Demonstrate the AST parser functionality"""
    print("=" * 70)
    print("  Agent 2 AST Parser - Demo")
    print("=" * 70)

    session = get_database_session()

    try:
        # Step 1: Parse the Python code
        print("\n[Step 1] Parsing sample Python code...")
        parser = PythonASTParser()
        parsed_result = parser.parse_file('demo/test_sample.py', SAMPLE_PYTHON_CODE)

        if not parsed_result:
            print("  [ERROR] Failed to parse code")
            return

        print(f"  [OK] Successfully parsed!")
        print(f"       - Functions: {len(parsed_result.functions)}")
        print(f"       - Classes:   {len(parsed_result.classes)}")
        print(f"       - Imports:   {len(parsed_result.imports)}")
        print(f"       - LOC:       {parsed_result.code_lines}")

        # Show some details
        print("\n  Functions found:")
        for func in parsed_result.functions[:5]:
            print(f"    - {func.name}() [lines {func.start_line}-{func.end_line}], complexity: {func.cyclomatic_complexity}")

        print("\n  Classes found:")
        for cls in parsed_result.classes:
            print(f"    - {cls.name} with {cls.method_count} methods")

        print("\n  Imports found:")
        for imp in parsed_result.imports[:5]:
            print(f"    - {imp.import_type} {imp.module_name}")

        # Step 2: Get or create a test file record
        print("\n[Step 2] Creating file record in database...")

        # Get the first available repository and commit
        repo = session.query(Repository).first()
        commit = session.query(Commit).filter(Commit.repository_id == repo.id).first()

        if not repo or not commit:
            print("  [ERROR] No repository or commit found in database")
            return

        # Check if test file already exists
        test_file = session.query(File).filter(
            File.repository_id == repo.id,
            File.path == 'demo/test_sample.py'
        ).first()

        if not test_file:
            test_file = File(
                repository_id=repo.id,
                path='demo/test_sample.py',
                filename='test_sample.py',
                extension='.py',
                directory='demo',
                language='Python',
                first_seen_at=datetime.utcnow(),
                last_modified_at=datetime.utcnow()
            )
            session.add(test_file)
            session.flush()
            print(f"  [OK] Created file record (ID: {test_file.id})")
        else:
            print(f"  [OK] Using existing file record (ID: {test_file.id})")

        # Step 3: Write parsed data to database
        print("\n[Step 3] Writing parsed data to database...")
        writer = DatabaseWriter(session)

        # Use the complete method that handles everything
        write_stats = writer.write_complete_parsed_file(
            parsed_file=parsed_result,
            file_id=test_file.id,
            repository_id=repo.id,
            commit_sha=commit.sha
        )

        if write_stats:
            print(f"  [OK] Parsing complete!")
            print(f"      - Functions inserted: {write_stats.functions_inserted}")
            print(f"      - Classes inserted:   {write_stats.classes_inserted}")
            print(f"      - Imports inserted:   {write_stats.imports_inserted}")
            print(f"      - Duration:          {write_stats.duration_ms}ms")

            session.commit()

            # Step 4: Query and display results
            print("\n[Step 4] Querying parsed results from database...")

            # Get the parsed_file record we just created
            from agents.parser.models import ParsedFile
            parsed_file = session.query(ParsedFile).filter(
                ParsedFile.file_id == test_file.id,
                ParsedFile.commit_sha == commit.sha
            ).first()

            if parsed_file:
                print(f"  ParsedFile record ID: {parsed_file.id}")

                result = session.execute(text("""
                    SELECT COUNT(*) FROM code_functions WHERE parsed_file_id = :parsed_file_id
                """), {"parsed_file_id": parsed_file.id})
                func_count = result.scalar()

                result = session.execute(text("""
                    SELECT COUNT(*) FROM code_imports WHERE parsed_file_id = :parsed_file_id
                """), {"parsed_file_id": parsed_file.id})
                import_count = result.scalar()

                print(f"  Database contains:")
                print(f"    - {func_count} functions")
                print(f"    - {import_count} imports")

                # Show most complex function
                result = session.execute(text("""
                    SELECT name, cyclomatic_complexity, start_line, end_line
                    FROM code_functions
                    WHERE parsed_file_id = :parsed_file_id
                    ORDER BY cyclomatic_complexity DESC
                    LIMIT 1
                """), {"parsed_file_id": parsed_file.id})

                most_complex = result.fetchone()
                if most_complex:
                    print(f"\n  Most complex function: {most_complex[0]}()")
                    print(f"    Complexity: {most_complex[1]}")
                    print(f"    Lines: {most_complex[2]}-{most_complex[3]}")

        print("\n" + "=" * 70)
        print("  Demo Complete! Agent 2 is working correctly.")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
    finally:
        session.close()


if __name__ == "__main__":
    demo_parser()
