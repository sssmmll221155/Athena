#!/usr/bin/env python3
"""
Athena Setup Verification Script
Verifies all components of the Week 1 foundation are working correctly.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.END} {text}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.END} {text}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ{Colors.END} {text}")


class SetupVerifier:
    """Verifies Athena setup"""

    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.total_checks = 0
        self.passed_checks = 0

    def test_imports(self) -> bool:
        """Test 1: Verify Python imports"""
        print_header("TEST 1: Python Imports")

        imports = {
            'sqlalchemy': 'SQLAlchemy ORM',
            'aiokafka': 'Async Kafka client',
            'aiohttp': 'Async HTTP client',
            'redis': 'Redis client',
            'pydantic': 'Data validation',
            'dotenv': 'Environment config',
        }

        passed = 0
        total = len(imports)

        for module, description in imports.items():
            try:
                __import__(module)
                print_success(f"{description} ({module})")
                passed += 1
            except ImportError as e:
                print_error(f"{description} ({module}): {e}")

        self.total_checks += total
        self.passed_checks += passed

        success = passed == total
        self.results['imports'] = {'passed': passed, 'total': total, 'success': success}

        if success:
            print_success(f"All {total} imports working!")
        else:
            print_error(f"Only {passed}/{total} imports working")

        return success

    def test_project_structure(self) -> bool:
        """Test 2: Verify project structure"""
        print_header("TEST 2: Project File Structure")

        required_files = {
            'agents/__init__.py': 'Agents package',
            'agents/crawler/__init__.py': 'Crawler package',
            'agents/crawler/models.py': 'SQLAlchemy models',
            'agents/crawler/kafka_producer.py': 'Kafka producer',
            'infrastructure/sql/schema.sql': 'Database schema',
            'infrastructure/kafka/create_topics.sh': 'Kafka topics script',
            'docker-compose.yml': 'Docker configuration',
            'integration_test.py': 'Integration tests',
            'requirements.txt': 'Python dependencies',
            '.env': 'Environment variables',
        }

        passed = 0
        total = len(required_files)

        for file_path, description in required_files.items():
            if os.path.exists(file_path):
                print_success(f"{description}: {file_path}")
                passed += 1
            else:
                print_error(f"{description} MISSING: {file_path}")

        self.total_checks += total
        self.passed_checks += passed

        success = passed == total
        self.results['structure'] = {'passed': passed, 'total': total, 'success': success}

        if success:
            print_success(f"All {total} required files exist!")
        else:
            print_error(f"Only {passed}/{total} files found")

        return success

    def test_database_connection(self) -> bool:
        """Test 3: Verify PostgreSQL connection"""
        print_header("TEST 3: PostgreSQL Database")

        try:
            from sqlalchemy import create_engine, text
            from dotenv import load_dotenv

            load_dotenv('.env')

            url = (
                f"postgresql://{os.getenv('POSTGRES_USER')}:"
                f"{os.getenv('POSTGRES_PASSWORD')}@"
                f"{os.getenv('POSTGRES_HOST')}:"
                f"{os.getenv('POSTGRES_PORT')}/"
                f"{os.getenv('POSTGRES_DB')}"
            )

            print_info(f"Connecting to PostgreSQL at {os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}...")
            engine = create_engine(url, pool_pre_ping=True)

            with engine.connect() as conn:
                # Test connection
                result = conn.execute(text('SELECT version();'))
                version = result.fetchone()[0]
                print_success(f"Connected to database")
                print_info(f"Version: {version[:80]}...")

                # Count tables
                result = conn.execute(text(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_schema = 'public';"
                ))
                table_count = result.fetchone()[0]

                if table_count >= 14:
                    print_success(f"Found {table_count} tables (expected 14)")
                    passed = 2
                else:
                    print_warning(f"Found {table_count} tables (expected 14)")
                    passed = 1

            self.total_checks += 2
            self.passed_checks += passed
            self.results['database'] = {'passed': passed, 'total': 2, 'success': passed == 2}

            return passed == 2

        except Exception as e:
            print_error(f"Database connection failed: {e}")
            self.total_checks += 2
            self.results['database'] = {'passed': 0, 'total': 2, 'success': False}
            return False

    async def test_kafka_connection(self) -> bool:
        """Test 4: Verify Kafka connection"""
        print_header("TEST 4: Kafka Message Broker")

        try:
            from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
            import subprocess

            # Test producer
            print_info("Testing Kafka producer...")
            producer = AIOKafkaProducer(
                bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
            )
            await producer.start()
            print_success("Kafka producer connected")
            await producer.stop()

            # Count topics
            print_info("Counting Kafka topics...")
            result = subprocess.run(
                ['docker', 'exec', 'athena-kafka', 'kafka-topics',
                 '--list', '--bootstrap-server', 'localhost:9092'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                topics = result.stdout.strip().split('\n')
                topic_count = len([t for t in topics if t])

                if topic_count >= 21:
                    print_success(f"Found {topic_count} Kafka topics (expected 21)")
                    passed = 2
                else:
                    print_warning(f"Found {topic_count} Kafka topics (expected 21)")
                    passed = 1
            else:
                print_error("Could not list Kafka topics")
                passed = 1

            self.total_checks += 2
            self.passed_checks += passed
            self.results['kafka'] = {'passed': passed, 'total': 2, 'success': passed == 2}

            return passed == 2

        except Exception as e:
            print_error(f"Kafka connection failed: {e}")
            self.total_checks += 2
            self.results['kafka'] = {'passed': 0, 'total': 2, 'success': False}
            return False

    def test_docker_services(self) -> bool:
        """Test 5: Verify Docker services"""
        print_header("TEST 5: Docker Services")

        try:
            import subprocess

            result = subprocess.run(
                ['docker-compose', 'ps', '--format', 'json'],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )

            if result.returncode != 0:
                print_error("docker-compose ps failed")
                self.total_checks += 1
                self.results['docker'] = {'passed': 0, 'total': 1, 'success': False}
                return False

            # Parse docker-compose output
            import json
            services = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        services.append(json.loads(line))
                    except:
                        pass

            expected_services = [
                'postgres', 'kafka', 'zookeeper', 'redis',
                'weaviate', 'kafka-ui', 'prometheus', 'grafana'
            ]

            running = 0
            for service_name in expected_services:
                service_data = next((s for s in services if service_name in s.get('Name', '')), None)
                if service_data:
                    status = service_data.get('State', 'unknown')
                    if 'running' in status.lower() or 'up' in status.lower():
                        print_success(f"{service_name}: {status}")
                        running += 1
                    else:
                        print_warning(f"{service_name}: {status}")
                else:
                    print_error(f"{service_name}: not found")

            total = len(expected_services)
            self.total_checks += total
            self.passed_checks += running

            success = running >= 7  # At least 7 of 8 should be running
            self.results['docker'] = {'passed': running, 'total': total, 'success': success}

            if success:
                print_success(f"{running}/{total} Docker services running")
            else:
                print_error(f"Only {running}/{total} Docker services running")

            return success

        except Exception as e:
            print_error(f"Docker check failed: {e}")
            self.total_checks += 1
            self.results['docker'] = {'passed': 0, 'total': 1, 'success': False}
            return False

    def test_env_config(self) -> bool:
        """Test 6: Verify environment configuration"""
        print_header("TEST 6: Environment Configuration")

        try:
            from dotenv import load_dotenv
            load_dotenv('.env')

            required_vars = {
                'POSTGRES_HOST': 'Database host',
                'POSTGRES_PORT': 'Database port',
                'POSTGRES_DB': 'Database name',
                'POSTGRES_USER': 'Database user',
                'POSTGRES_PASSWORD': 'Database password',
                'KAFKA_BOOTSTRAP_SERVERS': 'Kafka servers',
                'REDIS_HOST': 'Redis host',
            }

            passed = 0
            total = len(required_vars)

            for var, description in required_vars.items():
                value = os.getenv(var)
                if value:
                    # Mask passwords
                    if 'PASSWORD' in var or 'SECRET' in var:
                        display_value = '*' * 8
                    else:
                        display_value = value
                    print_success(f"{description} ({var}): {display_value}")
                    passed += 1
                else:
                    print_error(f"{description} ({var}): NOT SET")

            # Check GitHub token
            github_token = os.getenv('GITHUB_TOKEN')
            if github_token and github_token != 'ghp_your_github_personal_access_token_here':
                print_success("GitHub token: ********")
                print_info("GitHub token is configured")
            else:
                print_warning("GitHub token: Not configured (optional for Week 1)")

            self.total_checks += total
            self.passed_checks += passed

            success = passed == total
            self.results['env'] = {'passed': passed, 'total': total, 'success': success}

            if success:
                print_success(f"All {total} required environment variables set")
            else:
                print_error(f"Only {passed}/{total} environment variables set")

            return success

        except Exception as e:
            print_error(f"Environment check failed: {e}")
            self.total_checks += 1
            self.results['env'] = {'passed': 0, 'total': 1, 'success': False}
            return False

    def print_summary(self):
        """Print final summary"""
        print_header("VERIFICATION SUMMARY")

        for test_name, result in self.results.items():
            passed = result['passed']
            total = result['total']
            success = result['success']

            status = f"{passed}/{total}"
            if success:
                print_success(f"{test_name.upper()}: {status} checks passed")
            else:
                print_error(f"{test_name.upper()}: {status} checks passed")

        print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")

        percentage = int((self.passed_checks / self.total_checks) * 100) if self.total_checks > 0 else 0

        print(f"\n{Colors.BOLD}TOTAL: {self.passed_checks}/{self.total_checks} checks passed ({percentage}%){Colors.END}\n")

        if percentage >= 90:
            print(f"{Colors.GREEN}{Colors.BOLD}✅ EXCELLENT! Your setup is ready!{Colors.END}")
            print(f"{Colors.GREEN}Week 1 foundation is complete. You can start Week 2!{Colors.END}\n")
        elif percentage >= 75:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  GOOD! Most components working.{Colors.END}")
            print(f"{Colors.YELLOW}Fix the failing checks before proceeding.{Colors.END}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}❌ NEEDS ATTENTION{Colors.END}")
            print(f"{Colors.RED}Several components need to be fixed.{Colors.END}\n")

        print(f"{Colors.BLUE}{'='*70}{Colors.END}\n")

        return percentage >= 75

    async def run_all_tests(self) -> bool:
        """Run all verification tests"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}")
        print("🚀 " + "="*66)
        print("🚀 ATHENA SETUP VERIFICATION")
        print("🚀 " + "="*66)
        print(f"{Colors.END}\n")

        print_info(f"Starting verification at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_info(f"Working directory: {os.getcwd()}\n")

        # Run tests
        self.test_imports()
        self.test_project_structure()
        self.test_env_config()
        self.test_database_connection()
        await self.test_kafka_connection()
        self.test_docker_services()

        # Print summary
        return self.print_summary()


async def main():
    """Main entry point"""
    verifier = SetupVerifier()
    success = await verifier.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
