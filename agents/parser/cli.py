"""
Parser CLI Interface
Command-line interface for running the code parser.
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from agents.parser.pipeline import ParsingPipeline, PipelineStats
from agents.parser.config import ParserConfig, DatabaseConfig, GitHubConfig

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_level: str = 'INFO'):
    """Configure logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('parser.log')
        ]
    )


# ============================================================================
# Output Formatting
# ============================================================================

def print_banner():
    """Print CLI banner"""
    banner = """
    ╔═══════════════════════════════════════════╗
    ║     Athena Code Parser - Agent 2          ║
    ║     Python AST Parser & Code Analyzer     ║
    ╚═══════════════════════════════════════════╝
    """
    ascii_banner = """
    ===========================================
        Athena Code Parser - Agent 2
        Python AST Parser & Code Analyzer
    ===========================================
    """
    try:
        print(banner)
    except UnicodeEncodeError:
        print(ascii_banner)


def print_stats(stats: PipelineStats):
    """Print pipeline statistics"""
    print("\n" + "=" * 60)
    print("PARSING STATISTICS")
    print("=" * 60)

    print(f"\nFiles:")
    print(f"  Total:      {stats.total_files}")
    print(f"  Fetched:    {stats.fetched_files}")
    print(f"  Parsed:     {stats.parsed_files}")
    print(f"  Written:    {stats.written_files}")
    print(f"  Skipped:    {stats.skipped_files}")
    print(f"  Failed:     {stats.failed_files}")

    print(f"\nCode Elements:")
    print(f"  Functions:  {stats.total_functions}")
    print(f"  Classes:    {stats.total_classes}")
    print(f"  Imports:    {stats.total_imports}")

    print(f"\nTiming:")
    print(f"  Fetch:      {stats.fetch_duration_ms:,} ms")
    print(f"  Parse:      {stats.parse_duration_ms:,} ms")
    print(f"  Write:      {stats.write_duration_ms:,} ms")
    print(f"  Total:      {stats.total_duration_ms:,} ms")

    if stats.errors:
        print(f"\nErrors ({len(stats.errors)}):")
        for error in stats.errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more errors")

    print("=" * 60 + "\n")


def print_config(parser_config: ParserConfig, db_config: DatabaseConfig):
    """Print configuration"""
    print("\nConfiguration:")
    print(f"  Parser Version:      {parser_config.PARSER_VERSION}")
    print(f"  Supported Languages: {', '.join(parser_config.SUPPORTED_LANGUAGES)}")
    print(f"  Max File Size:       {parser_config.MAX_FILE_SIZE_MB} MB")
    print(f"  Concurrent Files:    {parser_config.CONCURRENT_FILES}")
    print(f"  Batch Insert Size:   {parser_config.BATCH_INSERT_SIZE}")
    print(f"  Database:            {db_config.database}@{db_config.host}:{db_config.port}")
    print()


# ============================================================================
# CLI Commands
# ============================================================================

async def cmd_backfill(args, session, parser_config, github_token):
    """Run backfill mode to parse unparsed files"""
    print("Running in BACKFILL mode...")
    print(f"Repository ID: {args.repository or 'ALL'}")
    print(f"Limit: {args.limit or 'UNLIMITED'}")
    print()

    pipeline = ParsingPipeline(
        session=session,
        config=parser_config,
        github_token=github_token
    )

    stats = await pipeline.run_backfill(
        repository_id=args.repository,
        limit=args.limit,
        show_progress=not args.no_progress
    )

    print_stats(stats)

    return 0 if stats.failed_files == 0 else 1


async def cmd_commit(args, session, parser_config, github_token):
    """Parse files for a specific commit"""
    print("Running in COMMIT mode...")
    print(f"Repository ID: {args.repository}")
    print(f"Commit SHA: {args.commit}")
    print(f"Limit: {args.limit or 'UNLIMITED'}")
    print()

    pipeline = ParsingPipeline(
        session=session,
        config=parser_config,
        github_token=github_token
    )

    stats = await pipeline.run_for_commit(
        repository_id=args.repository,
        commit_sha=args.commit,
        limit=args.limit,
        show_progress=not args.no_progress
    )

    print_stats(stats)

    return 0 if stats.failed_files == 0 else 1


async def cmd_stats(args, session):
    """Show parser statistics"""
    from sqlalchemy import text

    print("Parser Statistics")
    print("=" * 60)

    # Count parsed files
    result = session.execute(text("""
        SELECT
            COUNT(*) as total_parsed,
            COUNT(DISTINCT repository_id) as repositories,
            SUM(total_functions) as total_functions,
            SUM(total_classes) as total_classes,
            SUM(total_imports) as total_imports,
            AVG(average_complexity) as avg_complexity,
            MAX(max_complexity) as max_complexity
        FROM parsed_files
        WHERE parse_status = 'success'
    """))

    stats = result.fetchone()

    print(f"\nParsed Files:        {stats.total_parsed:,}")
    print(f"Repositories:        {stats.repositories:,}")
    print(f"Total Functions:     {stats.total_functions or 0:,}")
    print(f"Total Classes:       {stats.total_classes or 0:,}")
    print(f"Total Imports:       {stats.total_imports or 0:,}")
    print(f"Avg Complexity:      {stats.avg_complexity or 0:.2f}")
    print(f"Max Complexity:      {stats.max_complexity or 0:.0f}")

    # Top repositories
    print("\nTop Repositories by Parsed Files:")
    result = session.execute(text("""
        SELECT
            r.full_name,
            COUNT(pf.id) as parsed_files,
            SUM(pf.total_functions) as functions
        FROM repositories r
        JOIN parsed_files pf ON pf.repository_id = r.id
        WHERE pf.parse_status = 'success'
        GROUP BY r.id, r.full_name
        ORDER BY parsed_files DESC
        LIMIT 10
    """))

    for row in result:
        print(f"  {row.full_name:40} {row.parsed_files:6} files, {row.functions:8} functions")

    print("=" * 60 + "\n")

    return 0


# ============================================================================
# Main CLI
# ============================================================================

def create_argument_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='Athena Code Parser - Parse and analyze code from repositories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse all unparsed files (backfill)
  python -m agents.parser.cli --backfill

  # Parse specific repository
  python -m agents.parser.cli --backfill --repository 1

  # Parse specific commit
  python -m agents.parser.cli --commit abc123 --repository 1

  # Parse with limit
  python -m agents.parser.cli --backfill --limit 100

  # Show statistics
  python -m agents.parser.cli --stats
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--backfill',
        action='store_true',
        help='Parse all unparsed files'
    )
    mode_group.add_argument(
        '--commit',
        type=str,
        metavar='SHA',
        help='Parse files for specific commit SHA'
    )
    mode_group.add_argument(
        '--stats',
        action='store_true',
        help='Show parser statistics'
    )

    # Options
    parser.add_argument(
        '--repository',
        type=int,
        metavar='ID',
        help='Filter by repository ID'
    )
    parser.add_argument(
        '--limit',
        type=int,
        metavar='N',
        help='Maximum number of files to process'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )

    # Configuration
    parser.add_argument(
        '--max-file-size',
        type=int,
        metavar='MB',
        help='Maximum file size in MB (default: 1)'
    )
    parser.add_argument(
        '--concurrent',
        type=int,
        metavar='N',
        help='Maximum concurrent file fetches (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        metavar='N',
        help='Batch insert size (default: 100)'
    )

    # Logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    return parser


async def async_main():
    """Async main entry point"""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Print banner
    print_banner()

    try:
        # Load configuration
        parser_config = ParserConfig()
        db_config = DatabaseConfig()
        github_config = GitHubConfig()

        # Override config from CLI args
        if args.max_file_size:
            parser_config.MAX_FILE_SIZE_MB = args.max_file_size
            parser_config.MAX_FILE_SIZE_BYTES = args.max_file_size * 1024 * 1024

        if args.concurrent:
            parser_config.CONCURRENT_FILES = args.concurrent

        if args.batch_size:
            parser_config.BATCH_INSERT_SIZE = args.batch_size

        # Print configuration
        if not args.stats:
            print_config(parser_config, db_config)

        # Create database connection
        logger.info("Connecting to database...")
        engine = create_engine(
            db_config.get_connection_string(),
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            echo=db_config.echo
        )

        Session = sessionmaker(bind=engine)
        session = Session()

        # Execute command
        if args.backfill:
            exit_code = await cmd_backfill(args, session, parser_config, github_config.token)
        elif args.commit:
            if not args.repository:
                print("Error: --repository is required with --commit", file=sys.stderr)
                return 1
            exit_code = await cmd_commit(args, session, parser_config, github_config.token)
        elif args.stats:
            exit_code = await cmd_stats(args, session)
        else:
            parser.print_help()
            exit_code = 1

        # Cleanup
        session.close()
        engine.dispose()

        return exit_code

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 130

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nError: {e}", file=sys.stderr)
        return 1


def main():
    """Synchronous main entry point"""
    exit_code = asyncio.run(async_main())
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
