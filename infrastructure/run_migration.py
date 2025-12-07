"""
Database Migration Runner
Runs SQL migration scripts using Python and SQLAlchemy
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def split_sql_statements(sql_content: str) -> list:
    """
    Split SQL content into individual statements, properly handling:
    - PostgreSQL function definitions with $$ delimiters
    - Multi-line statements
    - Comments
    """
    statements = []
    current_statement = []
    in_dollar_quote = False
    dollar_tag = None

    lines = sql_content.split('\n')

    for line in lines:
        stripped = line.strip()

        # Skip empty lines and comments when not building a statement
        if not current_statement and (not stripped or stripped.startswith('--')):
            continue

        # Check for dollar-quoted strings (PostgreSQL functions)
        if '$$' in line or '$' in line:
            # Simple check for $$ or $tag$
            import re
            dollar_quotes = re.findall(r'\$[a-zA-Z_]*\$', line)
            for quote in dollar_quotes:
                if not in_dollar_quote:
                    in_dollar_quote = True
                    dollar_tag = quote
                elif quote == dollar_tag:
                    in_dollar_quote = False
                    dollar_tag = None

        current_statement.append(line)

        # If we hit a semicolon and we're not in a dollar-quoted section, that's the end
        if ';' in line and not in_dollar_quote:
            stmt = '\n'.join(current_statement).strip()
            if stmt and not stmt.startswith('--'):
                # Remove the trailing semicolon for execution
                if stmt.endswith(';'):
                    stmt = stmt[:-1].strip()
                if stmt:
                    statements.append(stmt)
            current_statement = []

    # Add any remaining statement
    if current_statement:
        stmt = '\n'.join(current_statement).strip()
        if stmt and not stmt.startswith('--'):
            if stmt.endswith(';'):
                stmt = stmt[:-1].strip()
            if stmt:
                statements.append(stmt)

    return statements


def get_database_url():
    """Get database connection URL from environment variables"""
    # Support both DB_* and POSTGRES_* environment variable formats
    db_host = os.getenv('DB_HOST') or os.getenv('POSTGRES_HOST', 'localhost')
    db_port = os.getenv('DB_PORT') or os.getenv('POSTGRES_PORT', '5432')
    db_name = os.getenv('DB_NAME') or os.getenv('POSTGRES_DB', 'athena')
    db_user = os.getenv('DB_USER') or os.getenv('POSTGRES_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD') or os.getenv('POSTGRES_PASSWORD', '')

    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def run_migration(sql_file_path: str):
    """Run SQL migration from file"""

    # Check if file exists
    if not Path(sql_file_path).exists():
        print(f"[ERROR] SQL file not found: {sql_file_path}")
        sys.exit(1)

    # Read SQL file
    print(f"[INFO] Reading migration file: {sql_file_path}")
    with open(sql_file_path, 'r', encoding='utf-8') as f:
        sql_content = f.read()

    # Create database connection
    db_url = get_database_url()
    print(f"[INFO] Connecting to database: {db_url.split('@')[1]}")  # Hide password

    try:
        # Use autocommit mode to execute each statement independently
        engine = create_engine(db_url, isolation_level="AUTOCOMMIT")

        with engine.connect() as conn:
            # Split SQL into individual statements (handle PostgreSQL $$ delimiters)
            statements = split_sql_statements(sql_content)

            print(f"[INFO] Executing {len(statements)} SQL statements...")

            success_count = 0
            skip_count = 0
            error_count = 0

            for i, statement in enumerate(statements, 1):
                if statement:
                    try:
                        conn.execute(text(statement))
                        success_count += 1
                        # Only show progress for every 10 statements to avoid spam
                        if i % 10 == 0 or i == len(statements):
                            print(f"   [OK] Executed {i}/{len(statements)} statements (success: {success_count}, skipped: {skip_count})")
                    except Exception as e:
                        error_msg = str(e).lower()
                        # Some statements might fail if objects already exist - that's ok
                        if "already exists" in error_msg or "duplicate" in error_msg:
                            skip_count += 1
                            if i % 10 == 0 or i == len(statements):
                                print(f"   [SKIP] Statement {i} already exists")
                        else:
                            error_count += 1
                            print(f"   [WARN] Error in statement {i}: {e}")
                            # Show first 200 chars of problematic SQL for debugging
                            print(f"   SQL: {statement[:200]}...")
                            # Don't raise - continue with other statements

        print(f"[SUCCESS] Migration completed! Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")

    except Exception as e:
        print(f"[FAILED] Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Athena Database Migration Runner")
    print("="*60 + "\n")

    # Default to agent2_schema.sql if no argument provided
    if len(sys.argv) > 1:
        sql_file = sys.argv[1]
    else:
        # Default to agent2_schema.sql in the same directory
        script_dir = Path(__file__).parent
        sql_file = script_dir / "sql" / "agent2_schema.sql"

    run_migration(str(sql_file))
