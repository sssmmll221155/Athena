"""
Query the ATHENA database to see collected data
"""
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# Build database connection string
db_connection = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'athena')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'athena_dev_pass')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'athena')}"
)

engine = create_engine(db_connection)

print("=" * 80)
print("ATHENA DATABASE STATUS")
print("=" * 80)

with engine.connect() as conn:
    # Check what tables exist
    result = conn.execute(text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """))
    tables = [row[0] for row in result]
    print(f"\n📊 Tables in database: {', '.join(tables)}\n")

    # Count records in each table
    print("TABLE COUNTS:")
    print("-" * 80)
    for table in tables:
        try:
            count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = count_result.scalar()
            print(f"  {table:30} {count:>10,} records")
        except Exception as e:
            print(f"  {table:30} ERROR: {str(e)[:40]}")

    # If we have commits, show sample data
    if 'commits' in tables:
        print("\n" + "=" * 80)
        print("SAMPLE COMMIT DATA (for ML feature extraction):")
        print("=" * 80)
        result = conn.execute(text("""
            SELECT
                c.sha,
                c.author_name,
                c.author_email,
                c.message,
                c.files_changed,
                c.insertions,
                c.deletions,
                c.committed_at,
                r.full_name as repo
            FROM commits c
            JOIN repositories r ON c.repository_id = r.id
            ORDER BY c.committed_at DESC
            LIMIT 10
        """))

        print(f"\n{'SHA':<12} {'Author':<20} {'Repo':<30} {'Files':<6} {'+/-':<10} {'Date':<20}")
        print("-" * 110)
        for row in result:
            sha_short = row[0][:7]
            author = row[1][:18] if row[1] else 'Unknown'
            repo = row[8][:28]
            files = row[4] or 0
            changes = f"+{row[5] or 0}/-{row[6] or 0}"
            date = str(row[7])[:19]
            print(f"{sha_short:<12} {author:<20} {repo:<30} {files:<6} {changes:<10} {date:<20}")

    # If we have repositories, show summary
    if 'repositories' in tables:
        print("\n" + "=" * 80)
        print("REPOSITORY SUMMARY:")
        print("=" * 80)
        result = conn.execute(text("""
            SELECT
                full_name,
                language,
                stars,
                forks,
                (SELECT COUNT(*) FROM commits WHERE repository_id = r.id) as commit_count
            FROM repositories r
            ORDER BY stars DESC
        """))

        print(f"\n{'Repository':<40} {'Language':<12} {'Stars':<8} {'Forks':<8} {'Commits':<10}")
        print("-" * 90)
        for row in result:
            print(f"{row[0]:<40} {row[1] or 'N/A':<12} {row[2]:>7,} {row[3]:>7,} {row[4]:>9,}")

print("\n" + "=" * 80)
print("✅ Query complete!")
print("=" * 80)
