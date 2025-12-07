"""Check what repositories are in the database"""
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def get_database_url():
    db_host = os.getenv('DB_HOST') or os.getenv('POSTGRES_HOST', 'localhost')
    db_port = os.getenv('DB_PORT') or os.getenv('POSTGRES_PORT', '5432')
    db_name = os.getenv('DB_NAME') or os.getenv('POSTGRES_DB', 'athena')
    db_user = os.getenv('DB_USER') or os.getenv('POSTGRES_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD') or os.getenv('POSTGRES_PASSWORD', '')
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

engine = create_engine(get_database_url())

with engine.connect() as conn:
    # Count repositories by language
    result = conn.execute(text("""
        SELECT language, COUNT(*) as repo_count, SUM(stars) as total_stars
        FROM repositories
        WHERE language IS NOT NULL
        GROUP BY language
        ORDER BY repo_count DESC
        LIMIT 10
    """))

    print("Repositories by language:")
    print("-" * 60)
    for row in result:
        print(f"  {row[0]:20s} - {row[1]:3d} repos, {row[2]:6d} stars")

    # Get Python repositories with commits
    result = conn.execute(text("""
        SELECT r.id, r.full_name, r.language, r.stars, COUNT(c.id) as commit_count
        FROM repositories r
        LEFT JOIN commits c ON r.id = c.repository_id
        WHERE r.language = 'Python'
        GROUP BY r.id, r.full_name, r.language, r.stars
        ORDER BY r.stars DESC
        LIMIT 10
    """))

    print("\nTop Python repositories with commits:")
    print("-" * 80)
    for row in result:
        print(f"  ID: {row[0]:3d} | {row[1]:45s} | {row[3]:6d} stars | {row[4]:4d} commits")
