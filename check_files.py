"""Quick script to check file counts in database"""
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
    result = conn.execute(text("SELECT COUNT(*) FROM files"))
    files_count = result.scalar()

    result = conn.execute(text("SELECT COUNT(*) FROM commit_files"))
    commit_files_count = result.scalar()

    print(f"Files in database: {files_count}")
    print(f"CommitFiles (file changes): {commit_files_count}")

    # Show some sample files
    result = conn.execute(text("""
        SELECT f.filename, f.extension, f.language
        FROM files f
        LIMIT 5
    """))
    print("\nSample files:")
    for row in result:
        print(f"  - {row[0]} ({row[2] or 'unknown'})")
