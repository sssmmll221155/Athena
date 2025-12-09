"""
Initialize ATHENA database - Create all tables
"""
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from agents.crawler.models import Base, init_db

load_dotenv()

# Build database connection string
db_connection = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'athena')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'athena_dev_pass')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'athena')}"
)

print("=" * 80)
print("ATHENA DATABASE INITIALIZATION")
print("=" * 80)
print(f"\nConnection: {db_connection.split('@')[1]}")  # Hide password
print("\nCreating tables...")

engine = create_engine(db_connection)
init_db(engine)

print("\n✓ Database initialized successfully!")
print("\nTables created:")
print("  - repositories")
print("  - commits")
print("  - files")
print("  - commit_files")
print("  - issues")
print("  - pull_requests")
print("  - models")
print("  - predictions")
print("  - feedback")
print("  - features")
print("  - embeddings")
print("  - sequential_patterns")
print("  - association_rules")
print("  - rl_episodes")
print("=" * 80)
