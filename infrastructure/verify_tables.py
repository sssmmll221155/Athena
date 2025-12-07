"""
Verify Agent 2 tables exist
"""

import os
from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv

load_dotenv()

def get_database_url():
    db_host = os.getenv('DB_HOST') or os.getenv('POSTGRES_HOST', 'localhost')
    db_port = os.getenv('DB_PORT') or os.getenv('POSTGRES_PORT', '5432')
    db_name = os.getenv('DB_NAME') or os.getenv('POSTGRES_DB', 'athena')
    db_user = os.getenv('DB_USER') or os.getenv('POSTGRES_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD') or os.getenv('POSTGRES_PASSWORD', '')
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

def verify_tables():
    engine = create_engine(get_database_url())
    inspector = inspect(engine)

    print("\n" + "="*60)
    print("  Agent 2 Parser - Table Verification")
    print("="*60 + "\n")

    # Expected tables for Agent 2
    expected_tables = [
        'parsed_files',
        'code_functions',
        'code_imports',
        'code_classes'
    ]

    # Expected views
    expected_views = [
        'code_complexity_view',
        'import_dependencies_view',
        'most_complex_functions',
        'file_statistics'
    ]

    existing_tables = inspector.get_table_names()

    print("[Tables]")
    for table in expected_tables:
        if table in existing_tables:
            # Get column count
            columns = inspector.get_columns(table)
            indexes = inspector.get_indexes(table)
            print(f"  [OK] {table:20s} ({len(columns)} columns, {len(indexes)} indexes)")
        else:
            print(f"  [MISSING] {table}")

    print("\n[Views]")
    with engine.connect() as conn:
        for view in expected_views:
            result = conn.execute(text(f"SELECT EXISTS (SELECT 1 FROM pg_views WHERE viewname = '{view}')"))
            exists = result.scalar()
            if exists:
                print(f"  [OK] {view}")
            else:
                print(f"  [MISSING] {view}")

    print("\n[Functions]")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'update_updated_at_column')"))
        exists = result.scalar()
        if exists:
            print(f"  [OK] update_updated_at_column()")
        else:
            print(f"  [MISSING] update_updated_at_column()")

    print("\n" + "="*60)
    print("  Verification Complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    verify_tables()
