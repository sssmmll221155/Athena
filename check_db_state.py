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
    result = conn.execute(text("""
        SELECT
            (SELECT COUNT(*) FROM commits) as total_commits,
            (SELECT COUNT(*) FROM files) as total_files,
            (SELECT COUNT(*) FROM parsed_files) as parsed_files,
            (SELECT COUNT(*) FROM code_functions) as functions,
            (SELECT COUNT(*) FROM code_imports) as imports
    """)).fetchone()

    print("total_commits | total_files | parsed_files | functions | imports")
    print("-------------+-------------+--------------+-----------+---------")
    print(f"{result[0]:>13} | {result[1]:>11} | {result[2]:>12} | {result[3]:>9} | {result[4]:>7}")
