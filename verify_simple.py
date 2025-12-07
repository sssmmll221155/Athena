"""Athena Setup Verification - Simple Version"""
import os
import sys

def check(condition, msg):
    if condition:
        print(f"[OK] {msg}")
        return 1
    else:
        print(f"[FAIL] {msg}")
        return 0

print("\n" + "="*70)
print("ATHENA SETUP VERIFICATION")
print("="*70 + "\n")

score = 0
total = 0

# Test 1: File Structure
print("TEST 1: File Structure")
print("-" * 70)
files = [
    'agents/__init__.py',
    'agents/crawler/__init__.py',
    'agents/crawler/models.py',
    'agents/crawler/kafka_producer.py',
    'infrastructure/sql/schema.sql',
    'infrastructure/kafka/create_topics.sh',
    'docker-compose.yml',
    'integration_test.py',
    '.env'
]
for f in files:
    total += 1
    score += check(os.path.exists(f), f)

# Test 2: Python Imports
print("\nTEST 2: Python Imports")
print("-" * 70)
modules = ['sqlalchemy', 'aiokafka', 'aiohttp', 'redis', 'pydantic', 'dotenv']
for m in modules:
    total += 1
    try:
        __import__(m)
        score += check(True, f"{m}")
    except:
        score += check(False, f"{m}")

# Test 3: Database Connection
print("\nTEST 3: Database Connection")
print("-" * 70)
total += 1
try:
    from sqlalchemy import create_engine, text
    from dotenv import load_dotenv
    load_dotenv('.env')
    url = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    engine = create_engine(url)
    with engine.connect() as conn:
        conn.execute(text('SELECT 1'))
    score += check(True, "PostgreSQL connected")
except Exception as e:
    score += check(False, f"PostgreSQL: {e}")

# Test 4: Kafka Connection  
print("\nTEST 4: Kafka Connection")
print("-" * 70)
total += 1
try:
    import asyncio
    from aiokafka import AIOKafkaProducer
    async def test_kafka():
        producer = AIOKafkaProducer(bootstrap_servers='localhost:9092')
        await producer.start()
        await producer.stop()
    asyncio.run(test_kafka())
    score += check(True, "Kafka connected")
except Exception as e:
    score += check(False, f"Kafka: {e}")

# Summary
print("\n" + "="*70)
print(f"RESULT: {score}/{total} checks passed ({int(score/total*100)}%)")
print("="*70 + "\n")

if score >= total * 0.9:
    print("SUCCESS: Your setup is ready!")
    sys.exit(0)
elif score >= total * 0.75:
    print("WARNING: Most components working, fix failing checks")
    sys.exit(0)
else:
    print("ERROR: Several components need attention")
    sys.exit(1)
