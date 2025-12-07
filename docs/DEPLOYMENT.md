# Athena Deployment Guide

Complete step-by-step deployment for Week 1 foundation.

## 📋 Prerequisites Checklist

Before starting, ensure you have:
- ✅ Docker Desktop installed and running
- ✅ Python 3.11+ installed
- ✅ Git installed
- ✅ GitHub Personal Access Token ([Create one here](https://github.com/settings/tokens))
- ✅ At least 8GB RAM and 20GB free disk space

---

## 🚀 Quick Start (10 Minutes)

### Step 1: Open Docker Desktop

1. Launch **Docker Desktop** application
2. Wait for it to fully start (whale icon in system tray should be stable)
3. Verify Docker is running:
   ```bash
   docker --version
   docker-compose --version
   ```

### Step 2: Configure Environment

1. Navigate to the athena directory:
   ```bash
   cd C:\Users\User\athena
   ```

2. Copy the environment template:
   ```bash
   copy .env.example .env
   ```

3. Edit `.env` file and add your GitHub token:
   ```bash
   notepad .env
   ```

   Update this line:
   ```
   GITHUB_TOKEN=ghp_your_github_personal_access_token_here
   ```

### Step 3: Start Docker Infrastructure

1. Start all services:
   ```bash
   docker-compose up -d
   ```

2. Verify all containers are running:
   ```bash
   docker-compose ps
   ```

   You should see these services running:
   - ✅ athena-postgres
   - ✅ athena-kafka
   - ✅ athena-zookeeper
   - ✅ athena-redis
   - ✅ athena-weaviate
   - ✅ athena-mlflow
   - ✅ athena-prometheus
   - ✅ athena-grafana
   - ✅ athena-kafka-ui

3. Wait 30 seconds for services to initialize

### Step 4: Create Kafka Topics

**For Windows (using Git Bash):**
```bash
bash create_topics.sh
```

**For PowerShell/CMD:**
```powershell
docker exec athena-kafka kafka-topics --create --if-not-exists --bootstrap-server localhost:9092 --topic raw_commits --partitions 6 --replication-factor 1
docker exec athena-kafka kafka-topics --create --if-not-exists --bootstrap-server localhost:9092 --topic raw_issues --partitions 3 --replication-factor 1
docker exec athena-kafka kafka-topics --create --if-not-exists --bootstrap-server localhost:9092 --topic crawler_events --partitions 3 --replication-factor 1
```

### Step 5: Setup Python Environment

1. Create virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate virtual environment:
   ```bash
   # Windows CMD
   venv\Scripts\activate

   # Windows PowerShell
   venv\Scripts\Activate.ps1

   # Git Bash
   source venv/Scripts/activate
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Step 6: Run Integration Test

```bash
python integration_test.py
```

**Expected Output:**
```
🚀 ====================================================================
🚀 ATHENA INTEGRATION TEST SUITE
🚀 ====================================================================

======================================================================
TEST 1: Infrastructure Check
======================================================================
✓ Database tables created/verified
✓ PostgreSQL connected and tables created
✓ Kafka connected and message sent
✓ Infrastructure check passed

======================================================================
TEST 2: Kafka Integration
======================================================================
✓ Sent crawler event
✓ Sent commit message
✓ Sent issue message
✓ Kafka stats: {'messages_sent': 4, 'messages_failed': 0, 'bytes_sent': ...}
✓ Kafka integration test passed

======================================================================
TEST 3: Database Integration
======================================================================
✓ Created test repository: octocat/Hello-World
✓ Verified repository in DB: octocat/Hello-World (id=1, stars=100)
✓ Database integration test passed

======================================================================
TEST 4: End-to-End Flow
======================================================================
✓ End-to-end test passed

======================================================================
TEST SUMMARY
======================================================================
✅ PASS - Infrastructure
✅ PASS - Kafka
✅ PASS - Database
✅ PASS - End To End
======================================================================
✅ ALL TESTS PASSED!

🎉 Athena is ready to go!
```

---

## 🔍 Verify Deployment

### Check Docker Containers
```bash
docker-compose ps
```

### Access Web Interfaces

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| Kafka UI | http://localhost:8080 | None |
| Grafana | http://localhost:3000 | admin / admin |
| MLflow | http://localhost:5000 | None |
| Prometheus | http://localhost:9090 | None |

### Check PostgreSQL Database

```bash
# Connect to database
docker exec -it athena-postgres psql -U athena -d athena

# List tables
\dt

# Check repositories table
SELECT * FROM repositories LIMIT 5;

# Exit
\q
```

### Check Kafka Topics

```bash
# List all topics
docker exec athena-kafka kafka-topics --list --bootstrap-server localhost:9092

# Describe a topic
docker exec athena-kafka kafka-topics --describe --bootstrap-server localhost:9092 --topic raw_commits

# Consume messages (last 10)
docker exec athena-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic raw_commits --from-beginning --max-messages 10
```

---

## 🛠️ Troubleshooting

### Issue: Docker containers won't start

**Solution:**
```bash
# Stop all containers
docker-compose down

# Remove volumes (WARNING: This deletes all data!)
docker-compose down -v

# Start fresh
docker-compose up -d
```

### Issue: Port already in use

**Solution:**
1. Check what's using the port:
   ```bash
   netstat -ano | findstr :5432
   netstat -ano | findstr :9092
   ```

2. Either:
   - Stop the conflicting service
   - Change port in `docker-compose.yml`

### Issue: Kafka topics creation fails

**Solution:**
```bash
# Wait for Kafka to fully start
docker logs athena-kafka

# Try creating topics manually
docker exec athena-kafka kafka-topics --create --if-not-exists --bootstrap-server localhost:9092 --topic raw_commits --partitions 6 --replication-factor 1
```

### Issue: Integration test fails on database connection

**Solution:**
```bash
# Check PostgreSQL is ready
docker logs athena-postgres

# Verify .env file has correct credentials
cat .env | grep POSTGRES

# Test connection manually
docker exec -it athena-postgres psql -U athena -d athena -c "SELECT 1;"
```

### Issue: Integration test fails on Kafka connection

**Solution:**
```bash
# Check Kafka is ready
docker logs athena-kafka

# Verify Kafka is listening
docker exec athena-kafka kafka-broker-api-versions --bootstrap-server localhost:9092
```

---

## 📦 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Docker Desktop                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│  │  PostgreSQL  │   │    Kafka     │   │    Redis     │       │
│  │ (TimescaleDB)│   │  + Zookeeper │   │   (Cache)    │       │
│  └──────────────┘   └──────────────┘   └──────────────┘       │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│  │   Weaviate   │   │    MLflow    │   │  Prometheus  │       │
│  │  (Vectors)   │   │  (Tracking)  │   │  + Grafana   │       │
│  └──────────────┘   └──────────────┘   └──────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ connects to
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                    Your Local Machine                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  athena/                                                        │
│  ├── models.py              (SQLAlchemy ORM)                    │
│  ├── kafka_producer.py      (Async Kafka producer)             │
│  ├── integration_test.py    (Tests)                            │
│  ├── schema.sql             (Database schema)                  │
│  └── .env                   (Configuration)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 What You've Built

### ✅ Complete Infrastructure Stack

- **Database:** PostgreSQL 16 + TimescaleDB for time-series optimization
- **Message Queue:** Kafka for reliable event streaming
- **Vector DB:** Weaviate for code embeddings
- **Caching:** Redis for fast lookups
- **Monitoring:** Prometheus + Grafana for observability
- **ML Tracking:** MLflow for experiment management

### ✅ Production-Ready Components

- **ORM Models:** 14 SQLAlchemy models with relationships and validation
- **Kafka Producer:** Async producer with batching and retry logic
- **Integration Tests:** End-to-end validation of data flow
- **Docker Setup:** Complete containerized environment

---

## 📚 Next Steps

### Week 2: GitHub Crawler

Create `crawler.py` to fetch real GitHub data:
```python
# Basic crawler structure
import asyncio
from github import Github
from kafka_producer import AthenaKafkaProducer, CommitMessage

async def crawl_repository(repo_name: str):
    # Fetch commits from GitHub
    # Transform to CommitMessage
    # Send to Kafka
    pass
```

### Week 3: Kafka Consumer

Create `consumer.py` to read from Kafka and write to database:
```python
# Basic consumer structure
from aiokafka import AIOKafkaConsumer
from models import Repository, Commit

async def consume_commits():
    # Read from raw_commits topic
    # Parse message
    # Insert into database
    pass
```

### Week 4: Feature Engineering

Start extracting features from code:
```python
# Basic feature extractor
from models import File, Feature

def extract_features(file_content: str):
    # Parse code
    # Calculate metrics
    # Store features
    pass
```

---

## 🔧 Maintenance Commands

### Stop Infrastructure
```bash
docker-compose stop
```

### Start Infrastructure
```bash
docker-compose start
```

### Restart Everything
```bash
docker-compose restart
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f athena-postgres
docker-compose logs -f athena-kafka
```

### Clean Up (Remove All Data)
```bash
docker-compose down -v
```

---

## 💡 Tips & Best Practices

### Development Workflow

1. **Keep Docker Desktop running** while developing
2. **Use virtual environment** for Python dependencies
3. **Check logs regularly** for errors
4. **Run integration tests** after changes
5. **Commit frequently** to Git

### Performance Tuning

- Adjust `KAFKA_BATCH_SIZE` in `.env` for throughput
- Increase `POSTGRES_MAX_CONNECTIONS` for concurrent writes
- Monitor memory usage in Docker Desktop settings

### Security

- Never commit `.env` file to Git
- Rotate GitHub token regularly
- Use strong passwords in production
- Enable authentication on all services in production

---

## 📞 Support

If you encounter issues:

1. Check logs: `docker-compose logs -f`
2. Verify all containers running: `docker-compose ps`
3. Review this guide's troubleshooting section
4. Check Docker Desktop has enough resources (Settings → Resources)

---

## ✅ Deployment Checklist

Before moving to next phase:

- [ ] All Docker containers running
- [ ] Kafka topics created (22 topics)
- [ ] Database schema applied (14 tables)
- [ ] Integration test passes
- [ ] Can access Kafka UI
- [ ] Can access Grafana
- [ ] GitHub token configured
- [ ] Virtual environment activated
- [ ] All Python dependencies installed

**Status:** Week 1 Foundation Complete! 🎉

