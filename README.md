# Athena

An enterprise-grade machine learning platform for automated code intelligence, defect prediction, and data-driven development insights.

## 🚀 Quick Start

### 1. Prerequisites
- Docker Desktop running
- Python 3.11+
- GitHub Personal Access Token

### 2. Setup (5 minutes)

```bash
# Clone and navigate to project directory
cd athena

# Configure environment variables
cp .env.example .env
# Edit .env and add your GITHUB_TOKEN

# Start Docker infrastructure
docker-compose up -d

# Initialize Kafka topics (wait 30 seconds for services to start)
bash create_topics.sh
```

### 3. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Run Integration Test

```bash
python integration_test.py
```

**Expected:** All tests pass ✅

## 📁 Project Structure

```
athena/
├── .env                    # Environment configuration (add GITHUB_TOKEN here)
├── .env.example            # Environment template
├── docker-compose.yml      # Docker infrastructure setup
├── Dockerfile              # Application container
├── requirements.txt        # Python dependencies
├── schema.sql              # Database schema (14 tables)
├── models.py               # SQLAlchemy ORM models
├── kafka_producer.py       # Async Kafka producer
├── create_topics.sh        # Kafka topics setup
├── integration_test.py     # End-to-end tests
├── DEPLOYMENT.md           # Detailed deployment guide
└── README.md               # This file
```

## 🏗️ Architecture

### Docker Services (9 containers)
- **PostgreSQL + TimescaleDB:** Time-series optimized database
- **Kafka + Zookeeper:** Event streaming
- **Redis:** Caching layer
- **Weaviate:** Vector database for embeddings
- **MLflow:** ML experiment tracking
- **Prometheus + Grafana:** Monitoring
- **Kafka UI:** Kafka management interface

### Python Components
- **models.py:** 14 SQLAlchemy models (Repository, Commit, File, Issue, etc.)
- **kafka_producer.py:** High-performance async Kafka producer
- **integration_test.py:** Validates entire data pipeline

## 🌐 Web Interfaces

| Service | URL | Credentials |
|---------|-----|-------------|
| Kafka UI | http://localhost:8080 | None |
| Grafana | http://localhost:3000 | admin/admin |
| MLflow | http://localhost:5000 | None |
| Prometheus | http://localhost:9090 | None |

## 📊 Database Schema

14 production-ready tables:
- **Core:** repositories, commits, files, issues, pull_requests
- **ML:** models, predictions, feedback, features, embeddings
- **RL:** rl_episodes
- **Patterns:** sequential_patterns, association_rules
- **Junction:** commit_files

## 🔍 Quick Commands

### Check Infrastructure Status
```bash
docker-compose ps
```

### View Logs
```bash
docker-compose logs -f athena-postgres
docker-compose logs -f athena-kafka
```

### Access PostgreSQL
```bash
docker exec -it athena-postgres psql -U athena -d athena
```

### List Kafka Topics
```bash
docker exec athena-kafka kafka-topics --list --bootstrap-server localhost:9092
```

### Stop/Start Infrastructure
```bash
docker-compose stop
docker-compose start
```

## 🧪 Running Tests

```bash
# Full integration test
python integration_test.py

# Expected output: ✅ ALL TESTS PASSED!
```

## Development Roadmap

1. **Data Ingestion:** GitHub crawler for repository data collection
2. **Stream Processing:** Kafka consumer for real-time data processing
3. **Feature Engineering:** Code metrics and complexity analysis
4. **Predictive Models:** Bug prediction and code churn analysis
5. **Deep Learning:** Code embeddings and graph neural networks
6. **Recommendation Engine:** Reinforcement learning-based suggestions
7. **Production Deployment:** Scalable production infrastructure

## 🛠️ Development Workflow

```bash
# 1. Start Docker infrastructure
docker-compose up -d

# 2. Activate Python environment
venv\Scripts\activate

# 3. Make changes to code
# ... edit files ...

# 4. Run tests
python integration_test.py

# 5. Commit changes
git add .
git commit -m "Your message"

# 6. Stop infrastructure when done
docker-compose stop
```

## 🐛 Troubleshooting

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed troubleshooting guide.

Common issues:
- **Port conflicts:** Check if ports 5432, 9092, 6379 are available
- **Docker not running:** Start Docker Desktop
- **Kafka connection fails:** Wait 30 seconds after docker-compose up
- **Integration test fails:** Check .env has GITHUB_TOKEN set

## 📖 Documentation

- **DEPLOYMENT.md** - Complete deployment guide with troubleshooting
- **SETUP.md** - Initial setup instructions
- **schema.sql** - Database schema with comments
- **models.py** - ORM models with docstrings
- **kafka_producer.py** - Producer API documentation

## 🎯 Current Status

**Week 1 Foundation: ✅ COMPLETE**

- [x] Docker infrastructure setup
- [x] Database schema (14 tables, views, indexes)
- [x] Kafka topics (22 topics configured)
- [x] Python ORM models (14 models with relationships)
- [x] Async Kafka producer (with batching & retry)
- [x] Integration tests (end-to-end validation)
- [x] Deployment documentation

**Next Milestone:** Week 2 - GitHub Crawler Implementation

## 🏆 Project Goals

Build a production-grade ML system that:
1. **Crawls** GitHub repositories at scale
2. **Streams** data through Kafka for reliability
3. **Stores** structured data in PostgreSQL
4. **Extracts** code features and embeddings
5. **Trains** ML models for bug prediction
6. **Learns** from developer feedback (RL)
7. **Provides** intelligent code recommendations

---

**Built with:** Python 3.11 • PostgreSQL 16 • TimescaleDB • Kafka • Redis • Weaviate • MLflow • Docker

**License:** MIT

**Author:** Athena Team

---

⭐ **Ready to start?** Follow the Quick Start guide above!
