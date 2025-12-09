# ATHENA Bug Prediction API

Production FastAPI server for real-time bug prediction using XGBoost ML model.

## Quick Start

```bash
# Start the API server
python start_api.py
```

The server will start on http://localhost:8000

## Endpoints

### 1. **Demo Page** - `GET /`
Interactive HTML demo page for testing predictions.

**URL:** http://localhost:8000

### 2. **Predict** - `POST /predict`
Predict bug probability for a single commit.

**Example Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "message": "fix: resolve null pointer exception in user service",
    "files_changed": 3,
    "insertions": 45,
    "deletions": 12,
    "repo_stars": 1200,
    "repo_language": "Python"
  }'
```

**Example Response:**
```json
{
  "is_bugfix": true,
  "probability": 0.87,
  "confidence": "high",
  "features_used": {
    "insertions": 45,
    "deletions": 12,
    "total_changes": 57,
    "files_changed": 3,
    "msg_length": 56,
    "msg_word_count": 8
  },
  "model_version": "bug_predictor_xgboost_20251208_003601",
  "predicted_at": "2025-12-08T00:40:15.123456"
}
```

### 3. **Batch Predict** - `POST /predict/batch`
Batch prediction for multiple commits (max 100).

**Example Request:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "commits": [
      {
        "message": "fix: null pointer",
        "files_changed": 2,
        "insertions": 10,
        "deletions": 5
      },
      {
        "message": "feat: add dashboard",
        "files_changed": 8,
        "insertions": 234,
        "deletions": 12
      }
    ]
  }'
```

**Example Response:**
```json
{
  "predictions": [
    {
      "is_bugfix": true,
      "probability": 0.76,
      "confidence": "medium",
      "model_version": "bug_predictor_xgboost_20251208_003601",
      "predicted_at": "2025-12-08T00:40:15.123456"
    },
    {
      "is_bugfix": false,
      "probability": 0.12,
      "confidence": "high",
      "model_version": "bug_predictor_xgboost_20251208_003601",
      "predicted_at": "2025-12-08T00:40:15.234567"
    }
  ],
  "total_commits": 2,
  "bugfix_count": 1,
  "average_probability": 0.44
}
```

### 4. **Model Info** - `GET /model/info`
Get model metadata and training information.

**Example Request:**
```bash
curl http://localhost:8000/model/info
```

**Example Response:**
```json
{
  "model_path": "models/bug_predictor_xgboost_20251208_003601.pkl",
  "model_version": "bug_predictor_xgboost_20251208_003601",
  "training_date": "20251208_003601",
  "feature_count": 23,
  "features": [
    "insertions",
    "deletions",
    "total_changes",
    "files_changed",
    "change_ratio"
  ],
  "accuracy_metrics": {
    "test_accuracy": 0.74,
    "test_auc_roc": 0.787,
    "test_f1_bugfix": 0.31,
    "train_accuracy": 0.99
  },
  "model_params": {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1
  }
}
```

### 5. **Health Check** - `GET /health`
Check API health and status.

**Example Request:**
```bash
curl http://localhost:8000/health
```

**Example Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 3600.5,
  "version": "1.0.0"
}
```

## Request Schema

### CommitData

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| message | string | Yes | Commit message (min 1 char) |
| files_changed | integer | Yes | Number of files changed (≥ 0) |
| insertions | integer | Yes | Lines inserted (≥ 0) |
| deletions | integer | Yes | Lines deleted (≥ 0) |
| author_email | string | No | Author email for context |
| author_name | string | No | Author name |
| repo_stars | integer | No | Repository stars (default: 0) |
| repo_language | string | No | Primary language (default: "Python") |
| hour | integer | No | Hour of commit 0-23 |
| day_of_week | integer | No | Day of week 0-6 (0=Mon) |
| is_merge | boolean | No | Is merge commit (default: false) |

## Response Schema

### PredictionResponse

| Field | Type | Description |
|-------|------|-------------|
| is_bugfix | boolean | Binary prediction (true = bug fix) |
| probability | float | Probability of bug fix (0.0-1.0) |
| confidence | string | Confidence level: "high", "medium", "low" |
| features_used | object | Extracted features used for prediction |
| model_version | string | Model version identifier |
| predicted_at | string | ISO timestamp of prediction |

### Confidence Levels

- **High**: Probability ≥ 0.8 or ≤ 0.2
- **Medium**: Probability ≥ 0.65 or ≤ 0.35
- **Low**: Probability between 0.35 and 0.65

## Features Used

The model uses **23 features** across 5 categories:

1. **Code Churn** (6 features)
   - insertions, deletions, total_changes
   - files_changed, change_ratio, files_per_change

2. **Commit Message** (5 features)
   - msg_length, msg_word_count
   - msg_has_question, msg_has_exclamation, msg_all_caps_ratio

3. **Temporal** (4 features)
   - hour, day_of_week, is_weekend, is_night

4. **Author** (5 features)
   - author_avg_changes, author_std_changes
   - author_commit_count, author_avg_files, author_total_merges

5. **Repository** (3 features)
   - repo_stars, is_merge, lang_Python

## Testing

### Using cURL

```bash
# Basic prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "message": "fix: memory leak in cache manager",
    "files_changed": 2,
    "insertions": 15,
    "deletions": 8
  }'

# With full context
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "message": "fix: memory leak in cache manager",
    "files_changed": 2,
    "insertions": 15,
    "deletions": 8,
    "author_email": "dev@example.com",
    "repo_stars": 5000,
    "repo_language": "Python",
    "hour": 14,
    "day_of_week": 2
  }'
```

### Using Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "message": "fix: resolve authentication timeout",
    "files_changed": 3,
    "insertions": 25,
    "deletions": 10
}

response = requests.post(url, json=data)
result = response.json()

print(f"Bug Fix: {result['is_bugfix']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Confidence: {result['confidence']}")
```

### Using JavaScript

```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: 'fix: handle edge case in validation',
    files_changed: 1,
    insertions: 12,
    deletions: 3
  })
})
.then(response => response.json())
.then(data => {
  console.log('Bug Fix:', data.is_bugfix);
  console.log('Probability:', data.probability);
  console.log('Confidence:', data.confidence);
});
```

## Interactive Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Both interfaces allow you to:
- View all endpoints and schemas
- Test API calls directly in the browser
- See example requests/responses
- Download OpenAPI spec

## CORS Configuration

The API includes CORS middleware configured to allow:
- **Origins**: All (`*`) - restrict in production
- **Methods**: All
- **Headers**: All
- **Credentials**: Enabled

To restrict origins in production, modify `api/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specify exact origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Logging

All predictions are logged to:
- **File**: `api/predictions.log`
- **Console**: stdout

Log format includes:
- Timestamp
- Prediction result (bugfix/normal)
- Probability
- Commit message (truncated)
- Files changed, insertions, deletions

Example log entry:
```
2025-12-08 00:45:23,456 - __main__ - INFO - Prediction - Bugfix: 1, Prob: 0.876, Msg: 'fix: resolve null pointer exception in user s...', Files: 3, +45/-12
```

## Error Handling

The API includes comprehensive error handling:

1. **Validation Errors** (422)
   - Invalid request format
   - Missing required fields
   - Out-of-range values

2. **Runtime Errors** (500)
   - Model prediction failures
   - Feature extraction errors

3. **Service Errors** (503)
   - Model not loaded

All errors return JSON with details:
```json
{
  "detail": "Error description",
  "error": "Specific error message"
}
```

## Production Deployment

For production deployment:

1. **Set specific CORS origins**
2. **Add authentication middleware**
3. **Enable HTTPS with reverse proxy (nginx)**
4. **Set up monitoring and alerts**
5. **Configure rate limiting**
6. **Use a production ASGI server**

Example with gunicorn:
```bash
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Performance

- **Cold start**: < 1 second (model loading)
- **Prediction latency**: < 50ms per commit
- **Batch prediction**: ~30ms per commit (100 commits)
- **Memory usage**: ~200MB (model in memory)

## Requirements

- Python 3.8+
- fastapi >= 0.100.0
- uvicorn >= 0.20.0
- xgboost >= 1.7.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- joblib >= 1.2.0

## Troubleshooting

### Model not found error
```
FileNotFoundError: No trained models found
```
**Solution**: Train a model first:
```bash
python train_bug_predictor.py
```

### Port already in use
```
OSError: [Errno 48] Address already in use
```
**Solution**: Change port or kill process:
```bash
# Change port in start_api.py
port=8001

# Or kill existing process
lsof -ti:8000 | xargs kill -9  # Unix
taskkill /F /PID <PID>          # Windows
```

### Import errors
```
ModuleNotFoundError: No module named 'fastapi'
```
**Solution**: Install dependencies:
```bash
pip install fastapi uvicorn xgboost pandas numpy joblib
```

## License

Part of the ATHENA project - AI-powered code intelligence platform.
