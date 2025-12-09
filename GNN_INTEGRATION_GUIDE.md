# GNN Integration Guide for ATHENA

This guide explains how to build, train, and deploy the Graph Neural Network layer for ATHENA.

---

## Overview

The GNN layer analyzes file dependency patterns from commit history to predict bug-prone files. It complements the existing XGBoost and RL models by adding graph-structured learning.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ATHENA ENSEMBLE                       │
├─────────────────┬──────────────────┬────────────────────┤
│  XGBoost (40%)  │    GNN (30%)     │     RL (30%)       │
│  Code Metrics   │  Co-change Graph │  Adaptive Threshold│
└─────────────────┴──────────────────┴────────────────────┘
```

---

## Installation

### 1. Install Dependencies

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric

# Install other GNN dependencies
pip install -r gnn/requirements.txt
```

For GPU support, use:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
```

### 2. Verify Installation

```bash
python test_gnn.py
```

This will test all GNN components and verify they're working correctly.

---

## Usage

### Step 1: Build the File Dependency Graph

Extract co-change patterns from the database and build the graph:

```bash
python -m gnn.graph_builder
```

**Output:**
- `gnn/data/file_graph_TIMESTAMP.gpickle` - NetworkX graph
- `gnn/data/file_graph_pyg_TIMESTAMP.pt` - PyTorch Geometric data
- `gnn/data/graph_mappings_TIMESTAMP.pkl` - File ID mappings
- `gnn/data/graph_stats_TIMESTAMP.txt` - Graph statistics

**What it does:**
- Queries `commit_files` table for co-change relationships
- Creates nodes for each file with features:
  - `change_frequency`: Number of commits modifying this file
  - `complexity_score`: Code complexity metric
  - `author_count`: Number of unique authors
  - `lines_of_code`: Total lines
  - `is_test_file`: Boolean flag
- Creates edges between files modified in the same commit
- Edge weights = co-change frequency

### Step 2: Train the GNN Model

Train the GraphSAGE model on the graph:

```bash
python -m gnn.train_gnn
```

**Output:**
- `gnn/models/gnn_model_TIMESTAMP.pt` - Trained model
- `gnn/models/training_history_TIMESTAMP.json` - Training metrics
- `gnn/models/evaluation_report_TIMESTAMP.txt` - Test set performance
- `gnn/plots/training_curves_TIMESTAMP.png` - Loss/accuracy plots

**Training details:**
- Architecture: 3-layer GraphSAGE with mean aggregation
- Hidden dimensions: [64, 32, 16]
- Training: 100 epochs with early stopping
- Split: 80% train, 20% test
- Loss: Binary cross-entropy with class weights
- Optimizer: Adam (lr=0.01, weight_decay=5e-4)

**Labels:**
Files are labeled as bug-prone if their bug-fix commit ratio is above the median.

### Step 3: Make Predictions

Test the trained model:

```bash
python -m gnn.predict
```

This will:
- Load the trained GNN model
- Predict bug probabilities for all files
- Display top 20 bug-prone files
- Test ensemble predictions

### Step 4: Integrate with API

#### Option A: Auto-integration (Recommended)

Add these lines to `api/main.py`:

```python
# At the top with other imports
from api.gnn_predict import get_gnn_router, initialize_gnn_predictor

# In the startup event (after model_manager.load_latest_model())
@app.on_event("startup")
async def startup_event():
    # ... existing code ...

    # Initialize GNN predictor
    initialize_gnn_predictor()
    logger.info("GNN predictor initialized")

# After app creation (after CORS middleware)
app.include_router(get_gnn_router())
```

#### Option B: Manual integration

Copy the code from `api/gnn_predict.py` into `api/main.py` and register the endpoints manually.

### Step 5: Test API Endpoints

Start the API:
```bash
python api/main.py
```

Test GNN endpoints:

```bash
# Get GNN status
curl http://localhost:8000/predict/gnn/status

# Predict for a file
curl -X POST http://localhost:8000/predict/gnn/file \
  -H "Content-Type: application/json" \
  -d '{"file_id": 123}'

# Get top bug-prone files
curl http://localhost:8000/predict/gnn/top-files?top_k=20

# Ensemble prediction
curl -X POST http://localhost:8000/predict/ensemble \
  -H "Content-Type: application/json" \
  -d '{"file_id": 123}'
```

---

## API Endpoints

### 1. `POST /predict/gnn/file`

Predict bug probability for a file using GNN.

**Request:**
```json
{
  "file_id": 123
}
```

**Response:**
```json
{
  "file_id": 123,
  "file_path": "src/main.py",
  "gnn_probability": 0.782,
  "is_bugprone": true,
  "confidence": "high",
  "predicted_at": "2024-12-08T10:30:00"
}
```

### 2. `POST /predict/ensemble`

Ensemble prediction combining GNN, XGBoost, and RL.

**Request:**
```json
{
  "file_id": 123
}
```

**Response:**
```json
{
  "file_id": 123,
  "file_path": "src/main.py",
  "gnn_probability": 0.782,
  "xgboost_probability": 0.854,
  "rl_probability": 0.731,
  "ensemble_probability": 0.789,
  "is_bugprone": true,
  "confidence": "high",
  "ensemble_weights": {
    "gnn": 0.3,
    "xgboost": 0.4,
    "rl": 0.3
  },
  "models_used": ["GNN", "XGBoost", "RL"],
  "predicted_at": "2024-12-08T10:30:00"
}
```

### 3. `GET /predict/gnn/top-files`

Get top K most bug-prone files.

**Query params:**
- `top_k`: Number of files (default: 20, max: 100)
- `threshold`: Minimum probability (default: 0.5)

**Response:**
```json
{
  "top_files": [
    {
      "file_path": "src/core/database.py",
      "probability": 0.892,
      "file_id": 456,
      "rank": 1
    },
    ...
  ],
  "total_files": 20,
  "threshold": 0.5
}
```

### 4. `GET /predict/gnn/status`

Get GNN model status and information.

**Response:**
```json
{
  "gnn_available": true,
  "ensemble_available": true,
  "models_loaded": ["GNN"],
  "gnn_info": {
    "num_files": 1234,
    "num_edges": 5678,
    "model_type": "sage",
    "best_test_acc": 0.743
  },
  "ensemble_weights": {
    "gnn": 0.3,
    "xgboost": 0.4,
    "rl": 0.3
  }
}
```

---

## How It Works

### Graph Construction

1. **Query Database**: Extract all commit-file relationships
2. **Create Nodes**: One node per file with aggregated features
3. **Create Edges**: Connect files modified in the same commit
4. **Edge Weights**: Number of times files co-changed

### GNN Learning

The GraphSAGE model learns by:

1. **Aggregating neighbor information**: Each file learns from files it co-changes with
2. **Multi-hop reasoning**: 3 layers → sees files up to 3 hops away
3. **Feature transformation**: Learns which features are important
4. **Graph structure**: Captures coupling and dependency patterns

### Prediction

For a new file:
1. Extract its features and neighbors
2. Aggregate information from co-changed files
3. Transform through 3 GraphSAGE layers
4. Output bug probability (sigmoid of logits)

---

## Ensemble Strategy

The ensemble combines three complementary models:

| Model | Weight | What it captures |
|-------|--------|------------------|
| **XGBoost** | 40% | Code metrics, commit patterns, message analysis |
| **GNN** | 30% | File dependencies, co-change patterns, graph structure |
| **RL** | 30% | Adaptive thresholds, context-aware prioritization |

**Ensemble formula:**
```
P_ensemble = 0.4 × P_xgboost + 0.3 × P_gnn + 0.3 × P_rl
```

---

## Performance Tips

### 1. Graph Building

- **Large repositories**: Use `repository_id` filter
- **Reduce edges**: Increase `min_cochange_freq` (e.g., 5)
- **Memory**: Process repositories in batches

### 2. Training

- **GPU acceleration**: Use CUDA if available
- **Batch size**: Increase for faster training (if memory allows)
- **Early stopping**: Adjust `patience` (default: 20)

### 3. Inference

- **Batch predictions**: Use `predict_all()` instead of individual predictions
- **Caching**: Predictions are cached in `_cached_predictions`
- **API**: Consider adding caching middleware

---

## Troubleshooting

### "No graph files found"

**Solution**: Run `python -m gnn.graph_builder` first

### "No trained models found"

**Solution**: Run `python -m gnn.train_gnn` first

### "CUDA out of memory"

**Solution**:
- Use CPU: `device='cpu'` in trainer
- Reduce hidden dimensions: `[32, 16, 8]`
- Reduce batch size

### "Import Error: torch_geometric not found"

**Solution**:
```bash
pip install torch-geometric
```

### Low accuracy

**Possible causes:**
- Insufficient data (< 100 files)
- Poor label quality (adjust bug-fix threshold)
- Hyperparameters need tuning

**Solutions:**
- Collect more commit history
- Adjust `hidden_channels`, `learning_rate`, `dropout`
- Try attention variant: `model_type='sage_attention'`

---

## Advanced Configuration

### Custom Model Architecture

Edit `gnn/model.py`:

```python
model = GraphSAGE(
    in_channels=5,
    hidden_channels=[128, 64, 32],  # Deeper network
    dropout=0.3,                     # Less dropout
    aggr='max'                        # Max aggregation
)
```

### Custom Labels

Edit `gnn/train_gnn.py` → `create_labels_from_database()`:

```python
# Example: Label based on issue count instead of bug-fix ratio
labels = (issue_counts > threshold).astype(int)
```

### Custom Ensemble Weights

Edit `api/gnn_predict.py`:

```python
ensemble_predictor = EnsemblePredictor(
    gnn_weight=0.5,      # Increase GNN weight
    xgboost_weight=0.3,
    rl_weight=0.2
)
```

---

## File Structure

```
athena/
├── gnn/
│   ├── __init__.py
│   ├── README.md
│   ├── requirements.txt
│   ├── graph_builder.py      # Build graph from database
│   ├── model.py               # GraphSAGE architecture
│   ├── train_gnn.py           # Training pipeline
│   ├── predict.py             # Inference and ensemble
│   ├── data/                  # Graph data (generated)
│   ├── models/                # Trained models (generated)
│   └── plots/                 # Visualizations (generated)
├── api/
│   ├── gnn_predict.py         # GNN API endpoints
│   └── main.py                # Main API (integrate here)
├── test_gnn.py                # Test suite
└── GNN_INTEGRATION_GUIDE.md   # This file
```

---

## Next Steps

1. ✅ Install dependencies
2. ✅ Run test suite (`python test_gnn.py`)
3. ✅ Build graph (`python -m gnn.graph_builder`)
4. ✅ Train model (`python -m gnn.train_gnn`)
5. ✅ Test predictions (`python -m gnn.predict`)
6. ✅ Integrate with API (see Step 4)
7. ✅ Deploy and monitor

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments in each module
3. Check logs in `api/predictions.log`

---

## References

- **GraphSAGE Paper**: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **ATHENA Architecture**: See main README.md
