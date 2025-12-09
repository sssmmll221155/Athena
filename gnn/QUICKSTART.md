# GNN Quick Start Guide

Get started with ATHENA's Graph Neural Network layer in 5 minutes!

## Prerequisites

- Python 3.8+
- PostgreSQL database with commit history data
- ATHENA project set up

## Installation

```bash
# 1. Install PyTorch (choose CPU or GPU version)
pip install torch torchvision torchaudio

# 2. Install PyTorch Geometric
pip install torch-geometric

# 3. Install GNN dependencies
pip install -r gnn/requirements.txt
```

## Usage

### Step 1: Build the Graph (5-10 minutes)

```bash
python -m gnn.graph_builder
```

This creates a file dependency graph from your commit history.

**Output**: `gnn/data/file_graph_pyg_*.pt`

### Step 2: Train the Model (10-30 minutes)

```bash
python -m gnn.train_gnn
```

This trains a GraphSAGE model to predict bug-prone files.

**Output**: `gnn/models/gnn_model_*.pt`

### Step 3: Make Predictions

```bash
python -m gnn.predict
```

This shows the top 20 most bug-prone files.

### Step 4: Integrate with API

Add to `api/main.py`:

```python
from api.gnn_predict import get_gnn_router, initialize_gnn_predictor

# In startup event
@app.on_event("startup")
async def startup_event():
    # ... existing code ...
    initialize_gnn_predictor()

# After app creation
app.include_router(get_gnn_router())
```

Start the API:
```bash
python api/main.py
```

Test it:
```bash
# Get top bug-prone files
curl http://localhost:8000/predict/gnn/top-files?top_k=10

# Get ensemble prediction
curl -X POST http://localhost:8000/predict/ensemble \
  -H "Content-Type: application/json" \
  -d '{"file_id": 123}'
```

## What's Next?

- Read the full guide: `GNN_INTEGRATION_GUIDE.md`
- Tune hyperparameters in `gnn/train_gnn.py`
- Adjust ensemble weights in `api/gnn_predict.py`
- Visualize the graph with NetworkX

## Troubleshooting

**"No module named 'torch_geometric'"**
```bash
pip install torch-geometric
```

**"No graph files found"**
```bash
python -m gnn.graph_builder
```

**"No trained models found"**
```bash
python -m gnn.train_gnn
```

For more help, see `GNN_INTEGRATION_GUIDE.md`.
