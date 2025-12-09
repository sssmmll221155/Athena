# GNN Layer for ATHENA

Graph Neural Network module for analyzing file dependencies and predicting bug-prone files based on co-change patterns.

## Overview

This module uses **GraphSAGE** (Graph Sample and Aggregate) to learn representations of files in the codebase by analyzing:
- **Co-change patterns**: Files frequently modified together in commits
- **File characteristics**: Change frequency, complexity, author diversity
- **Dependency structure**: Graph topology of file relationships

## Components

### 1. `graph_builder.py`
Constructs a graph from the commit history:
- **Nodes**: Files (with features: change_frequency, complexity, author_count)
- **Edges**: Co-change relationships (weighted by frequency)
- Saves as PyTorch Geometric `Data` object

### 2. `model.py`
GraphSAGE neural network:
- 3-layer architecture with mean aggregation
- Hidden dimensions: [64, 32, 16]
- Binary classification (bug-prone vs normal)

### 3. `train_gnn.py`
Training pipeline:
- Loads graph from PostgreSQL database
- Train/test split (80/20)
- Trains for 100 epochs
- Saves best model and plots learning curves

### 4. `predict.py`
Inference and ensemble:
- Loads trained GNN model
- Predicts bug probability for files
- Combines with XGBoost and RL predictions

## Installation

```bash
pip install -r gnn/requirements.txt
```

## Usage

### 1. Build Graph from Database
```bash
python -m gnn.graph_builder
```

### 2. Train GNN Model
```bash
python -m gnn.train_gnn
```

### 3. Make Predictions
```bash
python -m gnn.predict --file-path "src/main.py"
```

### 4. Use in API
The GNN predictions are automatically integrated into the API at:
- `POST /predict/gnn` - GNN-enhanced predictions
- Ensemble combines: 40% XGBoost + 30% GNN + 30% RL

## Model Architecture

```
Input: Node features [change_freq, complexity, author_count]
   ↓
GraphSAGE Layer 1 (64 hidden)
   ↓
ReLU + Dropout
   ↓
GraphSAGE Layer 2 (32 hidden)
   ↓
ReLU + Dropout
   ↓
GraphSAGE Layer 3 (16 hidden)
   ↓
Linear → Sigmoid → Bug Probability
```

## Graph Structure

- **Nodes**: Each file in the repository
- **Node Features**:
  - `change_frequency`: Total number of commits modifying this file
  - `complexity_score`: Code complexity metric (cyclomatic/cognitive)
  - `author_count`: Number of unique authors who modified this file
  - `lines_of_code`: Total lines in file
  - `is_test_file`: Boolean flag

- **Edges**: Co-change relationships
  - **Weight**: Number of times two files were modified in the same commit
  - **Direction**: Undirected (bidirectional)

## Performance

The GNN leverages graph structure to capture:
- Files that change together often (coupling)
- Hotspot files (central nodes with high degree)
- Clusters of related files
- Temporal patterns in file modifications

Ensemble performance improves by ~5-10% over XGBoost alone.
