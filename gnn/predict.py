"""
GNN Prediction Module for ATHENA
Load trained GNN and make predictions on files, with ensemble support.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import glob

import numpy as np
import torch
from torch_geometric.data import Data

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn.model import create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GNNPredictor:
    """
    Predictor for bug-prone files using trained GNN model.

    Supports:
    - Loading trained GNN models
    - Making predictions for individual files
    - Batch predictions for all files in graph
    - Ensemble predictions with XGBoost and RL models
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize GNN predictor.

        Args:
            model_path: Path to trained model (.pt file)
                       If None, loads the most recent model
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Model components
        self.model = None
        self.model_metadata = None
        self.graph_data = None
        self.file_to_idx = None
        self.idx_to_file = None
        self.file_features = None

        # Load model and graph
        self.load_model(model_path)
        self.load_graph()
        self.load_mappings()

        logger.info(f"GNN Predictor initialized on {self.device}")

    def load_model(self, model_path: Optional[str] = None):
        """
        Load trained GNN model from disk.

        Args:
            model_path: Path to model file. If None, loads most recent.
        """
        logger.info("Loading GNN model...")

        if model_path is None:
            # Find most recent model
            models_dir = Path("gnn/models")
            if not models_dir.exists():
                raise FileNotFoundError("No GNN models directory found. Train a model first.")

            model_files = list(models_dir.glob("gnn_model_*.pt"))
            if not model_files:
                raise FileNotFoundError("No trained GNN models found. Run train_gnn.py first.")

            model_path = max(model_files, key=os.path.getctime)

        logger.info(f"Loading model from: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_metadata = checkpoint

        # Create model
        self.model = create_model(
            in_channels=checkpoint['num_features'],
            model_type=checkpoint['model_type'],
            hidden_channels=checkpoint['hidden_channels']
        )

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully (Test Acc: {checkpoint['best_test_acc']:.4f})")

    def load_graph(self):
        """Load graph data from disk"""
        logger.info("Loading graph data...")

        # Find most recent graph
        data_dir = Path("gnn/data")
        graph_files = list(data_dir.glob("file_graph_pyg_*.pt"))

        if not graph_files:
            raise FileNotFoundError("No graph files found. Run graph_builder.py first.")

        graph_path = max(graph_files, key=os.path.getctime)
        logger.info(f"Loading graph from: {graph_path}")

        self.graph_data = torch.load(graph_path)
        logger.info(f"Graph loaded: {self.graph_data.num_nodes} nodes")

    def load_mappings(self):
        """Load file-to-index mappings"""
        logger.info("Loading graph mappings...")

        data_dir = Path("gnn/data")
        mapping_files = list(data_dir.glob("graph_mappings_*.pkl"))

        if not mapping_files:
            raise FileNotFoundError("Graph mappings not found")

        mapping_path = max(mapping_files, key=os.path.getctime)

        with open(mapping_path, 'rb') as f:
            mappings = pickle.load(f)

        self.file_to_idx = mappings['file_to_idx']
        self.idx_to_file = mappings['idx_to_file']
        self.file_features = mappings['file_features']

        logger.info(f"Mappings loaded: {len(self.file_to_idx)} files")

    @torch.no_grad()
    def predict_all(self) -> Dict[int, float]:
        """
        Predict bug probability for all files in the graph.

        Returns:
            Dictionary mapping file_id -> bug_probability
        """
        logger.info("Predicting bug probabilities for all files...")

        self.model.eval()

        # Move data to device
        x = self.graph_data.x.to(self.device)
        edge_index = self.graph_data.edge_index.to(self.device)
        edge_weight = self.graph_data.edge_attr.to(self.device) if hasattr(self.graph_data, 'edge_attr') else None

        # Forward pass
        logits = self.model(x, edge_index, edge_weight)
        probabilities = torch.sigmoid(logits).squeeze()

        # Convert to numpy
        probs_np = probabilities.cpu().numpy()

        # Map to file IDs
        predictions = {}
        for node_idx, prob in enumerate(probs_np):
            file_id = self.idx_to_file.get(node_idx)
            if file_id:
                predictions[file_id] = float(prob)

        logger.info(f"Predictions generated for {len(predictions)} files")

        return predictions

    def predict_file(self, file_id: Optional[int] = None,
                    file_path: Optional[str] = None) -> Optional[float]:
        """
        Predict bug probability for a single file.

        Args:
            file_id: Database file ID
            file_path: File path (alternative to file_id)

        Returns:
            Bug probability (0-1) or None if file not in graph
        """
        # Get file ID from path if needed
        if file_id is None and file_path is not None:
            # Find file ID by path
            for fid, features in self.file_features.items():
                if features.get('path') == file_path:
                    file_id = fid
                    break

        if file_id is None:
            logger.warning(f"File not found: {file_path}")
            return None

        # Check if file in graph
        if file_id not in self.file_to_idx:
            logger.warning(f"File ID {file_id} not in graph")
            return None

        # Get all predictions (cached for efficiency)
        if not hasattr(self, '_cached_predictions'):
            self._cached_predictions = self.predict_all()

        return self._cached_predictions.get(file_id)

    def get_top_bugprone_files(self, top_k: int = 20) -> List[Tuple[str, float, int]]:
        """
        Get top K most bug-prone files.

        Args:
            top_k: Number of files to return

        Returns:
            List of (file_path, bug_probability, file_id) tuples
        """
        logger.info(f"Finding top {top_k} bug-prone files...")

        # Get all predictions
        predictions = self.predict_all()

        # Sort by probability
        sorted_files = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Add file paths
        results = []
        for file_id, prob in sorted_files:
            file_path = self.file_features[file_id].get('path', 'unknown')
            results.append((file_path, prob, file_id))

        return results

    def get_embeddings(self) -> np.ndarray:
        """
        Get learned node embeddings from the GNN.

        Useful for:
        - Visualization (t-SNE, UMAP)
        - Clustering files
        - Transfer learning

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        logger.info("Extracting node embeddings...")

        self.model.eval()

        # Move data to device
        x = self.graph_data.x.to(self.device)
        edge_index = self.graph_data.edge_index.to(self.device)
        edge_weight = self.graph_data.edge_attr.to(self.device) if hasattr(self.graph_data, 'edge_attr') else None

        # Get embeddings
        embeddings = self.model.get_embeddings(x, edge_index, edge_weight)
        embeddings_np = embeddings.cpu().numpy()

        logger.info(f"Embeddings shape: {embeddings_np.shape}")

        return embeddings_np


class EnsemblePredictor:
    """
    Ensemble predictor combining GNN, XGBoost, and RL predictions.

    Weighted ensemble: 0.4 * XGBoost + 0.3 * GNN + 0.3 * RL
    """

    def __init__(
        self,
        gnn_weight: float = 0.3,
        xgboost_weight: float = 0.4,
        rl_weight: float = 0.3
    ):
        """
        Initialize ensemble predictor.

        Args:
            gnn_weight: Weight for GNN predictions
            xgboost_weight: Weight for XGBoost predictions
            rl_weight: Weight for RL predictions
        """
        self.gnn_weight = gnn_weight
        self.xgboost_weight = xgboost_weight
        self.rl_weight = rl_weight

        # Normalize weights
        total_weight = gnn_weight + xgboost_weight + rl_weight
        self.gnn_weight /= total_weight
        self.xgboost_weight /= total_weight
        self.rl_weight /= total_weight

        logger.info(f"Ensemble weights: GNN={self.gnn_weight:.2f}, "
                   f"XGBoost={self.xgboost_weight:.2f}, "
                   f"RL={self.rl_weight:.2f}")

        # Initialize GNN predictor
        try:
            self.gnn_predictor = GNNPredictor()
            self.gnn_available = True
        except Exception as e:
            logger.warning(f"GNN predictor not available: {e}")
            self.gnn_available = False

    def predict(
        self,
        file_id: int,
        xgboost_prob: Optional[float] = None,
        rl_prob: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Make ensemble prediction for a file.

        Args:
            file_id: File ID
            xgboost_prob: XGBoost prediction probability (0-1)
            rl_prob: RL prediction probability (0-1)

        Returns:
            Dictionary with individual and ensemble predictions
        """
        result = {
            'file_id': file_id,
            'gnn_prob': None,
            'xgboost_prob': xgboost_prob,
            'rl_prob': rl_prob,
            'ensemble_prob': None
        }

        # Get GNN prediction
        if self.gnn_available:
            result['gnn_prob'] = self.gnn_predictor.predict_file(file_id=file_id)

        # Calculate ensemble
        probs = []
        weights = []

        if result['gnn_prob'] is not None:
            probs.append(result['gnn_prob'])
            weights.append(self.gnn_weight)

        if xgboost_prob is not None:
            probs.append(xgboost_prob)
            weights.append(self.xgboost_weight)

        if rl_prob is not None:
            probs.append(rl_prob)
            weights.append(self.rl_weight)

        if probs:
            # Weighted average
            weights_sum = sum(weights)
            result['ensemble_prob'] = sum(p * w for p, w in zip(probs, weights)) / weights_sum

        return result


def main():
    """Test GNN predictor"""
    logger.info("=" * 60)
    logger.info("ATHENA - GNN Predictor Test")
    logger.info("=" * 60)

    # Initialize predictor
    predictor = GNNPredictor()

    # Get top bug-prone files
    top_files = predictor.get_top_bugprone_files(top_k=20)

    logger.info("\nTop 20 Bug-Prone Files:")
    logger.info("=" * 60)
    for i, (path, prob, file_id) in enumerate(top_files, 1):
        logger.info(f"{i:2d}. {path:50s} | Probability: {prob:.4f}")

    # Test ensemble predictor
    logger.info("\n" + "=" * 60)
    logger.info("Testing Ensemble Predictor")
    logger.info("=" * 60)

    ensemble = EnsemblePredictor(
        gnn_weight=0.3,
        xgboost_weight=0.4,
        rl_weight=0.3
    )

    # Test prediction
    if top_files:
        test_file_id = top_files[0][2]
        result = ensemble.predict(
            file_id=test_file_id,
            xgboost_prob=0.75,
            rl_prob=0.68
        )

        logger.info(f"\nEnsemble Prediction for File ID {test_file_id}:")
        logger.info(f"  GNN:      {result['gnn_prob']:.4f}" if result['gnn_prob'] else "  GNN:      N/A")
        logger.info(f"  XGBoost:  {result['xgboost_prob']:.4f}" if result['xgboost_prob'] else "  XGBoost:  N/A")
        logger.info(f"  RL:       {result['rl_prob']:.4f}" if result['rl_prob'] else "  RL:       N/A")
        logger.info(f"  Ensemble: {result['ensemble_prob']:.4f}" if result['ensemble_prob'] else "  Ensemble: N/A")

    logger.info("\n✓ GNN Predictor test completed!")


if __name__ == "__main__":
    main()
