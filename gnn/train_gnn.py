"""
Training Pipeline for ATHENA GNN
Trains GraphSAGE model to predict bug-prone files based on co-change patterns.
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import pickle
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import File, Commit, CommitFile
from gnn.model import create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GNNTrainer:
    """
    Trainer for GraphSAGE model on file dependency graph.
    """

    def __init__(
        self,
        model_type: str = 'sage',
        hidden_channels: list = [64, 32, 16],
        learning_rate: float = 0.01,
        dropout: float = 0.5,
        weight_decay: float = 5e-4,
        device: str = 'auto'
    ):
        """
        Initialize GNN trainer.

        Args:
            model_type: Type of GNN model ('sage' or 'sage_attention')
            hidden_channels: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            dropout: Dropout probability
            weight_decay: L2 regularization weight
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.model_type = model_type
        self.hidden_channels = hidden_channels
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.weight_decay = weight_decay

        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Model components (initialized later)
        self.model = None
        self.optimizer = None
        self.criterion = None

        # Data
        self.data = None
        self.train_mask = None
        self.test_mask = None

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'test_auc': []
        }

        # Best model tracking
        self.best_test_acc = 0.0
        self.best_model_state = None

        # Database connection
        load_dotenv()
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            self.engine = create_engine(db_url)
            self.Session = sessionmaker(bind=self.engine)
        else:
            self.engine = None
            logger.warning("No DATABASE_URL found - will not load labels from database")

    def load_graph(self, graph_path: Optional[str] = None) -> Data:
        """
        Load graph from disk.

        Args:
            graph_path: Path to PyTorch Geometric graph file (.pt)
                       If None, loads the most recent graph

        Returns:
            PyTorch Geometric Data object
        """
        logger.info("Loading graph...")

        if graph_path is None:
            # Find most recent graph file
            data_dir = Path("gnn/data")
            graph_files = list(data_dir.glob("file_graph_pyg_*.pt"))

            if not graph_files:
                raise FileNotFoundError(
                    "No graph files found. Run graph_builder.py first."
                )

            graph_path = max(graph_files, key=os.path.getctime)

        logger.info(f"Loading graph from: {graph_path}")
        data = torch.load(graph_path)

        logger.info(f"Graph loaded: {data.num_nodes} nodes, "
                   f"{data.num_edges} edges, "
                   f"{data.num_node_features} features")

        self.data = data
        return data

    def create_labels_from_database(self) -> torch.Tensor:
        """
        Create binary labels for nodes based on bug-fix frequency.

        A file is labeled as bug-prone (label=1) if it has been modified
        in bug-fix commits more frequently than the median.

        Returns:
            Binary labels [num_nodes]
        """
        logger.info("Creating labels from database...")

        if self.engine is None:
            raise RuntimeError("Database connection not available")

        # Load file-to-index mapping
        data_dir = Path("gnn/data")
        mapping_files = list(data_dir.glob("graph_mappings_*.pkl"))

        if not mapping_files:
            raise FileNotFoundError("Graph mappings not found")

        mapping_path = max(mapping_files, key=os.path.getctime)
        with open(mapping_path, 'rb') as f:
            mappings = pickle.load(f)

        file_to_idx = mappings['file_to_idx']
        idx_to_file = mappings['idx_to_file']

        # Query bug-fix commits
        session = self.Session()
        try:
            # Bug-fix pattern keywords
            bugfix_keywords = ['fix', 'bug', 'patch', 'error', 'issue', 'resolve']

            # Get bug-fix commit counts per file
            query = session.query(
                CommitFile.file_id,
                Commit.message
            ).join(
                Commit,
                CommitFile.commit_id == Commit.id
            )

            commit_files_df = pd.read_sql(query.statement, session.bind)

            # Identify bug-fix commits
            commit_files_df['is_bugfix'] = commit_files_df['message'].str.lower().apply(
                lambda msg: any(keyword in msg for keyword in bugfix_keywords)
            )

            # Count bug-fixes per file
            bugfix_counts = commit_files_df.groupby('file_id')['is_bugfix'].sum()

            # Get total commits per file
            total_commits = commit_files_df.groupby('file_id').size()

            # Calculate bug-fix ratio
            bugfix_ratio = bugfix_counts / total_commits
            bugfix_ratio = bugfix_ratio.fillna(0)

            # Create labels: 1 if above median, 0 otherwise
            threshold = bugfix_ratio.median()
            logger.info(f"Bug-fix ratio threshold: {threshold:.3f}")

            labels = np.zeros(self.data.num_nodes, dtype=np.int64)

            for node_idx in range(self.data.num_nodes):
                file_id = idx_to_file.get(node_idx)
                if file_id and file_id in bugfix_ratio:
                    labels[node_idx] = 1 if bugfix_ratio[file_id] > threshold else 0

            # Log label distribution
            num_positive = labels.sum()
            num_negative = len(labels) - num_positive
            logger.info(f"Label distribution: {num_positive} bug-prone, "
                       f"{num_negative} normal files")

        finally:
            session.close()

        return torch.tensor(labels, dtype=torch.long)

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Split nodes into train/test sets.

        Args:
            test_size: Fraction of nodes for testing
            random_state: Random seed for reproducibility
        """
        logger.info(f"Splitting data (test_size={test_size})...")

        num_nodes = self.data.num_nodes

        # Create masks
        indices = np.arange(num_nodes)
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=self.data.y.numpy() if hasattr(self.data, 'y') else None
        )

        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        self.train_mask[train_idx] = True
        self.test_mask[test_idx] = True

        logger.info(f"Train nodes: {self.train_mask.sum().item()}")
        logger.info(f"Test nodes: {self.test_mask.sum().item()}")

    def initialize_model(self):
        """Initialize model, optimizer, and loss criterion"""
        logger.info(f"Initializing {self.model_type} model...")

        # Create model
        self.model = create_model(
            in_channels=self.data.num_node_features,
            model_type=self.model_type,
            hidden_channels=self.hidden_channels,
            dropout=self.dropout
        )
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Loss criterion (binary cross-entropy with logits)
        # Calculate class weights for imbalanced data
        if hasattr(self.data, 'y'):
            num_positive = (self.data.y == 1).sum().item()
            num_negative = (self.data.y == 0).sum().item()
            pos_weight = torch.tensor([num_negative / num_positive]).to(self.device)
            logger.info(f"Class weight for positive class: {pos_weight.item():.2f}")
        else:
            pos_weight = torch.tensor([1.0]).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            (train_loss, train_accuracy)
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move data to device
        x = self.data.x.to(self.device)
        edge_index = self.data.edge_index.to(self.device)
        edge_weight = self.data.edge_attr.to(self.device) if hasattr(self.data, 'edge_attr') else None
        y = self.data.y.to(self.device)

        # Forward pass
        logits = self.model(x, edge_index, edge_weight).squeeze()

        # Compute loss on training nodes
        train_logits = logits[self.train_mask]
        train_labels = y[self.train_mask].float()

        loss = self.criterion(train_logits, train_labels)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Calculate accuracy
        with torch.no_grad():
            train_preds = (torch.sigmoid(train_logits) >= 0.5).long()
            train_acc = (train_preds == y[self.train_mask]).float().mean()

        return loss.item(), train_acc.item()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        # Move data to device
        x = self.data.x.to(self.device)
        edge_index = self.data.edge_index.to(self.device)
        edge_weight = self.data.edge_attr.to(self.device) if hasattr(self.data, 'edge_attr') else None
        y = self.data.y.to(self.device)

        # Forward pass
        logits = self.model(x, edge_index, edge_weight).squeeze()

        # Test set predictions
        test_logits = logits[self.test_mask]
        test_labels = y[self.test_mask]

        # Loss
        test_loss = self.criterion(test_logits, test_labels.float()).item()

        # Predictions and probabilities
        test_probs = torch.sigmoid(test_logits)
        test_preds = (test_probs >= 0.5).long()

        # Move to CPU for sklearn metrics
        test_labels_np = test_labels.cpu().numpy()
        test_preds_np = test_preds.cpu().numpy()
        test_probs_np = test_probs.cpu().numpy()

        # Calculate metrics
        metrics = {
            'loss': test_loss,
            'accuracy': accuracy_score(test_labels_np, test_preds_np),
            'precision': precision_score(test_labels_np, test_preds_np, zero_division=0),
            'recall': recall_score(test_labels_np, test_preds_np, zero_division=0),
            'f1': f1_score(test_labels_np, test_preds_np, zero_division=0),
            'auc_roc': roc_auc_score(test_labels_np, test_probs_np) if len(np.unique(test_labels_np)) > 1 else 0.0
        }

        return metrics

    def train(self, num_epochs: int = 100, early_stopping_patience: int = 20):
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of training epochs
            early_stopping_patience: Stop if no improvement for N epochs
        """
        logger.info("=" * 60)
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info("=" * 60)

        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch()

            # Evaluate
            test_metrics = self.evaluate()

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_metrics['loss'])
            self.history['test_acc'].append(test_metrics['accuracy'])
            self.history['test_auc'].append(test_metrics['auc_roc'])

            # Check for best model
            if test_metrics['accuracy'] > self.best_test_acc:
                self.best_test_acc = test_metrics['accuracy']
                self.best_model_state = self.model.state_dict().copy()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # Log progress
            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Test Loss: {test_metrics['loss']:.4f} "
                    f"Acc: {test_metrics['accuracy']:.4f} "
                    f"AUC: {test_metrics['auc_roc']:.4f} "
                    f"F1: {test_metrics['f1']:.4f}"
                )

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch} "
                           f"(no improvement for {early_stopping_patience} epochs)")
                break

        logger.info("=" * 60)
        logger.info(f"Training completed!")
        logger.info(f"Best test accuracy: {self.best_test_acc:.4f} at epoch {best_epoch}")
        logger.info("=" * 60)

        # Load best model
        self.model.load_state_dict(self.best_model_state)

    def save_model(self, output_dir: str = "gnn/models"):
        """
        Save trained model and training results.

        Args:
            output_dir: Directory to save model files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        model_path = output_path / f"gnn_model_{timestamp}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'hidden_channels': self.hidden_channels,
            'num_features': self.data.num_node_features,
            'best_test_acc': self.best_test_acc,
            'history': self.history
        }, model_path)
        logger.info(f"Model saved to: {model_path}")

        # Save training history
        history_path = output_path / f"training_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to: {history_path}")

        # Final evaluation report
        final_metrics = self.evaluate()
        report_path = output_path / f"evaluation_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("GNN MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model Type: {self.model_type}\n")
            f.write(f"Hidden Channels: {self.hidden_channels}\n")
            f.write(f"Training Date: {timestamp}\n\n")
            f.write("TEST SET METRICS:\n")
            f.write("-" * 60 + "\n")
            for metric, value in final_metrics.items():
                f.write(f"{metric.upper():15s}: {value:.4f}\n")

        logger.info(f"Evaluation report saved to: {report_path}")

        return {
            'model': str(model_path),
            'history': str(history_path),
            'report': str(report_path)
        }

    def plot_training_curves(self, output_dir: str = "gnn/plots"):
        """
        Plot and save training curves.

        Args:
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot loss
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['test_loss'], label='Test Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Test Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot accuracy and AUC
        axes[1].plot(self.history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[1].plot(self.history['test_acc'], label='Test Accuracy', linewidth=2)
        axes[1].plot(self.history['test_auc'], label='Test AUC-ROC', linewidth=2, linestyle='--')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('Model Performance', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = output_path / f"training_curves_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to: {plot_path}")

        plt.close()


def main():
    """Main training function"""
    logger.info("=" * 60)
    logger.info("ATHENA - GNN Training Pipeline")
    logger.info("=" * 60)

    # Initialize trainer
    trainer = GNNTrainer(
        model_type='sage',
        hidden_channels=[64, 32, 16],
        learning_rate=0.01,
        dropout=0.5
    )

    # Load graph
    trainer.load_graph()

    # Create labels from database
    labels = trainer.create_labels_from_database()
    trainer.data.y = labels

    # Split data
    trainer.split_data(test_size=0.2, random_state=42)

    # Initialize model
    trainer.initialize_model()

    # Train
    trainer.train(num_epochs=100, early_stopping_patience=20)

    # Save model and results
    saved_paths = trainer.save_model()
    trainer.plot_training_curves()

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)
    logger.info("Saved files:")
    for key, path in saved_paths.items():
        logger.info(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
