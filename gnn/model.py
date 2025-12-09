"""
GraphSAGE Model for ATHENA
Graph Neural Network for predicting bug-prone files based on co-change patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from typing import Optional


class GraphSAGE(nn.Module):
    """
    GraphSAGE model for bug-prone file prediction.

    Architecture:
        - 3 GraphSAGE layers with mean aggregation
        - Hidden dimensions: [64, 32, 16]
        - Dropout for regularization
        - Binary classification output (bug-prone or not)

    The model learns to aggregate information from neighboring files
    (files that co-change) to predict if a file is bug-prone.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list = [64, 32, 16],
        dropout: float = 0.5,
        aggr: str = 'mean'
    ):
        """
        Initialize GraphSAGE model.

        Args:
            in_channels: Number of input node features
            hidden_channels: List of hidden layer dimensions
            dropout: Dropout probability for regularization
            aggr: Aggregation method ('mean', 'max', 'sum')
        """
        super(GraphSAGE, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # GraphSAGE layers
        self.conv1 = SAGEConv(in_channels, hidden_channels[0], aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels[0], hidden_channels[1], aggr=aggr)
        self.conv3 = SAGEConv(hidden_channels[1], hidden_channels[2], aggr=aggr)

        # Output layer for binary classification
        self.classifier = nn.Linear(hidden_channels[2], 1)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass through the network.

        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            logits: Output logits [num_nodes, 1]
        """
        # Layer 1: GraphSAGE + ReLU + Dropout
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2: GraphSAGE + ReLU + Dropout
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3: GraphSAGE + ReLU + Dropout
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Classification layer
        x = self.classifier(x)

        return x

    def predict_proba(self, x, edge_index, edge_weight=None):
        """
        Predict probabilities for bug-prone files.

        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            probabilities: Sigmoid probabilities [num_nodes, 1]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index, edge_weight)
            probabilities = torch.sigmoid(logits)
        return probabilities

    def get_embeddings(self, x, edge_index, edge_weight=None):
        """
        Get node embeddings from the final GraphSAGE layer.

        Useful for:
        - Visualization (t-SNE, UMAP)
        - Clustering files by behavior
        - Transfer learning

        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            embeddings: Node embeddings [num_nodes, hidden_channels[-1]]
        """
        self.eval()
        with torch.no_grad():
            # Layer 1
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)

            # Layer 2
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)

            # Layer 3 (final embeddings)
            x = self.conv3(x, edge_index, edge_weight=edge_weight)

        return x


class GraphSAGEWithAttention(nn.Module):
    """
    Enhanced GraphSAGE with attention mechanism.

    This variant uses attention to weight the importance of neighboring files
    when aggregating information, allowing the model to focus on the most
    relevant co-change relationships.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list = [64, 32, 16],
        dropout: float = 0.5,
        heads: int = 4
    ):
        """
        Initialize GraphSAGE with attention.

        Args:
            in_channels: Number of input node features
            hidden_channels: List of hidden layer dimensions
            dropout: Dropout probability
            heads: Number of attention heads
        """
        super(GraphSAGEWithAttention, self).__init__()

        from torch_geometric.nn import GATConv

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.heads = heads

        # GAT layers with multi-head attention
        self.conv1 = GATConv(
            in_channels,
            hidden_channels[0] // heads,
            heads=heads,
            dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_channels[0],
            hidden_channels[1] // heads,
            heads=heads,
            dropout=dropout
        )
        self.conv3 = GATConv(
            hidden_channels[1],
            hidden_channels[2],
            heads=1,
            concat=False,
            dropout=dropout
        )

        # Output layer
        self.classifier = nn.Linear(hidden_channels[2], 1)

    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass with attention"""
        # Note: GATConv doesn't use edge_weight in the same way as SAGEConv
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)

        x = self.classifier(x)
        return x


def create_model(
    in_channels: int,
    model_type: str = 'sage',
    hidden_channels: list = [64, 32, 16],
    dropout: float = 0.5,
    **kwargs
) -> nn.Module:
    """
    Factory function to create GNN models.

    Args:
        in_channels: Number of input features
        model_type: Type of model ('sage' or 'sage_attention')
        hidden_channels: Hidden layer dimensions
        dropout: Dropout probability
        **kwargs: Additional model-specific arguments

    Returns:
        GNN model instance
    """
    if model_type == 'sage':
        return GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            aggr=kwargs.get('aggr', 'mean')
        )
    elif model_type == 'sage_attention':
        return GraphSAGEWithAttention(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            heads=kwargs.get('heads', 4)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    print("Testing GraphSAGE model...")

    # Create dummy data
    num_nodes = 100
    num_features = 5
    num_edges = 500

    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_weight = torch.rand(num_edges)

    # Test standard GraphSAGE
    model = GraphSAGE(in_channels=num_features)
    print(f"\nModel architecture:\n{model}")

    # Forward pass
    logits = model(x, edge_index, edge_weight)
    print(f"\nLogits shape: {logits.shape}")

    # Predict probabilities
    probs = model.predict_proba(x, edge_index, edge_weight)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probability range: [{probs.min():.3f}, {probs.max():.3f}]")

    # Get embeddings
    embeddings = model.get_embeddings(x, edge_index, edge_weight)
    print(f"Embeddings shape: {embeddings.shape}")

    # Test attention variant
    print("\n" + "=" * 60)
    print("Testing GraphSAGE with Attention...")
    model_att = GraphSAGEWithAttention(in_channels=num_features, heads=4)
    print(f"\nModel architecture:\n{model_att}")

    logits_att = model_att(x, edge_index, edge_weight)
    print(f"Logits shape: {logits_att.shape}")

    print("\n✓ All tests passed!")
