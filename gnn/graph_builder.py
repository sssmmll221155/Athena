"""
Graph Builder for ATHENA GNN
Constructs a file dependency graph from commit history using co-change patterns.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pickle

import numpy as np
import pandas as pd
import networkx as nx
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import File, CommitFile, Commit, Repository

try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch Geometric not available. Graph will be saved as NetworkX only.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FileGraphBuilder:
    """
    Builds a graph representation of file dependencies from commit history.

    Nodes: Files in the repository
    Node Features:
        - change_frequency: Number of commits modifying this file
        - complexity_score: Code complexity metric
        - author_count: Number of unique authors
        - lines_of_code: Total lines in file
        - is_test_file: Boolean flag

    Edges: Co-change relationships (files modified in same commit)
    Edge Weights: Co-change frequency (number of times files changed together)
    """

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize graph builder.

        Args:
            db_url: Database connection URL. If None, reads from .env
        """
        # Load environment variables
        load_dotenv()

        # Setup database connection
        if db_url is None:
            db_url = os.getenv('DATABASE_URL')
            if not db_url:
                raise ValueError("DATABASE_URL not found in environment")

        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

        # Graph data structures
        self.file_to_idx: Dict[int, int] = {}  # file_id -> node_index
        self.idx_to_file: Dict[int, int] = {}  # node_index -> file_id
        self.file_features: Dict[int, Dict] = {}  # file_id -> features
        self.cochange_matrix: Dict[Tuple[int, int], int] = defaultdict(int)  # (file1, file2) -> count

        # NetworkX graph
        self.graph = nx.Graph()

    def extract_file_features(self, repository_id: Optional[int] = None) -> pd.DataFrame:
        """
        Extract features for each file from the database.

        Args:
            repository_id: Optional filter for specific repository

        Returns:
            DataFrame with file features
        """
        logger.info("Extracting file features from database...")

        session = self.Session()
        try:
            # Query file statistics
            query = session.query(
                File.id,
                File.path,
                File.repository_id,
                File.total_commits,
                File.total_authors,
                File.lines_of_code,
                File.complexity_score,
                File.is_test_file,
                File.extension
            )

            if repository_id:
                query = query.filter(File.repository_id == repository_id)

            # Filter out deleted files
            query = query.filter(File.is_deleted == False)

            files_df = pd.read_sql(query.statement, session.bind)

            logger.info(f"Extracted features for {len(files_df)} files")

            return files_df

        finally:
            session.close()

    def extract_cochange_patterns(self, repository_id: Optional[int] = None,
                                  min_cochange_freq: int = 2) -> pd.DataFrame:
        """
        Extract co-change patterns from commit history.

        Two files are considered co-changed if they were modified in the same commit.

        Args:
            repository_id: Optional filter for specific repository
            min_cochange_freq: Minimum co-change frequency to include edge

        Returns:
            DataFrame with co-change pairs and frequencies
        """
        logger.info("Extracting co-change patterns from commit history...")

        session = self.Session()
        try:
            # Get all commit-file relationships
            query = session.query(
                CommitFile.commit_id,
                CommitFile.file_id
            )

            if repository_id:
                query = query.filter(CommitFile.repository_id == repository_id)

            # Order by commit to group files in same commit
            query = query.order_by(CommitFile.commit_id)

            commit_files_df = pd.read_sql(query.statement, session.bind)

            logger.info(f"Found {len(commit_files_df)} commit-file relationships")

            # Group files by commit
            commit_groups = commit_files_df.groupby('commit_id')['file_id'].apply(list).reset_index()

            # Build co-change matrix
            cochange_counts = defaultdict(int)

            for _, row in commit_groups.iterrows():
                file_ids = row['file_id']

                # Create edges between all pairs of files in this commit
                if len(file_ids) > 1:
                    for i in range(len(file_ids)):
                        for j in range(i + 1, len(file_ids)):
                            file1, file2 = sorted([file_ids[i], file_ids[j]])
                            cochange_counts[(file1, file2)] += 1

            # Convert to DataFrame
            cochange_data = []
            for (file1, file2), count in cochange_counts.items():
                if count >= min_cochange_freq:
                    cochange_data.append({
                        'file1_id': file1,
                        'file2_id': file2,
                        'cochange_count': count
                    })

            cochange_df = pd.DataFrame(cochange_data)

            logger.info(f"Found {len(cochange_df)} co-change pairs "
                       f"(min frequency: {min_cochange_freq})")

            return cochange_df

        finally:
            session.close()

    def build_graph(self, repository_id: Optional[int] = None,
                   min_cochange_freq: int = 2) -> nx.Graph:
        """
        Build NetworkX graph from file features and co-change patterns.

        Args:
            repository_id: Optional filter for specific repository
            min_cochange_freq: Minimum co-change frequency to include edge

        Returns:
            NetworkX graph with file nodes and co-change edges
        """
        logger.info("Building file dependency graph...")

        # Extract features and co-change patterns
        files_df = self.extract_file_features(repository_id)
        cochange_df = self.extract_cochange_patterns(repository_id, min_cochange_freq)

        # Create file_id to node index mapping
        file_ids = files_df['id'].unique()
        self.file_to_idx = {file_id: idx for idx, file_id in enumerate(file_ids)}
        self.idx_to_file = {idx: file_id for file_id, idx in self.file_to_idx.items()}

        # Add nodes with features
        for _, row in files_df.iterrows():
            file_id = row['id']
            node_idx = self.file_to_idx[file_id]

            # Node features
            features = {
                'file_id': file_id,
                'path': row['path'],
                'change_frequency': row['total_commits'] or 0,
                'author_count': row['total_authors'] or 0,
                'lines_of_code': row['lines_of_code'] or 0,
                'complexity_score': row['complexity_score'] or 0.0,
                'is_test_file': 1 if row['is_test_file'] else 0,
                'extension': row['extension'] or ''
            }

            self.file_features[file_id] = features
            self.graph.add_node(node_idx, **features)

        # Add edges with weights
        edge_count = 0
        for _, row in cochange_df.iterrows():
            file1_id = row['file1_id']
            file2_id = row['file2_id']

            # Only add edge if both files are in the graph
            if file1_id in self.file_to_idx and file2_id in self.file_to_idx:
                node1 = self.file_to_idx[file1_id]
                node2 = self.file_to_idx[file2_id]
                weight = row['cochange_count']

                self.graph.add_edge(node1, node2, weight=weight)
                self.cochange_matrix[(file1_id, file2_id)] = weight
                edge_count += 1

        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")

        # Calculate graph statistics
        self._log_graph_statistics()

        return self.graph

    def _log_graph_statistics(self):
        """Log graph statistics and properties"""
        logger.info("=" * 60)
        logger.info("GRAPH STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Nodes: {self.graph.number_of_nodes()}")
        logger.info(f"Edges: {self.graph.number_of_edges()}")

        if self.graph.number_of_nodes() > 0:
            # Degree statistics
            degrees = [d for n, d in self.graph.degree()]
            logger.info(f"Average degree: {np.mean(degrees):.2f}")
            logger.info(f"Max degree: {np.max(degrees)}")
            logger.info(f"Min degree: {np.min(degrees)}")

            # Connectivity
            is_connected = nx.is_connected(self.graph)
            logger.info(f"Is connected: {is_connected}")

            if not is_connected:
                components = list(nx.connected_components(self.graph))
                logger.info(f"Number of components: {len(components)}")
                logger.info(f"Largest component size: {max(len(c) for c in components)}")

            # Density
            density = nx.density(self.graph)
            logger.info(f"Density: {density:.4f}")

        logger.info("=" * 60)

    def to_pytorch_geometric(self) -> Optional[Data]:
        """
        Convert NetworkX graph to PyTorch Geometric Data object.

        Returns:
            PyTorch Geometric Data object or None if PyTorch not available
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch Geometric not available")
            return None

        logger.info("Converting to PyTorch Geometric format...")

        # Extract node features as matrix
        num_nodes = self.graph.number_of_nodes()

        # Feature matrix: [change_frequency, complexity_score, author_count,
        #                  lines_of_code, is_test_file]
        feature_matrix = np.zeros((num_nodes, 5))

        for node_idx in range(num_nodes):
            if node_idx in self.graph.nodes:
                node_data = self.graph.nodes[node_idx]
                feature_matrix[node_idx] = [
                    node_data['change_frequency'],
                    node_data['complexity_score'],
                    node_data['author_count'],
                    node_data['lines_of_code'],
                    node_data['is_test_file']
                ]

        # Normalize features (z-score normalization)
        for i in range(feature_matrix.shape[1]):
            col = feature_matrix[:, i]
            if col.std() > 0:
                feature_matrix[:, i] = (col - col.mean()) / col.std()

        # Extract edge list and weights
        edge_list = []
        edge_weights = []

        for (u, v, data) in self.graph.edges(data=True):
            edge_list.append([u, v])
            edge_list.append([v, u])  # Add reverse edge for undirected graph
            weight = data.get('weight', 1.0)
            edge_weights.append(weight)
            edge_weights.append(weight)

        # Convert to tensors
        x = torch.tensor(feature_matrix, dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)

        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

        logger.info(f"PyTorch Geometric Data created:")
        logger.info(f"  - Nodes: {data.num_nodes}")
        logger.info(f"  - Edges: {data.num_edges}")
        logger.info(f"  - Features: {data.num_node_features}")

        return data

    def save_graph(self, output_dir: str = "gnn/data"):
        """
        Save graph to disk in multiple formats.

        Args:
            output_dir: Directory to save graph files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save NetworkX graph
        nx_path = output_path / f"file_graph_{timestamp}.gpickle"
        nx.write_gpickle(self.graph, nx_path)
        logger.info(f"NetworkX graph saved to: {nx_path}")

        # Save mappings
        mappings_path = output_path / f"graph_mappings_{timestamp}.pkl"
        with open(mappings_path, 'wb') as f:
            pickle.dump({
                'file_to_idx': self.file_to_idx,
                'idx_to_file': self.idx_to_file,
                'file_features': self.file_features
            }, f)
        logger.info(f"Graph mappings saved to: {mappings_path}")

        # Save PyTorch Geometric data if available
        if TORCH_AVAILABLE:
            pyg_data = self.to_pytorch_geometric()
            if pyg_data is not None:
                pyg_path = output_path / f"file_graph_pyg_{timestamp}.pt"
                torch.save(pyg_data, pyg_path)
                logger.info(f"PyTorch Geometric data saved to: {pyg_path}")

        # Save graph statistics
        stats_path = output_path / f"graph_stats_{timestamp}.txt"
        with open(stats_path, 'w') as f:
            f.write("GRAPH STATISTICS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Nodes: {self.graph.number_of_nodes()}\n")
            f.write(f"Edges: {self.graph.number_of_edges()}\n")

            if self.graph.number_of_nodes() > 0:
                degrees = [d for n, d in self.graph.degree()]
                f.write(f"Average degree: {np.mean(degrees):.2f}\n")
                f.write(f"Max degree: {np.max(degrees)}\n")
                f.write(f"Min degree: {np.min(degrees)}\n")
                f.write(f"Density: {nx.density(self.graph):.4f}\n")

        logger.info(f"Graph statistics saved to: {stats_path}")

        return {
            'networkx': str(nx_path),
            'mappings': str(mappings_path),
            'pytorch': str(pyg_path) if TORCH_AVAILABLE else None,
            'stats': str(stats_path)
        }


def main():
    """Main function to build and save file dependency graph"""
    logger.info("=" * 60)
    logger.info("ATHENA - File Dependency Graph Builder")
    logger.info("=" * 60)

    try:
        # Create graph builder
        builder = FileGraphBuilder()

        # Build graph (use all repositories, min co-change frequency = 2)
        graph = builder.build_graph(repository_id=None, min_cochange_freq=2)

        # Save graph
        saved_paths = builder.save_graph()

        logger.info("=" * 60)
        logger.info("Graph building completed successfully!")
        logger.info("=" * 60)
        logger.info("Saved files:")
        for key, path in saved_paths.items():
            if path:
                logger.info(f"  - {key}: {path}")

    except Exception as e:
        logger.error(f"Error building graph: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
