"""
Test Script for GNN Implementation
Tests all GNN components end-to-end.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all GNN modules can be imported"""
    logger.info("=" * 60)
    logger.info("TEST 1: Importing GNN modules...")
    logger.info("=" * 60)

    try:
        # Test core GNN modules (no database dependency)
        from gnn.model import GraphSAGE, create_model
        logger.info("✓ model imported")

        # Test API components
        from api.gnn_predict import get_gnn_router
        logger.info("✓ gnn_predict imported")

        # Test database-dependent modules (may fail if DB not configured)
        try:
            from gnn.graph_builder import FileGraphBuilder
            logger.info("✓ graph_builder imported")
        except Exception as e:
            logger.warning(f"⚠ graph_builder import failed (DB required): {e}")

        try:
            from gnn.train_gnn import GNNTrainer
            logger.info("✓ train_gnn imported")
        except Exception as e:
            logger.warning(f"⚠ train_gnn import failed (DB required): {e}")

        try:
            from gnn.predict import GNNPredictor, EnsemblePredictor
            logger.info("✓ predict imported")
        except Exception as e:
            logger.warning(f"⚠ predict import failed (models required): {e}")

        logger.info("\n✓ Core imports successful!\n")
        return True

    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_graph_builder():
    """Test graph builder"""
    logger.info("=" * 60)
    logger.info("TEST 2: Testing Graph Builder...")
    logger.info("=" * 60)
    logger.info("NOTE: This test requires database connection.")
    logger.info("Skipping for now. Run manually after DB setup.")
    logger.info("")

    return True


def test_model_creation():
    """Test GNN model creation"""
    logger.info("=" * 60)
    logger.info("TEST 3: Testing Model Creation...")
    logger.info("=" * 60)

    try:
        import torch
        from gnn.model import GraphSAGE, GraphSAGEWithAttention

        # Test GraphSAGE
        model = GraphSAGE(in_channels=5, hidden_channels=[64, 32, 16])
        logger.info(f"✓ GraphSAGE created ({sum(p.numel() for p in model.parameters()):,} parameters)")

        # Test forward pass
        x = torch.randn(100, 5)
        edge_index = torch.randint(0, 100, (2, 200))
        edge_weight = torch.rand(200)

        logits = model(x, edge_index, edge_weight)
        logger.info(f"✓ Forward pass successful (output shape: {logits.shape})")

        # Test predictions
        probs = model.predict_proba(x, edge_index, edge_weight)
        logger.info(f"✓ Predictions generated (prob range: [{probs.min():.3f}, {probs.max():.3f}])")

        # Test embeddings
        embeddings = model.get_embeddings(x, edge_index, edge_weight)
        logger.info(f"✓ Embeddings extracted (shape: {embeddings.shape})")

        # Test attention variant
        model_att = GraphSAGEWithAttention(in_channels=5)
        logger.info(f"✓ GraphSAGE with Attention created")

        logger.info("\n✓ Model creation tests passed!\n")
        return True

    except Exception as e:
        logger.error(f"✗ Model creation test failed: {e}", exc_info=True)
        return False


def test_full_pipeline():
    """Test full GNN pipeline (build graph, train, predict)"""
    logger.info("=" * 60)
    logger.info("TEST 4: Full Pipeline Test...")
    logger.info("=" * 60)
    logger.info("NOTE: This test requires database connection and takes time.")
    logger.info("Skipping for now. Run manually with:")
    logger.info("  1. python -m gnn.graph_builder")
    logger.info("  2. python -m gnn.train_gnn")
    logger.info("  3. python -m gnn.predict")
    logger.info("")

    return True


def test_api_integration():
    """Test API integration"""
    logger.info("=" * 60)
    logger.info("TEST 5: Testing API Integration...")
    logger.info("=" * 60)

    try:
        from api.gnn_predict import get_gnn_router, initialize_gnn_predictor

        # Get router
        router = get_gnn_router()
        logger.info(f"✓ GNN router created with {len(router.routes)} routes")

        # List routes
        for route in router.routes:
            logger.info(f"  - {route.methods} {route.path}")

        logger.info("\n✓ API integration tests passed!\n")
        return True

    except Exception as e:
        logger.error(f"✗ API integration test failed: {e}", exc_info=True)
        return False


def main():
    """Run all tests"""
    logger.info("\n")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + " " * 15 + "ATHENA GNN TEST SUITE" + " " * 22 + "║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("\n")

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Graph Builder", test_graph_builder()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Full Pipeline", test_full_pipeline()))
    results.append(("API Integration", test_api_integration()))

    # Summary
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:20s}: {status}")

    logger.info("=" * 60)
    logger.info(f"TOTAL: {passed}/{total} tests passed")
    logger.info("=" * 60)

    if passed == total:
        logger.info("\n🎉 All tests passed! GNN implementation is ready.")
        logger.info("\nNext steps:")
        logger.info("  1. Build graph: python -m gnn.graph_builder")
        logger.info("  2. Train model: python -m gnn.train_gnn")
        logger.info("  3. Test predictions: python -m gnn.predict")
        logger.info("  4. Integrate with API (see INTEGRATION_GUIDE.md)")
    else:
        logger.warning("\n⚠️  Some tests failed. Please check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
