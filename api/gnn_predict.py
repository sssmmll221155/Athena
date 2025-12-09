"""
GNN-Enhanced API Endpoints for ATHENA
Adds Graph Neural Network predictions with ensemble support.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import GNN modules
try:
    from gnn.predict import GNNPredictor, EnsemblePredictor
    GNN_AVAILABLE = True
    logger.info("GNN modules loaded successfully")
except ImportError as e:
    GNN_AVAILABLE = False
    logger.warning(f"GNN modules not available: {e}")
    logger.warning("GNN endpoints will be disabled")

# Try to import existing API components
try:
    from api.main import model_manager
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost model manager not available")

try:
    from api.rl_predict import get_rl_predictor
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    logger.warning("RL predictor not available")


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class FilePathRequest(BaseModel):
    """Request for file-based prediction"""
    file_path: str = Field(..., description="Path to file in repository")
    repository_id: Optional[int] = Field(None, description="Repository ID")


class FileIDRequest(BaseModel):
    """Request for file ID-based prediction"""
    file_id: int = Field(..., description="Database file ID", ge=1)


class GNNPredictionResponse(BaseModel):
    """GNN prediction response for a file"""
    file_id: int
    file_path: Optional[str] = None
    gnn_probability: float = Field(..., ge=0.0, le=1.0)
    is_bugprone: bool
    confidence: str
    predicted_at: str


class EnsemblePredictionResponse(BaseModel):
    """Ensemble prediction combining GNN, XGBoost, and RL"""
    file_id: int
    file_path: Optional[str] = None

    # Individual model predictions
    gnn_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    xgboost_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    rl_probability: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Ensemble result
    ensemble_probability: float = Field(..., ge=0.0, le=1.0)
    is_bugprone: bool
    confidence: str

    # Weights used
    ensemble_weights: Dict[str, float]

    # Metadata
    models_used: List[str]
    predicted_at: str


class TopFilesResponse(BaseModel):
    """Response for top bug-prone files"""
    top_files: List[Dict]
    total_files: int
    threshold: float


# ============================================================================
# Global GNN and Ensemble Predictors
# ============================================================================

gnn_predictor = None
ensemble_predictor = None


def initialize_gnn_predictor():
    """Initialize GNN predictor on startup"""
    global gnn_predictor, ensemble_predictor

    if not GNN_AVAILABLE:
        logger.warning("GNN not available - skipping initialization")
        return

    try:
        logger.info("Initializing GNN predictor...")
        gnn_predictor = GNNPredictor()
        logger.info("GNN predictor initialized successfully")

        # Initialize ensemble predictor
        logger.info("Initializing ensemble predictor...")
        ensemble_predictor = EnsemblePredictor(
            gnn_weight=0.3,
            xgboost_weight=0.4,
            rl_weight=0.3
        )
        logger.info("Ensemble predictor initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize GNN predictor: {e}", exc_info=True)
        gnn_predictor = None
        ensemble_predictor = None


# ============================================================================
# FastAPI Router for GNN Endpoints
# ============================================================================

router = APIRouter(prefix="/predict", tags=["GNN Predictions"])


@router.post("/gnn/file", response_model=GNNPredictionResponse)
async def predict_file_gnn(request: FileIDRequest):
    """
    Predict bug probability for a file using GNN.

    The GNN model analyzes file co-change patterns and dependency structure
    to predict if a file is bug-prone.

    Args:
        file_id: Database file ID

    Returns:
        GNN prediction with probability and confidence level
    """
    if gnn_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="GNN predictor not available. Train a GNN model first."
        )

    try:
        # Get prediction
        probability = gnn_predictor.predict_file(file_id=request.file_id)

        if probability is None:
            raise HTTPException(
                status_code=404,
                detail=f"File ID {request.file_id} not found in graph"
            )

        # Get file path
        file_path = None
        if request.file_id in gnn_predictor.file_features:
            file_path = gnn_predictor.file_features[request.file_id].get('path')

        # Determine prediction
        is_bugprone = probability >= 0.5

        # Confidence level
        if probability >= 0.8 or probability <= 0.2:
            confidence = "high"
        elif probability >= 0.65 or probability <= 0.35:
            confidence = "medium"
        else:
            confidence = "low"

        logger.info(f"GNN Prediction - File ID: {request.file_id}, "
                   f"Probability: {probability:.3f}, "
                   f"Bug-prone: {is_bugprone}")

        return GNNPredictionResponse(
            file_id=request.file_id,
            file_path=file_path,
            gnn_probability=probability,
            is_bugprone=is_bugprone,
            confidence=confidence,
            predicted_at=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GNN prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"GNN prediction failed: {str(e)}")


@router.post("/ensemble", response_model=EnsemblePredictionResponse)
async def predict_ensemble(request: FileIDRequest):
    """
    Make ensemble prediction combining GNN, XGBoost, and RL models.

    Ensemble weights:
    - XGBoost: 40% (code metrics and commit patterns)
    - GNN: 30% (file dependencies and co-change patterns)
    - RL: 30% (adaptive threshold optimization)

    The ensemble provides more robust predictions by combining
    complementary models that capture different aspects of code quality.

    Args:
        file_id: Database file ID

    Returns:
        Ensemble prediction with individual model probabilities
    """
    if ensemble_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Ensemble predictor not available"
        )

    try:
        # Get GNN prediction
        gnn_prob = gnn_predictor.predict_file(file_id=request.file_id) if gnn_predictor else None

        # TODO: Get XGBoost prediction for this file
        # This would require mapping file_id to commit features
        xgboost_prob = None

        # TODO: Get RL prediction for this file
        rl_prob = None

        # Make ensemble prediction
        result = ensemble_predictor.predict(
            file_id=request.file_id,
            xgboost_prob=xgboost_prob,
            rl_prob=rl_prob
        )

        if result['ensemble_prob'] is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not generate predictions for file ID {request.file_id}"
            )

        # Get file path
        file_path = None
        if gnn_predictor and request.file_id in gnn_predictor.file_features:
            file_path = gnn_predictor.file_features[request.file_id].get('path')

        # Determine prediction
        ensemble_prob = result['ensemble_prob']
        is_bugprone = ensemble_prob >= 0.5

        # Confidence level
        if ensemble_prob >= 0.8 or ensemble_prob <= 0.2:
            confidence = "high"
        elif ensemble_prob >= 0.65 or ensemble_prob <= 0.35:
            confidence = "medium"
        else:
            confidence = "low"

        # Track which models were used
        models_used = []
        if result['gnn_prob'] is not None:
            models_used.append('GNN')
        if result['xgboost_prob'] is not None:
            models_used.append('XGBoost')
        if result['rl_prob'] is not None:
            models_used.append('RL')

        logger.info(f"Ensemble Prediction - File ID: {request.file_id}, "
                   f"Ensemble: {ensemble_prob:.3f}, "
                   f"GNN: {result['gnn_prob']:.3f if result['gnn_prob'] else 'N/A'}, "
                   f"Models: {models_used}")

        return EnsemblePredictionResponse(
            file_id=request.file_id,
            file_path=file_path,
            gnn_probability=result['gnn_prob'],
            xgboost_probability=result['xgboost_prob'],
            rl_probability=result['rl_prob'],
            ensemble_probability=ensemble_prob,
            is_bugprone=is_bugprone,
            confidence=confidence,
            ensemble_weights={
                'gnn': ensemble_predictor.gnn_weight,
                'xgboost': ensemble_predictor.xgboost_weight,
                'rl': ensemble_predictor.rl_weight
            },
            models_used=models_used,
            predicted_at=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ensemble prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ensemble prediction failed: {str(e)}")


@router.get("/gnn/top-files", response_model=TopFilesResponse)
async def get_top_bugprone_files(top_k: int = 20, threshold: float = 0.5):
    """
    Get top K most bug-prone files according to GNN.

    This endpoint is useful for:
    - Code review prioritization
    - Technical debt identification
    - Testing focus areas
    - Refactoring candidates

    Args:
        top_k: Number of files to return (max 100)
        threshold: Minimum probability threshold (0-1)

    Returns:
        List of top bug-prone files with probabilities
    """
    if gnn_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="GNN predictor not available"
        )

    try:
        # Limit top_k
        top_k = min(top_k, 100)

        # Get top files
        top_files = gnn_predictor.get_top_bugprone_files(top_k=top_k)

        # Filter by threshold
        filtered_files = [
            {
                'file_path': path,
                'probability': prob,
                'file_id': file_id,
                'rank': i + 1
            }
            for i, (path, prob, file_id) in enumerate(top_files)
            if prob >= threshold
        ]

        logger.info(f"Top files request: top_k={top_k}, threshold={threshold}, "
                   f"results={len(filtered_files)}")

        return TopFilesResponse(
            top_files=filtered_files,
            total_files=len(filtered_files),
            threshold=threshold
        )

    except Exception as e:
        logger.error(f"Top files error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get top files: {str(e)}")


@router.get("/gnn/status")
async def get_gnn_status():
    """
    Get GNN model status and information.

    Returns:
        GNN availability, model metadata, and performance metrics
    """
    status = {
        'gnn_available': gnn_predictor is not None,
        'ensemble_available': ensemble_predictor is not None,
        'models_loaded': []
    }

    if gnn_predictor:
        status['models_loaded'].append('GNN')
        status['gnn_info'] = {
            'num_files': len(gnn_predictor.file_to_idx),
            'num_edges': gnn_predictor.graph_data.num_edges,
            'model_type': gnn_predictor.model_metadata.get('model_type'),
            'best_test_acc': gnn_predictor.model_metadata.get('best_test_acc')
        }

    if ensemble_predictor:
        status['ensemble_weights'] = {
            'gnn': ensemble_predictor.gnn_weight,
            'xgboost': ensemble_predictor.xgboost_weight,
            'rl': ensemble_predictor.rl_weight
        }

    return status


# ============================================================================
# Initialization Function
# ============================================================================

def get_gnn_router():
    """
    Get GNN router for FastAPI app.

    Usage in main.py:
        from api.gnn_predict import get_gnn_router, initialize_gnn_predictor

        # On startup
        initialize_gnn_predictor()

        # Include router
        app.include_router(get_gnn_router())
    """
    return router


if __name__ == "__main__":
    # Test GNN API components
    logger.info("Testing GNN API components...")

    if GNN_AVAILABLE:
        initialize_gnn_predictor()

        if gnn_predictor:
            logger.info("✓ GNN predictor initialized")

        if ensemble_predictor:
            logger.info("✓ Ensemble predictor initialized")

        logger.info("\nGNN API endpoints ready:")
        logger.info("  - POST /predict/gnn/file")
        logger.info("  - POST /predict/ensemble")
        logger.info("  - GET  /predict/gnn/top-files")
        logger.info("  - GET  /predict/gnn/status")
    else:
        logger.error("✗ GNN not available")
