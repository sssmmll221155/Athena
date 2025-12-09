"""
Start the ATHENA Bug Prediction API server
"""
import os
import sys

# Ensure we're in the correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

if __name__ == "__main__":
    import uvicorn
    from api.main import app

    print("=" * 80)
    print("ATHENA BUG PREDICTION API")
    print("=" * 80)
    print("\nStarting server...")
    print("\nEndpoints:")
    print("  - Demo Page:        http://localhost:8000")
    print("  - API Docs:         http://localhost:8000/docs")
    print("  - Predict:          POST http://localhost:8000/predict")
    print("  - Batch Predict:    POST http://localhost:8000/predict/batch")
    print("  - Model Info:       GET  http://localhost:8000/model/info")
    print("  - Health Check:     GET  http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80)
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=False  # Set to True for development
    )
