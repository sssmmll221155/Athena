"""
Test script for ATHENA Bug Prediction API

Demonstrates all API endpoints with example requests.
"""
import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def print_json(data):
    """Pretty print JSON"""
    print(json.dumps(data, indent=2))

def test_health():
    """Test health check endpoint"""
    print_section("1. HEALTH CHECK - GET /health")

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print_json(response.json())

def test_single_prediction():
    """Test single commit prediction"""
    print_section("2. SINGLE PREDICTION - POST /predict")

    # Test Case 1: Bug fix commit
    print("Test Case 1: Bug Fix Commit")
    print("-" * 80)
    bug_fix_data = {
        "message": "fix: resolve null pointer exception in user authentication service",
        "files_changed": 3,
        "insertions": 45,
        "deletions": 12,
        "repo_stars": 5000,
        "repo_language": "Python",
        "author_email": "senior.dev@company.com"
    }

    print("Request:")
    print_json(bug_fix_data)

    response = requests.post(f"{BASE_URL}/predict", json=bug_fix_data)
    print(f"\nResponse (Status {response.status_code}):")
    result = response.json()
    print_json(result)

    print(f"\n{'='*80}")
    print(f"PREDICTION: {'[BUG FIX]' if result['is_bugfix'] else '[NORMAL]'}")
    print(f"Probability: {result['probability']:.2%}")
    print(f"Confidence: {result['confidence'].upper()}")
    print(f"{'='*80}")

    # Test Case 2: Feature addition
    print("\n\nTest Case 2: Feature Addition")
    print("-" * 80)
    feature_data = {
        "message": "feat: add new interactive dashboard with real-time charts and analytics",
        "files_changed": 8,
        "insertions": 324,
        "deletions": 18,
        "repo_stars": 3200,
        "repo_language": "Python"
    }

    print("Request:")
    print_json(feature_data)

    response = requests.post(f"{BASE_URL}/predict", json=feature_data)
    print(f"\nResponse (Status {response.status_code}):")
    result = response.json()
    print_json(result)

    print(f"\n{'='*80}")
    print(f"PREDICTION: {'[BUG FIX]' if result['is_bugfix'] else '[NORMAL]'}")
    print(f"Probability: {result['probability']:.2%}")
    print(f"Confidence: {result['confidence'].upper()}")
    print(f"{'='*80}")

def test_batch_prediction():
    """Test batch prediction"""
    print_section("3. BATCH PREDICTION - POST /predict/batch")

    batch_data = {
        "commits": [
            {
                "message": "fix: memory leak in cache manager",
                "files_changed": 2,
                "insertions": 15,
                "deletions": 8
            },
            {
                "message": "feat: implement OAuth2 authentication",
                "files_changed": 12,
                "insertions": 456,
                "deletions": 23
            },
            {
                "message": "chore: update dependencies to latest versions",
                "files_changed": 1,
                "insertions": 5,
                "deletions": 5
            },
            {
                "message": "fix: handle edge case in input validation",
                "files_changed": 1,
                "insertions": 12,
                "deletions": 3
            },
            {
                "message": "docs: update API documentation with new endpoints",
                "files_changed": 3,
                "insertions": 89,
                "deletions": 12
            }
        ]
    }

    print("Request: 5 commits")
    for i, commit in enumerate(batch_data['commits'], 1):
        print(f"  {i}. '{commit['message'][:50]}...' "
              f"(files: {commit['files_changed']}, "
              f"+{commit['insertions']}/-{commit['deletions']})")

    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
    result = response.json()

    print(f"\nResponse (Status {response.status_code}):")
    print(f"\nSummary:")
    print(f"  Total Commits: {result['total_commits']}")
    print(f"  Bugfix Count: {result['bugfix_count']}")
    print(f"  Average Probability: {result['average_probability']:.2%}")

    print("\nIndividual Predictions:")
    print("-" * 80)
    for i, pred in enumerate(result['predictions'], 1):
        status = "[BUG]" if pred['is_bugfix'] else "[ OK]"
        print(f"{i}. {status} | Prob: {pred['probability']:>5.1%} | "
              f"Conf: {pred['confidence']:>6} | {batch_data['commits'][i-1]['message'][:45]}")

def test_model_info():
    """Test model info endpoint"""
    print_section("4. MODEL INFO - GET /model/info")

    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")

    result = response.json()
    print(f"\nModel Version: {result['model_version']}")
    print(f"Training Date: {result['training_date']}")
    print(f"Feature Count: {result['feature_count']}")

    print("\nAccuracy Metrics:")
    for metric, value in result['accuracy_metrics'].items():
        print(f"  {metric:20}: {value:.3f}")

    print("\nModel Parameters:")
    for param, value in result['model_params'].items():
        print(f"  {param:20}: {value}")

    print(f"\nFeatures ({len(result['features'])}):")
    for i, feature in enumerate(result['features'], 1):
        print(f"  {i:2}. {feature}")

def run_all_tests():
    """Run all API tests"""
    print("\n" + "=" * 80)
    print("ATHENA BUG PREDICTION API - TEST SUITE")
    print("=" * 80)

    try:
        # Test 1: Health check
        test_health()

        # Test 2: Single predictions
        test_single_prediction()

        # Test 3: Batch prediction
        test_batch_prediction()

        # Test 4: Model info
        test_model_info()

        print_section("ALL TESTS COMPLETED SUCCESSFULLY")

        print("Next Steps:")
        print("  1. Open demo page: http://localhost:8000")
        print("  2. View API docs: http://localhost:8000/docs")
        print("  3. Integrate with your application using the API")

    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API server")
        print("Make sure the server is running:")
        print("  python start_api.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
