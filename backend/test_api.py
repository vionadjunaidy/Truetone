"""
Simple test script for the emotion detection API
Run this after starting the Flask server to verify it's working
"""

import requests
import json

API_URL = "http://localhost:5000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_analyze():
    """Test the analyze endpoint"""
    print("\nTesting analyze endpoint...")
    
    test_cases = [
        {
            "text": "I'm so happy today! Everything is going great!",
            "gender": "female"
        },
        {
            "text": "This is really frustrating and annoying.",
            "gender": "male"
        },
        {
            "text": "I feel calm and peaceful right now.",
            "gender": "female"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print(f"  Text: {test_case['text']}")
        print(f"  Gender: {test_case['gender']}")
        
        try:
            response = requests.post(
                f"{API_URL}/api/analyze",
                json=test_case,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"  Result: {json.dumps(result, indent=4)}")
            else:
                print(f"  Error: {response.text}")
                
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("Emotion Detection API Test")
    print("=" * 50)
    
    # Test health endpoint
    if test_health():
        print("\n✓ Health check passed!")
    else:
        print("\n✗ Health check failed! Make sure the server is running.")
        print("  Start the server with: python app.py")
        exit(1)
    
    # Test analyze endpoint
    test_analyze()
    
    print("\n" + "=" * 50)
    print("Testing complete!")
    print("=" * 50)
