"""
Test script to verify default LLM provider is used when not specified
"""
import requests
import json

API_URL = "http://localhost:8000"

print("=" * 60)
print("Testing Default LLM Provider Configuration")
print("=" * 60)

# Test 1: Request WITHOUT llm_provider (should use default from config)
print("\n✓ Test 1: Request WITHOUT llm_provider")
print("  Expected: Should use default_llm from config (OPEN-API)")

payload1 = {
    "user_id": "test_default_user",
    "prompt": "What is 1+1?",
    "role": "student"
    # Note: llm_provider is NOT specified
}

try:
    response1 = requests.post(f"{API_URL}/generate", json=payload1, timeout=30)
    if response1.status_code == 200:
        result1 = response1.json()
        print(f"  ✓ Status: {response1.status_code}")
        print(f"  ✓ Session ID: {result1.get('session_id')}")
        print(f"  ✓ Response preview: {result1.get('response')[:80]}...")
    else:
        print(f"  ✗ Failed with status: {response1.status_code}")
        print(f"  Error: {response1.text}")
except Exception as e:
    print(f"  ✗ Error: {str(e)}")

# Test 2: Request WITH llm_provider specified
print("\n✓ Test 2: Request WITH llm_provider='GEMINI'")
print("  Expected: Should use GEMINI provider")

payload2 = {
    "user_id": "test_gemini_user",
    "prompt": "What is 2+2?",
    "role": "student",
    "llm_provider": "GEMINI"
}

try:
    response2 = requests.post(f"{API_URL}/generate", json=payload2, timeout=30)
    if response2.status_code == 200:
        result2 = response2.json()
        print(f"  ✓ Status: {response2.status_code}")
        print(f"  ✓ Session ID: {result2.get('session_id')}")
        print(f"  ✓ Response preview: {result2.get('response')[:80]}...")
    else:
        print(f"  ✗ Failed with status: {response2.status_code}")
        print(f"  Error: {response2.text}")
except Exception as e:
    print(f"  ✗ Error: {str(e)}")

print("\n" + "=" * 60)
print("Check the server logs to see which provider was used!")
print("Look for lines like:")
print("  'Using default LLM provider from config: OPEN-API'")
print("  'Using user-specified LLM provider: GEMINI'")
print("=" * 60)
