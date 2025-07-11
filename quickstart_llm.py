#!/usr/bin/env python3
"""
QNTI LLM+MCP Quick Start Script
Test the LLM integration with sample queries
"""

import requests
import json
import time
import sys
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5000"
API_KEY = "qnti-secret-key"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def test_llm_status():
    """Test LLM service status"""
    print("🔍 Testing LLM service status...")
    try:
        response = requests.get(f"{BASE_URL}/llm/status", headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            print("✓ LLM service is running")
            print(f"  Model: {data.get('model', 'Unknown')}")
            print(f"  Memory: {data.get('memory_status', 'Unknown')}")
            print(f"  Scheduler: {data.get('scheduler_status', 'Unknown')}")
            return True
        else:
            print(f"✗ LLM service error: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False

def test_chat():
    """Test chat functionality"""
    print("\n💬 Testing chat functionality...")
    
    test_messages = [
        "Hello! Can you help me analyze my trading performance?",
        "What are the key market indicators I should watch today?",
        "How can I improve my risk management strategy?"
    ]
    
    for message in test_messages:
        print(f"\n📝 Query: {message}")
        try:
            response = requests.post(
                f"{BASE_URL}/llm/chat",
                headers=HEADERS,
                json={
                    "message": message,
                    "context_window": 10,
                    "user_id": "quickstart_test"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Response: {data.get('response', 'No response')[:200]}...")
            else:
                print(f"✗ Chat error: {response.status_code}")
                print(f"  Response: {response.text}")
                
        except Exception as e:
            print(f"✗ Chat request failed: {e}")

def test_daily_brief():
    """Test daily brief generation"""
    print("\n📊 Testing daily brief generation...")
    try:
        response = requests.get(f"{BASE_URL}/llm/daily-brief", headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            print("✓ Daily brief generated successfully")
            print(f"  Brief: {data.get('brief', 'No brief available')[:300]}...")
        else:
            print(f"✗ Daily brief error: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ Daily brief request failed: {e}")

def test_context_upload():
    """Test context upload functionality"""
    print("\n📤 Testing context upload...")
    
    sample_context = {
        "document_type": "analysis",
        "content": "Sample trading analysis: EURUSD showing strong bullish momentum with RSI at 65. Support at 1.0850, resistance at 1.0950. Recommend long position with tight stop loss.",
        "metadata": {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/llm/context/upload",
            headers=HEADERS,
            json=sample_context
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Context uploaded successfully")
            print(f"  Document ID: {data.get('document_id', 'Unknown')}")
        else:
            print(f"✗ Context upload error: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ Context upload failed: {e}")

def test_analysis():
    """Test trade analysis functionality"""
    print("\n🔍 Testing trade analysis...")
    
    analysis_request = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "include_news": True,
        "context_window": 15
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/llm/analyze",
            headers=HEADERS,
            json=analysis_request
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Analysis completed successfully")
            print(f"  Analysis: {data.get('analysis', 'No analysis available')[:200]}...")
        else:
            print(f"✗ Analysis error: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ Analysis request failed: {e}")

def test_news_fetch():
    """Test news fetching functionality"""
    print("\n📰 Testing news fetch...")
    
    news_request = {
        "query": "forex trading market analysis",
        "limit": 5
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/llm/news/fetch",
            headers=HEADERS,
            json=news_request
        )
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            print(f"✓ Fetched {len(articles)} news articles")
            for i, article in enumerate(articles[:3]):
                print(f"  {i+1}. {article.get('title', 'No title')[:80]}...")
        else:
            print(f"✗ News fetch error: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ News fetch failed: {e}")

def main():
    """Main test function"""
    print("🚀 QNTI LLM+MCP Quick Start Test")
    print("=" * 50)
    print(f"Testing against: {BASE_URL}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test sequence
    tests = [
        ("LLM Status", test_llm_status),
        ("Chat", test_chat),
        ("Context Upload", test_context_upload),
        ("Analysis", test_analysis),
        ("Daily Brief", test_daily_brief),
        ("News Fetch", test_news_fetch)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            test_func()
            passed += 1
            print(f"✓ {test_name} test completed")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} test failed: {e}")
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"📊 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! Your LLM integration is working correctly.")
        print("\nNext steps:")
        print("1. Configure NewsAPI key in qnti_llm_config.json")
        print("2. Start using the LLM endpoints in your trading workflow")
        print("3. Check the dashboard for LLM status at http://localhost:5000")
    else:
        print("\n⚠️  Some tests failed. Please check:")
        print("1. QNTI main system is running on port 5000")
        print("2. LLM integration is properly installed")
        print("3. Ollama service is running")
        print("4. Dependencies are installed correctly")
        print("\nRun: python setup_llm_integration.py test")

if __name__ == "__main__":
    main() 