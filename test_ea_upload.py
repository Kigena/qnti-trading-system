#!/usr/bin/env python3
"""Test script to verify EA upload and parsing functionality"""

import requests
import json
import time

def test_ea_parsing():
    print("Testing EA parsing functionality...")
    
    # Read the test EA file
    with open('test_ea_sample.mq4', 'r') as f:
        ea_code = f.read()
    
    # Prepare the request
    url = "http://localhost:5000/api/ea/parse"
    headers = {"Content-Type": "application/json"}
    data = {
        "ea_code": ea_code,
        "filename": "TestEASample.mq4"
    }
    
    try:
        # Make the request
        response = requests.post(url, json=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ EA parsing successful!")
                profile = result.get('profile', {})
                print(f"EA Name: {profile.get('name', 'Unknown')}")
                print(f"Parameters: {len(profile.get('parameters', []))}")
                print(f"Trading Rules: {len(profile.get('trading_rules', []))}")
                print(f"Magic Number: {profile.get('magic_number', 'Not found')}")
                return True
            else:
                print(f"❌ EA parsing failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - is the system running?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ea_save():
    print("\nTesting EA profile save functionality...")
    
    # Read the test EA file
    with open('test_ea_sample.mq4', 'r') as f:
        ea_code = f.read()
    
    # Prepare the request for saving
    url = "http://localhost:5000/api/ea/save-profile"
    headers = {"Content-Type": "application/json"}
    data = {
        "name": "QNTI Test EA",
        "description": "Test EA for QNTI system validation",
        "symbols": ["EURUSD", "GBPUSD"],
        "timeframes": ["H1", "H4"],
        "magic_number": 123456,
        "original_code": ea_code,
        "profile": {
            "parameters": [
                {
                    "name": "LotSize",
                    "type": "double",
                    "default_value": "0.01",
                    "description": "Lot size for trading"
                },
                {
                    "name": "StopLoss",
                    "type": "int",
                    "default_value": "50",
                    "description": "Stop loss in pips"
                },
                {
                    "name": "TakeProfit",
                    "type": "int",
                    "default_value": "100",
                    "description": "Take profit in pips"
                }
            ],
            "trading_rules": [
                {
                    "type": "entry",
                    "direction": "buy",
                    "conditions": ["MA crossover bullish"],
                    "actions": ["Open buy position"]
                },
                {
                    "type": "entry",
                    "direction": "sell",
                    "conditions": ["MA crossover bearish"],
                    "actions": ["Open sell position"]
                }
            ],
            "indicators": ["Moving Average"],
            "execution_status": "parsed"
        }
    }
    
    try:
        response = requests.post(url, json=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ EA profile save successful!")
                return True
            else:
                print(f"❌ EA profile save failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ea_management():
    print("\nTesting EA management functionality...")
    
    # Test getting EA list
    url = "http://localhost:5000/api/eas"
    
    try:
        response = requests.get(url, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            eas = response.json()
            print(f"✅ Retrieved {len(eas)} EAs from system")
            
            # Display first few EAs
            for i, ea in enumerate(eas[:3]):
                print(f"  EA {i+1}: {ea.get('name', 'Unknown')} - Magic: {ea.get('magic_number', 'N/A')}")
            
            return True
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🚀 QNTI EA Upload & Management Test Suite")
    print("="*50)
    
    # Wait a moment for system to be ready
    time.sleep(2)
    
    # Run tests
    tests = [
        ("EA Parsing", test_ea_parsing),
        ("EA Profile Save", test_ea_save),
        ("EA Management", test_ea_management)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! EA upload and management system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the system configuration.")

if __name__ == "__main__":
    main()