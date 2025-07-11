#!/usr/bin/env python3
"""Test script to verify analytics endpoints are working"""

import requests
import json
import time

def test_analytics_endpoints():
    print("🔬 Testing QNTI Advanced Analytics Endpoints")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    endpoints = [
        "/api/analytics/performance-summary",
        "/api/analytics/advanced-metrics", 
        "/api/analytics/risk-profile",
        "/api/analytics/market-insights",
        "/api/analytics/ai-recommendations",
        "/api/analytics/comprehensive"
    ]
    
    results = []
    
    for endpoint in endpoints:
        try:
            print(f"\n📡 Testing {endpoint}...")
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ SUCCESS - Status: {response.status_code}")
                print(f"   Response size: {len(response.text)} chars")
                
                # Show some sample data
                if isinstance(data, dict):
                    if 'metrics' in data:
                        print(f"   Metrics count: {len(data['metrics'])}")
                    elif 'insights' in data:
                        print(f"   Insights count: {len(data['insights'])}")
                    elif 'recommendations' in data:
                        print(f"   Recommendations count: {len(data['recommendations'])}")
                    else:
                        print(f"   Data keys: {list(data.keys())[:5]}")
                
                results.append((endpoint, True, response.status_code))
            else:
                print(f"❌ FAILED - Status: {response.status_code}")
                print(f"   Error: {response.text[:200]}")
                results.append((endpoint, False, response.status_code))
                
        except requests.exceptions.ConnectionError:
            print(f"❌ CONNECTION ERROR - Server not running?")
            results.append((endpoint, False, "Connection Error"))
        except Exception as e:
            print(f"❌ ERROR - {e}")
            results.append((endpoint, False, str(e)))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for endpoint, success, status in results:
        status_text = "✅ PASS" if success else "❌ FAIL"
        print(f"  {endpoint}: {status_text} ({status})")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(endpoints)} endpoints working")
    
    if passed == len(endpoints):
        print("🎉 All analytics endpoints are working correctly!")
    else:
        print(f"⚠️  {len(endpoints) - passed} endpoints failed")
    
    return passed == len(endpoints)

if __name__ == "__main__":
    success = test_analytics_endpoints()
    exit(0 if success else 1)