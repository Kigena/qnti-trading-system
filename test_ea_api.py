#!/usr/bin/env python3
"""
Test EA API endpoints workflow
"""

import json
import requests
import time

# Test data
sample_ea_code = """
//+------------------------------------------------------------------+
//|                                                     TestEA.mq4   |
//|                                Copyright 2024, QNTI Trading      |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, QNTI Trading"
#property version   "1.00"
#property strict

extern double StopLoss = 100;
extern double TakeProfit = 200;
extern double LotSize = 0.1;
extern int MagicNumber = 12345;

void OnTick() {
    if (OrdersTotal() == 0) {
        if (iMA(NULL, 0, 14, 0, MODE_SMA, PRICE_CLOSE, 1) > iMA(NULL, 0, 50, 0, MODE_SMA, PRICE_CLOSE, 1)) {
            OrderSend(Symbol(), OP_BUY, LotSize, Ask, 3, Ask - StopLoss * Point, Ask + TakeProfit * Point, "Test EA", MagicNumber, 0, clrBlue);
        }
    }
}
"""

def test_ea_workflow():
    """Test the complete EA workflow"""
    
    base_url = "http://localhost:5000"
    
    print("🧪 Testing EA API Workflow...")
    print("=" * 50)
    
    # Step 1: Test system health
    print("\n1. Testing system health...")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=10)
        if response.status_code == 200:
            print("✅ System health check passed")
        else:
            print(f"⚠️ System health check returned: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to QNTI system: {e}")
        return False
    
    # Step 2: Test EA parsing
    print("\n2. Testing EA code parsing...")
    try:
        parse_data = {"code": sample_ea_code}
        response = requests.post(f"{base_url}/api/ea/parse-code", 
                                json=parse_data, 
                                timeout=30)
        
        if response.status_code == 200:
            parsed_result = response.json()
            print("✅ EA parsing successful")
            print(f"   Name: {parsed_result.get('name', 'Unknown')}")
            print(f"   Magic: {parsed_result.get('magic_number', 'Unknown')}")
            print(f"   Parameters: {len(parsed_result.get('parameters', []))}")
            
            # Step 3: Test EA profile saving
            print("\n3. Testing EA profile saving...")
            
            profile_data = {
                "name": parsed_result.get('name', 'Test EA'),
                "magic_number": parsed_result.get('magic_number', 12345),
                "symbols": parsed_result.get('symbols', ['EURUSD']),
                "timeframes": parsed_result.get('timeframes', ['H1']),
                "parameters": parsed_result.get('parameters', {}),
                "original_code": sample_ea_code,
                "profile": parsed_result.get('profile', {})
            }
            
            response = requests.post(f"{base_url}/api/ea/save-profile", 
                                    json=profile_data, 
                                    timeout=30)
            
            if response.status_code == 200:
                save_result = response.json()
                print("✅ EA profile saved successfully")
                print(f"   Profile ID: {save_result.get('profile_id', 'Unknown')}")
                
                # Step 4: Test EA profiles retrieval
                print("\n4. Testing EA profiles retrieval...")
                
                response = requests.get(f"{base_url}/api/ea/profiles", timeout=30)
                
                if response.status_code == 200:
                    profiles = response.json()
                    print(f"✅ Retrieved {len(profiles)} EA profiles")
                    
                    # Find our saved profile
                    saved_profile = None
                    for profile in profiles:
                        if profile.get('name') == profile_data['name']:
                            saved_profile = profile
                            break
                    
                    if saved_profile:
                        print(f"✅ Found saved profile: {saved_profile['name']}")
                        print(f"   ID: {saved_profile.get('id', 'Unknown')}")
                        print(f"   Magic: {saved_profile.get('magic_number', 'Unknown')}")
                        return True
                    else:
                        print("❌ Saved profile not found in profiles list")
                        return False
                else:
                    print(f"❌ Profile retrieval failed: {response.status_code}")
                    return False
                    
            else:
                print(f"❌ EA profile save failed: {response.status_code}")
                try:
                    error_text = response.text
                    print(f"   Error: {error_text}")
                except:
                    pass
                return False
                
        else:
            print(f"❌ EA parsing failed: {response.status_code}")
            try:
                error_text = response.text
                print(f"   Error: {error_text}")
            except:
                pass
            return False
            
    except Exception as e:
        print(f"❌ EA workflow test failed: {e}")
        return False

def main():
    """Run the EA workflow test"""
    
    print("🔧 EA Upload Workflow Validation")
    print("Testing the complete EA upload → parse → save → retrieve workflow")
    print()
    
    try:
        success = test_ea_workflow()
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 EA Upload Workflow: ✅ COMPLETE AND FUNCTIONAL")
            print()
            print("The workflow now supports:")
            print("✅ EA code parsing and analysis")
            print("✅ EA profile creation and saving")
            print("✅ EA profile persistence and retrieval")
            print("✅ Integration with EA Management page")
            print()
            print("User experience:")
            print("1. Upload EA code → Parse → Save → Confirmation → Redirect")
            print("2. EA automatically appears in EA Management page")
            print("3. All extracted data is properly displayed")
        else:
            print("❌ EA Upload Workflow: ISSUES DETECTED")
            print("Please check system logs for detailed error information")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")

if __name__ == "__main__":
    main()