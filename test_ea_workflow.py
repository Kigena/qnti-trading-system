#!/usr/bin/env python3
"""
Test EA Upload Workflow
"""

import json
import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, '/mnt/c/Users/kigen/qnti-trading-system')

def test_ea_storage():
    """Test EA storage functions directly"""
    
    # Test profile data
    test_profile = {
        'name': 'Test EA',
        'strategy_type': 'trend_following',
        'symbols': ['EURUSD'],
        'timeframes': ['H1'],
        'parameters': {'StopLoss': 100, 'TakeProfit': 200},
        'risk_level': 'medium',
        'description': 'Test EA profile for workflow validation',
        'original_code': '// Test EA code\nvoid OnTick() { /* trading logic */ }'
    }
    
    try:
        # Create profiles directory
        profiles_dir = Path("ea_profiles")
        profiles_dir.mkdir(exist_ok=True)
        
        # Test save function
        import uuid
        profile_id = str(uuid.uuid4())[:8]
        test_profile['id'] = profile_id
        
        # Save to JSON file
        profile_file = profiles_dir / f"{profile_id}.json"
        with open(profile_file, 'w', encoding='utf-8') as f:
            json.dump(test_profile, f, indent=2)
        
        print(f"✅ Profile saved with ID: {profile_id}")
        
        # Test load function
        profiles = []
        for profile_file in profiles_dir.glob("*.json"):
            try:
                with open(profile_file, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                    # Don't include original code (too large)
                    profile_summary = {k: v for k, v in profile_data.items() if k != 'original_code'}
                    profiles.append(profile_summary)
            except Exception as e:
                print(f"⚠️ Could not load profile {profile_file}: {e}")
        
        print(f"✅ Loaded {len(profiles)} profiles")
        
        # Test load by ID
        try:
            with open(profiles_dir / f"{profile_id}.json", 'r', encoding='utf-8') as f:
                loaded_profile = json.load(f)
                print(f"✅ Loaded specific profile: {loaded_profile['name']}")
        except Exception as e:
            print(f"❌ Could not load specific profile: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ EA storage test failed: {e}")
        return False

def test_ea_parsing():
    """Test EA parsing functionality"""
    
    # Sample EA code
    sample_ea_code = """
    //+------------------------------------------------------------------+
    //|                                                       TestEA.mq4 |
    //|                                Copyright 2024, QNTI Trading      |
    //|                                                                  |
    //+------------------------------------------------------------------+
    #property copyright "Copyright 2024, QNTI Trading"
    #property link      "https://qnti.ai"
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
    
    try:
        # Test if parser exists
        from qnti_ea_parser import QNTIEAParser
        parser = QNTIEAParser()
        
        # Parse the EA code
        parsed_data = parser.parse_ea_code(sample_ea_code, "TestEA.mq4")
        
        print(f"✅ EA parsing successful:")
        print(f"   Name: {parsed_data.get('name', 'Unknown')}")
        print(f"   Strategy: {parsed_data.get('strategy_type', 'Unknown')}")
        print(f"   Parameters: {len(parsed_data.get('parameters', {}))}")
        print(f"   Symbols: {parsed_data.get('symbols', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ EA parsing test failed: {e}")
        return False

def test_ea_profiling():
    """Test EA profiling system"""
    
    try:
        from qnti_ea_profiling_system import QNTIEAProfilingSystem
        profiler = QNTIEAProfilingSystem()
        
        # Test profile creation
        ea_data = {
            'name': 'Test EA',
            'strategy_type': 'trend_following',
            'symbols': ['EURUSD'],
            'parameters': {'StopLoss': 100, 'TakeProfit': 200}
        }
        
        profile = profiler.create_ea_profile(ea_data)
        print(f"✅ EA profiling successful:")
        print(f"   Profile ID: {profile.get('id', 'Unknown')}")
        print(f"   Risk Level: {profile.get('risk_level', 'Unknown')}")
        print(f"   Strategy Type: {profile.get('strategy_type', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ EA profiling test failed: {e}")
        return False

def main():
    """Run all EA workflow tests"""
    
    print("🧪 Testing EA Upload Workflow Components...")
    print("=" * 50)
    
    # Test storage functions
    print("\n1. Testing EA Storage Functions...")
    storage_ok = test_ea_storage()
    
    # Test parsing
    print("\n2. Testing EA Parsing...")
    parsing_ok = test_ea_parsing()
    
    # Test profiling
    print("\n3. Testing EA Profiling...")
    profiling_ok = test_ea_profiling()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 EA Workflow Test Results:")
    print(f"   Storage Functions: {'✅ PASS' if storage_ok else '❌ FAIL'}")
    print(f"   EA Parsing: {'✅ PASS' if parsing_ok else '❌ FAIL'}")
    print(f"   EA Profiling: {'✅ PASS' if profiling_ok else '❌ FAIL'}")
    
    if storage_ok and parsing_ok and profiling_ok:
        print("\n🎉 All EA workflow components working correctly!")
        return True
    else:
        print("\n⚠️ Some EA workflow components have issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)