#!/usr/bin/env python3

import MetaTrader5 as mt5
import json

def check_available_symbols():
    """Check what symbols are available on the MT5 server"""
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    print("🔍 Checking available symbols on your MT5 server...")
    print("=" * 60)
    
    # Get all symbols
    all_symbols = mt5.symbols_get()
    
    if all_symbols is None:
        print("❌ Failed to get symbols from MT5")
        mt5.shutdown()
        return
    
    # Filter for common trading symbols
    forex_pairs = []
    metals = []
    indices = []
    crypto = []
    
    for symbol in all_symbols:
        symbol_name = symbol.name
        
        # Categorize symbols
        if any(pair in symbol_name for pair in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']):
            if len(symbol_name) == 6 and symbol_name.isalpha():
                forex_pairs.append(symbol_name)
        elif any(metal in symbol_name for metal in ['XAU', 'XAG', 'GOLD', 'SILVER']):
            metals.append(symbol_name)
        elif any(index in symbol_name for index in ['US30', 'SPX', 'NAS', 'DAX', 'FTSE']):
            indices.append(symbol_name)
        elif any(crypto in symbol_name for crypto in ['BTC', 'ETH', 'LTC']):
            crypto.append(symbol_name)
    
    # Display results
    print(f"📊 **FOREX PAIRS** ({len(forex_pairs)} available):")
    for pair in sorted(forex_pairs)[:20]:  # Show first 20
        print(f"   ✅ {pair}")
    if len(forex_pairs) > 20:
        print(f"   ... and {len(forex_pairs) - 20} more")
    
    print(f"\n🥇 **METALS** ({len(metals)} available):")
    for metal in sorted(metals):
        print(f"   ✅ {metal}")
    
    print(f"\n📈 **INDICES** ({len(indices)} available):")
    for index in sorted(indices)[:10]:  # Show first 10
        print(f"   ✅ {index}")
    if len(indices) > 10:
        print(f"   ... and {len(indices) - 10} more")
    
    print(f"\n₿ **CRYPTO** ({len(crypto)} available):")
    for coin in sorted(crypto):
        print(f"   ✅ {coin}")
    
    # Suggested configuration
    print("\n" + "=" * 60)
    print("🔧 **SUGGESTED SYMBOL CONFIGURATION:**")
    
    # Pick the most common forex pairs that are available
    suggested_forex = []
    common_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURGBP']
    for pair in common_pairs:
        if pair in forex_pairs:
            suggested_forex.append(pair)
    
    # Add metals if available
    suggested_metals = []
    if metals:
        suggested_metals = metals[:2]  # Add up to 2 metals
    
    all_suggested = suggested_forex[:6] + suggested_metals  # Max 8 symbols total
    
    print(f"\"symbols\": {json.dumps(all_suggested[:8])}")
    
    print(f"\n📋 **NEXT STEPS:**")
    print(f"1. Copy the suggested symbols above")
    print(f"2. Update your mt5_config.json file")
    print(f"3. Restart the QNTI system")
    print(f"4. You'll see more market data in your dashboard!")
    
    # Cleanup
    mt5.shutdown()

if __name__ == "__main__":
    check_available_symbols() 