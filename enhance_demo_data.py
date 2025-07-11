#!/usr/bin/env python3
"""
Enhance QNTI Demo Data - Make mock data more realistic for testing
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

def generate_realistic_trades(count=50):
    """Generate realistic trading data"""
    
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "GOLD", "SILVER"]
    trade_types = ["BUY", "SELL"]
    
    trades = []
    running_balance = 10000.0
    
    for i in range(count):
        symbol = random.choice(symbols)
        trade_type = random.choice(trade_types)
        
        # Generate realistic profit/loss
        if random.random() < 0.6:  # 60% win rate
            profit = random.uniform(50, 300)
        else:
            profit = random.uniform(-200, -50)
        
        # Generate realistic timestamps
        days_ago = random.randint(1, 30)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        
        open_time = datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        close_time = open_time + timedelta(minutes=random.randint(15, 480))  # 15 min to 8 hours
        
        running_balance += profit
        
        trade = {
            "id": f"TRADE_{i+1:04d}",
            "symbol": symbol,
            "type": trade_type,
            "volume": round(random.uniform(0.01, 0.5), 2),
            "open_price": round(random.uniform(1.0, 2.0), 5),
            "close_price": round(random.uniform(1.0, 2.0), 5),
            "profit": round(profit, 2),
            "swap": round(random.uniform(-5, 5), 2),
            "commission": round(random.uniform(-2, 0), 2),
            "open_time": open_time.isoformat(),
            "close_time": close_time.isoformat(),
            "magic_number": random.choice([12345, 67890, 11111, 22222]),
            "comment": f"EA Trade {i+1}",
            "status": "closed",
            "running_balance": round(running_balance, 2)
        }
        
        trades.append(trade)
    
    return trades

def generate_realistic_market_data():
    """Generate realistic market data"""
    
    symbols = {
        "EURUSD": {"bid": 1.0845, "ask": 1.0847, "spread": 0.0002},
        "GBPUSD": {"bid": 1.2634, "ask": 1.2636, "spread": 0.0002},
        "USDJPY": {"bid": 149.23, "ask": 149.25, "spread": 0.02},
        "USDCHF": {"bid": 0.8756, "ask": 0.8758, "spread": 0.0002},
        "AUDUSD": {"bid": 0.6587, "ask": 0.6589, "spread": 0.0002},
        "USDCAD": {"bid": 1.3542, "ask": 1.3544, "spread": 0.0002},
        "GOLD": {"bid": 2045.50, "ask": 2045.80, "spread": 0.30},
        "SILVER": {"bid": 24.85, "ask": 24.87, "spread": 0.02}
    }
    
    # Add small random variations
    for symbol, data in symbols.items():
        variation = random.uniform(-0.001, 0.001)
        data["bid"] += variation
        data["ask"] += variation
        data["last"] = (data["bid"] + data["ask"]) / 2
        data["change"] = round(random.uniform(-0.005, 0.005), 5)
        data["change_percent"] = round(random.uniform(-0.5, 0.5), 2)
        data["time"] = datetime.now().isoformat()
    
    return symbols

def generate_realistic_account_data():
    """Generate realistic account data"""
    
    balance = 10000.0
    equity = balance + random.uniform(-200, 200)
    margin = random.uniform(0, 500)
    free_margin = equity - margin
    
    return {
        "login": 12345678,
        "server": "Demo-Server",
        "name": "QNTI Demo Account",
        "company": "Demo Broker",
        "currency": "USD",
        "leverage": 100,
        "balance": round(balance, 2),
        "equity": round(equity, 2),
        "margin": round(margin, 2),
        "free_margin": round(free_margin, 2),
        "margin_level": round((equity / margin * 100) if margin > 0 else 1000, 2),
        "profit": round(equity - balance, 2),
        "credit": 0.0,
        "account_number": 12345678
    }

def update_demo_data():
    """Update the system with enhanced demo data"""
    
    print("🔄 Generating enhanced demo data...")
    
    # Generate realistic trades
    trades = generate_realistic_trades(50)
    
    # Generate market data
    market_data = generate_realistic_market_data()
    
    # Generate account data
    account_data = generate_realistic_account_data()
    
    # Save to files
    demo_data_dir = Path("demo_data")
    demo_data_dir.mkdir(exist_ok=True)
    
    with open(demo_data_dir / "trades.json", "w") as f:
        json.dump(trades, f, indent=2)
    
    with open(demo_data_dir / "market_data.json", "w") as f:
        json.dump(market_data, f, indent=2)
    
    with open(demo_data_dir / "account_data.json", "w") as f:
        json.dump(account_data, f, indent=2)
    
    print(f"✅ Generated {len(trades)} realistic trades")
    print(f"✅ Generated market data for {len(market_data)} symbols")
    print(f"✅ Generated account data (Balance: ${account_data['balance']}, Equity: ${account_data['equity']})")
    print(f"✅ Demo data saved to {demo_data_dir}")
    
    # Update the system to use this data
    print("\n🔧 To use this enhanced demo data:")
    print("1. Restart the QNTI system")
    print("2. The system will automatically load the demo data")
    print("3. You'll see realistic equity curves and trading history")
    
    return {
        "trades": trades,
        "market_data": market_data,
        "account_data": account_data
    }

if __name__ == "__main__":
    data = update_demo_data()
    
    # Print summary
    print("\n📊 Demo Data Summary:")
    print(f"   Total Trades: {len(data['trades'])}")
    print(f"   Symbols: {list(data['market_data'].keys())}")
    print(f"   Account Balance: ${data['account_data']['balance']}")
    print(f"   Account Equity: ${data['account_data']['equity']}")
    
    total_profit = sum(trade['profit'] for trade in data['trades'])
    winning_trades = len([t for t in data['trades'] if t['profit'] > 0])
    win_rate = (winning_trades / len(data['trades'])) * 100
    
    print(f"   Total Profit: ${total_profit:.2f}")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Winning Trades: {winning_trades}/{len(data['trades'])}")
    
    print("\n🎯 Demo data is ready! Restart the QNTI system to see realistic trading data.")