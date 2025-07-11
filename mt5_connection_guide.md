# MT5 Connection Guide for QNTI System

## Current Status
The QNTI system is running in **MOCK MODE** because MT5 is not connected. Here are your options:

## Option 1: Windows Environment (Recommended)
### Requirements:
- Windows 10/11 (MetaTrader5 Python module only works on Windows)
- MetaTrader 5 Terminal installed
- Active MT5 trading account

### Steps:
1. **Install MT5 Terminal**: Download from MetaQuotes website
2. **Install Python MT5 module**:
   ```bash
   pip install MetaTrader5
   ```
3. **Configure MT5 connection** in `mt5_config.json`:
   ```json
   {
     "account": 12345678,
     "password": "your_password",
     "server": "YourBroker-Demo",
     "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
     "timeout": 60000,
     "retry_count": 3
   }
   ```

## Option 2: Demo Mode with Simulated Data
### Current Implementation:
- Mock MT5 data for testing
- Simulated trades and account info
- Demo equity curve generation

### To Enable Real-like Demo Data:
1. **Update mock data sources** in `qnti_mt5_integration.py`
2. **Connect to financial data APIs** (Alpha Vantage, Yahoo Finance)
3. **Import historical trading data** from CSV files

## Option 3: Paper Trading Mode
### Features:
- Real market data from APIs
- Simulated trading without real money
- Full system testing capabilities

### Implementation:
1. **Connect to market data APIs**
2. **Enable paper trading mode**
3. **Track simulated positions**

## Quick Setup for Demo Mode

### 1. Enable Enhanced Mock Mode
Update the system to use more realistic demo data:

```python
# In qnti_mt5_integration.py
DEMO_MODE = True
ACCOUNT_BALANCE = 10000.0
ACCOUNT_EQUITY = 10000.0
```

### 2. Import Sample Trading Data
Load historical trades to populate the equity curve:

```python
# Sample data format
trades_data = [
    {"symbol": "EURUSD", "profit": 150.0, "open_time": "2024-01-01T10:00:00"},
    {"symbol": "GBPUSD", "profit": -75.0, "open_time": "2024-01-01T14:00:00"},
    # ... more trades
]
```

### 3. Connect to Live Market Data
Use free APIs for real-time prices:
- Alpha Vantage (free tier)
- Yahoo Finance
- FXCM API

## Current System Capabilities (Mock Mode)
✅ **Working Features:**
- Dashboard visualization
- EA management
- Advanced analytics
- Risk management
- Web interface

❌ **Missing (MT5 Required):**
- Real-time price feeds
- Actual trade execution
- Live account data
- Real equity curve

## Next Steps
1. **Choose your preferred connection method**
2. **Configure MT5 settings**
3. **Test connection**
4. **Enable real data feeds**

## Demo Data Enhancement
I can help you enhance the mock data to be more realistic for testing purposes while you set up the MT5 connection.