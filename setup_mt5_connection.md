# Setup MT5 Connection to Windows

## Quick Setup for Real MT5 Connection

### Step 1: Get your MT5 Account Details
1. Open MetaTrader 5 on your Windows machine
2. Go to **Tools → Options → Account** 
3. Note down:
   - **Login number** (your account number)
   - **Server name** (e.g., "ICMarkets-Demo" or "ICMarkets-Live01")
   - **Password** (your MT5 password)

### Step 2: Configure QNTI Connection
Update the `mt5_config.json` file with your details:

```json
{
  "account": {
    "login": YOUR_ACCOUNT_NUMBER,
    "password": "YOUR_PASSWORD", 
    "server": "YOUR_SERVER_NAME"
  },
  "symbols": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "GOLD", "SILVER"],
  "ea_monitoring": {
    "check_interval": 30,
    "log_directory": "MQL5/Files/EA_Logs",
    "enable_file_monitoring": true
  },
  "risk_management": {
    "max_daily_loss": 1000,
    "max_open_trades": 10,
    "max_lot_size": 1.0,
    "emergency_close_drawdown": 0.20
  }
}
```

### Step 3: Install MetaTrader5 Python Package
On Windows, install the MT5 Python package:

```bash
pip install MetaTrader5
```

### Step 4: Enable Auto Trading in MT5
1. In your MT5 terminal, click the **Auto Trading** button (should be green)
2. Go to **Tools → Options → Expert Advisors**
3. Check "Allow automated trading"
4. Check "Allow DLL imports"

### Step 5: Test Connection
The QNTI system will automatically connect when you restart it with proper credentials.

## Alternative: WSL Bridge Setup

If you want to keep the system on Linux/WSL and connect to Windows MT5:

### Option A: Network Bridge
- Configure MT5 to accept network connections
- Use TCP/IP bridge between WSL and Windows

### Option B: Shared Memory
- Use Windows-WSL shared memory bridge
- Configure MT5 DLL for cross-platform access

## Example Configuration

Replace the values in `mt5_config.json`:

```json
{
  "account": {
    "login": 12345678,
    "password": "YourMT5Password",
    "server": "ICMarkets-Demo"
  }
}
```

After updating the config, restart the QNTI system and it will connect to your real MT5 account.