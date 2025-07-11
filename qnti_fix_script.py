#!/usr/bin/env python3
"""
QNTI Quick Fix Script
Applies all the necessary fixes to resolve the identified issues
Run this script to automatically fix configuration and encoding issues
"""

import json
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

def backup_files():
    """Create backup of existing files"""
    backup_dir = Path("qnti_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = [
        "qnti_main_system.py",
        "qnti_mt5_integration.py", 
        "qnti_dashboard.html",
        "qnti_config.json",
        "mt5_config.json",
        "vision_config.json"
    ]
    
    for file_name in files_to_backup:
        if Path(file_name).exists():
            shutil.copy2(file_name, backup_dir / file_name)
            print(f"✓ Backed up {file_name}")
    
    print(f"✓ Backup created in: {backup_dir}")
    return backup_dir

def create_fixed_config_files():
    """Create properly structured configuration files"""
    
    # Main QNTI Configuration
    qnti_config = {
        "system": {
            "auto_trading": False,
            "vision_auto_analysis": True,
            "ea_monitoring": True,
            "api_port": 5000,
            "debug_mode": True,
            "max_concurrent_trades": 10,
            "risk_management": {
                "max_daily_loss": 1000,
                "max_drawdown": 0.20,
                "position_size_limit": 1.0,
                "emergency_close_drawdown": 0.20
            }
        },
        "integration": {
            "mt5_enabled": True,
            "vision_enabled": True,
            "dashboard_enabled": True,
            "webhook_enabled": False,
            "telegram_notifications": False
        },
        "ea_monitoring": {
            "check_interval": 30,
            "log_directory": "MQL5/Files/EA_Logs",
            "enable_file_monitoring": True
        },
        "scheduling": {
            "vision_analysis_interval": 300,
            "health_check_interval": 60,
            "performance_update_interval": 30,
            "backup_interval": 3600
        },
        "alerts": {
            "email_alerts": False,
            "telegram_alerts": False,
            "webhook_alerts": False,
            "log_alerts": True
        },
        "vision": {
            "primary_symbols": ["EURUSD", "GBPUSD", "USDJPY"],
            "timeframes": ["H1", "H4"]
        }
    }
    
    # MT5 Configuration
    mt5_config = {
        "account": {
            "login": 0,
            "password": "",
            "server": "",
            "timeout": 5000
        },
        "symbols": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"],
        "ea_monitoring": {
            "check_interval": 30,
            "log_directory": "MQL5/Files/EA_Logs",
            "enable_file_monitoring": True
        },
        "risk_management": {
            "max_daily_loss": 1000,
            "max_open_trades": 10,
            "max_lot_size": 1.0,
            "emergency_close_drawdown": 0.20
        }
    }
    
    # Vision Configuration
    vision_config = {
        "vision": {
            "openai_api_key": "YOUR_OPENAI_API_KEY_HERE",
            "model_name": "gpt-4-vision-preview",
            "max_tokens": 1500,
            "temperature": 0.1,
            "analysis_prompt_template": "",
            "enable_batch_analysis": True,
            "batch_size": 3,
            "rate_limit_delay": 2.0
        },
        "screenshot": {
            "window_title": "TradingView",
            "chart_region": None,
            "capture_interval": 30,
            "save_screenshots": True,
            "screenshot_directory": "qnti_screenshots",
            "max_screenshots": 1000
        },
        "symbols": {
            "primary": ["EURUSD", "GBPUSD", "USDJPY"],
            "secondary": ["USDCHF", "AUDUSD", "USDCAD"],
            "timeframes": ["M15", "H1", "H4", "D1"]
        },
        "trading_rules": {
            "min_confidence": 0.7,
            "max_risk_per_trade": 0.02,
            "min_risk_reward": 1.5,
            "enable_auto_trading": False,
            "require_manual_approval": True
        }
    }
    
    # Write configuration files
    with open("qnti_config.json", "w", encoding="utf-8") as f:
        json.dump(qnti_config, f, indent=2, ensure_ascii=False)
    print("✓ Created qnti_config.json")
    
    with open("mt5_config.json", "w", encoding="utf-8") as f:
        json.dump(mt5_config, f, indent=2, ensure_ascii=False)
    print("✓ Created mt5_config.json")
    
    with open("vision_config.json", "w", encoding="utf-8") as f:
        json.dump(vision_config, f, indent=2, ensure_ascii=False)
    print("✓ Created vision_config.json")

def fix_import_issues():
    """Fix common import and path issues"""
    
    # Create __init__.py files if needed
    init_content = '# QNTI Module\n'
    
    if not Path("__init__.py").exists():
        with open("__init__.py", "w", encoding="utf-8") as f:
            f.write(init_content)
        print("✓ Created __init__.py")

def create_startup_script():
    """Create a convenient startup script"""
    
    if os.name == 'nt':  # Windows
        startup_script = """@echo off
echo Starting Quantum Nexus Trading Intelligence...
echo.

REM Set UTF-8 encoding
chcp 65001 >nul 2>&1

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python and add it to PATH.
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "qnti_main_system.py" (
    echo ERROR: qnti_main_system.py not found.
    echo Please ensure all QNTI files are in the current directory.
    pause
    exit /b 1
)

echo Starting QNTI in safe mode (no auto-trading)...
echo Press Ctrl+C to stop the system
echo.

python qnti_main_system.py --no-auto-trading --debug

echo.
echo QNTI has stopped.
pause
"""
        with open("start_qnti.bat", "w", encoding="utf-8") as f:
            f.write(startup_script)
        print("✓ Created start_qnti.bat")
        
    else:  # Linux/Mac
        startup_script = """#!/bin/bash
echo "Starting Quantum Nexus Trading Intelligence..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Please install Python3."
    exit 1
fi

# Check if required files exist
if [ ! -f "qnti_main_system.py" ]; then
    echo "ERROR: qnti_main_system.py not found."
    echo "Please ensure all QNTI files are in the current directory."
    exit 1
fi

echo "Starting QNTI in safe mode (no auto-trading)..."
echo "Press Ctrl+C to stop the system"
echo

python3 qnti_main_system.py --no-auto-trading --debug

echo
echo "QNTI has stopped."
"""
        with open("start_qnti.sh", "w", encoding="utf-8") as f:
            f.write(startup_script)
        os.chmod("start_qnti.sh", 0o755)
        print("✓ Created start_qnti.sh")

def create_requirements_file():
    """Create comprehensive requirements.txt"""
    
    requirements = """# QNTI - Quantum Nexus Trading Intelligence Requirements
# Core trading and analysis
MetaTrader5>=5.0.45
pandas>=2.0.0
numpy>=1.24.0

# Computer vision and AI
opencv-python>=4.8.0
pyautogui>=0.9.54
pygetwindow>=0.0.9
Pillow>=10.0.0
openai>=1.0.0

# Web framework and APIs
flask>=2.3.0
flask-socketio>=5.3.0
flask-cors>=4.0.0
requests>=2.31.0

# Utilities and scheduling
schedule>=1.2.0
psutil>=5.9.0
python-socketio>=5.8.0
python-engineio>=4.7.0

# Data processing
scipy>=1.10.0
scikit-learn>=1.3.0

# Additional Windows support (install only on Windows)
pywin32>=306; sys_platform == "win32"

# Additional Linux support (install only on Linux)
python-xlib>=0.33; sys_platform == "linux"
"""
    
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements.strip())
    print("✓ Created requirements.txt")

def create_directory_structure():
    """Create necessary directories"""
    
    directories = [
        "qnti_data",
        "qnti_screenshots", 
        "qnti_backups",
        "logs",
        "MQL5/Files/EA_Logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_readme():
    """Create a comprehensive README file"""
    
    readme_content = """# Quantum Nexus Trading Intelligence (QNTI)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `vision_config.json` and add your OpenAI API key:
```json
{
  "vision": {
    "openai_api_key": "YOUR_OPENAI_API_KEY_HERE"
  }
}
```

### 3. Configure MT5 (Optional)
Edit `mt5_config.json` with your MetaTrader 5 account details:
```json
{
  "account": {
    "login": your_account_number,
    "password": "your_password",
    "server": "your_broker_server"
  }
}
```

### 4. Start the System

#### Windows:
```bash
start_qnti.bat
```

#### Linux/Mac:
```bash
./start_qnti.sh
```

#### Manual start:
```bash
python qnti_main_system.py --no-auto-trading --debug
```

### 5. Access Dashboard
Open your browser to: http://localhost:5000

## System Components

- **qnti_main_system.py** - Main orchestration system
- **qnti_core_system.py** - Core trade management
- **qnti_mt5_integration.py** - MetaTrader 5 integration
- **qnti_vision_analysis.py** - Chart analysis with AI
- **qnti_dashboard.html** - Web dashboard interface

## Configuration Files

- **qnti_config.json** - Main system configuration
- **mt5_config.json** - MetaTrader 5 settings
- **vision_config.json** - AI vision analysis settings

## Safety Features

- System starts in safe mode (no auto-trading) by default
- Emergency stop functionality
- Comprehensive risk management
- Full audit logging

## Troubleshooting

1. **MT5 Connection Issues**: Ensure MetaTrader 5 is installed and running
2. **Vision Analysis Errors**: Check OpenAI API key and credits
3. **Port Conflicts**: Change port in config if 5000 is in use
4. **Import Errors**: Ensure all files are in the same directory

## Support

Check the logs in the `logs/` directory for detailed error information.
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("✓ Created README.md")

def check_file_encodings():
    """Check and fix file encodings"""
    
    python_files = ["qnti_main_system.py", "qnti_mt5_integration.py", 
                   "qnti_core_system.py", "qnti_vision_analysis.py"]
    
    for file_name in python_files:
        if Path(file_name).exists():
            try:
                # Try to read and re-save with UTF-8 encoding
                with open(file_name, "r", encoding="utf-8") as f:
                    content = f.read()
                
                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(content)
                
                print(f"✓ Fixed encoding for {file_name}")
                
            except UnicodeDecodeError:
                print(f"⚠ Warning: {file_name} has encoding issues - manual fix required")
            except Exception as e:
                print(f"⚠ Warning: Could not process {file_name}: {e}")

def main():
    """Main fix script execution"""
    
    print("=" * 60)
    print("QNTI Quick Fix Script")
    print("=" * 60)
    print()
    
    # Create backup
    print("1. Creating backup of existing files...")
    backup_dir = backup_files()
    print()
    
    # Create directory structure
    print("2. Creating directory structure...")
    create_directory_structure()
    print()
    
    # Create configuration files
    print("3. Creating/fixing configuration files...")
    create_fixed_config_files()
    print()
    
    # Fix import issues
    print("4. Fixing import issues...")
    fix_import_issues()
    print()
    
    # Create requirements file
    print("5. Creating requirements.txt...")
    create_requirements_file()
    print()
    
    # Create startup scripts
    print("6. Creating startup scripts...")
    create_startup_script()
    print()
    
    # Create README
    print("7. Creating README.md...")
    create_readme()
    print()
    
    # Check file encodings
    print("8. Checking file encodings...")
    check_file_encodings()
    print()
    
    print("=" * 60)
    print("✅ QNTI Fix Script Completed Successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Replace your existing files with the fixed versions from the artifacts")
    print("2. Edit vision_config.json to add your OpenAI API key")
    print("3. Run: pip install -r requirements.txt")
    print("4. Start the system with the startup script or manual command")
    print()
    print(f"Backup of your original files: {backup_dir}")
    print()
    print("🚀 Ready to run QNTI!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error running fix script: {e}")
        sys.exit(1)