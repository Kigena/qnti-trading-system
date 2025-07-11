#!/bin/bash

# QNTI Trading System - Simple Docker Entrypoint Script
set -e

# Default environment values
export QNTI_ENV=${QNTI_ENV:-production}
export FLASK_ENV=${FLASK_ENV:-production}
export PYTHONPATH="/app:$PYTHONPATH"

echo "[INFO] Starting QNTI Trading System..."
echo "[INFO] Environment: $QNTI_ENV"
echo "[INFO] Python path: $PYTHONPATH"

# Create necessary directories
mkdir -p /app/logs /app/qnti_data /app/ea_profiles /app/chart_uploads /app/qnti_screenshots /app/qnti_backups /app/qnti_memory

# Create a basic configuration file
cat > /app/qnti_config.json << 'EOF'
{
    "system": {
        "auto_trading": false,
        "vision_auto_analysis": true,
        "ea_monitoring": true,
        "api_port": 5000,
        "debug_mode": false,
        "max_concurrent_trades": 10
    },
    "integration": {
        "mt5_enabled": false,
        "vision_enabled": true,
        "dashboard_enabled": true,
        "webhook_enabled": false,
        "telegram_notifications": false,
        "redis_enabled": false,
        "postgresql_enabled": false
    }
}
EOF

echo "[INFO] Configuration created successfully"
echo "[INFO] Starting application with command: $@"

# Execute the command
exec "$@"