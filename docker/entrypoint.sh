#!/bin/bash

# QNTI Trading System - Docker Entrypoint Script
# Handles initialization, health checks, and graceful shutdown

set -e

# Color output functions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Default environment values
export QNTI_ENV=${QNTI_ENV:-production}
export FLASK_ENV=${FLASK_ENV:-production}
export PYTHONPATH="/app:$PYTHONPATH"

# Database connection parameters
export QNTI_DB_HOST=${QNTI_DB_HOST:-qnti-postgres}
export QNTI_DB_PORT=${QNTI_DB_PORT:-5432}
export QNTI_DB_NAME=${QNTI_DB_NAME:-qnti_trading}
export QNTI_DB_USER=${QNTI_DB_USER:-qnti_user}
export QNTI_DB_PASSWORD=${QNTI_DB_PASSWORD:-qnti_password}
export QNTI_DB_SCHEMA=${QNTI_DB_SCHEMA:-qnti}

# Redis connection parameters
export QNTI_REDIS_HOST=${QNTI_REDIS_HOST:-qnti-redis}
export QNTI_REDIS_PORT=${QNTI_REDIS_PORT:-6379}
export QNTI_REDIS_PASSWORD=${QNTI_REDIS_PASSWORD:-}

# MT5 connection parameters
export QNTI_MT5_ENABLED=${QNTI_MT5_ENABLED:-false}
export QNTI_MT5_LOGIN=${QNTI_MT5_LOGIN:-}
export QNTI_MT5_PASSWORD=${QNTI_MT5_PASSWORD:-}
export QNTI_MT5_SERVER=${QNTI_MT5_SERVER:-}

# API Keys (optional)
export OPENAI_API_KEY=${OPENAI_API_KEY:-}
export TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN:-}

# Application settings
export QNTI_DEBUG=${QNTI_DEBUG:-false}
export QNTI_LOG_LEVEL=${QNTI_LOG_LEVEL:-INFO}
export QNTI_WORKERS=${QNTI_WORKERS:-4}
export QNTI_MAX_REQUESTS=${QNTI_MAX_REQUESTS:-1000}

# Function to wait for a service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1

    log_info "Waiting for $service_name to be ready at $host:$port..."

    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            log_success "$service_name is ready!"
            return 0
        fi
        
        log_warn "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 2
        ((attempt++))
    done

    log_error "$service_name is not ready after $max_attempts attempts"
    return 1
}

# Function to check database connection
check_database() {
    log_info "Checking database connection..."
    
    python3 -c "
import os
import psycopg2
import sys

try:
    conn = psycopg2.connect(
        host=os.environ.get('QNTI_DB_HOST'),
        port=int(os.environ.get('QNTI_DB_PORT', '5432')),
        database=os.environ.get('QNTI_DB_NAME'),
        user=os.environ.get('QNTI_DB_USER'),
        password=os.environ.get('QNTI_DB_PASSWORD')
    )
    
    with conn.cursor() as cursor:
        cursor.execute('SELECT 1')
        result = cursor.fetchone()
        
    conn.close()
    
    if result and result[0] == 1:
        print('Database connection successful')
        sys.exit(0)
    else:
        print('Database connection failed')
        sys.exit(1)
        
except Exception as e:
    print(f'Database connection error: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "Database connection successful"
        return 0
    else
        log_error "Database connection failed"
        return 1
    fi
}

# Function to check Redis connection
check_redis() {
    log_info "Checking Redis connection..."
    
    python3 -c "
import os
import redis
import sys

try:
    r = redis.Redis(
        host=os.environ.get('QNTI_REDIS_HOST'),
        port=int(os.environ.get('QNTI_REDIS_PORT', '6379')),
        password=os.environ.get('QNTI_REDIS_PASSWORD') or None,
        decode_responses=True
    )
    
    r.ping()
    print('Redis connection successful')
    sys.exit(0)
    
except Exception as e:
    print(f'Redis connection error: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "Redis connection successful"
        return 0
    else
        log_error "Redis connection failed"
        return 1
    fi
}

# Function to initialize database schema
init_database() {
    log_info "Initializing database schema..."
    
    python3 -c "
import os
import sys
sys.path.insert(0, '/app')

try:
    from db_migration.database_config import get_database_manager
    
    # Test database connection
    db_manager = get_database_manager()
    
    if db_manager.test_connection():
        print('Database schema initialization successful')
        sys.exit(0)
    else:
        print('Database schema initialization failed')
        sys.exit(1)
        
except Exception as e:
    print(f'Database schema initialization error: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "Database schema initialized"
        return 0
    else
        log_error "Database schema initialization failed"
        return 1
    fi
}

# Function to create default configuration
create_default_config() {
    log_info "Creating default configuration..."
    
    # Create database configuration
    cat > /app/db_config.json << EOF
{
    "host": "${QNTI_DB_HOST}",
    "port": ${QNTI_DB_PORT},
    "database": "${QNTI_DB_NAME}",
    "username": "${QNTI_DB_USER}",
    "password": "${QNTI_DB_PASSWORD}",
    "schema": "${QNTI_DB_SCHEMA}",
    "ssl_mode": "prefer",
    "connect_timeout": 30,
    "command_timeout": 60,
    "pool_min_conn": 2,
    "pool_max_conn": 10,
    "pool_max_overflow": 20,
    "pool_recycle": 3600,
    "pool_pre_ping": true
}
EOF

    # Create Redis configuration
    cat > /app/redis_config.json << EOF
{
    "host": "${QNTI_REDIS_HOST}",
    "port": ${QNTI_REDIS_PORT},
    "password": "${QNTI_REDIS_PASSWORD}",
    "db": 0,
    "decode_responses": true,
    "socket_timeout": 5,
    "socket_connect_timeout": 5,
    "retry_on_timeout": true,
    "max_connections": 50
}
EOF

    # Update QNTI configuration
    python3 -c "
import json
import os

config = {
    'system': {
        'auto_trading': False,
        'vision_auto_analysis': True,
        'ea_monitoring': True,
        'api_port': 5000,
        'debug_mode': os.environ.get('QNTI_DEBUG', 'false').lower() == 'true',
        'max_concurrent_trades': 10,
        'risk_management': {
            'max_daily_loss': 1000,
            'max_drawdown': 0.20,
            'position_size_limit': 1.0,
            'emergency_close_drawdown': 0.20
        }
    },
    'integration': {
        'mt5_enabled': os.environ.get('QNTI_MT5_ENABLED', 'false').lower() == 'true',
        'vision_enabled': True,
        'dashboard_enabled': True,
        'webhook_enabled': False,
        'telegram_notifications': bool(os.environ.get('TELEGRAM_BOT_TOKEN')),
        'redis_enabled': True,
        'postgresql_enabled': True
    },
    'database': {
        'host': os.environ.get('QNTI_DB_HOST'),
        'port': int(os.environ.get('QNTI_DB_PORT', '5432')),
        'name': os.environ.get('QNTI_DB_NAME'),
        'user': os.environ.get('QNTI_DB_USER'),
        'schema': os.environ.get('QNTI_DB_SCHEMA')
    },
    'redis': {
        'host': os.environ.get('QNTI_REDIS_HOST'),
        'port': int(os.environ.get('QNTI_REDIS_PORT', '6379'))
    },
    'mt5': {
        'enabled': os.environ.get('QNTI_MT5_ENABLED', 'false').lower() == 'true',
        'login': os.environ.get('QNTI_MT5_LOGIN', ''),
        'password': os.environ.get('QNTI_MT5_PASSWORD', ''),
        'server': os.environ.get('QNTI_MT5_SERVER', '')
    },
    'api_keys': {
        'openai': os.environ.get('OPENAI_API_KEY', ''),
        'telegram': os.environ.get('TELEGRAM_BOT_TOKEN', '')
    },
    'scheduling': {
        'vision_analysis_interval': 300,
        'health_check_interval': 60,
        'performance_update_interval': 30,
        'backup_interval': 3600
    },
    'alerts': {
        'email_alerts': False,
        'telegram_alerts': bool(os.environ.get('TELEGRAM_BOT_TOKEN')),
        'webhook_alerts': False,
        'log_alerts': True
    },
    'vision': {
        'primary_symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],
        'timeframes': ['H1', 'H4']
    }
}

with open('/app/qnti_config.json', 'w') as f:
    json.dump(config, f, indent=2)
    
print('Configuration created successfully')
"
    
    log_success "Default configuration created"
}

# Function to handle graceful shutdown
graceful_shutdown() {
    log_info "Received shutdown signal, stopping QNTI gracefully..."
    
    # Send SIGTERM to all child processes
    if [ ! -z "$QNTI_PID" ]; then
        kill -TERM "$QNTI_PID" 2>/dev/null || true
        wait "$QNTI_PID" 2>/dev/null || true
    fi
    
    log_success "QNTI stopped gracefully"
    exit 0
}

# Function to perform health check
health_check() {
    log_info "Performing health check..."
    
    # Check if application is responding
    if curl -f http://localhost:5000/health >/dev/null 2>&1; then
        log_success "Health check passed"
        return 0
    else
        log_error "Health check failed"
        return 1
    fi
}

# Set up signal handlers
trap graceful_shutdown SIGTERM SIGINT

# Main initialization
log_info "Starting QNTI Trading System..."
log_info "Environment: $QNTI_ENV"
log_info "Python path: $PYTHONPATH"

# Wait for dependencies
if ! wait_for_service "$QNTI_DB_HOST" "$QNTI_DB_PORT" "PostgreSQL"; then
    log_error "PostgreSQL is not available, exiting..."
    exit 1
fi

if ! wait_for_service "$QNTI_REDIS_HOST" "$QNTI_REDIS_PORT" "Redis"; then
    log_error "Redis is not available, exiting..."
    exit 1
fi

# Check connections
if ! check_database; then
    log_error "Database connection failed, exiting..."
    exit 1
fi

if ! check_redis; then
    log_error "Redis connection failed, exiting..."
    exit 1
fi

# Initialize database schema
if ! init_database; then
    log_warn "Database schema initialization failed, but continuing..."
fi

# Create configuration
create_default_config

# Set permissions
chown -R qnti:qnti /app/logs /app/qnti_data /app/ea_profiles /app/chart_uploads /app/qnti_screenshots /app/qnti_backups /app/qnti_memory 2>/dev/null || true

# Start application
log_info "Starting QNTI application..."
log_info "Command: $@"

# Execute the command
exec "$@" &
QNTI_PID=$!

# Wait for the process
wait $QNTI_PID