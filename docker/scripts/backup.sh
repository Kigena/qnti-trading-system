#!/bin/bash

# QNTI Trading System - Backup Script
# This script handles database and application data backups

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKUP_DIR="$PROJECT_DIR/docker/volumes/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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

# Function to backup PostgreSQL database
backup_postgres() {
    log_info "Backing up PostgreSQL database..."
    
    local backup_file="$BACKUP_DIR/postgres_backup_$TIMESTAMP.sql"
    
    # Create backup directory if it doesn't exist
    mkdir -p "$BACKUP_DIR"
    
    # Backup database
    docker-compose exec -T qnti-postgres pg_dump -U qnti_user -d qnti_trading > "$backup_file"
    
    # Compress backup
    gzip "$backup_file"
    
    log_success "PostgreSQL backup completed: ${backup_file}.gz"
}

# Function to backup Redis data
backup_redis() {
    log_info "Backing up Redis data..."
    
    local backup_file="$BACKUP_DIR/redis_backup_$TIMESTAMP.rdb"
    
    # Create backup directory if it doesn't exist
    mkdir -p "$BACKUP_DIR"
    
    # Force Redis to save current state
    docker-compose exec qnti-redis redis-cli BGSAVE
    
    # Wait for background save to complete
    sleep 5
    
    # Copy Redis dump file
    docker cp qnti-redis:/data/dump.rdb "$backup_file"
    
    # Compress backup
    gzip "$backup_file"
    
    log_success "Redis backup completed: ${backup_file}.gz"
}

# Function to backup application data
backup_app_data() {
    log_info "Backing up application data..."
    
    local backup_file="$BACKUP_DIR/app_data_backup_$TIMESTAMP.tar.gz"
    
    # Create backup directory if it doesn't exist
    mkdir -p "$BACKUP_DIR"
    
    # Backup application data
    tar -czf "$backup_file" -C "$PROJECT_DIR" \
        qnti_data \
        ea_profiles \
        chart_uploads \
        qnti_screenshots \
        qnti_memory \
        qnti_config.json \
        vision_config.json \
        mt5_config.json \
        --exclude="*.log" \
        --exclude="*.tmp" \
        --exclude="__pycache__" \
        --exclude="*.pyc"
    
    log_success "Application data backup completed: $backup_file"
}

# Function to backup configuration files
backup_config() {
    log_info "Backing up configuration files..."
    
    local backup_file="$BACKUP_DIR/config_backup_$TIMESTAMP.tar.gz"
    
    # Create backup directory if it doesn't exist
    mkdir -p "$BACKUP_DIR"
    
    # Backup configuration files
    tar -czf "$backup_file" -C "$PROJECT_DIR" \
        .env \
        docker-compose.yml \
        docker/ \
        --exclude="docker/volumes" \
        --exclude="docker/ssl/key.pem"
    
    log_success "Configuration backup completed: $backup_file"
}

# Function to upload backup to S3 (optional)
upload_to_s3() {
    local backup_file="$1"
    
    if [ -n "$AWS_S3_BUCKET" ] && [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
        log_info "Uploading backup to S3..."
        
        # Upload to S3 using AWS CLI
        aws s3 cp "$backup_file" "s3://$AWS_S3_BUCKET/qnti-backups/$(basename "$backup_file")"
        
        log_success "Backup uploaded to S3: s3://$AWS_S3_BUCKET/qnti-backups/$(basename "$backup_file")"
    else
        log_info "S3 credentials not configured, skipping upload"
    fi
}

# Function to cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up old backups..."
    
    local retention_days="${BACKUP_RETENTION_DAYS:-7}"
    
    # Remove backups older than retention period
    find "$BACKUP_DIR" -name "*.gz" -type f -mtime +$retention_days -delete
    find "$BACKUP_DIR" -name "*.sql" -type f -mtime +$retention_days -delete
    find "$BACKUP_DIR" -name "*.tar.gz" -type f -mtime +$retention_days -delete
    
    log_success "Old backups cleaned up (retention: $retention_days days)"
}

# Function to restore PostgreSQL database
restore_postgres() {
    local backup_file="$1"
    
    if [ -z "$backup_file" ]; then
        log_error "Please specify backup file to restore"
        return 1
    fi
    
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    log_info "Restoring PostgreSQL database from: $backup_file"
    
    # Stop application to prevent database conflicts
    docker-compose stop qnti-app
    
    # Drop existing database and recreate
    docker-compose exec qnti-postgres dropdb -U qnti_user qnti_trading --if-exists
    docker-compose exec qnti-postgres createdb -U qnti_user qnti_trading
    
    # Restore database
    if [[ "$backup_file" == *.gz ]]; then
        gunzip -c "$backup_file" | docker-compose exec -T qnti-postgres psql -U qnti_user -d qnti_trading
    else
        docker-compose exec -T qnti-postgres psql -U qnti_user -d qnti_trading < "$backup_file"
    fi
    
    # Restart application
    docker-compose start qnti-app
    
    log_success "PostgreSQL database restored successfully"
}

# Function to restore Redis data
restore_redis() {
    local backup_file="$1"
    
    if [ -z "$backup_file" ]; then
        log_error "Please specify backup file to restore"
        return 1
    fi
    
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    log_info "Restoring Redis data from: $backup_file"
    
    # Stop Redis container
    docker-compose stop qnti-redis
    
    # Restore Redis dump file
    if [[ "$backup_file" == *.gz ]]; then
        gunzip -c "$backup_file" > /tmp/dump.rdb
        docker cp /tmp/dump.rdb qnti-redis:/data/dump.rdb
        rm /tmp/dump.rdb
    else
        docker cp "$backup_file" qnti-redis:/data/dump.rdb
    fi
    
    # Restart Redis container
    docker-compose start qnti-redis
    
    log_success "Redis data restored successfully"
}

# Function to list available backups
list_backups() {
    log_info "Available backups:"
    
    if [ -d "$BACKUP_DIR" ]; then
        ls -la "$BACKUP_DIR"/ | grep -E "\.(sql|rdb|tar)\.gz$|\.sql$|\.rdb$|\.tar\.gz$"
    else
        log_info "No backup directory found"
    fi
}

# Function to show help
show_help() {
    echo "QNTI Trading System - Backup Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  backup        Create full backup (default)"
    echo "  postgres      Backup PostgreSQL database only"
    echo "  redis         Backup Redis data only"
    echo "  app-data      Backup application data only"
    echo "  config        Backup configuration files only"
    echo "  restore-db    Restore PostgreSQL database"
    echo "  restore-redis Restore Redis data"
    echo "  list          List available backups"
    echo "  cleanup       Clean up old backups"
    echo "  help          Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  BACKUP_RETENTION_DAYS  Number of days to keep backups (default: 7)"
    echo "  AWS_S3_BUCKET          S3 bucket for backup storage (optional)"
    echo "  AWS_ACCESS_KEY_ID      AWS access key (optional)"
    echo "  AWS_SECRET_ACCESS_KEY  AWS secret key (optional)"
    echo ""
}

# Main backup function
main_backup() {
    log_info "Starting full backup..."
    
    backup_postgres
    backup_redis
    backup_app_data
    backup_config
    
    # Upload to S3 if configured
    for backup_file in "$BACKUP_DIR"/*_$TIMESTAMP.*; do
        if [ -f "$backup_file" ]; then
            upload_to_s3 "$backup_file"
        fi
    done
    
    cleanup_old_backups
    
    log_success "Full backup completed successfully!"
}

# Parse command line arguments
case "${1:-backup}" in
    "backup")
        main_backup
        ;;
    "postgres")
        backup_postgres
        ;;
    "redis")
        backup_redis
        ;;
    "app-data")
        backup_app_data
        ;;
    "config")
        backup_config
        ;;
    "restore-db")
        restore_postgres "$2"
        ;;
    "restore-redis")
        restore_redis "$2"
        ;;
    "list")
        list_backups
        ;;
    "cleanup")
        cleanup_old_backups
        ;;
    "help")
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac