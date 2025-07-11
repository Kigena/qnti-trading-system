#!/bin/bash

# QNTI Trading System - Volume Initialization Script
# This script initializes all required volumes and directories

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Color output functions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to create volume directories
create_volume_directories() {
    log_info "Creating volume directories..."
    
    # Create base volumes directory
    mkdir -p "$PROJECT_DIR/docker/volumes"
    
    # Create service-specific volume directories
    mkdir -p "$PROJECT_DIR/docker/volumes/postgres"
    mkdir -p "$PROJECT_DIR/docker/volumes/redis"
    mkdir -p "$PROJECT_DIR/docker/volumes/qnti_data"
    mkdir -p "$PROJECT_DIR/docker/volumes/logs"
    mkdir -p "$PROJECT_DIR/docker/volumes/backups"
    mkdir -p "$PROJECT_DIR/docker/volumes/prometheus"
    mkdir -p "$PROJECT_DIR/docker/volumes/grafana"
    
    # Create SSL directory
    mkdir -p "$PROJECT_DIR/docker/ssl"
    
    # Create application data directories
    mkdir -p "$PROJECT_DIR/ea_profiles"
    mkdir -p "$PROJECT_DIR/chart_uploads"
    mkdir -p "$PROJECT_DIR/qnti_screenshots"
    mkdir -p "$PROJECT_DIR/qnti_memory"
    
    # Set proper permissions
    chmod 755 "$PROJECT_DIR/docker/volumes"
    chmod 755 "$PROJECT_DIR/docker/volumes"/*
    chmod 755 "$PROJECT_DIR/ea_profiles"
    chmod 755 "$PROJECT_DIR/chart_uploads"
    chmod 755 "$PROJECT_DIR/qnti_screenshots"
    chmod 755 "$PROJECT_DIR/qnti_memory"
    
    log_success "Volume directories created successfully"
}

# Function to initialize configuration files
initialize_configs() {
    log_info "Initializing configuration files..."
    
    # Create default .env if it doesn't exist
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        log_info "Created default .env file from template"
    fi
    
    # Create database config directory
    mkdir -p "$PROJECT_DIR/docker/postgres/archive"
    chmod 755 "$PROJECT_DIR/docker/postgres/archive"
    
    log_success "Configuration files initialized"
}

# Function to set up file permissions
set_permissions() {
    log_info "Setting up file permissions..."
    
    # Make scripts executable
    find "$PROJECT_DIR/docker/scripts" -name "*.sh" -exec chmod +x {} \;
    find "$PROJECT_DIR/docker/postgres/init-scripts" -name "*.sh" -exec chmod +x {} \;
    chmod +x "$PROJECT_DIR/docker/entrypoint.sh"
    
    # Set proper ownership for volume directories
    # Note: In production, these should be owned by specific service users
    chown -R $(whoami):$(whoami) "$PROJECT_DIR/docker/volumes" 2>/dev/null || true
    
    log_success "File permissions configured"
}

# Function to validate Docker setup
validate_docker() {
    log_info "Validating Docker setup..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    log_success "Docker setup validated"
}

# Function to show initialization summary
show_summary() {
    log_info "Initialization Summary:"
    
    echo ""
    echo "Created Directories:"
    echo "  - docker/volumes/postgres"
    echo "  - docker/volumes/redis"
    echo "  - docker/volumes/qnti_data"
    echo "  - docker/volumes/logs"
    echo "  - docker/volumes/backups"
    echo "  - docker/volumes/prometheus"
    echo "  - docker/volumes/grafana"
    echo "  - docker/ssl"
    echo "  - ea_profiles"
    echo "  - chart_uploads"
    echo "  - qnti_screenshots"
    echo "  - qnti_memory"
    echo ""
    echo "Configuration Files:"
    echo "  - .env (from .env.example)"
    echo "  - docker-compose.yml"
    echo "  - All service configurations"
    echo ""
    echo "Next Steps:"
    echo "  1. Review and update .env file with your settings"
    echo "  2. Run: ./docker/scripts/deploy.sh"
    echo "  3. Access the application at: https://localhost"
    echo ""
    echo "For development:"
    echo "  - Run: ./docker/scripts/dev-setup.sh"
    echo "  - Access at: http://localhost:5001"
    echo ""
}

# Main function
main() {
    log_info "Initializing QNTI Trading System Docker environment..."
    
    validate_docker
    create_volume_directories
    initialize_configs
    set_permissions
    show_summary
    
    log_success "Docker environment initialization completed!"
}

# Run main function
main "$@"