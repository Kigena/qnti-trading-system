#!/bin/bash

# QNTI Trading System - Production Deployment Script
# This script handles the complete deployment of the QNTI system

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKER_DIR="$PROJECT_DIR/docker"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"
SECURITY_COMPOSE_FILE="$DOCKER_DIR/security/docker-compose.security.yml"
ENV_FILE="$PROJECT_DIR/.env"

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

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker and try again."
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Function to create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p "$PROJECT_DIR/docker/volumes/postgres"
    mkdir -p "$PROJECT_DIR/docker/volumes/redis"
    mkdir -p "$PROJECT_DIR/docker/volumes/qnti_data"
    mkdir -p "$PROJECT_DIR/docker/volumes/logs"
    mkdir -p "$PROJECT_DIR/docker/volumes/backups"
    mkdir -p "$PROJECT_DIR/docker/volumes/prometheus"
    mkdir -p "$PROJECT_DIR/docker/volumes/grafana"
    mkdir -p "$PROJECT_DIR/docker/ssl"
    
    # Set proper permissions
    chmod 755 "$PROJECT_DIR/docker/volumes"
    chmod 755 "$PROJECT_DIR/docker/volumes"/*
    
    log_success "Directories created successfully"
}

# Function to generate SSL certificates
generate_ssl_certificates() {
    log_info "Generating SSL certificates..."
    
    SSL_DIR="$PROJECT_DIR/docker/ssl"
    
    if [ ! -f "$SSL_DIR/cert.pem" ] || [ ! -f "$SSL_DIR/key.pem" ]; then
        log_info "Generating self-signed SSL certificates..."
        
        openssl req -x509 -newkey rsa:4096 -keyout "$SSL_DIR/key.pem" -out "$SSL_DIR/cert.pem" \
            -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        chmod 600 "$SSL_DIR/key.pem"
        chmod 644 "$SSL_DIR/cert.pem"
        
        log_success "SSL certificates generated"
    else
        log_info "SSL certificates already exist"
    fi
}

# Function to setup environment
setup_environment() {
    log_info "Setting up environment..."
    
    if [ ! -f "$ENV_FILE" ]; then
        log_info "Creating environment file from template..."
        cp "$PROJECT_DIR/.env.example" "$ENV_FILE"
        
        # Generate random secrets
        SECRET_KEY=$(openssl rand -base64 32)
        SECURITY_SALT=$(openssl rand -base64 32)
        
        # Update environment file
        sed -i "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" "$ENV_FILE"
        sed -i "s/SECURITY_PASSWORD_SALT=.*/SECURITY_PASSWORD_SALT=$SECURITY_SALT/" "$ENV_FILE"
        
        log_warn "Environment file created. Please review and update $ENV_FILE before deployment."
    else
        log_info "Environment file already exists"
    fi
}

# Function to build images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_DIR"
    
    # Build QNTI application image
    docker build -t qnti-app:latest --target production .
    
    log_success "Docker images built successfully"
}

# Function to deploy services
deploy_services() {
    log_info "Deploying services..."
    
    cd "$PROJECT_DIR"
    
    # Deploy with security configurations
    docker-compose -f "$COMPOSE_FILE" -f "$SECURITY_COMPOSE_FILE" up -d
    
    log_success "Services deployed successfully"
}

# Function to wait for services to be healthy
wait_for_services() {
    log_info "Waiting for services to be healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up (healthy)"; then
            log_success "Services are healthy"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: Waiting for services..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Services did not become healthy within the timeout period"
    return 1
}

# Function to run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    cd "$PROJECT_DIR"
    
    # Wait for database to be ready
    docker-compose exec qnti-postgres pg_isready -U qnti_user -d qnti_trading
    
    # Run migrations
    docker-compose exec qnti-app python -c "
from db_migration.database_config import get_database_manager
db_manager = get_database_manager()
if db_manager.test_connection():
    print('Database migration completed successfully')
else:
    print('Database migration failed')
    exit(1)
"
    
    log_success "Database migrations completed"
}

# Function to setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    cd "$PROJECT_DIR"
    
    # Deploy monitoring services
    docker-compose -f "$COMPOSE_FILE" -f "$SECURITY_COMPOSE_FILE" --profile monitoring up -d
    
    log_success "Monitoring services deployed"
}

# Function to show deployment status
show_status() {
    log_info "Deployment status:"
    
    cd "$PROJECT_DIR"
    
    echo ""
    echo "Services:"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    echo "Logs (last 20 lines):"
    docker-compose -f "$COMPOSE_FILE" logs --tail=20
    
    echo ""
    echo "Access URLs:"
    echo "  - QNTI Application: https://localhost:443"
    echo "  - HTTP (redirects to HTTPS): http://localhost:80"
    echo "  - Prometheus (monitoring): http://localhost:9090"
    echo "  - Grafana (dashboards): http://localhost:3000"
    echo ""
    echo "Database Access:"
    echo "  - PostgreSQL: localhost:5432"
    echo "  - Redis: localhost:6379"
    echo ""
    echo "Default Credentials:"
    echo "  - Grafana: admin/admin"
    echo "  - Database: qnti_user/qnti_password"
    echo ""
}

# Function to cleanup resources
cleanup() {
    log_info "Cleaning up resources..."
    
    cd "$PROJECT_DIR"
    
    # Stop and remove containers
    docker-compose -f "$COMPOSE_FILE" -f "$SECURITY_COMPOSE_FILE" down
    
    # Remove unused images
    docker image prune -f
    
    log_success "Cleanup completed"
}

# Function to show help
show_help() {
    echo "QNTI Trading System - Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy        Full deployment (default)"
    echo "  build         Build Docker images only"
    echo "  start         Start services"
    echo "  stop          Stop services"
    echo "  restart       Restart services"
    echo "  status        Show deployment status"
    echo "  logs          Show service logs"
    echo "  cleanup       Stop services and cleanup"
    echo "  monitoring    Deploy monitoring services"
    echo "  help          Show this help message"
    echo ""
}

# Main deployment function
main_deploy() {
    log_info "Starting QNTI Trading System deployment..."
    
    check_prerequisites
    create_directories
    generate_ssl_certificates
    setup_environment
    build_images
    deploy_services
    wait_for_services
    run_migrations
    show_status
    
    log_success "Deployment completed successfully!"
    log_info "You can now access the QNTI Trading System at https://localhost"
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        main_deploy
        ;;
    "build")
        check_prerequisites
        build_images
        ;;
    "start")
        cd "$PROJECT_DIR"
        docker-compose -f "$COMPOSE_FILE" -f "$SECURITY_COMPOSE_FILE" up -d
        ;;
    "stop")
        cd "$PROJECT_DIR"
        docker-compose -f "$COMPOSE_FILE" -f "$SECURITY_COMPOSE_FILE" down
        ;;
    "restart")
        cd "$PROJECT_DIR"
        docker-compose -f "$COMPOSE_FILE" -f "$SECURITY_COMPOSE_FILE" restart
        ;;
    "status")
        show_status
        ;;
    "logs")
        cd "$PROJECT_DIR"
        docker-compose -f "$COMPOSE_FILE" logs -f
        ;;
    "cleanup")
        cleanup
        ;;
    "monitoring")
        setup_monitoring
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