#!/bin/bash

# QNTI Trading System - Development Environment Setup Script
# This script sets up the development environment for the QNTI system

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKER_DIR="$PROJECT_DIR/docker"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"
DEV_COMPOSE_FILE="$PROJECT_DIR/docker-compose.dev.yml"
ENV_FILE="$PROJECT_DIR/.env.development"

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
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3 and try again."
        exit 1
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed. Please install Git and try again."
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Function to create development compose file
create_dev_compose() {
    log_info "Creating development compose file..."
    
    cat > "$DEV_COMPOSE_FILE" << 'EOF'
# QNTI Trading System - Development Environment
# This extends the main docker-compose.yml for development

version: '3.8'

services:
  # Development version of QNTI Application
  qnti-app:
    build:
      context: .
      dockerfile: Dockerfile
      target: development
    environment:
      - QNTI_ENV=development
      - FLASK_ENV=development
      - QNTI_DEBUG=true
      - QNTI_RELOAD=true
    volumes:
      - .:/app:cached
      - /app/node_modules
    ports:
      - "5001:5000"
      - "5678:5678"  # Debug port
    command: ["python", "qnti_main_system.py", "--debug", "--host", "0.0.0.0", "--port", "5000"]
    depends_on:
      - qnti-postgres
      - qnti-redis
    
  # Development PostgreSQL with external port
  qnti-postgres:
    ports:
      - "5433:5432"
    environment:
      - POSTGRES_DB=qnti_trading_dev
      - POSTGRES_USER=qnti_dev_user
      - POSTGRES_PASSWORD=qnti_dev_password
    
  # Development Redis with external port
  qnti-redis:
    ports:
      - "6380:6379"
    command: >
      redis-server
      --appendonly yes
      --requirepass qnti_dev_redis_pass
    
  # Development Nginx
  qnti-nginx:
    ports:
      - "8080:80"
      - "8443:443"
    depends_on:
      - qnti-app
    
  # Jupyter Notebook for development
  qnti-jupyter:
    build:
      context: .
      dockerfile: Dockerfile
      target: development
    ports:
      - "8888:8888"
    volumes:
      - .:/app:cached
      - jupyter_data:/root/.jupyter
    command: >
      jupyter notebook
      --ip=0.0.0.0
      --port=8888
      --no-browser
      --allow-root
      --NotebookApp.token=''
      --NotebookApp.password=''
    environment:
      - JUPYTER_ENABLE_LAB=yes
    depends_on:
      - qnti-postgres
      - qnti-redis

volumes:
  jupyter_data:
EOF
    
    log_success "Development compose file created"
}

# Function to setup development environment
setup_dev_environment() {
    log_info "Setting up development environment..."
    
    # Copy development environment file
    if [ ! -f "$ENV_FILE" ]; then
        log_info "Creating development environment file..."
        
        # The .env.development file should already exist
        if [ -f "$PROJECT_DIR/.env.development" ]; then
            cp "$PROJECT_DIR/.env.development" "$PROJECT_DIR/.env"
        else
            log_error "Development environment template not found"
            exit 1
        fi
    fi
    
    log_success "Development environment configured"
}

# Function to create development directories
create_dev_directories() {
    log_info "Creating development directories..."
    
    mkdir -p "$PROJECT_DIR/docker/volumes/postgres_dev"
    mkdir -p "$PROJECT_DIR/docker/volumes/redis_dev"
    mkdir -p "$PROJECT_DIR/docker/volumes/qnti_data_dev"
    mkdir -p "$PROJECT_DIR/docker/volumes/logs_dev"
    mkdir -p "$PROJECT_DIR/docker/volumes/jupyter"
    
    # Set proper permissions
    chmod 755 "$PROJECT_DIR/docker/volumes"
    chmod 755 "$PROJECT_DIR/docker/volumes"/*
    
    log_success "Development directories created"
}

# Function to install development dependencies
install_dev_dependencies() {
    log_info "Installing development dependencies..."
    
    cd "$PROJECT_DIR"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements_api.txt
    pip install -r requirements_llm.txt
    pip install -r requirements_simulation.txt
    
    # Install development dependencies
    pip install \
        pytest \
        pytest-cov \
        black \
        flake8 \
        mypy \
        ipython \
        jupyter \
        pre-commit \
        bandit \
        safety \
        autopep8 \
        isort
    
    log_success "Development dependencies installed"
}

# Function to setup pre-commit hooks
setup_pre_commit() {
    log_info "Setting up pre-commit hooks..."
    
    cd "$PROJECT_DIR"
    
    # Create pre-commit configuration
    cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements
      - id: check-json
      - id: pretty-format-json
        args: ['--autofix']
      
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3
        
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88', '--extend-ignore=E203,W503']
        
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ['--profile', 'black']
        
  - repo: https://github.com/pycqa/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-r', '.', '-f', 'json', '-o', 'bandit-report.json']
        
  - repo: https://github.com/Lucas-C/pre-commit-hooks-safety
    rev: v1.3.2
    hooks:
      - id: python-safety-dependencies-check
EOF
    
    # Install pre-commit hooks
    source venv/bin/activate
    pre-commit install
    
    log_success "Pre-commit hooks configured"
}

# Function to create development scripts
create_dev_scripts() {
    log_info "Creating development scripts..."
    
    # Create test script
    cat > "$PROJECT_DIR/run_tests.sh" << 'EOF'
#!/bin/bash
# Run tests for QNTI Trading System

source venv/bin/activate
python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing
EOF
    
    # Create format script
    cat > "$PROJECT_DIR/format_code.sh" << 'EOF'
#!/bin/bash
# Format code for QNTI Trading System

source venv/bin/activate
black .
isort .
flake8 .
EOF
    
    # Create development server script
    cat > "$PROJECT_DIR/run_dev.sh" << 'EOF'
#!/bin/bash
# Run development server for QNTI Trading System

source venv/bin/activate
export FLASK_ENV=development
export QNTI_ENV=development
export QNTI_DEBUG=true
python qnti_main_system.py --debug --host 0.0.0.0 --port 5000
EOF
    
    # Make scripts executable
    chmod +x "$PROJECT_DIR/run_tests.sh"
    chmod +x "$PROJECT_DIR/format_code.sh"
    chmod +x "$PROJECT_DIR/run_dev.sh"
    
    log_success "Development scripts created"
}

# Function to build development images
build_dev_images() {
    log_info "Building development images..."
    
    cd "$PROJECT_DIR"
    
    # Build development image
    docker build -t qnti-app:dev --target development .
    
    log_success "Development images built"
}

# Function to start development environment
start_dev_environment() {
    log_info "Starting development environment..."
    
    cd "$PROJECT_DIR"
    
    # Start development services
    docker-compose -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" up -d
    
    log_success "Development environment started"
}

# Function to show development status
show_dev_status() {
    log_info "Development environment status:"
    
    cd "$PROJECT_DIR"
    
    echo ""
    echo "Services:"
    docker-compose -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" ps
    
    echo ""
    echo "Access URLs:"
    echo "  - QNTI Application: http://localhost:5001"
    echo "  - Jupyter Notebook: http://localhost:8888"
    echo "  - Nginx: http://localhost:8080"
    echo "  - Database: localhost:5433"
    echo "  - Redis: localhost:6380"
    echo ""
    echo "Development Commands:"
    echo "  - Run tests: ./run_tests.sh"
    echo "  - Format code: ./format_code.sh"
    echo "  - Run dev server: ./run_dev.sh"
    echo "  - Activate venv: source venv/bin/activate"
    echo ""
}

# Function to show help
show_help() {
    echo "QNTI Trading System - Development Setup Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup         Full development setup (default)"
    echo "  build         Build development images"
    echo "  start         Start development environment"
    echo "  stop          Stop development environment"
    echo "  status        Show development status"
    echo "  logs          Show development logs"
    echo "  shell         Open shell in development container"
    echo "  test          Run tests"
    echo "  format        Format code"
    echo "  clean         Clean development environment"
    echo "  help          Show this help message"
    echo ""
}

# Main setup function
main_setup() {
    log_info "Setting up QNTI Trading System development environment..."
    
    check_prerequisites
    create_dev_compose
    setup_dev_environment
    create_dev_directories
    install_dev_dependencies
    setup_pre_commit
    create_dev_scripts
    build_dev_images
    start_dev_environment
    show_dev_status
    
    log_success "Development environment setup completed!"
}

# Parse command line arguments
case "${1:-setup}" in
    "setup")
        main_setup
        ;;
    "build")
        build_dev_images
        ;;
    "start")
        start_dev_environment
        ;;
    "stop")
        cd "$PROJECT_DIR"
        docker-compose -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" down
        ;;
    "status")
        show_dev_status
        ;;
    "logs")
        cd "$PROJECT_DIR"
        docker-compose -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" logs -f
        ;;
    "shell")
        cd "$PROJECT_DIR"
        docker-compose -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" exec qnti-app /bin/bash
        ;;
    "test")
        cd "$PROJECT_DIR"
        ./run_tests.sh
        ;;
    "format")
        cd "$PROJECT_DIR"
        ./format_code.sh
        ;;
    "clean")
        cd "$PROJECT_DIR"
        docker-compose -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" down -v
        docker system prune -f
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