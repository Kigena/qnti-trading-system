# QNTI Trading System - Docker Deployment Guide

This guide provides comprehensive instructions for deploying the QNTI Trading System using Docker containers.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Architecture](#architecture)
5. [Configuration](#configuration)
6. [Deployment](#deployment)
7. [Development Setup](#development-setup)
8. [Monitoring](#monitoring)
9. [Security](#security)
10. [Backup and Recovery](#backup-and-recovery)
11. [Troubleshooting](#troubleshooting)
12. [Scaling](#scaling)

## Overview

The QNTI Trading System is containerized using Docker with the following components:

- **QNTI Application**: Flask-based trading system with WebSocket support
- **PostgreSQL**: Primary database for trading data
- **Redis**: Cache and session store
- **Nginx**: Reverse proxy with SSL termination
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Visualization and dashboards

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: Minimum 50GB free space
- **CPU**: 4+ cores recommended

### Software Requirements

- Docker 20.10+
- Docker Compose 1.29+
- Git
- OpenSSL (for SSL certificates)
- curl or wget

### Installation

#### Ubuntu/Debian
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.15.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Log out and back in for group changes to take effect
```

#### macOS
```bash
# Install Docker Desktop
brew install --cask docker

# Install Docker Compose (usually included with Docker Desktop)
brew install docker-compose
```

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd qnti-trading-system
```

### 2. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### 3. Deploy the System
```bash
# Make deployment script executable
chmod +x docker/scripts/deploy.sh

# Run deployment
./docker/scripts/deploy.sh
```

### 4. Access the System
- **Main Application**: https://localhost
- **Monitoring**: http://localhost:9090 (Prometheus)
- **Dashboards**: http://localhost:3000 (Grafana)

## Architecture

### Container Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network                           │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   Nginx     │    │    QNTI     │    │ PostgreSQL  │    │
│  │   (Proxy)   │◄───┤     App     │◄───┤ (Database)  │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│         │                   │                              │
│         │                   ▼                              │
│         │            ┌─────────────┐                       │
│         │            │    Redis    │                       │
│         │            │   (Cache)   │                       │
│         │            └─────────────┘                       │
│         │                                                  │
│         ▼                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   Client    │    │ Prometheus  │    │   Grafana   │    │
│  │  (Browser)  │    │(Monitoring) │    │(Dashboard)  │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Client Request** → Nginx → QNTI App
2. **Database Operations** → QNTI App → PostgreSQL
3. **Cache Operations** → QNTI App → Redis
4. **Metrics Collection** → Prometheus → All Services
5. **Visualization** → Grafana → Prometheus

## Configuration

### Environment Variables

#### Application Settings
```bash
# Application environment
QNTI_ENV=production
FLASK_ENV=production
QNTI_DEBUG=false
QNTI_LOG_LEVEL=INFO

# Performance settings
QNTI_WORKERS=4
QNTI_MAX_REQUESTS=1000

# Security
SECRET_KEY=your-secret-key-here
SECURITY_PASSWORD_SALT=your-salt-here
```

#### Database Configuration
```bash
# PostgreSQL settings
QNTI_DB_HOST=qnti-postgres
QNTI_DB_PORT=5432
QNTI_DB_NAME=qnti_trading
QNTI_DB_USER=qnti_user
QNTI_DB_PASSWORD=secure-password-here
QNTI_DB_SCHEMA=qnti
```

#### Redis Configuration
```bash
# Redis settings
QNTI_REDIS_HOST=qnti-redis
QNTI_REDIS_PORT=6379
QNTI_REDIS_PASSWORD=secure-redis-password
```

#### MT5 Integration (Optional)
```bash
# MT5 configuration
QNTI_MT5_ENABLED=false
QNTI_MT5_LOGIN=your-mt5-login
QNTI_MT5_PASSWORD=your-mt5-password
QNTI_MT5_SERVER=your-mt5-server
```

#### API Keys (Optional)
```bash
# External API keys
OPENAI_API_KEY=your-openai-key
TELEGRAM_BOT_TOKEN=your-telegram-token
```

### SSL/TLS Configuration

#### Self-Signed Certificates (Development)
```bash
# Generate self-signed certificates
mkdir -p docker/ssl
openssl req -x509 -newkey rsa:4096 -keyout docker/ssl/key.pem -out docker/ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

#### Production Certificates
```bash
# Using Let's Encrypt
certbot certonly --standalone -d your-domain.com
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem docker/ssl/cert.pem
cp /etc/letsencrypt/live/your-domain.com/privkey.pem docker/ssl/key.pem
```

## Deployment

### Production Deployment

#### 1. Prepare Environment
```bash
# Create production environment
cp .env.example .env
nano .env  # Configure production settings
```

#### 2. Deploy Services
```bash
# Deploy with security configurations
./docker/scripts/deploy.sh

# Or manually:
docker-compose -f docker-compose.yml -f docker/security/docker-compose.security.yml up -d
```

#### 3. Verify Deployment
```bash
# Check service status
docker-compose ps

# Check logs
docker-compose logs -f qnti-app

# Test health endpoints
curl -k https://localhost/health
```

### Staging Deployment

#### 1. Configure Staging Environment
```bash
# Use staging environment
cp .env.example .env.staging
# Edit staging-specific settings
```

#### 2. Deploy to Staging
```bash
# Deploy with staging configuration
COMPOSE_FILE=docker-compose.yml:docker-compose.staging.yml ./docker/scripts/deploy.sh
```

### Rolling Updates

#### 1. Update Application
```bash
# Build new image
docker build -t qnti-app:latest .

# Update services one by one
docker-compose up -d --no-deps qnti-app
```

#### 2. Database Migrations
```bash
# Run migrations
docker-compose exec qnti-app python -c "
from db_migration.database_config import get_database_manager
db_manager = get_database_manager()
# Add migration logic here
"
```

## Development Setup

### 1. Development Environment
```bash
# Setup development environment
./docker/scripts/dev-setup.sh

# Or manually:
cp .env.development .env
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### 2. Development Services
- **Application**: http://localhost:5001
- **Jupyter Notebook**: http://localhost:8888
- **Database**: localhost:5433
- **Redis**: localhost:6380

### 3. Development Workflow
```bash
# Run tests
./run_tests.sh

# Format code
./format_code.sh

# Access development shell
docker-compose exec qnti-app /bin/bash
```

## Monitoring

### Prometheus Configuration

#### Metrics Collection
- Application metrics: `/metrics`
- Business metrics: `/metrics/business`
- Trading metrics: `/metrics/trading`

#### Custom Metrics
```python
from prometheus_client import Counter, Histogram, Gauge

# Example custom metrics
trade_counter = Counter('qnti_trades_total', 'Total trades processed')
response_time = Histogram('qnti_response_time_seconds', 'Response time')
active_positions = Gauge('qnti_active_positions', 'Number of active positions')
```

### Grafana Dashboards

#### Access Grafana
- **URL**: http://localhost:3000
- **Username**: admin
- **Password**: admin (change on first login)

#### Pre-configured Dashboards
- **QNTI Overview**: System health and trading metrics
- **Infrastructure**: Container and resource metrics
- **Trading Performance**: Trading-specific KPIs

### Alerting

#### Prometheus Alerts
```yaml
# Example alert rules
groups:
  - name: qnti_alerts
    rules:
      - alert: QNTIAppDown
        expr: up{job="qnti-app"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "QNTI application is down"
```

## Security

### Container Security

#### 1. Security Configurations
```bash
# Apply security settings
docker-compose -f docker-compose.yml -f docker/security/docker-compose.security.yml up -d
```

#### 2. Security Features
- **No new privileges**: Prevents privilege escalation
- **Read-only root filesystems**: Where possible
- **Dropped capabilities**: Minimal required capabilities
- **AppArmor/SELinux**: Additional access controls
- **Seccomp profiles**: Syscall filtering

### Network Security

#### 1. Network Isolation
```yaml
# Custom network configuration
networks:
  qnti-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

#### 2. Firewall Rules
```bash
# Example iptables rules
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -j DROP
```

### Data Security

#### 1. Database Security
- Encrypted connections
- Strong passwords
- Limited user privileges
- Regular security updates

#### 2. Redis Security
- Authentication enabled
- Dangerous commands disabled
- Network restrictions

## Backup and Recovery

### Automated Backups

#### 1. Configure Backup
```bash
# Set backup schedule
export BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
export BACKUP_RETENTION_DAYS=7
```

#### 2. Run Backup
```bash
# Manual backup
./docker/scripts/backup.sh

# Automated backup (cron)
0 2 * * * /path/to/qnti-trading-system/docker/scripts/backup.sh
```

### Backup Components

#### 1. Database Backup
```bash
# PostgreSQL backup
./docker/scripts/backup.sh postgres

# Redis backup
./docker/scripts/backup.sh redis
```

#### 2. Application Data Backup
```bash
# Application data
./docker/scripts/backup.sh app-data

# Configuration files
./docker/scripts/backup.sh config
```

### Recovery Procedures

#### 1. Database Recovery
```bash
# Restore PostgreSQL
./docker/scripts/backup.sh restore-db /path/to/backup.sql.gz

# Restore Redis
./docker/scripts/backup.sh restore-redis /path/to/backup.rdb.gz
```

#### 2. Disaster Recovery
```bash
# Full system restore
1. Restore configuration files
2. Restore database
3. Restore application data
4. Restart services
```

## Troubleshooting

### Common Issues

#### 1. Container Won't Start
```bash
# Check logs
docker-compose logs qnti-app

# Check resource usage
docker stats

# Verify configuration
docker-compose config
```

#### 2. Database Connection Issues
```bash
# Test database connectivity
docker-compose exec qnti-app python -c "
from db_migration.database_config import get_database_manager
db_manager = get_database_manager()
print(db_manager.test_connection())
"
```

#### 3. Memory Issues
```bash
# Check memory usage
docker stats --no-stream

# Adjust memory limits
# Edit docker-compose.yml deploy.resources.limits.memory
```

### Debug Mode

#### 1. Enable Debug Mode
```bash
# Set debug environment
export QNTI_DEBUG=true
export FLASK_DEBUG=1

# Restart with debug
docker-compose restart qnti-app
```

#### 2. Access Debug Information
```bash
# Application logs
docker-compose logs -f qnti-app

# Database logs
docker-compose logs -f qnti-postgres

# System metrics
curl http://localhost:9090/metrics
```

## Scaling

### Horizontal Scaling

#### 1. Scale Application
```bash
# Scale app containers
docker-compose up -d --scale qnti-app=3

# Add load balancer configuration
# Update nginx upstream block
```

#### 2. Database Scaling
```bash
# PostgreSQL read replicas
# Add replica configuration to docker-compose.yml
```

### Vertical Scaling

#### 1. Resource Limits
```yaml
# Increase container resources
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      cpus: '2.0'
      memory: 4G
```

#### 2. Performance Tuning
```bash
# PostgreSQL tuning
# Edit docker/postgres/postgresql.conf

# Redis tuning
# Edit docker/redis/redis.conf
```

### Auto-scaling

#### 1. Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml qnti
```

#### 2. Kubernetes
```bash
# Convert to Kubernetes
kompose convert

# Deploy to Kubernetes
kubectl apply -f qnti-deployment.yaml
```

## Best Practices

### 1. Security Best Practices
- Use strong passwords and secrets
- Regularly update base images
- Enable security scanning
- Implement proper logging
- Use least privilege principles

### 2. Performance Best Practices
- Optimize Docker images
- Use appropriate resource limits
- Implement caching strategies
- Monitor performance metrics
- Regular performance testing

### 3. Operational Best Practices
- Implement health checks
- Use structured logging
- Regular backups
- Documentation updates
- Monitoring and alerting

## Support

For issues and questions:
- Check the troubleshooting section
- Review container logs
- Consult the monitoring dashboards
- Contact the development team

---

**Note**: This deployment guide covers the essential aspects of containerizing and deploying the QNTI Trading System. For production deployments, additional considerations such as high availability, disaster recovery, and compliance requirements may apply.