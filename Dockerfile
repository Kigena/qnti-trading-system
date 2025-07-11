# QNTI Trading System - Multi-Stage Docker Build
# Production-ready containerization with security and optimization

# ==============================================================================
# Stage 1: Base Python Environment
# ==============================================================================
FROM python:3.11-slim-bullseye as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r qnti && useradd -r -g qnti qnti

# Create application directory
WORKDIR /app

# ==============================================================================
# Stage 2: Dependencies Installation
# ==============================================================================
FROM base as dependencies

# Copy requirements files
COPY requirements.txt requirements_api.txt requirements_llm.txt requirements_simulation.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_api.txt && \
    pip install --no-cache-dir -r requirements_llm.txt && \
    pip install --no-cache-dir -r requirements_simulation.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    gunicorn \
    psycopg2-binary \
    redis \
    prometheus-client \
    structlog \
    uvloop \
    httptools

# ==============================================================================
# Stage 3: Application Build
# ==============================================================================
FROM dependencies as builder

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p \
    /app/logs \
    /app/qnti_data \
    /app/ea_profiles \
    /app/chart_uploads \
    /app/qnti_screenshots \
    /app/qnti_backups \
    /app/qnti_memory

# Set proper permissions
RUN chown -R qnti:qnti /app && \
    chmod -R 755 /app

# Remove development files and optimize
RUN find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + && \
    find /app -name "*.pytest_cache" -type d -exec rm -rf {} + && \
    find /app -name ".git" -type d -exec rm -rf {} + && \
    rm -rf /app/tests /app/docs /app/*.md

# ==============================================================================
# Stage 4: Production Image
# ==============================================================================
FROM python:3.11-slim-bullseye as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    DEBIAN_FRONTEND=noninteractive \
    QNTI_ENV=production

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    libffi7 \
    libssl1.1 \
    libxml2 \
    libxslt1.1 \
    zlib1g \
    libjpeg62-turbo \
    libpng16-16 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r qnti && useradd -r -g qnti qnti

# Copy Python packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Create application directory
WORKDIR /app

# Copy application from builder stage
COPY --from=builder --chown=qnti:qnti /app .

# Create runtime directories
RUN mkdir -p \
    /app/logs \
    /app/qnti_data \
    /app/ea_profiles \
    /app/chart_uploads \
    /app/qnti_screenshots \
    /app/qnti_backups \
    /app/qnti_memory \
    /var/log/qnti \
    && chown -R qnti:qnti /app /var/log/qnti

# Copy entrypoint script
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Switch to non-root user
USER qnti

# Expose ports
EXPOSE 5000

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--worker-class", "eventlet", "--worker-connections", "1000", "--timeout", "120", "--keepalive", "2", "--max-requests", "1000", "--max-requests-jitter", "50", "--preload", "qnti_main_system:app"]

# ==============================================================================
# Stage 5: Development Image
# ==============================================================================
FROM dependencies as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    black>=23.0.0 \
    flake8>=6.0.0 \
    mypy>=1.0.0 \
    ipython>=8.0.0 \
    jupyter>=1.0.0 \
    flask-debug-toolbar>=0.13.0

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p \
    /app/logs \
    /app/qnti_data \
    /app/ea_profiles \
    /app/chart_uploads \
    /app/qnti_screenshots \
    /app/qnti_backups \
    /app/qnti_memory

# Set proper permissions
RUN chown -R qnti:qnti /app && \
    chmod -R 755 /app

# Switch to non-root user
USER qnti

# Expose ports (including debug port)
EXPOSE 5000 5678

# Set development environment
ENV QNTI_ENV=development \
    FLASK_ENV=development \
    FLASK_DEBUG=1

# Development entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default development command
CMD ["python", "qnti_main_system.py", "--debug", "--host", "0.0.0.0", "--port", "5000"]

# ==============================================================================
# Metadata
# ==============================================================================
LABEL maintainer="QNTI Team" \
      version="1.0.0" \
      description="Quantum Nexus Trading Intelligence System" \
      org.opencontainers.image.title="QNTI Trading System" \
      org.opencontainers.image.description="Advanced AI-powered trading system with MT5 integration" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="QNTI Team" \
      org.opencontainers.image.licenses="MIT"