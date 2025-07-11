#!/usr/bin/env python3
"""
QNTI Automation Setup - Install and configure automation tools
Sets up Puppeteer, Selenium, and other dependencies for comprehensive testing
"""

import subprocess
import sys
import os
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('QNTI_AUTOMATION_SETUP')

def run_command(command, description=""):
    """Run shell command with error handling"""
    logger.info(f"Running: {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False, e.stderr

def install_python_packages():
    """Install required Python packages"""
    packages = [
        'pyppeteer',
        'selenium', 
        'requests',
        'aiohttp',
        'beautifulsoup4',
        'Pillow',
        'numpy',
        'pandas'
    ]
    
    logger.info("📦 Installing Python packages...")
    
    for package in packages:
        success, output = run_command(
            f"{sys.executable} -m pip install {package}",
            f"Installing {package}"
        )
        if not success:
            logger.warning(f"Failed to install {package}: {output}")

def setup_puppeteer():
    """Setup Puppeteer browser automation"""
    logger.info("🤖 Setting up Puppeteer...")
    
    # Install pyppeteer and download Chromium
    success, output = run_command(
        f"{sys.executable} -c \"import pyppeteer; pyppeteer.chromium_downloader.download_chromium()\"",
        "Downloading Chromium for Puppeteer"
    )
    
    if success:
        logger.info("✅ Puppeteer setup complete")
    else:
        logger.warning("⚠️ Puppeteer setup failed")

def setup_selenium():
    """Setup Selenium WebDriver"""
    logger.info("🕷️ Setting up Selenium...")
    
    # Try to install webdriver-manager for automatic driver management
    success, output = run_command(
        f"{sys.executable} -m pip install webdriver-manager",
        "Installing webdriver-manager"
    )
    
    if success:
        # Download Chrome driver
        run_command(
            f"{sys.executable} -c \"from webdriver_manager.chrome import ChromeDriverManager; ChromeDriverManager().install()\"",
            "Installing Chrome WebDriver"
        )
        logger.info("✅ Selenium setup complete")
    else:
        logger.warning("⚠️ Selenium automatic setup failed - manual driver setup may be required")

def create_automation_config():
    """Create automation configuration file"""
    config = {
        "qnti_url": "http://localhost:5000",
        "automation_settings": {
            "browser_headless": True,
            "timeout_seconds": 30,
            "screenshot_on_error": True,
            "max_retries": 3
        },
        "test_scenarios": {
            "quick_test": {
                "duration": 60,
                "concurrent_users": 2,
                "ea_strategies": ["Trend Following Strategy", "Mean Reversion Strategy"]
            },
            "standard_test": {
                "duration": 300,
                "concurrent_users": 5,
                "ea_strategies": [
                    "Trend Following Strategy",
                    "Mean Reversion Strategy", 
                    "Momentum Strategy",
                    "Scalping Strategy"
                ]
            },
            "load_test": {
                "duration": 1800,
                "concurrent_users": 20,
                "stress_testing": True
            }
        },
        "performance_thresholds": {
            "max_response_time": 5.0,
            "min_success_rate": 0.95,
            "max_error_rate": 0.05
        }
    }
    
    config_file = Path("automation_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"📋 Created automation config: {config_file}")

def create_automation_scripts():
    """Create convenient automation scripts"""
    
    # Quick test script
    quick_test_script = """#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qnti_automation_suite import QNTIAutomationSuite, SimulationConfig

async def quick_test():
    config = SimulationConfig(
        qnti_url="http://localhost:5000",
        simulation_duration=60,
        max_concurrent_users=2,
        ea_strategies_to_test=["Quick Test Strategy"],
    )
    
    suite = QNTIAutomationSuite(config)
    results = await suite.run_comprehensive_simulation()
    
    print(f"Quick test completed: {results.success_rate:.1%} success rate")
    return results

if __name__ == "__main__":
    asyncio.run(quick_test())
"""
    
    with open("quick_test.py", 'w') as f:
        f.write(quick_test_script)
    
    # Load test script
    load_test_script = """#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qnti_automation_suite import QNTIAutomationSuite, SimulationConfig

async def load_test():
    config = SimulationConfig(
        qnti_url="http://localhost:5000",
        simulation_duration=1800,  # 30 minutes
        max_concurrent_users=20,
    )
    
    suite = QNTIAutomationSuite(config)
    results = await suite.run_comprehensive_simulation()
    
    print(f"Load test completed: {results.success_rate:.1%} success rate")
    print(f"Average response time: {results.avg_response_time:.3f}s")
    return results

if __name__ == "__main__":
    asyncio.run(load_test())
"""
    
    with open("load_test.py", 'w') as f:
        f.write(load_test_script)
    
    # Make scripts executable
    if os.name != 'nt':  # Unix/Linux/Mac
        os.chmod("quick_test.py", 0o755)
        os.chmod("load_test.py", 0o755)
    
    logger.info("📜 Created automation convenience scripts")

def create_github_actions_workflow():
    """Create GitHub Actions workflow for CI/CD"""
    workflow_dir = Path(".github/workflows")
    workflow_dir.mkdir(parents=True, exist_ok=True)
    
    workflow_content = """name: QNTI Automation Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run automated tests daily at 2 AM UTC
    - cron: '0 2 * * *'

jobs:
  automation-tests:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python setup_automation.py
        pip install -r requirements.txt
    
    - name: Start QNTI System
      run: |
        python qnti_main_system.py &
        sleep 30  # Wait for system to start
    
    - name: Run Quick Automation Tests
      run: |
        python quick_test.py
    
    - name: Upload Test Results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.python-version }}
        path: |
          qnti_simulation_report_*.json
          *.png
          qnti_automation.log
    
    - name: Check Test Results
      run: |
        # Parse test results and fail if success rate < 95%
        python -c "
        import json, glob, sys
        reports = glob.glob('qnti_simulation_report_*.json')
        if reports:
            with open(reports[-1]) as f:
                data = json.load(f)
            success_rate = data['simulation_summary']['success_rate']
            if success_rate < 0.95:
                print(f'Test failure: Success rate {success_rate:.1%} < 95%')
                sys.exit(1)
            print(f'Tests passed: Success rate {success_rate:.1%}')
        else:
            print('No test reports found')
            sys.exit(1)
        "

  load-tests:
    runs-on: ubuntu-latest
    # Only run load tests on main branch
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python setup_automation.py
        pip install -r requirements.txt
    
    - name: Start QNTI System
      run: |
        python qnti_main_system.py &
        sleep 30
    
    - name: Run Load Tests
      run: |
        timeout 3600 python load_test.py  # 1 hour timeout
    
    - name: Upload Load Test Results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: load-test-results
        path: |
          qnti_simulation_report_*.json
          *.png
          qnti_automation.log
"""
    
    workflow_file = workflow_dir / "qnti_automation.yml"
    with open(workflow_file, 'w') as f:
        f.write(workflow_content)
    
    logger.info(f"🔄 Created GitHub Actions workflow: {workflow_file}")

def create_docker_automation():
    """Create Docker setup for automation testing"""
    dockerfile_content = """FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    wget \\
    gnupg \\
    unzip \\
    curl \\
    chromium \\
    chromium-driver \\
    && rm -rf /var/lib/apt/lists/*

# Set up Chrome for Puppeteer
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true \\
    PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
COPY setup_automation.py .
RUN python setup_automation.py

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Default command
CMD ["python", "qnti_automation_suite.py", "--headless", "--url", "http://localhost:5000"]
"""
    
    with open("Dockerfile.automation", 'w') as f:
        f.write(dockerfile_content)
    
    # Docker Compose for automation testing
    docker_compose_content = """version: '3.8'

services:
  qnti-system:
    build: .
    ports:
      - "5000:5000"
    environment:
      - PYTHONPATH=/app
    volumes:
      - ./qnti_data:/app/qnti_data
    command: python qnti_main_system.py
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/system/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  automation-tests:
    build:
      context: .
      dockerfile: Dockerfile.automation
    depends_on:
      qnti-system:
        condition: service_healthy
    environment:
      - QNTI_URL=http://qnti-system:5000
    volumes:
      - ./automation_results:/app/automation_results
    command: python qnti_automation_suite.py --url http://qnti-system:5000 --headless

  load-tests:
    build:
      context: .
      dockerfile: Dockerfile.automation
    depends_on:
      qnti-system:
        condition: service_healthy
    environment:
      - QNTI_URL=http://qnti-system:5000
    volumes:
      - ./load_test_results:/app/load_test_results
    command: python load_test.py
    profiles:
      - load-testing
"""
    
    with open("docker-compose.automation.yml", 'w') as f:
        f.write(docker_compose_content)
    
    logger.info("🐳 Created Docker automation setup")

def main():
    """Main setup function"""
    logger.info("🚀 Setting up QNTI Automation Suite...")
    
    # Install Python packages
    install_python_packages()
    
    # Setup browser automation
    setup_puppeteer()
    setup_selenium()
    
    # Create configuration and scripts
    create_automation_config()
    create_automation_scripts()
    
    # Create CI/CD workflows
    create_github_actions_workflow()
    create_docker_automation()
    
    # Create directories for results
    os.makedirs("automation_results", exist_ok=True)
    os.makedirs("load_test_results", exist_ok=True)
    os.makedirs("screenshots", exist_ok=True)
    
    logger.info("✅ QNTI Automation Suite setup complete!")
    
    print("\n" + "="*60)
    print("🎯 QNTI AUTOMATION SUITE READY")
    print("="*60)
    print("📋 Available commands:")
    print("  python qnti_automation_suite.py --quick      # Quick test")
    print("  python quick_test.py                         # Quick test script")
    print("  python load_test.py                          # Load test script")
    print("  python qnti_automation_suite.py --help       # Full options")
    print("\n🐳 Docker commands:")
    print("  docker-compose -f docker-compose.automation.yml up")
    print("  docker-compose -f docker-compose.automation.yml --profile load-testing up")
    print("\n📊 Test results will be saved to:")
    print("  • qnti_simulation_report_*.json")
    print("  • automation_results/")
    print("  • Screenshots: *.png")
    print("="*60)

if __name__ == "__main__":
    main()