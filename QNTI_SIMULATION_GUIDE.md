# QNTI Trading System - User Path Simulation Guide

## Overview

This comprehensive simulation suite tests all user paths and interactions in the QNTI trading system. It includes backend API testing, frontend browser automation, performance testing, and security testing.

## Components

### 1. Backend API Simulation (`qnti_automated_user_path_simulation.py`)
- Tests all REST API endpoints
- Simulates realistic user workflows
- Performs load testing with concurrent users
- Tests WebSocket connections
- Validates error handling

### 2. Frontend Browser Automation (`qnti_browser_automation.py`)
- Tests UI interactions using Selenium WebDriver
- Validates responsive design across screen sizes
- Tests form validation and user workflows
- Captures screenshots for debugging
- Tests accessibility features

### 3. Master Test Runner (`run_qnti_full_simulation.py`)
- Orchestrates all test suites
- Generates comprehensive reports
- Includes performance and security testing
- Provides detailed recommendations

## Prerequisites

### System Requirements
- Python 3.7 or higher
- Chrome browser (for frontend testing)
- QNTI server running on `http://localhost:5000`

### Required Python Packages
```bash
pip install requests websocket-client Pillow selenium
```

### Chrome WebDriver
The system will attempt to use Chrome WebDriver. Ensure Chrome is installed and WebDriver is available in your PATH.

## Quick Start

### Option 1: Using Batch/Shell Scripts (Recommended)

**Windows:**
```cmd
run_simulation.bat
```

**Linux/Mac:**
```bash
./run_simulation.sh
```

### Option 2: Direct Python Execution

**Full Simulation:**
```bash
python run_qnti_full_simulation.py
```

**Backend Only:**
```bash
python run_qnti_full_simulation.py --backend-only
```

**Frontend Only:**
```bash
python run_qnti_full_simulation.py --frontend-only
```

**Headless Mode:**
```bash
python run_qnti_full_simulation.py --headless
```

## Test Categories

### 1. Dashboard Access Tests
- Main dashboard loading
- Navigation menu functionality
- Page responsiveness
- Status indicator validation

### 2. System Health Monitoring
- Health check endpoints
- System status validation
- Vision system status
- Real-time connection testing

### 3. Trade Management Tests
- Active trades retrieval
- Trade history queries
- Different timeframe testing
- Trade data validation

### 4. EA Management Tests
- EA registration and control
- Platform scanning
- Auto-detection functionality
- Performance recalculation
- EA history retrieval

### 5. AI Vision Analysis Workflow
- Image upload functionality
- Analysis execution
- Result retrieval
- Multi-symbol/timeframe testing
- Error handling validation

### 6. Frontend UI Tests
- Form validation
- Dropdown interactions
- Trading actions panel
- Responsive design testing
- Accessibility compliance

### 7. Performance Tests
- Light load (3 users, 15s)
- Medium load (5 users, 30s)
- Heavy load (10 users, 20s)
- Response time analysis
- Concurrent user simulation

### 8. Security Tests
- SQL injection prevention
- XSS attack prevention
- Unauthorized access testing
- Input validation
- Security score calculation

## Configuration Options

### Command Line Arguments

```bash
python run_qnti_full_simulation.py [OPTIONS]

Options:
  --url URL              QNTI server URL (default: http://localhost:5000)
  --headless            Run browser tests in headless mode
  --backend-only        Run only backend tests
  --frontend-only       Run only frontend tests
  --performance-only    Run only performance tests
  --security-only       Run only security tests
```

### Environment Variables

```bash
export QNTI_SERVER_URL="http://localhost:5000"
export QNTI_HEADLESS="true"
export QNTI_LOG_LEVEL="INFO"
```

## Test Results and Reports

### Generated Files

1. **Comprehensive Report:** `qnti_full_simulation_report_YYYYMMDD_HHMMSS.json`
2. **Backend Results:** `qnti_simulation_results_YYYYMMDD_HHMMSS.json`
3. **Frontend Report:** `qnti_browser_test_report_YYYYMMDD_HHMMSS.json`
4. **Log Files:** `qnti_full_simulation.log`, `qnti_user_simulation.log`, `qnti_browser_automation.log`
5. **Screenshots:** `screenshots/` directory with test screenshots

### Report Structure

```json
{
  "simulation_overview": {
    "start_time": "2024-01-01T10:00:00",
    "duration_seconds": 120.5,
    "base_url": "http://localhost:5000"
  },
  "system_health": {
    "overall_score": 95.2,
    "backend_success_rate": 98.5,
    "frontend_success_rate": 92.0,
    "server_accessible": true
  },
  "test_results": {
    "backend": {...},
    "frontend": {...},
    "performance": {...},
    "security": {...}
  },
  "recommendations": [
    "All tests passed successfully - system is performing well"
  ]
}
```

## Interpreting Results

### Success Rates
- **90-100%**: Excellent performance
- **80-89%**: Good performance with minor issues
- **70-79%**: Moderate performance, needs attention
- **Below 70%**: Poor performance, requires immediate attention

### Performance Metrics
- **Response Time < 1s**: Excellent
- **Response Time 1-3s**: Good
- **Response Time 3-5s**: Acceptable
- **Response Time > 5s**: Poor

### Security Score
- **100%**: All attacks blocked
- **90-99%**: Minor vulnerabilities
- **80-89%**: Moderate security concerns
- **Below 80%**: Significant security issues

## Troubleshooting

### Common Issues

1. **Server Not Accessible**
   - Ensure QNTI server is running
   - Check firewall settings
   - Verify URL is correct

2. **Chrome WebDriver Issues**
   - Install Chrome browser
   - Download ChromeDriver
   - Add to system PATH

3. **Permission Errors**
   - Run as administrator (Windows)
   - Use sudo if needed (Linux/Mac)
   - Check file permissions

4. **Module Import Errors**
   - Install required packages
   - Check Python path
   - Verify virtual environment

### Debug Mode

Enable debug logging:
```bash
python run_qnti_full_simulation.py --url http://localhost:5000 2>&1 | tee debug.log
```

### Selective Testing

Run specific test categories:
```bash
# Only backend API tests
python qnti_automated_user_path_simulation.py

# Only frontend browser tests  
python qnti_browser_automation.py

# Only performance tests
python run_qnti_full_simulation.py --performance-only
```

## Advanced Usage

### Custom Test Scenarios

Create custom test scenarios by modifying the simulation scripts:

```python
# Add custom test to backend simulation
def test_custom_workflow(self):
    results = []
    # Your custom test logic here
    return results

# Add to test suite
test_suite.append(('Custom Workflow', self.test_custom_workflow))
```

### Integration with CI/CD

Add to your CI/CD pipeline:

```yaml
# GitHub Actions example
- name: Run QNTI Simulation
  run: |
    python run_qnti_full_simulation.py --headless
    
- name: Upload Results
  uses: actions/upload-artifact@v2
  with:
    name: qnti-simulation-results
    path: qnti_*_report_*.json
```

### Monitoring and Alerting

Set up monitoring based on results:

```python
# Example monitoring script
def check_system_health():
    with open('latest_report.json') as f:
        report = json.load(f)
    
    health_score = report['system_health']['overall_score']
    
    if health_score < 80:
        send_alert(f"QNTI system health degraded: {health_score}%")
```

## Best Practices

1. **Regular Testing**: Run simulations regularly to catch issues early
2. **Baseline Metrics**: Establish baseline performance metrics
3. **Trend Analysis**: Monitor trends over time
4. **Environment Consistency**: Test in environments similar to production
5. **Test Data Management**: Use consistent test data for reproducible results

## Support and Maintenance

### Updating Tests
- Update test data regularly
- Add new test scenarios for new features
- Maintain browser compatibility
- Update dependencies regularly

### Performance Optimization
- Adjust concurrent user counts based on system capacity
- Optimize test execution order
- Use parallel testing where possible
- Monitor resource usage during tests

## Conclusion

This simulation suite provides comprehensive coverage of all QNTI user paths, ensuring system reliability and performance. Regular execution helps maintain system quality and identifies issues before they impact users.

For questions or issues, refer to the log files and generated reports for detailed diagnostic information. 