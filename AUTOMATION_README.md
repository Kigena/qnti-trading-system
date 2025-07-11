# QNTI Automation & Testing Suite

A comprehensive automation suite for testing the **Quantum Nexus Trading Intelligence (QNTI)** system, including EA Generation, stress testing, and performance benchmarking using Puppeteer, Selenium, and advanced load testing tools.

## 🚀 Quick Start

### 1. Setup Automation Environment
```bash
# Install automation dependencies
python setup_automation.py

# This installs:
# - Puppeteer (pyppeteer) + Chromium
# - Selenium WebDriver 
# - Load testing tools
# - Creates configuration files
```

### 2. Run Quick Test (5 minutes)
```bash
# Quick functional test
python run_full_simulation.py --quick

# Or use the convenience script
python quick_test.py
```

### 3. Run Full Simulation (30+ minutes)
```bash
# Comprehensive testing suite
python run_full_simulation.py

# With custom parameters
python run_full_simulation.py --duration 3600 --users 20
```

## 📋 Available Testing Suites

### 1. **Functional Testing** (`qnti_automation_suite.py`)
- **UI Automation**: Puppeteer + Selenium browser testing
- **API Testing**: Comprehensive endpoint validation  
- **EA Generation Workflows**: End-to-end EA creation testing
- **AI Integration**: Tests LLM integration and recommendations

**Features:**
- ✅ Dashboard navigation testing
- ✅ EA Generation form automation
- ✅ Workflow progress monitoring
- ✅ AI insights validation
- ✅ Performance metrics collection

### 2. **Stress Testing** (`qnti_stress_testing.py`)
- **High Load Simulation**: Up to 50+ concurrent users
- **EA Generation Stress**: Multiple simultaneous EA creations
- **Resource Monitoring**: CPU, memory, response time tracking
- **Bottleneck Identification**: Automated performance analysis

**Features:**
- 🔥 Concurrent user simulation
- 🔥 API endpoint stress testing
- 🔥 System resource monitoring
- 🔥 Performance grading (A-F scale)
- 🔥 Scalability analysis

### 3. **Full System Simulation** (`run_full_simulation.py`)
- **Orchestrated Testing**: Combines all test suites
- **System Health Checks**: Automatic QNTI startup
- **Comprehensive Reporting**: Unified performance assessment
- **CI/CD Integration**: GitHub Actions workflow included

## 🛠️ Usage Examples

### Basic Usage

```bash
# Quick 5-minute test
python run_full_simulation.py --quick

# Functional tests only (no stress testing)
python run_full_simulation.py --functional-only

# Stress tests only
python run_full_simulation.py --stress-only

# Custom duration and user count
python run_full_simulation.py --duration 1800 --users 25
```

### Advanced Usage

```bash
# Comprehensive automation suite
python qnti_automation_suite.py --url http://localhost:5000 --duration 3600 --users 10

# Stress testing with custom parameters
python qnti_stress_testing.py --duration 1800 --users 50 --rps 100 --ea-concurrent 15

# Load testing with specific scenarios
python load_test.py  # Uses predefined load test configuration
```

### Docker Usage

```bash
# Run automation in Docker
docker-compose -f docker-compose.automation.yml up

# Run load tests specifically
docker-compose -f docker-compose.automation.yml --profile load-testing up
```

## 📊 Test Reports & Results

### Generated Files
- **`qnti_simulation_report_*.json`** - Detailed functional test results
- **`qnti_stress_test_report_*.json`** - Stress test performance analysis  
- **`qnti_full_simulation_report_*.json`** - Comprehensive system assessment
- **`*.png`** - Screenshots from browser automation
- **`qnti_automation.log`** - Detailed execution logs

### Sample Report Structure
```json
{
  "simulation_summary": {
    "total_tests": 150,
    "successful_tests": 142,
    "success_rate": 0.947,
    "avg_response_time": 1.234,
    "duration": 1800
  },
  "performance_statistics": {
    "request_stats": {...},
    "response_time_stats": {...},
    "resource_stats": {...}
  },
  "ea_generation_results": [...],
  "recommendations": [...]
}
```

## 🎯 Test Scenarios

### 1. **Normal Operation Testing**
- Standard user workflows
- Typical EA generation patterns
- Expected load levels
- Basic error handling

### 2. **High Load Testing**  
- 50+ concurrent users
- 100+ requests/second
- Multiple simultaneous EA generations
- System resource limits

### 3. **Error Injection Testing**
- Network timeout simulation
- Invalid input handling
- Server error responses
- Recovery mechanisms

### 4. **Endurance Testing**
- Extended operation periods (30+ minutes)
- Memory leak detection
- Performance degradation analysis
- Stability assessment

## 🔧 Configuration

### Automation Config (`automation_config.json`)
```json
{
  "qnti_url": "http://localhost:5000",
  "automation_settings": {
    "browser_headless": true,
    "timeout_seconds": 30,
    "screenshot_on_error": true,
    "max_retries": 3
  },
  "test_scenarios": {
    "quick_test": {
      "duration": 60,
      "concurrent_users": 2
    },
    "load_test": {
      "duration": 1800,
      "concurrent_users": 20
    }
  }
}
```

### Environment Variables
```bash
export QNTI_URL="http://localhost:5000"
export PUPPETEER_HEADLESS="true"
export AUTOMATION_TIMEOUT="30"
export MAX_CONCURRENT_USERS="50"
```

## 📈 Performance Benchmarks

### Response Time Targets
- **Health Check**: < 100ms
- **EA Indicators**: < 500ms  
- **EA Creation**: < 2s
- **Workflow Status**: < 200ms
- **Dashboard Load**: < 1s

### Throughput Targets
- **Standard Load**: 50 requests/second
- **Peak Load**: 100+ requests/second
- **EA Generations**: 5+ concurrent workflows
- **Success Rate**: > 95%

### Resource Limits
- **CPU Usage**: < 80% average
- **Memory Usage**: < 2GB
- **Response Time P95**: < 5s
- **Error Rate**: < 5%

## 🚨 Troubleshooting

### Common Issues

**1. Browser Automation Fails**
```bash
# Install/reinstall Puppeteer
python -c "import pyppeteer; pyppeteer.chromium_downloader.download_chromium()"

# Check Chrome/Chromium installation
which chromium-browser
```

**2. QNTI System Not Starting**
```bash
# Check if port is in use
netstat -tulpn | grep 5000

# Start QNTI manually
python qnti_main_system.py

# Check logs
tail -f qnti_main.log
```

**3. High Failure Rates**
```bash
# Check system resources
htop
free -h

# Reduce concurrent users
python run_full_simulation.py --users 10

# Run with debugging
python run_full_simulation.py --quick 2>&1 | tee debug.log
```

**4. Dependencies Missing**
```bash
# Reinstall automation dependencies
pip install -r requirements.txt
python setup_automation.py

# Check Python version (requires 3.9+)
python --version
```

## 🔄 CI/CD Integration

### GitHub Actions
The automation suite includes a complete GitHub Actions workflow (`.github/workflows/qnti_automation.yml`) that:

- ✅ Runs on every push/PR
- ✅ Tests multiple Python versions (3.9, 3.10, 3.11)
- ✅ Automatically starts QNTI system
- ✅ Executes full test suite
- ✅ Uploads test results and screenshots
- ✅ Fails build if success rate < 95%

### Scheduled Testing
- **Daily Tests**: Automated testing at 2 AM UTC
- **Load Tests**: Weekly comprehensive stress testing
- **Performance Monitoring**: Continuous benchmark tracking

## 📝 Customization

### Adding New Test Scenarios

1. **Create Test Function**
```python
async def custom_test_scenario(self):
    """Custom test scenario"""
    # Your test logic here
    pass
```

2. **Add to Test Suite**
```python
# In QNTIAutomationSuite class
async def run_comprehensive_simulation(self):
    # ... existing tests ...
    await self.custom_test_scenario()
```

3. **Configure Parameters**
```python
# Add to SimulationConfig
custom_test_duration: int = 300
custom_test_parameters: Dict[str, Any] = {}
```

### Custom EA Generation Tests

```python
# Custom EA configuration for testing
custom_ea_config = {
    'ea_name': 'Custom Test Strategy',
    'description': 'Custom test configuration',
    'symbols': ['EURUSD', 'GBPUSD'],
    'timeframes': ['H1', 'H4'],
    'indicators': [
        {'name': 'Custom_Indicator', 'params': {'period': 20}}
    ],
    'method': 'genetic_algorithm'
}
```

## 🏆 Best Practices

### Running Tests
- ✅ Start with quick tests before full simulation
- ✅ Monitor system resources during testing
- ✅ Review logs for any warnings or errors
- ✅ Compare results with previous benchmarks
- ✅ Run tests in isolated environment

### Performance Optimization
- ✅ Use headless browser mode for faster execution
- ✅ Limit concurrent operations based on system capacity
- ✅ Implement proper cleanup after test completion
- ✅ Cache test data when possible
- ✅ Use appropriate timeouts for different operations

### Test Data Management
- ✅ Archive test reports regularly
- ✅ Compare performance trends over time
- ✅ Backup screenshots and logs
- ✅ Document any configuration changes
- ✅ Maintain test environment consistency

## 📞 Support & Contributing

### Getting Help
- Check logs in `qnti_automation.log`
- Review generated JSON reports for details
- Use `--quick` flag for faster debugging
- Enable verbose logging for troubleshooting

### Contributing
- Follow existing code patterns
- Add comprehensive error handling
- Include logging for debugging
- Document any new test scenarios
- Update this README for new features

---

**🎯 Ready to test your QNTI system? Start with:**
```bash
python setup_automation.py
python run_full_simulation.py --quick
```

The automation suite will handle the rest! 🚀