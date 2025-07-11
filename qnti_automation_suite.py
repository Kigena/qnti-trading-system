#!/usr/bin/env python3
"""
QNTI Automation Suite - Comprehensive System Simulation
Uses Puppeteer, Selenium, and custom automation to simulate complete QNTI EA Generation workflows
"""

import asyncio
import logging
import time
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import subprocess
import requests
import concurrent.futures
from dataclasses import dataclass, field
import threading
import os

# Browser automation imports
try:
    from pyppeteer import launch
    from pyppeteer.page import Page
    PUPPETEER_AVAILABLE = True
except ImportError:
    PUPPETEER_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait, Select
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qnti_automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('QNTI_AUTOMATION')

@dataclass
class SimulationConfig:
    """Configuration for QNTI system simulation"""
    # System settings
    qnti_url: str = "http://localhost:5000"
    simulation_duration: int = 3600  # 1 hour
    max_concurrent_users: int = 5
    
    # EA Generation settings
    ea_strategies_to_test: List[str] = field(default_factory=lambda: [
        "Trend Following Strategy",
        "Mean Reversion Strategy", 
        "Momentum Strategy",
        "Scalping Strategy",
        "Breakout Strategy"
    ])
    
    indicators_pool: List[str] = field(default_factory=lambda: [
        "SMA", "EMA", "RSI", "MACD", "Bollinger Bands",
        "Stochastic", "ATR", "CCI", "Williams %R", "ROC"
    ])
    
    symbols_pool: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"
    ])
    
    timeframes_pool: List[str] = field(default_factory=lambda: [
        "M15", "H1", "H4", "D1"
    ])
    
    # Testing scenarios
    test_scenarios: List[str] = field(default_factory=lambda: [
        "normal_operation",
        "high_load",
        "error_injection",
        "network_delays",
        "resource_constraints"
    ])
    
    # Performance thresholds
    max_response_time: float = 5.0  # seconds
    min_success_rate: float = 0.95  # 95%
    max_error_rate: float = 0.05   # 5%

@dataclass
class SimulationResults:
    """Results from automation simulation"""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tests: int = 0
    successful_tests: int = 0
    failed_tests: int = 0
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = float('inf')
    errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    ea_generation_results: List[Dict] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.successful_tests / self.total_tests
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

class QNTIAPISimulator:
    """Simulate QNTI API interactions"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        
    async def test_system_health(self) -> Dict[str, Any]:
        """Test system health endpoint"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/api/system/health", timeout=10)
            response_time = time.time() - start_time
            
            return {
                'success': response.status_code == 200,
                'response_time': response_time,
                'data': response.json() if response.status_code == 200 else None,
                'error': None if response.status_code == 200 else f"HTTP {response.status_code}"
            }
        except Exception as e:
            return {
                'success': False,
                'response_time': time.time() - start_time,
                'data': None,
                'error': str(e)
            }
    
    async def test_ea_indicators(self) -> Dict[str, Any]:
        """Test EA indicators endpoint"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/api/ea/indicators", timeout=10)
            response_time = time.time() - start_time
            
            return {
                'success': response.status_code == 200,
                'response_time': response_time,
                'data': response.json() if response.status_code == 200 else None,
                'error': None if response.status_code == 200 else f"HTTP {response.status_code}"
            }
        except Exception as e:
            return {
                'success': False,
                'response_time': time.time() - start_time,
                'data': None,
                'error': str(e)
            }
    
    async def test_ea_workflow_creation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test EA workflow creation"""
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/ea/workflow/start",
                json=config,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            response_time = time.time() - start_time
            
            return {
                'success': response.status_code == 200,
                'response_time': response_time,
                'data': response.json() if response.status_code == 200 else None,
                'error': None if response.status_code == 200 else f"HTTP {response.status_code}"
            }
        except Exception as e:
            return {
                'success': False,
                'response_time': time.time() - start_time,
                'data': None,
                'error': str(e)
            }
    
    async def test_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Test workflow status monitoring"""
        try:
            start_time = time.time()
            response = self.session.get(
                f"{self.base_url}/api/ea/workflow/status/{workflow_id}",
                timeout=10
            )
            response_time = time.time() - start_time
            
            return {
                'success': response.status_code == 200,
                'response_time': response_time,
                'data': response.json() if response.status_code == 200 else None,
                'error': None if response.status_code == 200 else f"HTTP {response.status_code}"
            }
        except Exception as e:
            return {
                'success': False,
                'response_time': time.time() - start_time,
                'data': None,
                'error': str(e)
            }

class QNTIPuppeteerSimulator:
    """Simulate QNTI web interface using Puppeteer"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.browser = None
        self.page = None
        
    async def launch_browser(self, headless: bool = True):
        """Launch Puppeteer browser"""
        if not PUPPETEER_AVAILABLE:
            raise ImportError("Puppeteer (pyppeteer) not available")
        
        self.browser = await launch(
            headless=headless,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu'
            ]
        )
        self.page = await self.browser.newPage()
        
        # Set viewport
        await self.page.setViewport({'width': 1920, 'height': 1080})
        
        # Set user agent
        await self.page.setUserAgent(
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
    
    async def close_browser(self):
        """Close browser"""
        if self.browser:
            await self.browser.close()
    
    async def simulate_dashboard_visit(self) -> Dict[str, Any]:
        """Simulate visiting the main dashboard"""
        try:
            start_time = time.time()
            
            # Navigate to dashboard
            await self.page.goto(self.base_url, waitUntil='networkidle2')
            
            # Wait for dashboard to load
            await self.page.waitForSelector('.dashboard-card', timeout=10000)
            
            # Check if AI insights are loaded
            ai_insights = await self.page.querySelector('.ai-insight-box')
            
            # Take screenshot
            await self.page.screenshot({'path': 'simulation_dashboard.png'})
            
            response_time = time.time() - start_time
            
            return {
                'success': True,
                'response_time': response_time,
                'ai_insights_present': ai_insights is not None,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'response_time': time.time() - start_time,
                'error': str(e)
            }
    
    async def simulate_ea_generation_workflow(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate complete EA generation workflow"""
        try:
            start_time = time.time()
            
            # Navigate to EA Generation page
            await self.page.goto(f"{self.base_url}/ea-generation", waitUntil='networkidle2')
            
            # Wait for page to load
            await self.page.waitForSelector('#eaForm', timeout=10000)
            
            # Fill in EA name
            await self.page.type('#eaName', config['ea_name'])
            
            # Fill in description
            await self.page.type('#eaDescription', config['description'])
            
            # Select symbols
            for symbol in config['symbols']:
                await self.page.select('#eaSymbols', symbol)
            
            # Select timeframes
            for timeframe in config['timeframes']:
                await self.page.select('#eaTimeframes', timeframe)
            
            # Wait for indicators to load
            await self.page.waitForSelector('.indicator-grid', timeout=10000)
            
            # Select indicators
            for indicator in config['indicators']:
                indicator_checkbox = f"#indicator_{indicator['name']}"
                try:
                    await self.page.click(indicator_checkbox)
                except:
                    logger.warning(f"Could not select indicator: {indicator['name']}")
            
            # Select optimization method
            await self.page.select('#optimizationMethod', config.get('method', 'genetic_algorithm'))
            
            # Submit form
            await self.page.click('button[type="submit"]')
            
            # Wait for workflow to start
            await self.page.waitForSelector('#workflowStatus.status-running', timeout=5000)
            
            # Monitor workflow progress
            workflow_completed = False
            timeout_count = 0
            max_timeout = 60  # 60 seconds timeout
            
            while not workflow_completed and timeout_count < max_timeout:
                try:
                    # Check if workflow is completed
                    completed_element = await self.page.querySelector('#workflowStatus.status-completed')
                    failed_element = await self.page.querySelector('#workflowStatus.status-failed')
                    
                    if completed_element:
                        workflow_completed = True
                        success = True
                    elif failed_element:
                        workflow_completed = True
                        success = False
                    else:
                        await asyncio.sleep(1)
                        timeout_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error checking workflow status: {e}")
                    await asyncio.sleep(1)
                    timeout_count += 1
            
            # Take screenshot of final state
            await self.page.screenshot({'path': f'simulation_ea_generation_{int(time.time())}.png'})
            
            response_time = time.time() - start_time
            
            return {
                'success': workflow_completed and success if workflow_completed else False,
                'response_time': response_time,
                'workflow_completed': workflow_completed,
                'timed_out': timeout_count >= max_timeout,
                'error': None if workflow_completed else 'Workflow timeout or failure'
            }
            
        except Exception as e:
            return {
                'success': False,
                'response_time': time.time() - start_time,
                'error': str(e)
            }

class QNTISeleniumSimulator:
    """Simulate QNTI web interface using Selenium (fallback)"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.driver = None
        
    def launch_browser(self, headless: bool = True):
        """Launch Selenium browser"""
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium not available")
        
        options = Options()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        try:
            self.driver = webdriver.Chrome(options=options)
        except Exception as e:
            logger.error(f"Failed to launch Chrome: {e}")
            # Try Firefox as fallback
            from selenium.webdriver.firefox.options import Options as FirefoxOptions
            firefox_options = FirefoxOptions()
            if headless:
                firefox_options.add_argument('--headless')
            self.driver = webdriver.Firefox(options=firefox_options)
    
    def close_browser(self):
        """Close browser"""
        if self.driver:
            self.driver.quit()
    
    def simulate_dashboard_navigation(self) -> Dict[str, Any]:
        """Simulate dashboard navigation"""
        try:
            start_time = time.time()
            
            # Navigate to dashboard
            self.driver.get(self.base_url)
            
            # Wait for dashboard to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "dashboard-card"))
            )
            
            # Check navigation links
            nav_links = self.driver.find_elements(By.CSS_SELECTOR, ".nav-links a")
            
            # Take screenshot
            self.driver.save_screenshot('selenium_dashboard.png')
            
            response_time = time.time() - start_time
            
            return {
                'success': True,
                'response_time': response_time,
                'nav_links_count': len(nav_links),
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'response_time': time.time() - start_time,
                'error': str(e)
            }

class QNTILoadTestSimulator:
    """Simulate high load scenarios"""
    
    def __init__(self, base_url: str, max_workers: int = 10):
        self.base_url = base_url
        self.max_workers = max_workers
        self.api_simulator = QNTIAPISimulator(base_url)
        
    async def simulate_concurrent_users(self, num_users: int, duration: int) -> List[Dict]:
        """Simulate multiple concurrent users"""
        results = []
        
        async def user_session(user_id: int):
            """Simulate a single user session"""
            session_start = time.time()
            session_results = []
            
            while time.time() - session_start < duration:
                # Random delay between actions
                await asyncio.sleep(random.uniform(1, 5))
                
                # Random action
                action = random.choice([
                    'health_check',
                    'indicators_check', 
                    'ea_creation'
                ])
                
                if action == 'health_check':
                    result = await self.api_simulator.test_system_health()
                elif action == 'indicators_check':
                    result = await self.api_simulator.test_ea_indicators()
                else:  # ea_creation
                    ea_config = self._generate_random_ea_config()
                    result = await self.api_simulator.test_ea_workflow_creation(ea_config)
                
                result['user_id'] = user_id
                result['action'] = action
                result['timestamp'] = time.time()
                session_results.append(result)
            
            return session_results
        
        # Run concurrent user sessions
        tasks = [user_session(i) for i in range(num_users)]
        user_results = await asyncio.gather(*tasks)
        
        # Flatten results
        for user_result in user_results:
            results.extend(user_result)
        
        return results
    
    def _generate_random_ea_config(self) -> Dict[str, Any]:
        """Generate random EA configuration for testing"""
        indicators = random.sample([
            'SMA', 'EMA', 'RSI', 'MACD', 'Bollinger Bands',
            'Stochastic', 'ATR', 'CCI'
        ], k=random.randint(2, 5))
        
        return {
            'ea_name': f"Test Strategy {random.randint(1000, 9999)}",
            'description': f"Automated test strategy generated at {datetime.now()}",
            'symbols': random.sample(['EURUSD', 'GBPUSD', 'USDJPY'], k=random.randint(1, 2)),
            'timeframes': random.sample(['H1', 'H4', 'D1'], k=random.randint(1, 2)),
            'indicators': [{'name': ind, 'params': {}} for ind in indicators],
            'method': random.choice(['genetic_algorithm', 'grid_search', 'bayesian']),
            'auto_proceed': True
        }

class QNTIAutomationSuite:
    """Main automation suite orchestrator"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.results = SimulationResults(start_time=datetime.now())
        self.api_simulator = QNTIAPISimulator(config.qnti_url)
        self.load_simulator = QNTILoadTestSimulator(config.qnti_url, config.max_concurrent_users)
        
    async def run_comprehensive_simulation(self) -> SimulationResults:
        """Run comprehensive QNTI system simulation"""
        logger.info("🚀 Starting QNTI Comprehensive Automation Suite")
        
        try:
            # Phase 1: Basic connectivity and health checks
            await self._run_health_checks()
            
            # Phase 2: API functionality tests
            await self._run_api_tests()
            
            # Phase 3: UI automation tests
            await self._run_ui_tests()
            
            # Phase 4: Load testing
            await self._run_load_tests()
            
            # Phase 5: EA generation workflow tests
            await self._run_ea_workflow_tests()
            
            # Phase 6: Performance benchmarking
            await self._run_performance_tests()
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            self.results.errors.append(str(e))
        
        finally:
            self.results.end_time = datetime.now()
            await self._generate_final_report()
        
        return self.results
    
    async def _run_health_checks(self):
        """Run basic system health checks"""
        logger.info("📊 Running health checks...")
        
        health_result = await self.api_simulator.test_system_health()
        self._update_results(health_result, "health_check")
        
        if health_result['success']:
            logger.info("✅ System health check passed")
        else:
            logger.error(f"❌ System health check failed: {health_result['error']}")
    
    async def _run_api_tests(self):
        """Run API functionality tests"""
        logger.info("🔌 Running API tests...")
        
        # Test indicators endpoint
        indicators_result = await self.api_simulator.test_ea_indicators()
        self._update_results(indicators_result, "api_indicators")
        
        if indicators_result['success']:
            logger.info("✅ Indicators API test passed")
        else:
            logger.error(f"❌ Indicators API test failed: {indicators_result['error']}")
    
    async def _run_ui_tests(self):
        """Run UI automation tests"""
        logger.info("🖥️ Running UI tests...")
        
        if PUPPETEER_AVAILABLE:
            await self._run_puppeteer_tests()
        elif SELENIUM_AVAILABLE:
            await self._run_selenium_tests()
        else:
            logger.warning("⚠️ No browser automation libraries available")
    
    async def _run_puppeteer_tests(self):
        """Run Puppeteer UI tests"""
        puppeteer_sim = QNTIPuppeteerSimulator(self.config.qnti_url)
        
        try:
            await puppeteer_sim.launch_browser(headless=True)
            
            # Test dashboard
            dashboard_result = await puppeteer_sim.simulate_dashboard_visit()
            self._update_results(dashboard_result, "ui_dashboard")
            
            # Test EA generation workflow
            ea_config = {
                'ea_name': 'Automation Test Strategy',
                'description': 'Strategy created by automation suite',
                'symbols': ['EURUSD'],
                'timeframes': ['H1'],
                'indicators': [{'name': 'SMA'}, {'name': 'RSI'}],
                'method': 'genetic_algorithm'
            }
            
            ea_result = await puppeteer_sim.simulate_ea_generation_workflow(ea_config)
            self._update_results(ea_result, "ui_ea_generation")
            self.results.ea_generation_results.append(ea_result)
            
        finally:
            await puppeteer_sim.close_browser()
    
    async def _run_selenium_tests(self):
        """Run Selenium UI tests as fallback"""
        selenium_sim = QNTISeleniumSimulator(self.config.qnti_url)
        
        try:
            selenium_sim.launch_browser(headless=True)
            
            dashboard_result = selenium_sim.simulate_dashboard_navigation()
            self._update_results(dashboard_result, "ui_selenium_dashboard")
            
        finally:
            selenium_sim.close_browser()
    
    async def _run_load_tests(self):
        """Run load testing scenarios"""
        logger.info("⚡ Running load tests...")
        
        # Test with multiple concurrent users
        load_results = await self.load_simulator.simulate_concurrent_users(
            num_users=self.config.max_concurrent_users,
            duration=60  # 1 minute load test
        )
        
        # Analyze load test results
        successful_requests = sum(1 for r in load_results if r['success'])
        total_requests = len(load_results)
        avg_response_time = sum(r['response_time'] for r in load_results) / total_requests if total_requests > 0 else 0
        
        load_summary = {
            'success': successful_requests / total_requests > 0.9 if total_requests > 0 else False,
            'response_time': avg_response_time,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'error': None if successful_requests / total_requests > 0.9 else 'High failure rate'
        }
        
        self._update_results(load_summary, "load_test")
        self.results.performance_metrics['load_test'] = load_results
    
    async def _run_ea_workflow_tests(self):
        """Run EA workflow specific tests"""
        logger.info("🤖 Running EA workflow tests...")
        
        for strategy_name in self.config.ea_strategies_to_test[:3]:  # Test first 3 strategies
            ea_config = {
                'ea_name': strategy_name,
                'description': f'Automated test of {strategy_name}',
                'symbols': random.sample(self.config.symbols_pool, k=2),
                'timeframes': random.sample(self.config.timeframes_pool, k=2),
                'indicators': [
                    {'name': ind, 'params': {}} 
                    for ind in random.sample(self.config.indicators_pool, k=3)
                ],
                'method': random.choice(['genetic_algorithm', 'grid_search']),
                'auto_proceed': True
            }
            
            workflow_result = await self.api_simulator.test_ea_workflow_creation(ea_config)
            self._update_results(workflow_result, f"ea_workflow_{strategy_name}")
            
            # If workflow started successfully, monitor it
            if workflow_result['success'] and workflow_result['data']:
                workflow_id = workflow_result['data'].get('workflow_id')
                if workflow_id:
                    await self._monitor_workflow(workflow_id)
    
    async def _monitor_workflow(self, workflow_id: str):
        """Monitor a specific workflow"""
        max_monitoring_time = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_monitoring_time:
            status_result = await self.api_simulator.test_workflow_status(workflow_id)
            
            if status_result['success'] and status_result['data']:
                status = status_result['data'].get('status')
                if status in ['completed', 'failed']:
                    self._update_results(status_result, f"workflow_monitoring_{workflow_id}")
                    break
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def _run_performance_tests(self):
        """Run performance benchmarking tests"""
        logger.info("📈 Running performance tests...")
        
        # Measure response times for various endpoints
        endpoints_to_test = [
            ('health', lambda: self.api_simulator.test_system_health()),
            ('indicators', lambda: self.api_simulator.test_ea_indicators()),
        ]
        
        performance_results = {}
        
        for endpoint_name, test_func in endpoints_to_test:
            # Run multiple iterations
            times = []
            for _ in range(10):
                result = await test_func()
                if result['success']:
                    times.append(result['response_time'])
                await asyncio.sleep(0.1)
            
            if times:
                performance_results[endpoint_name] = {
                    'avg_response_time': sum(times) / len(times),
                    'min_response_time': min(times),
                    'max_response_time': max(times),
                    'samples': len(times)
                }
        
        self.results.performance_metrics['endpoint_performance'] = performance_results
    
    def _update_results(self, test_result: Dict[str, Any], test_name: str):
        """Update simulation results"""
        self.results.total_tests += 1
        
        if test_result['success']:
            self.results.successful_tests += 1
        else:
            self.results.failed_tests += 1
            if test_result.get('error'):
                self.results.errors.append(f"{test_name}: {test_result['error']}")
        
        # Update response time statistics
        response_time = test_result.get('response_time', 0)
        if response_time > 0:
            if self.results.max_response_time < response_time:
                self.results.max_response_time = response_time
            if self.results.min_response_time > response_time:
                self.results.min_response_time = response_time
            
            # Update average response time
            total_response_time = self.results.avg_response_time * (self.results.total_tests - 1) + response_time
            self.results.avg_response_time = total_response_time / self.results.total_tests
    
    async def _generate_final_report(self):
        """Generate comprehensive simulation report"""
        logger.info("📋 Generating final simulation report...")
        
        report = {
            'simulation_summary': {
                'start_time': self.results.start_time.isoformat(),
                'end_time': self.results.end_time.isoformat() if self.results.end_time else None,
                'duration': self.results.duration,
                'total_tests': self.results.total_tests,
                'successful_tests': self.results.successful_tests,
                'failed_tests': self.results.failed_tests,
                'success_rate': self.results.success_rate,
                'avg_response_time': self.results.avg_response_time,
                'max_response_time': self.results.max_response_time,
                'min_response_time': self.results.min_response_time
            },
            'performance_metrics': self.results.performance_metrics,
            'ea_generation_results': self.results.ea_generation_results,
            'errors': self.results.errors,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report to file
        report_file = f"qnti_simulation_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Simulation report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 QNTI AUTOMATION SUITE RESULTS")
        print("="*60)
        print(f"Duration: {self.results.duration:.1f} seconds")
        print(f"Total Tests: {self.results.total_tests}")
        print(f"Success Rate: {self.results.success_rate:.1%}")
        print(f"Average Response Time: {self.results.avg_response_time:.3f}s")
        print(f"Max Response Time: {self.results.max_response_time:.3f}s")
        
        if self.results.errors:
            print(f"\n❌ Errors ({len(self.results.errors)}):")
            for error in self.results.errors[:5]:  # Show first 5 errors
                print(f"  • {error}")
        
        print("="*60)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on simulation results"""
        recommendations = []
        
        if self.results.success_rate < self.config.min_success_rate:
            recommendations.append(
                f"Success rate ({self.results.success_rate:.1%}) is below threshold "
                f"({self.config.min_success_rate:.1%}). Investigate failing tests."
            )
        
        if self.results.avg_response_time > self.config.max_response_time:
            recommendations.append(
                f"Average response time ({self.results.avg_response_time:.3f}s) exceeds "
                f"threshold ({self.config.max_response_time}s). Consider performance optimization."
            )
        
        if len(self.results.errors) > self.results.total_tests * self.config.max_error_rate:
            recommendations.append(
                f"Error rate is high ({len(self.results.errors)}/{self.results.total_tests}). "
                "Review error logs and improve error handling."
            )
        
        if not recommendations:
            recommendations.append("System performance is within acceptable parameters.")
        
        return recommendations

# CLI interface
async def main():
    """Main entry point for automation suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QNTI Automation Suite")
    parser.add_argument('--url', default='http://localhost:5000', help='QNTI system URL')
    parser.add_argument('--duration', type=int, default=3600, help='Simulation duration in seconds')
    parser.add_argument('--users', type=int, default=5, help='Max concurrent users')
    parser.add_argument('--headless', action='store_true', help='Run browser tests in headless mode')
    parser.add_argument('--quick', action='store_true', help='Run quick test suite only')
    
    args = parser.parse_args()
    
    # Create configuration
    config = SimulationConfig(
        qnti_url=args.url,
        simulation_duration=60 if args.quick else args.duration,
        max_concurrent_users=2 if args.quick else args.users
    )
    
    if args.quick:
        config.ea_strategies_to_test = config.ea_strategies_to_test[:2]
    
    # Run simulation
    automation_suite = QNTIAutomationSuite(config)
    results = await automation_suite.run_comprehensive_simulation()
    
    return results

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Automation suite interrupted by user")
    except Exception as e:
        logger.error(f"Automation suite failed: {e}")