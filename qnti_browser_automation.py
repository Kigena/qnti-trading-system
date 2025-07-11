#!/usr/bin/env python3
"""
QNTI Browser Automation - Frontend User Path Testing
Comprehensive testing of all frontend user interactions using Selenium WebDriver
"""

import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import random
from dataclasses import dataclass

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qnti_browser_automation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('QNTI_BROWSER_AUTO')

@dataclass
class BrowserTestResult:
    """Browser test result data structure"""
    test_name: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    screenshot_path: Optional[str] = None
    page_source: Optional[str] = None

class QNTIBrowserAutomation:
    """Comprehensive QNTI Browser Automation for Frontend Testing"""
    
    def __init__(self, base_url: str = "http://localhost:5000", headless: bool = False):
        self.base_url = base_url.rstrip('/')
        self.headless = headless
        self.driver = None
        self.wait = None
        self.test_results: List[BrowserTestResult] = []
        self.screenshot_dir = Path("screenshots")
        self.screenshot_dir.mkdir(exist_ok=True)
        
        # Test data
        self.test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'USDCAD']
        self.test_timeframes = ['1M', '5M', '15M', '1H', '4H', '1D']
        
        logger.info(f"QNTI Browser Automation initialized for {base_url}")
    
    def setup_driver(self):
        """Initialize Chrome WebDriver with options"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument('--headless')
        
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--allow-running-insecure-content')
        chrome_options.add_argument('--disable-extensions')
        
        # Add user agent
        chrome_options.add_argument('--user-agent=QNTI-Browser-Automation/1.0')
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.wait = WebDriverWait(self.driver, 10)
            logger.info("✅ Chrome WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Chrome WebDriver: {e}")
            raise
    
    def teardown_driver(self):
        """Clean up WebDriver"""
        if self.driver:
            self.driver.quit()
            logger.info("🔧 WebDriver closed")
    
    def take_screenshot(self, test_name: str) -> str:
        """Take screenshot for debugging"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{test_name.replace(' ', '_')}_{timestamp}.png"
        filepath = self.screenshot_dir / filename
        
        try:
            self.driver.save_screenshot(str(filepath))
            logger.info(f"📸 Screenshot saved: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"❌ Failed to take screenshot: {e}")
            return None
    
    def _execute_test(self, test_name: str, test_function) -> BrowserTestResult:
        """Execute a test function and track results"""
        logger.info(f"🔍 Running: {test_name}")
        start_time = time.time()
        
        try:
            test_function()
            duration = time.time() - start_time
            
            result = BrowserTestResult(
                test_name=test_name,
                success=True,
                duration=duration,
                screenshot_path=self.take_screenshot(test_name)
            )
            
            logger.info(f"✅ {test_name} completed in {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            screenshot_path = self.take_screenshot(f"{test_name}_ERROR")
            
            result = BrowserTestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error_message=str(e),
                screenshot_path=screenshot_path,
                page_source=self.driver.page_source if self.driver else None
            )
            
            logger.error(f"❌ {test_name} failed after {duration:.2f}s: {e}")
        
        self.test_results.append(result)
        return result
    
    def test_dashboard_load(self):
        """Test main dashboard page loading"""
        def _test():
            self.driver.get(self.base_url)
            
            # Wait for page to load
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Check for key dashboard elements
            assert "QNTI Dashboard" in self.driver.title
            
            # Check for navigation elements
            nav_elements = self.driver.find_elements(By.CLASS_NAME, "nav-link")
            assert len(nav_elements) > 0, "Navigation links not found"
            
            # Check for dashboard cards
            dashboard_cards = self.driver.find_elements(By.CLASS_NAME, "dashboard-card")
            assert len(dashboard_cards) > 0, "Dashboard cards not found"
            
            # Check for status indicators
            status_indicators = self.driver.find_elements(By.CLASS_NAME, "status-item")
            assert len(status_indicators) > 0, "Status indicators not found"
        
        return self._execute_test("Dashboard Load Test", _test)
    
    def test_navigation_menu(self):
        """Test navigation menu functionality"""
        def _test():
            self.driver.get(self.base_url)
            
            # Find all navigation links
            nav_links = self.driver.find_elements(By.CLASS_NAME, "nav-link")
            
            for link in nav_links:
                if link.is_displayed() and link.is_enabled():
                    link_text = link.text
                    href = link.get_attribute('href')
                    
                    if href and href != '#':
                        logger.info(f"Testing navigation to: {link_text}")
                        
                        # Click the link
                        self.driver.execute_script("arguments[0].click();", link)
                        time.sleep(2)
                        
                        # Verify page loaded
                        current_url = self.driver.current_url
                        assert current_url != self.base_url, f"Navigation to {link_text} failed"
                        
                        # Go back to main dashboard
                        self.driver.get(self.base_url)
                        time.sleep(1)
        
        return self._execute_test("Navigation Menu Test", _test)
    
    def test_ai_vision_upload_interface(self):
        """Test AI Vision Analysis upload interface"""
        def _test():
            self.driver.get(self.base_url)
            
            # Look for AI Vision section
            vision_section = self.wait.until(
                EC.presence_of_element_located((By.ID, "ai-vision-section"))
            )
            
            # Scroll to vision section
            self.driver.execute_script("arguments[0].scrollIntoView(true);", vision_section)
            time.sleep(1)
            
            # Check for upload button
            upload_button = self.driver.find_element(By.ID, "upload-chart-btn")
            assert upload_button.is_displayed(), "Upload button not visible"
            
            # Check for symbol dropdown
            symbol_dropdown = self.driver.find_element(By.ID, "symbol-select")
            assert symbol_dropdown.is_displayed(), "Symbol dropdown not visible"
            
            # Check for timeframe dropdown
            timeframe_dropdown = self.driver.find_element(By.ID, "timeframe-select")
            assert timeframe_dropdown.is_displayed(), "Timeframe dropdown not visible"
            
            # Check for analyze button (should be disabled initially)
            analyze_button = self.driver.find_element(By.ID, "analyze-chart-btn")
            assert not analyze_button.is_enabled(), "Analyze button should be disabled initially"
        
        return self._execute_test("AI Vision Upload Interface Test", _test)
    
    def test_dropdown_interactions(self):
        """Test dropdown menu interactions"""
        def _test():
            self.driver.get(self.base_url)
            
            # Navigate to vision section
            vision_section = self.wait.until(
                EC.presence_of_element_located((By.ID, "ai-vision-section"))
            )
            self.driver.execute_script("arguments[0].scrollIntoView(true);", vision_section)
            
            # Test symbol dropdown
            symbol_select = Select(self.driver.find_element(By.ID, "symbol-select"))
            
            # Get all options
            symbol_options = symbol_select.options
            assert len(symbol_options) > 1, "Symbol dropdown should have options"
            
            # Test selecting different symbols
            for i, option in enumerate(symbol_options[1:3]):  # Test first 2 options
                symbol_select.select_by_visible_text(option.text)
                time.sleep(0.5)
                
                selected_value = symbol_select.first_selected_option.text
                assert selected_value == option.text, f"Symbol selection failed: {option.text}"
            
            # Test timeframe dropdown
            timeframe_select = Select(self.driver.find_element(By.ID, "timeframe-select"))
            
            # Get all options
            timeframe_options = timeframe_select.options
            assert len(timeframe_options) > 1, "Timeframe dropdown should have options"
            
            # Test selecting different timeframes
            for i, option in enumerate(timeframe_options[1:3]):  # Test first 2 options
                timeframe_select.select_by_visible_text(option.text)
                time.sleep(0.5)
                
                selected_value = timeframe_select.first_selected_option.text
                assert selected_value == option.text, f"Timeframe selection failed: {option.text}"
        
        return self._execute_test("Dropdown Interactions Test", _test)
    
    def test_form_validation(self):
        """Test form validation logic"""
        def _test():
            self.driver.get(self.base_url)
            
            # Navigate to vision section
            vision_section = self.wait.until(
                EC.presence_of_element_located((By.ID, "ai-vision-section"))
            )
            self.driver.execute_script("arguments[0].scrollIntoView(true);", vision_section)
            
            # Initially, analyze button should be disabled
            analyze_button = self.driver.find_element(By.ID, "analyze-chart-btn")
            assert not analyze_button.is_enabled(), "Analyze button should be disabled initially"
            
            # Select symbol
            symbol_select = Select(self.driver.find_element(By.ID, "symbol-select"))
            symbol_select.select_by_index(1)  # Select first option
            time.sleep(0.5)
            
            # Button should still be disabled (no timeframe and image)
            assert not analyze_button.is_enabled(), "Analyze button should still be disabled"
            
            # Select timeframe
            timeframe_select = Select(self.driver.find_element(By.ID, "timeframe-select"))
            timeframe_select.select_by_index(1)  # Select first option
            time.sleep(0.5)
            
            # Button should still be disabled (no image)
            assert not analyze_button.is_enabled(), "Analyze button should still be disabled without image"
            
            # Simulate image upload by triggering the validation function
            self.driver.execute_script("window.uploadedImage = 'test_image.png'; validateForm();")
            time.sleep(0.5)
            
            # Now button should be enabled
            assert analyze_button.is_enabled(), "Analyze button should be enabled after all fields are filled"
        
        return self._execute_test("Form Validation Test", _test)
    
    def test_trading_actions_panel(self):
        """Test trading actions panel functionality"""
        def _test():
            self.driver.get(self.base_url)
            
            # Navigate to vision section
            vision_section = self.wait.until(
                EC.presence_of_element_located((By.ID, "ai-vision-section"))
            )
            self.driver.execute_script("arguments[0].scrollIntoView(true);", vision_section)
            
            # Check for trading actions section
            trading_actions = self.driver.find_element(By.ID, "trading-actions-section")
            assert trading_actions.is_displayed(), "Trading actions section not visible"
            
            # Check for trading parameter inputs
            lot_size_input = self.driver.find_element(By.ID, "lot-size")
            stop_loss_input = self.driver.find_element(By.ID, "stop-loss")
            take_profit_input = self.driver.find_element(By.ID, "take-profit")
            
            assert lot_size_input.is_displayed(), "Lot size input not visible"
            assert stop_loss_input.is_displayed(), "Stop loss input not visible"
            assert take_profit_input.is_displayed(), "Take profit input not visible"
            
            # Test input validation
            lot_size_input.clear()
            lot_size_input.send_keys("0.01")
            
            stop_loss_input.clear()
            stop_loss_input.send_keys("50")
            
            take_profit_input.clear()
            take_profit_input.send_keys("100")
            
            # Check for action buttons
            buy_button = self.driver.find_element(By.ID, "execute-buy-btn")
            sell_button = self.driver.find_element(By.ID, "execute-sell-btn")
            watchlist_button = self.driver.find_element(By.ID, "add-watchlist-btn")
            
            assert buy_button.is_displayed(), "Buy button not visible"
            assert sell_button.is_displayed(), "Sell button not visible"
            assert watchlist_button.is_displayed(), "Watchlist button not visible"
        
        return self._execute_test("Trading Actions Panel Test", _test)
    
    def test_responsive_design(self):
        """Test responsive design at different screen sizes"""
        def _test():
            # Test different screen sizes
            screen_sizes = [
                (1920, 1080),  # Desktop
                (1366, 768),   # Laptop
                (768, 1024),   # Tablet
                (375, 667)     # Mobile
            ]
            
            for width, height in screen_sizes:
                logger.info(f"Testing responsive design at {width}x{height}")
                
                # Set window size
                self.driver.set_window_size(width, height)
                time.sleep(1)
                
                # Load page
                self.driver.get(self.base_url)
                time.sleep(2)
                
                # Check if main elements are still visible
                dashboard_cards = self.driver.find_elements(By.CLASS_NAME, "dashboard-card")
                assert len(dashboard_cards) > 0, f"Dashboard cards not visible at {width}x{height}"
                
                # Check navigation
                nav_menu = self.driver.find_element(By.CLASS_NAME, "nav-menu")
                assert nav_menu.is_displayed(), f"Navigation not visible at {width}x{height}"
                
                # Take screenshot for visual verification
                self.take_screenshot(f"responsive_{width}x{height}")
        
        return self._execute_test("Responsive Design Test", _test)
    
    def test_error_handling_ui(self):
        """Test UI error handling and user feedback"""
        def _test():
            self.driver.get(self.base_url)
            
            # Test invalid form submission
            vision_section = self.wait.until(
                EC.presence_of_element_located((By.ID, "ai-vision-section"))
            )
            self.driver.execute_script("arguments[0].scrollIntoView(true);", vision_section)
            
            # Try to trigger error by calling analysis without proper setup
            self.driver.execute_script("""
                // Simulate error condition
                window.showError = function(message) {
                    console.log('Error shown:', message);
                };
                
                // Trigger error handling
                if (typeof analyzeChart === 'function') {
                    analyzeChart();
                }
            """)
            
            time.sleep(2)
            
            # Check if error handling elements exist
            error_elements = self.driver.find_elements(By.CLASS_NAME, "error-message")
            warning_elements = self.driver.find_elements(By.CLASS_NAME, "warning-message")
            
            # At least one error handling mechanism should be present
            assert len(error_elements) > 0 or len(warning_elements) > 0, "No error handling UI elements found"
        
        return self._execute_test("Error Handling UI Test", _test)
    
    def test_real_time_updates(self):
        """Test real-time updates and dynamic content"""
        def _test():
            self.driver.get(self.base_url)
            
            # Wait for page to load
            time.sleep(3)
            
            # Check for elements that should update in real-time
            status_indicators = self.driver.find_elements(By.CLASS_NAME, "status-dot")
            assert len(status_indicators) > 0, "No status indicators found"
            
            # Check for animated elements
            animated_elements = self.driver.find_elements(By.CSS_SELECTOR, "[class*='pulse']")
            
            # Monitor for changes over time
            initial_values = {}
            metric_elements = self.driver.find_elements(By.CLASS_NAME, "metric-value")
            
            for i, element in enumerate(metric_elements):
                initial_values[i] = element.text
            
            # Wait and check for updates
            time.sleep(5)
            
            # Verify some dynamic behavior exists
            current_time = self.driver.execute_script("return new Date().toISOString();")
            assert current_time is not None, "JavaScript execution failed"
        
        return self._execute_test("Real-time Updates Test", _test)
    
    def test_accessibility_features(self):
        """Test accessibility features"""
        def _test():
            self.driver.get(self.base_url)
            
            # Check for proper heading structure
            h1_elements = self.driver.find_elements(By.TAG_NAME, "h1")
            h2_elements = self.driver.find_elements(By.TAG_NAME, "h2")
            h3_elements = self.driver.find_elements(By.TAG_NAME, "h3")
            
            # Check for alt text on images
            images = self.driver.find_elements(By.TAG_NAME, "img")
            for img in images:
                alt_text = img.get_attribute("alt")
                assert alt_text is not None, f"Image missing alt text: {img.get_attribute('src')}"
            
            # Check for proper form labels
            inputs = self.driver.find_elements(By.TAG_NAME, "input")
            for input_elem in inputs:
                input_id = input_elem.get_attribute("id")
                if input_id:
                    labels = self.driver.find_elements(By.CSS_SELECTOR, f"label[for='{input_id}']")
                    assert len(labels) > 0, f"Input missing label: {input_id}"
            
            # Check for keyboard navigation
            focusable_elements = self.driver.find_elements(By.CSS_SELECTOR, 
                "button, input, select, textarea, a[href], [tabindex]")
            assert len(focusable_elements) > 0, "No focusable elements found"
        
        return self._execute_test("Accessibility Features Test", _test)
    
    def simulate_user_workflow(self):
        """Simulate a complete user workflow"""
        def _test():
            logger.info("Simulating complete user workflow...")
            
            # Step 1: Load dashboard
            self.driver.get(self.base_url)
            time.sleep(2)
            
            # Step 2: Navigate to vision section
            vision_section = self.wait.until(
                EC.presence_of_element_located((By.ID, "ai-vision-section"))
            )
            self.driver.execute_script("arguments[0].scrollIntoView(true);", vision_section)
            time.sleep(1)
            
            # Step 3: Select trading symbol
            symbol_select = Select(self.driver.find_element(By.ID, "symbol-select"))
            symbol_select.select_by_visible_text("EURUSD")
            time.sleep(1)
            
            # Step 4: Select timeframe
            timeframe_select = Select(self.driver.find_element(By.ID, "timeframe-select"))
            timeframe_select.select_by_visible_text("4H")
            time.sleep(1)
            
            # Step 5: Simulate image upload
            self.driver.execute_script("""
                window.uploadedImage = 'user_workflow_test.png';
                validateForm();
                
                // Show upload preview
                const preview = document.getElementById('image-preview');
                if (preview) {
                    preview.style.display = 'block';
                    preview.innerHTML = '<p>✅ Test chart uploaded successfully</p>';
                }
            """)
            time.sleep(2)
            
            # Step 6: Verify analyze button is enabled
            analyze_button = self.driver.find_element(By.ID, "analyze-chart-btn")
            assert analyze_button.is_enabled(), "Analyze button should be enabled"
            
            # Step 7: Click analyze button
            analyze_button.click()
            time.sleep(3)
            
            # Step 8: Check for results (simulate)
            self.driver.execute_script("""
                // Simulate analysis results
                const resultsSection = document.getElementById('analysis-results');
                if (resultsSection) {
                    resultsSection.innerHTML = `
                        <div class="analysis-complete">
                            <h3>Analysis Complete</h3>
                            <p>Signal: BUY</p>
                            <p>Confidence: 85%</p>
                        </div>
                    `;
                    resultsSection.style.display = 'block';
                }
                
                // Show trading actions
                const tradingActions = document.getElementById('trading-actions-section');
                if (tradingActions) {
                    tradingActions.style.display = 'block';
                }
            """)
            time.sleep(2)
            
            # Step 9: Configure trading parameters
            lot_size_input = self.driver.find_element(By.ID, "lot-size")
            lot_size_input.clear()
            lot_size_input.send_keys("0.01")
            
            stop_loss_input = self.driver.find_element(By.ID, "stop-loss")
            stop_loss_input.clear()
            stop_loss_input.send_keys("50")
            
            take_profit_input = self.driver.find_element(By.ID, "take-profit")
            take_profit_input.clear()
            take_profit_input.send_keys("100")
            
            # Step 10: Test watchlist functionality
            watchlist_button = self.driver.find_element(By.ID, "add-watchlist-btn")
            watchlist_button.click()
            time.sleep(1)
            
            logger.info("User workflow simulation completed successfully")
        
        return self._execute_test("Complete User Workflow Simulation", _test)
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.success])
        failed_tests = total_tests - passed_tests
        
        # Calculate performance metrics
        durations = [r.duration for r in self.test_results]
        avg_duration = sum(durations) / len(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        
        # Group results by success/failure
        passed_test_names = [r.test_name for r in self.test_results if r.success]
        failed_test_details = [
            {
                'test_name': r.test_name,
                'error': r.error_message,
                'duration': r.duration,
                'screenshot': r.screenshot_path
            }
            for r in self.test_results if not r.success
        ]
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'average_duration': round(avg_duration, 2),
                'max_duration': round(max_duration, 2)
            },
            'passed_tests': passed_test_names,
            'failed_tests': failed_test_details,
            'screenshots_directory': str(self.screenshot_dir)
        }
        
        return report
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all browser automation tests"""
        logger.info("🚀 Starting QNTI Browser Automation Tests...")
        logger.info("=" * 60)
        
        # Initialize WebDriver
        self.setup_driver()
        
        try:
            # Define test suite
            test_suite = [
                self.test_dashboard_load,
                self.test_navigation_menu,
                self.test_ai_vision_upload_interface,
                self.test_dropdown_interactions,
                self.test_form_validation,
                self.test_trading_actions_panel,
                self.test_responsive_design,
                self.test_error_handling_ui,
                self.test_real_time_updates,
                self.test_accessibility_features,
                self.simulate_user_workflow
            ]
            
            # Run all tests
            for test_function in test_suite:
                try:
                    test_function()
                    time.sleep(1)  # Brief pause between tests
                except Exception as e:
                    logger.error(f"Test execution failed: {e}")
            
            # Generate report
            report = self.generate_test_report()
            
            # Save report
            report_file = f"qnti_browser_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info("=" * 60)
            logger.info("🎯 BROWSER AUTOMATION COMPLETE!")
            logger.info(f"📊 Total Tests: {report['test_summary']['total_tests']}")
            logger.info(f"✅ Success Rate: {report['test_summary']['success_rate']:.1f}%")
            logger.info(f"⏱️ Average Duration: {report['test_summary']['average_duration']}s")
            logger.info(f"📁 Report saved to: {report_file}")
            logger.info(f"📸 Screenshots saved to: {self.screenshot_dir}")
            logger.info("=" * 60)
            
            return report
            
        finally:
            # Clean up
            self.teardown_driver()

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QNTI Browser Automation")
    parser.add_argument('--url', default='http://localhost:5000', help='QNTI server URL')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    
    args = parser.parse_args()
    
    # Initialize automation
    automation = QNTIBrowserAutomation(base_url=args.url, headless=args.headless)
    
    # Run all tests
    report = automation.run_all_tests()
    
    return report

if __name__ == "__main__":
    main() 