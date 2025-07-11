#!/usr/bin/env python3
"""
QNTI Automated User Path Simulation
Comprehensive testing of all user workflows and interactions in the QNTI trading system
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests
import websocket
import threading
from dataclasses import dataclass
import io
import base64
from PIL import Image, ImageDraw

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qnti_user_simulation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('QNTI_USER_SIM')

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    success: bool
    response_time: float
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    response_data: Optional[Dict] = None

class QNTIUserPathSimulator:
    """Comprehensive QNTI User Path Simulator"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'QNTI-User-Simulator/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        # Test results storage
        self.test_results: List[TestResult] = []
        self.simulation_start_time = datetime.now()
        
        # WebSocket connection
        self.ws_connected = False
        self.ws_messages = []
        
        # Test data
        self.test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'USDCAD']
        self.test_timeframes = ['1M', '5M', '15M', '1H', '4H', '1D']
        
        logger.info(f"QNTI User Path Simulator initialized for {base_url}")
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> TestResult:
        """Make HTTP request and track results"""
        test_name = f"{method.upper()} {endpoint}"
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.request(method, url, **kwargs)
            response_time = time.time() - start_time
            
            try:
                response_data = response.json()
            except:
                response_data = {'raw_response': response.text[:500]}
            
            result = TestResult(
                test_name=test_name,
                success=response.status_code < 400,
                response_time=response_time,
                status_code=response.status_code,
                response_data=response_data
            )
            
            if result.success:
                logger.info(f"✅ {test_name} - {response.status_code} ({response_time:.3f}s)")
            else:
                logger.error(f"❌ {test_name} - {response.status_code} ({response_time:.3f}s)")
                result.error_message = response_data.get('error', 'Unknown error')
            
        except Exception as e:
            response_time = time.time() - start_time
            result = TestResult(
                test_name=test_name,
                success=False,
                response_time=response_time,
                error_message=str(e)
            )
            logger.error(f"❌ {test_name} - Exception: {e}")
        
        self.test_results.append(result)
        return result
    
    def _generate_test_chart_image(self) -> bytes:
        """Generate a test chart image for vision analysis"""
        # Create a simple chart-like image
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw some chart-like elements
        # Background
        draw.rectangle([50, 50, 750, 550], fill='black')
        
        # Grid lines
        for i in range(6):
            y = 100 + i * 75
            draw.line([100, y, 700, y], fill='gray', width=1)
        
        for i in range(7):
            x = 100 + i * 100
            draw.line([x, 100, x, 500], fill='gray', width=1)
        
        # Candlestick-like bars
        for i in range(20):
            x = 120 + i * 30
            high = random.randint(150, 200)
            low = random.randint(400, 450)
            open_price = random.randint(high + 20, low - 20)
            close_price = random.randint(high + 20, low - 20)
            
            # High-low line
            draw.line([x, high, x, low], fill='white', width=1)
            
            # Body
            body_color = 'green' if close_price > open_price else 'red'
            draw.rectangle([x-5, min(open_price, close_price), x+5, max(open_price, close_price)], 
                         fill=body_color)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()
    
    def test_dashboard_access(self) -> List[TestResult]:
        """Test main dashboard access and page loads"""
        logger.info("🔍 Testing Dashboard Access...")
        results = []
        
        # Test main dashboard
        results.append(self._make_request('GET', '/'))
        
        # Test dashboard pages
        dashboard_pages = [
            '/dashboard/analytics_reports.html',
            '/dashboard/ea_management.html'
        ]
        
        for page in dashboard_pages:
            results.append(self._make_request('GET', page))
        
        return results
    
    def test_system_health_monitoring(self) -> List[TestResult]:
        """Test system health and status endpoints"""
        logger.info("🔍 Testing System Health Monitoring...")
        results = []
        
        # Basic health check
        results.append(self._make_request('GET', '/api/health'))
        results.append(self._make_request('GET', '/api/system/health'))
        
        # Vision system status
        results.append(self._make_request('GET', '/api/vision/status'))
        
        return results
    
    def test_trade_management(self) -> List[TestResult]:
        """Test trade-related endpoints"""
        logger.info("🔍 Testing Trade Management...")
        results = []
        
        # Get active trades
        results.append(self._make_request('GET', '/api/trades'))
        results.append(self._make_request('GET', '/api/trades/active'))
        
        # Get trade history with different timeframes
        for timeframe in self.test_timeframes:
            results.append(self._make_request('GET', f'/api/trades/history?timeframe={timeframe}'))
        
        return results
    
    def test_ea_management(self) -> List[TestResult]:
        """Test Expert Advisor management endpoints"""
        logger.info("🔍 Testing EA Management...")
        results = []
        
        # Get EA list
        results.append(self._make_request('GET', '/api/eas'))
        
        # Platform scanning
        results.append(self._make_request('POST', '/api/eas/scan-platform'))
        
        # Auto-detect EAs
        results.append(self._make_request('POST', '/api/eas/auto-detect'))
        
        # Recalculate performance
        results.append(self._make_request('POST', '/api/eas/recalculate-performance'))
        
        # Test EA registration
        test_ea_data = {
            "name": "TestEA_Simulator",
            "magic_number": 999999,
            "symbol": "EURUSD",
            "strategy": "Test Strategy",
            "description": "Automated test EA"
        }
        results.append(self._make_request('POST', '/api/eas/register', json=test_ea_data))
        
        # Test EA control
        control_data = {
            "action": "pause",
            "parameters": {"reason": "Automated test"}
        }
        results.append(self._make_request('POST', '/api/eas/TestEA_Simulator/control', json=control_data))
        
        # Get EA history
        results.append(self._make_request('GET', '/api/eas/TestEA_Simulator/history?days=7'))
        
        # Get EA details
        results.append(self._make_request('GET', '/api/eas/TestEA_Simulator/details'))
        
        return results
    
    def test_vision_analysis_workflow(self) -> List[TestResult]:
        """Test complete AI vision analysis workflow"""
        logger.info("🔍 Testing Vision Analysis Workflow...")
        results = []
        
        # Check vision status
        results.append(self._make_request('GET', '/api/vision/status'))
        
        # Upload test chart image
        test_image = self._generate_test_chart_image()
        files = {'image': ('test_chart.png', test_image, 'image/png')}
        
        # Note: requests with files need special handling
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/api/vision/upload", files=files)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                upload_result = response.json()
                analysis_id = upload_result.get('analysis_id')
                
                result = TestResult(
                    test_name="POST /api/vision/upload",
                    success=True,
                    response_time=response_time,
                    status_code=response.status_code,
                    response_data=upload_result
                )
                logger.info(f"✅ Vision upload successful - Analysis ID: {analysis_id}")
                
                # Test analysis with different parameters
                if analysis_id:
                    # Analyze with different symbols and timeframes
                    for symbol in self.test_symbols[:3]:  # Test first 3 symbols
                        for timeframe in ['1H', '4H']:  # Test key timeframes
                            results.append(self._make_request(
                                'POST', 
                                f'/api/vision/analyze/{analysis_id}',
                                json={'symbol': symbol, 'timeframe': timeframe}
                            ))
                            time.sleep(1)  # Rate limiting
                    
                    # Get analysis results
                    results.append(self._make_request('GET', f'/api/vision/analyze/{analysis_id}'))
                    
                    # Get recent analyses
                    results.append(self._make_request('GET', '/api/vision/recent'))
            else:
                result = TestResult(
                    test_name="POST /api/vision/upload",
                    success=False,
                    response_time=response_time,
                    status_code=response.status_code,
                    error_message=response.text
                )
                logger.error(f"❌ Vision upload failed: {response.status_code}")
            
            results.append(result)
            
        except Exception as e:
            result = TestResult(
                test_name="POST /api/vision/upload",
                success=False,
                response_time=0,
                error_message=str(e)
            )
            results.append(result)
            logger.error(f"❌ Vision upload exception: {e}")
        
        return results
    
    def test_system_controls(self) -> List[TestResult]:
        """Test system control endpoints"""
        logger.info("🔍 Testing System Controls...")
        results = []
        
        # Toggle auto trading
        results.append(self._make_request('POST', '/api/system/toggle-auto-trading'))
        time.sleep(1)
        results.append(self._make_request('POST', '/api/system/toggle-auto-trading'))  # Toggle back
        
        # Toggle vision analysis
        results.append(self._make_request('POST', '/api/vision/toggle-auto-analysis'))
        time.sleep(1)
        results.append(self._make_request('POST', '/api/vision/toggle-auto-analysis'))  # Toggle back
        
        return results
    
    def test_websocket_connection(self) -> TestResult:
        """Test WebSocket real-time connection"""
        logger.info("🔍 Testing WebSocket Connection...")
        
        def on_message(ws, message):
            self.ws_messages.append(json.loads(message))
            logger.info(f"📡 WebSocket message received: {message[:100]}...")
        
        def on_error(ws, error):
            logger.error(f"❌ WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            self.ws_connected = False
            logger.info("📡 WebSocket connection closed")
        
        def on_open(ws):
            self.ws_connected = True
            logger.info("📡 WebSocket connection opened")
            # Send test message
            ws.send(json.dumps({"type": "test", "message": "User simulation test"}))
        
        try:
            ws_url = self.base_url.replace('http', 'ws') + '/socket.io/?EIO=4&transport=websocket'
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Run WebSocket in separate thread
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection and test
            time.sleep(3)
            
            if self.ws_connected:
                ws.close()
                return TestResult(
                    test_name="WebSocket Connection",
                    success=True,
                    response_time=0,
                    response_data={"messages_received": len(self.ws_messages)}
                )
            else:
                return TestResult(
                    test_name="WebSocket Connection",
                    success=False,
                    response_time=0,
                    error_message="Failed to establish WebSocket connection"
                )
        
        except Exception as e:
            return TestResult(
                test_name="WebSocket Connection",
                success=False,
                response_time=0,
                error_message=str(e)
            )
    
    def test_error_handling(self) -> List[TestResult]:
        """Test error handling and edge cases"""
        logger.info("🔍 Testing Error Handling...")
        results = []
        
        # Test invalid endpoints
        invalid_endpoints = [
            '/api/nonexistent',
            '/api/trades/invalid',
            '/api/eas/nonexistent/control',
            '/api/vision/invalid-id'
        ]
        
        for endpoint in invalid_endpoints:
            results.append(self._make_request('GET', endpoint))
        
        # Test invalid POST data
        invalid_post_data = [
            ('/api/eas/register', {"invalid": "data"}),
            ('/api/vision/analyze/invalid-id', {"symbol": "INVALID"}),
            ('/api/eas/test/control', {"action": "invalid"})
        ]
        
        for endpoint, data in invalid_post_data:
            results.append(self._make_request('POST', endpoint, json=data))
        
        return results
    
    def simulate_realistic_user_session(self) -> List[TestResult]:
        """Simulate a realistic user session with multiple interactions"""
        logger.info("🔍 Simulating Realistic User Session...")
        results = []
        
        # 1. User opens dashboard
        results.append(self._make_request('GET', '/'))
        time.sleep(1)
        
        # 2. Check system health
        results.append(self._make_request('GET', '/api/health'))
        time.sleep(0.5)
        
        # 3. View active trades
        results.append(self._make_request('GET', '/api/trades'))
        time.sleep(1)
        
        # 4. Check EAs
        results.append(self._make_request('GET', '/api/eas'))
        time.sleep(1)
        
        # 5. View trade history for different timeframes
        for timeframe in ['1W', '1M', '6M']:
            results.append(self._make_request('GET', f'/api/trades/history?timeframe={timeframe}'))
            time.sleep(0.5)
        
        # 6. Upload and analyze chart
        test_image = self._generate_test_chart_image()
        files = {'image': ('user_chart.png', test_image, 'image/png')}
        
        try:
            response = self.session.post(f"{self.base_url}/api/vision/upload", files=files)
            if response.status_code == 200:
                analysis_id = response.json().get('analysis_id')
                if analysis_id:
                    # Analyze with user's preferred symbol
                    results.append(self._make_request(
                        'POST', 
                        f'/api/vision/analyze/{analysis_id}',
                        json={'symbol': 'EURUSD', 'timeframe': '4H'}
                    ))
                    time.sleep(2)
                    
                    # Get results
                    results.append(self._make_request('GET', f'/api/vision/analyze/{analysis_id}'))
        except Exception as e:
            logger.error(f"User session vision test failed: {e}")
        
        # 7. Check vision status
        results.append(self._make_request('GET', '/api/vision/status'))
        
        # 8. Scan for new EAs
        results.append(self._make_request('POST', '/api/eas/scan-platform'))
        time.sleep(1)
        
        # 9. Final health check
        results.append(self._make_request('GET', '/api/health'))
        
        return results
    
    def run_load_test(self, concurrent_users: int = 5, duration_seconds: int = 30) -> List[TestResult]:
        """Run load test with multiple concurrent users"""
        logger.info(f"🔍 Running Load Test - {concurrent_users} users for {duration_seconds}s...")
        results = []
        
        def user_load_test(user_id: int):
            """Single user load test"""
            user_results = []
            end_time = time.time() + duration_seconds
            
            while time.time() < end_time:
                # Random user actions
                actions = [
                    lambda: self._make_request('GET', '/api/health'),
                    lambda: self._make_request('GET', '/api/trades'),
                    lambda: self._make_request('GET', '/api/eas'),
                    lambda: self._make_request('GET', '/api/vision/status'),
                    lambda: self._make_request('GET', f'/api/trades/history?timeframe={random.choice(self.test_timeframes)}')
                ]
                
                action = random.choice(actions)
                result = action()
                result.test_name = f"LoadTest-User{user_id}-{result.test_name}"
                user_results.append(result)
                
                time.sleep(random.uniform(0.5, 2.0))  # Random delay between actions
            
            return user_results
        
        # Run concurrent users
        threads = []
        for user_id in range(concurrent_users):
            thread = threading.Thread(target=lambda uid=user_id: results.extend(user_load_test(uid)))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        return results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.success])
        failed_tests = total_tests - successful_tests
        
        # Calculate performance metrics
        response_times = [r.response_time for r in self.test_results if r.response_time > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        
        # Group results by test category
        test_categories = {}
        for result in self.test_results:
            category = result.test_name.split('-')[0] if '-' in result.test_name else 'API'
            if category not in test_categories:
                test_categories[category] = {'total': 0, 'passed': 0, 'failed': 0}
            
            test_categories[category]['total'] += 1
            if result.success:
                test_categories[category]['passed'] += 1
            else:
                test_categories[category]['failed'] += 1
        
        # Identify failed tests
        failed_test_details = [
            {
                'test_name': r.test_name,
                'error': r.error_message,
                'status_code': r.status_code
            }
            for r in self.test_results if not r.success
        ]
        
        simulation_duration = (datetime.now() - self.simulation_start_time).total_seconds()
        
        report = {
            'simulation_summary': {
                'start_time': self.simulation_start_time.isoformat(),
                'duration_seconds': simulation_duration,
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0
            },
            'performance_metrics': {
                'average_response_time': round(avg_response_time, 3),
                'max_response_time': round(max_response_time, 3),
                'min_response_time': round(min_response_time, 3),
                'total_requests': total_tests,
                'requests_per_second': round(total_tests / simulation_duration, 2) if simulation_duration > 0 else 0
            },
            'test_categories': test_categories,
            'failed_tests': failed_test_details,
            'websocket_test': {
                'connected': self.ws_connected,
                'messages_received': len(self.ws_messages)
            }
        }
        
        return report
    
    def run_full_simulation(self) -> Dict[str, Any]:
        """Run complete user path simulation"""
        logger.info("🚀 Starting QNTI Comprehensive User Path Simulation...")
        logger.info("=" * 60)
        
        all_results = []
        
        # Run all test categories
        test_suites = [
            ('Dashboard Access', self.test_dashboard_access),
            ('System Health', self.test_system_health_monitoring),
            ('Trade Management', self.test_trade_management),
            ('EA Management', self.test_ea_management),
            ('Vision Analysis', self.test_vision_analysis_workflow),
            ('System Controls', self.test_system_controls),
            ('Error Handling', self.test_error_handling),
            ('Realistic User Session', self.simulate_realistic_user_session),
        ]
        
        for suite_name, test_function in test_suites:
            logger.info(f"\n📋 Running {suite_name} Tests...")
            try:
                results = test_function()
                all_results.extend(results)
                passed = len([r for r in results if r.success])
                total = len(results)
                logger.info(f"✅ {suite_name}: {passed}/{total} tests passed")
            except Exception as e:
                logger.error(f"❌ {suite_name} failed: {e}")
        
        # Test WebSocket connection
        logger.info(f"\n📡 Testing WebSocket Connection...")
        ws_result = self.test_websocket_connection()
        all_results.append(ws_result)
        
        # Run load test
        logger.info(f"\n⚡ Running Load Test...")
        load_results = self.run_load_test(concurrent_users=3, duration_seconds=15)
        all_results.extend(load_results)
        
        # Generate final report
        report = self.generate_comprehensive_report()
        
        # Save detailed results
        detailed_results = {
            'report': report,
            'all_test_results': [
                {
                    'test_name': r.test_name,
                    'success': r.success,
                    'response_time': r.response_time,
                    'status_code': r.status_code,
                    'error_message': r.error_message
                }
                for r in self.test_results
            ]
        }
        
        # Save to file
        results_file = f"qnti_simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("🎯 SIMULATION COMPLETE!")
        logger.info(f"📊 Total Tests: {report['simulation_summary']['total_tests']}")
        logger.info(f"✅ Success Rate: {report['simulation_summary']['success_rate']:.1f}%")
        logger.info(f"⚡ Avg Response Time: {report['performance_metrics']['average_response_time']}s")
        logger.info(f"📁 Detailed results saved to: {results_file}")
        logger.info("=" * 60)
        
        return report

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QNTI User Path Simulation")
    parser.add_argument('--url', default='http://localhost:5000', help='QNTI server URL')
    parser.add_argument('--load-test', action='store_true', help='Run extended load test')
    parser.add_argument('--users', type=int, default=5, help='Number of concurrent users for load test')
    parser.add_argument('--duration', type=int, default=30, help='Load test duration in seconds')
    
    args = parser.parse_args()
    
    # Initialize simulator
    simulator = QNTIUserPathSimulator(base_url=args.url)
    
    if args.load_test:
        logger.info(f"Running extended load test with {args.users} users for {args.duration}s")
        results = simulator.run_load_test(concurrent_users=args.users, duration_seconds=args.duration)
        report = simulator.generate_comprehensive_report()
    else:
        # Run full simulation
        report = simulator.run_full_simulation()
    
    return report

if __name__ == "__main__":
    main() 