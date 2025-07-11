#!/usr/bin/env python3
"""
QNTI Full System Simulation - Master Test Runner
Comprehensive testing of all user paths including backend API and frontend browser automation
"""

import os
import sys
import json
import time
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import argparse

# Import our simulation modules
try:
    from qnti_automated_user_path_simulation import QNTIUserPathSimulator
except ImportError:
    print("❌ Backend simulation module not found. Please ensure qnti_automated_user_path_simulation.py exists.")
    sys.exit(1)

try:
    from qnti_browser_automation import QNTIBrowserAutomation
except ImportError:
    print("❌ Browser automation module not found. Please ensure qnti_browser_automation.py exists.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qnti_full_simulation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('QNTI_FULL_SIM')

class QNTIFullSystemSimulation:
    """Master class for running comprehensive QNTI system simulation"""
    
    def __init__(self, base_url: str = "http://localhost:5000", headless: bool = False):
        self.base_url = base_url.rstrip('/')
        self.headless = headless
        self.simulation_start_time = datetime.now()
        self.results = {}
        
        # Initialize simulators
        self.backend_simulator = QNTIUserPathSimulator(base_url=base_url)
        self.frontend_simulator = QNTIBrowserAutomation(base_url=base_url, headless=headless)
        
        logger.info(f"QNTI Full System Simulation initialized for {base_url}")
    
    def check_server_availability(self) -> bool:
        """Check if QNTI server is running and accessible"""
        logger.info("🔍 Checking QNTI server availability...")
        
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ QNTI server is running and accessible")
                return True
            else:
                logger.warning(f"⚠️ QNTI server responded with status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ QNTI server not accessible: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install required dependencies for simulation"""
        logger.info("📦 Installing required dependencies...")
        
        dependencies = [
            "requests",
            "websocket-client",
            "Pillow",
            "selenium"
        ]
        
        for dep in dependencies:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep], 
                                    capture_output=True, text=True)
                logger.info(f"✅ Installed {dep}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {dep}: {e}")
                return False
        
        return True
    
    def run_backend_simulation(self) -> Dict[str, Any]:
        """Run backend API simulation"""
        logger.info("🔧 Starting Backend API Simulation...")
        
        try:
            backend_results = self.backend_simulator.run_full_simulation()
            logger.info("✅ Backend simulation completed successfully")
            return backend_results
        except Exception as e:
            logger.error(f"❌ Backend simulation failed: {e}")
            return {"error": str(e), "success": False}
    
    def run_frontend_simulation(self) -> Dict[str, Any]:
        """Run frontend browser automation"""
        logger.info("🌐 Starting Frontend Browser Automation...")
        
        try:
            frontend_results = self.frontend_simulator.run_all_tests()
            logger.info("✅ Frontend simulation completed successfully")
            return frontend_results
        except Exception as e:
            logger.error(f"❌ Frontend simulation failed: {e}")
            return {"error": str(e), "success": False}
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance and load tests"""
        logger.info("⚡ Starting Performance Tests...")
        
        try:
            # Run load test with different configurations
            load_results = []
            
            # Light load test
            logger.info("Running light load test (3 users, 15 seconds)...")
            light_load = self.backend_simulator.run_load_test(concurrent_users=3, duration_seconds=15)
            load_results.append({
                "test_type": "light_load",
                "users": 3,
                "duration": 15,
                "results": len(light_load)
            })
            
            # Medium load test
            logger.info("Running medium load test (5 users, 30 seconds)...")
            medium_load = self.backend_simulator.run_load_test(concurrent_users=5, duration_seconds=30)
            load_results.append({
                "test_type": "medium_load",
                "users": 5,
                "duration": 30,
                "results": len(medium_load)
            })
            
            # Heavy load test
            logger.info("Running heavy load test (10 users, 20 seconds)...")
            heavy_load = self.backend_simulator.run_load_test(concurrent_users=10, duration_seconds=20)
            load_results.append({
                "test_type": "heavy_load",
                "users": 10,
                "duration": 20,
                "results": len(heavy_load)
            })
            
            return {
                "load_tests": load_results,
                "total_requests": len(light_load) + len(medium_load) + len(heavy_load),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Performance tests failed: {e}")
            return {"error": str(e), "success": False}
    
    def run_security_tests(self) -> Dict[str, Any]:
        """Run basic security tests"""
        logger.info("🔒 Starting Security Tests...")
        
        security_results = []
        
        try:
            import requests
            
            # Test SQL injection attempts
            sql_injection_payloads = [
                "' OR '1'='1",
                "'; DROP TABLE users; --",
                "1' UNION SELECT * FROM users--"
            ]
            
            for payload in sql_injection_payloads:
                try:
                    response = requests.get(f"{self.base_url}/api/trades", 
                                          params={"filter": payload}, timeout=5)
                    security_results.append({
                        "test": "SQL Injection",
                        "payload": payload,
                        "status_code": response.status_code,
                        "blocked": response.status_code >= 400
                    })
                except Exception as e:
                    security_results.append({
                        "test": "SQL Injection",
                        "payload": payload,
                        "error": str(e),
                        "blocked": True
                    })
            
            # Test XSS attempts
            xss_payloads = [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>"
            ]
            
            for payload in xss_payloads:
                try:
                    response = requests.post(f"{self.base_url}/api/eas/register", 
                                           json={"name": payload}, timeout=5)
                    security_results.append({
                        "test": "XSS",
                        "payload": payload,
                        "status_code": response.status_code,
                        "blocked": response.status_code >= 400
                    })
                except Exception as e:
                    security_results.append({
                        "test": "XSS",
                        "payload": payload,
                        "error": str(e),
                        "blocked": True
                    })
            
            # Test unauthorized access
            unauthorized_endpoints = [
                "/api/system/shutdown",
                "/api/admin/users",
                "/api/config/secret"
            ]
            
            for endpoint in unauthorized_endpoints:
                try:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                    security_results.append({
                        "test": "Unauthorized Access",
                        "endpoint": endpoint,
                        "status_code": response.status_code,
                        "blocked": response.status_code >= 400
                    })
                except Exception as e:
                    security_results.append({
                        "test": "Unauthorized Access",
                        "endpoint": endpoint,
                        "error": str(e),
                        "blocked": True
                    })
            
            blocked_attempts = len([r for r in security_results if r.get("blocked", False)])
            total_attempts = len(security_results)
            
            return {
                "security_tests": security_results,
                "total_attempts": total_attempts,
                "blocked_attempts": blocked_attempts,
                "security_score": (blocked_attempts / total_attempts * 100) if total_attempts > 0 else 0,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Security tests failed: {e}")
            return {"error": str(e), "success": False}
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive simulation report"""
        simulation_duration = (datetime.now() - self.simulation_start_time).total_seconds()
        
        # Calculate overall metrics
        backend_success = self.results.get("backend", {}).get("simulation_summary", {}).get("success_rate", 0)
        frontend_success = self.results.get("frontend", {}).get("test_summary", {}).get("success_rate", 0)
        
        # Overall system health score
        system_health_score = (backend_success + frontend_success) / 2
        
        # Performance metrics
        performance_data = self.results.get("performance", {})
        security_data = self.results.get("security", {})
        
        comprehensive_report = {
            "simulation_overview": {
                "start_time": self.simulation_start_time.isoformat(),
                "duration_seconds": round(simulation_duration, 2),
                "base_url": self.base_url,
                "headless_mode": self.headless
            },
            "system_health": {
                "overall_score": round(system_health_score, 2),
                "backend_success_rate": backend_success,
                "frontend_success_rate": frontend_success,
                "server_accessible": self.check_server_availability()
            },
            "test_results": {
                "backend": self.results.get("backend", {}),
                "frontend": self.results.get("frontend", {}),
                "performance": performance_data,
                "security": security_data
            },
            "recommendations": self._generate_recommendations(),
            "summary": {
                "total_backend_tests": self.results.get("backend", {}).get("simulation_summary", {}).get("total_tests", 0),
                "total_frontend_tests": self.results.get("frontend", {}).get("test_summary", {}).get("total_tests", 0),
                "total_performance_requests": performance_data.get("total_requests", 0),
                "total_security_tests": security_data.get("total_attempts", 0),
                "security_score": security_data.get("security_score", 0)
            }
        }
        
        return comprehensive_report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Backend recommendations
        backend_results = self.results.get("backend", {})
        if backend_results.get("simulation_summary", {}).get("success_rate", 0) < 90:
            recommendations.append("Backend API reliability needs improvement - success rate below 90%")
        
        # Frontend recommendations
        frontend_results = self.results.get("frontend", {})
        if frontend_results.get("test_summary", {}).get("success_rate", 0) < 90:
            recommendations.append("Frontend UI stability needs improvement - success rate below 90%")
        
        # Performance recommendations
        performance_results = self.results.get("performance", {})
        if performance_results.get("success", False):
            recommendations.append("Consider implementing caching for better performance under load")
        
        # Security recommendations
        security_results = self.results.get("security", {})
        security_score = security_results.get("security_score", 0)
        if security_score < 100:
            recommendations.append(f"Security vulnerabilities detected - {security_score}% of attacks blocked")
        
        if not recommendations:
            recommendations.append("All tests passed successfully - system is performing well")
        
        return recommendations
    
    def save_results(self, report: Dict[str, Any]) -> str:
        """Save simulation results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"qnti_full_simulation_report_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📁 Comprehensive report saved to: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
            return None
    
    def run_full_simulation(self) -> Dict[str, Any]:
        """Run complete system simulation"""
        logger.info("🚀 Starting QNTI Full System Simulation...")
        logger.info("=" * 80)
        
        # Check server availability
        if not self.check_server_availability():
            logger.error("❌ QNTI server is not accessible. Please start the server first.")
            return {"error": "Server not accessible", "success": False}
        
        # Install dependencies
        if not self.install_dependencies():
            logger.warning("⚠️ Some dependencies may be missing. Continuing with simulation...")
        
        # Run backend simulation
        logger.info("\n" + "=" * 40)
        logger.info("BACKEND API SIMULATION")
        logger.info("=" * 40)
        self.results["backend"] = self.run_backend_simulation()
        
        # Run frontend simulation
        logger.info("\n" + "=" * 40)
        logger.info("FRONTEND BROWSER AUTOMATION")
        logger.info("=" * 40)
        self.results["frontend"] = self.run_frontend_simulation()
        
        # Run performance tests
        logger.info("\n" + "=" * 40)
        logger.info("PERFORMANCE TESTING")
        logger.info("=" * 40)
        self.results["performance"] = self.run_performance_tests()
        
        # Run security tests
        logger.info("\n" + "=" * 40)
        logger.info("SECURITY TESTING")
        logger.info("=" * 40)
        self.results["security"] = self.run_security_tests()
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        # Save results
        report_file = self.save_results(report)
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("🎯 QNTI FULL SYSTEM SIMULATION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"📊 Overall System Health Score: {report['system_health']['overall_score']:.1f}%")
        logger.info(f"🔧 Backend Success Rate: {report['system_health']['backend_success_rate']:.1f}%")
        logger.info(f"🌐 Frontend Success Rate: {report['system_health']['frontend_success_rate']:.1f}%")
        logger.info(f"🔒 Security Score: {report['summary']['security_score']:.1f}%")
        logger.info(f"⚡ Total Performance Requests: {report['summary']['total_performance_requests']}")
        logger.info(f"⏱️ Total Duration: {report['simulation_overview']['duration_seconds']}s")
        
        if report_file:
            logger.info(f"📁 Comprehensive report: {report_file}")
        
        logger.info("\n🔍 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            logger.info(f"  {i}. {rec}")
        
        logger.info("=" * 80)
        
        return report

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="QNTI Full System Simulation")
    parser.add_argument('--url', default='http://localhost:5000', help='QNTI server URL')
    parser.add_argument('--headless', action='store_true', help='Run browser tests in headless mode')
    parser.add_argument('--backend-only', action='store_true', help='Run only backend tests')
    parser.add_argument('--frontend-only', action='store_true', help='Run only frontend tests')
    parser.add_argument('--performance-only', action='store_true', help='Run only performance tests')
    parser.add_argument('--security-only', action='store_true', help='Run only security tests')
    
    args = parser.parse_args()
    
    # Initialize full simulation
    full_sim = QNTIFullSystemSimulation(base_url=args.url, headless=args.headless)
    
    # Run specific test suites based on arguments
    if args.backend_only:
        logger.info("Running backend tests only...")
        results = full_sim.run_backend_simulation()
    elif args.frontend_only:
        logger.info("Running frontend tests only...")
        results = full_sim.run_frontend_simulation()
    elif args.performance_only:
        logger.info("Running performance tests only...")
        results = full_sim.run_performance_tests()
    elif args.security_only:
        logger.info("Running security tests only...")
        results = full_sim.run_security_tests()
    else:
        # Run full simulation
        results = full_sim.run_full_simulation()
    
    return results

if __name__ == "__main__":
    main() 