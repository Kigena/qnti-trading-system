#!/usr/bin/env python3
"""
QNTI Full System Simulation Runner
Orchestrates complete system testing including functional, load, and stress testing
"""

import asyncio
import logging
import sys
import time
import json
import subprocess
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import argparse

# Import our testing modules
from qnti_automation_suite import QNTIAutomationSuite, SimulationConfig
from qnti_stress_testing import QNTIStressTestSuite, StressTestConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'qnti_full_simulation_{int(time.time())}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('QNTI_FULL_SIMULATION')

class QNTISystemChecker:
    """Check if QNTI system is running and healthy"""
    
    def __init__(self, url: str = "http://localhost:5000"):
        self.url = url
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Check if QNTI system is running and responsive"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                # Check basic connectivity
                async with session.get(f"{self.url}/api/system/health", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'status': 'healthy',
                            'response_time': response.headers.get('X-Response-Time', 'N/A'),
                            'data': data
                        }
                    else:
                        return {
                            'status': 'unhealthy',
                            'error': f'HTTP {response.status}'
                        }
        except asyncio.TimeoutError:
            return {
                'status': 'timeout',
                'error': 'System health check timed out'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def start_qnti_if_needed(self) -> bool:
        """Start QNTI system if it's not running"""
        health = await self.check_system_health()
        
        if health['status'] == 'healthy':
            logger.info("✅ QNTI system is already running")
            return True
        
        logger.info("🚀 Starting QNTI system...")
        
        try:
            # Start QNTI system in background
            subprocess.Popen([
                sys.executable, 'qnti_main_system.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for system to start (up to 2 minutes)
            for _ in range(24):  # 24 * 5 seconds = 2 minutes
                await asyncio.sleep(5)
                health = await self.check_system_health()
                if health['status'] == 'healthy':
                    logger.info("✅ QNTI system started successfully")
                    return True
            
            logger.error("❌ QNTI system failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to start QNTI system: {e}")
            return False

class SimulationReport:
    """Generate comprehensive simulation report"""
    
    def __init__(self):
        self.results = {
            'simulation_metadata': {
                'start_time': datetime.now().isoformat(),
                'end_time': None,
                'total_duration': 0,
                'qnti_version': 'Unknown',
                'python_version': sys.version,
                'platform': sys.platform
            },
            'system_health': {},
            'functional_tests': {},
            'stress_tests': {},
            'performance_benchmarks': {},
            'overall_assessment': {},
            'recommendations': []
        }
    
    def add_system_health(self, health_data: Dict[str, Any]):
        """Add system health check results"""
        self.results['system_health'] = health_data
    
    def add_functional_tests(self, test_results: Dict[str, Any]):
        """Add functional test results"""
        self.results['functional_tests'] = test_results
    
    def add_stress_tests(self, stress_results: Dict[str, Any]):
        """Add stress test results"""
        self.results['stress_tests'] = stress_results
    
    def calculate_overall_assessment(self):
        """Calculate overall system assessment"""
        scores = []
        
        # Functional test score
        if 'functional_tests' in self.results and self.results['functional_tests']:
            functional_success_rate = self.results['functional_tests'].get('success_rate', 0)
            scores.append(('functional', functional_success_rate * 100))
        
        # Stress test score
        if 'stress_tests' in self.results and self.results['stress_tests']:
            stress_grade = self.results['stress_tests'].get('performance_grade', 'F')
            grade_scores = {'A': 95, 'B': 85, 'C': 75, 'D': 65, 'F': 50}
            scores.append(('stress', grade_scores.get(stress_grade, 50)))
        
        # System health score
        if 'system_health' in self.results and self.results['system_health']:
            health_status = self.results['system_health'].get('status', 'error')
            health_score = 100 if health_status == 'healthy' else 0
            scores.append(('health', health_score))
        
        if scores:
            overall_score = sum(score for _, score in scores) / len(scores)
            
            if overall_score >= 90:
                rating = "Excellent"
                color = "🟢"
            elif overall_score >= 80:
                rating = "Good"
                color = "🟡"
            elif overall_score >= 70:
                rating = "Fair"
                color = "🟠"
            else:
                rating = "Poor"
                color = "🔴"
            
            self.results['overall_assessment'] = {
                'score': overall_score,
                'rating': rating,
                'color': color,
                'component_scores': dict(scores)
            }
        
        return self.results['overall_assessment']
    
    def generate_recommendations(self):
        """Generate recommendations based on all test results"""
        recommendations = []
        
        # Functional test recommendations
        if 'functional_tests' in self.results:
            success_rate = self.results['functional_tests'].get('success_rate', 0)
            if success_rate < 0.95:
                recommendations.append({
                    'category': 'Functional',
                    'priority': 'High',
                    'issue': f'Functional test success rate is {success_rate:.1%}',
                    'recommendation': 'Review failed functional tests and fix underlying issues'
                })
            
            avg_response_time = self.results['functional_tests'].get('avg_response_time', 0)
            if avg_response_time > 2.0:
                recommendations.append({
                    'category': 'Performance',
                    'priority': 'Medium',
                    'issue': f'Average response time is {avg_response_time:.2f}s',
                    'recommendation': 'Optimize slow endpoints and consider caching strategies'
                })
        
        # Stress test recommendations
        if 'stress_tests' in self.results:
            bottlenecks = self.results['stress_tests'].get('identified_bottlenecks', [])
            for bottleneck in bottlenecks[:3]:  # Top 3 bottlenecks
                recommendations.append({
                    'category': 'Stress Testing',
                    'priority': 'High',
                    'issue': bottleneck,
                    'recommendation': 'Address this bottleneck to improve system scalability'
                })
        
        # System health recommendations
        if 'system_health' in self.results:
            if self.results['system_health'].get('status') != 'healthy':
                recommendations.append({
                    'category': 'System Health',
                    'priority': 'Critical',
                    'issue': 'System health check failed',
                    'recommendation': 'Investigate system startup and configuration issues'
                })
        
        # General recommendations
        if not recommendations:
            recommendations.append({
                'category': 'General',
                'priority': 'Low',
                'issue': 'No major issues detected',
                'recommendation': 'Continue monitoring and consider implementing additional test scenarios'
            })
        
        self.results['recommendations'] = recommendations
        return recommendations
    
    def save_report(self, filename: str = None):
        """Save comprehensive report to file"""
        if filename is None:
            filename = f"qnti_full_simulation_report_{int(time.time())}.json"
        
        # Finalize report
        self.results['simulation_metadata']['end_time'] = datetime.now().isoformat()
        start_time = datetime.fromisoformat(self.results['simulation_metadata']['start_time'])
        end_time = datetime.fromisoformat(self.results['simulation_metadata']['end_time'])
        self.results['simulation_metadata']['total_duration'] = (end_time - start_time).total_seconds()
        
        # Calculate final assessment
        self.calculate_overall_assessment()
        self.generate_recommendations()
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"📊 Comprehensive simulation report saved: {filename}")
        return filename
    
    def print_summary(self):
        """Print executive summary"""
        assessment = self.results.get('overall_assessment', {})
        
        print("\n" + "="*80)
        print("🎯 QNTI FULL SYSTEM SIMULATION RESULTS")
        print("="*80)
        
        if assessment:
            print(f"{assessment.get('color', '⚪')} Overall Rating: {assessment.get('rating', 'Unknown')} ({assessment.get('score', 0):.1f}/100)")
            
            if 'component_scores' in assessment:
                print("\n📊 Component Scores:")
                for component, score in assessment['component_scores'].items():
                    print(f"   {component.capitalize()}: {score:.1f}/100")
        
        # System health summary
        if 'system_health' in self.results:
            health = self.results['system_health']
            status_emoji = "✅" if health.get('status') == 'healthy' else "❌"
            print(f"\n{status_emoji} System Health: {health.get('status', 'Unknown')}")
        
        # Test summaries
        if 'functional_tests' in self.results and self.results['functional_tests']:
            ft = self.results['functional_tests']
            print(f"\n🔧 Functional Tests: {ft.get('successful_tests', 0)}/{ft.get('total_tests', 0)} passed ({ft.get('success_rate', 0):.1%})")
            print(f"   Average Response Time: {ft.get('avg_response_time', 0):.3f}s")
        
        if 'stress_tests' in self.results and self.results['stress_tests']:
            st = self.results['stress_tests']
            grade = st.get('performance_grade', 'N/A')
            print(f"\n🔥 Stress Tests: Grade {grade}")
            
            if 'performance_statistics' in st:
                stats = st['performance_statistics']
                if 'request_stats' in stats:
                    rs = stats['request_stats']
                    print(f"   Requests: {rs.get('total', 0)} total, {rs.get('success_rate', 0):.1%} success rate")
        
        # Top recommendations
        if 'recommendations' in self.results and self.results['recommendations']:
            print(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(self.results['recommendations'][:3], 1):
                priority_emoji = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(rec['priority'], "⚪")
                print(f"   {i}. {priority_emoji} [{rec['category']}] {rec['recommendation']}")
        
        print("\n📋 Detailed reports saved to JSON files")
        print("="*80)

class QNTIFullSimulationRunner:
    """Main orchestrator for full QNTI system simulation"""
    
    def __init__(self, qnti_url: str = "http://localhost:5000"):
        self.qnti_url = qnti_url
        self.system_checker = QNTISystemChecker(qnti_url)
        self.report = SimulationReport()
    
    async def run_full_simulation(self, config: Dict[str, Any]) -> SimulationReport:
        """Run comprehensive simulation suite"""
        logger.info("🚀 Starting QNTI Full System Simulation")
        
        try:
            # Phase 1: System health and startup
            await self._check_system_health()
            
            # Phase 2: Functional testing
            await self._run_functional_tests(config)
            
            # Phase 3: Stress testing (if enabled)
            if config.get('include_stress_tests', True):
                await self._run_stress_tests(config)
            
            # Phase 4: Performance benchmarking
            await self._run_performance_benchmarks(config)
            
        except Exception as e:
            logger.error(f"❌ Simulation failed: {e}")
            self.report.results['overall_assessment'] = {
                'score': 0,
                'rating': 'Failed',
                'color': '🔴',
                'error': str(e)
            }
        
        finally:
            # Generate final report
            report_file = self.report.save_report()
            self.report.print_summary()
        
        return self.report
    
    async def _check_system_health(self):
        """Check and ensure system health"""
        logger.info("🏥 Checking system health...")
        
        # Start system if needed
        system_ready = await self.system_checker.start_qnti_if_needed()
        
        if system_ready:
            health = await self.system_checker.check_system_health()
            self.report.add_system_health(health)
            logger.info(f"✅ System health: {health['status']}")
        else:
            health = {'status': 'failed', 'error': 'Could not start QNTI system'}
            self.report.add_system_health(health)
            logger.error("❌ System health check failed")
    
    async def _run_functional_tests(self, config: Dict[str, Any]):
        """Run functional testing suite"""
        logger.info("🔧 Running functional tests...")
        
        # Configure functional tests
        sim_config = SimulationConfig(
            qnti_url=self.qnti_url,
            simulation_duration=config.get('functional_test_duration', 300),
            max_concurrent_users=config.get('functional_test_users', 5),
            ea_strategies_to_test=config.get('ea_strategies', [
                "Functional Test Strategy 1",
                "Functional Test Strategy 2"
            ])
        )
        
        # Run automation suite
        automation_suite = QNTIAutomationSuite(sim_config)
        results = await automation_suite.run_comprehensive_simulation()
        
        # Convert results to dict format
        functional_results = {
            'total_tests': results.total_tests,
            'successful_tests': results.successful_tests,
            'failed_tests': results.failed_tests,
            'success_rate': results.success_rate,
            'avg_response_time': results.avg_response_time,
            'max_response_time': results.max_response_time,
            'min_response_time': results.min_response_time,
            'errors': results.errors,
            'ea_generation_results': results.ea_generation_results,
            'performance_metrics': results.performance_metrics
        }
        
        self.report.add_functional_tests(functional_results)
        logger.info(f"✅ Functional tests completed: {results.success_rate:.1%} success rate")
    
    async def _run_stress_tests(self, config: Dict[str, Any]):
        """Run stress testing suite"""
        logger.info("🔥 Running stress tests...")
        
        # Configure stress tests
        stress_config = StressTestConfig(
            qnti_url=self.qnti_url,
            test_duration=config.get('stress_test_duration', 1800),
            max_concurrent_users=config.get('stress_test_users', 50),
            requests_per_second=config.get('stress_rps', 100),
            concurrent_ea_generations=config.get('stress_ea_concurrent', 10)
        )
        
        # Run stress testing suite
        stress_suite = QNTIStressTestSuite(stress_config)
        results = await stress_suite.run_comprehensive_stress_test()
        
        self.report.add_stress_tests(results)
        grade = results.get('performance_grade', 'Unknown')
        logger.info(f"🔥 Stress tests completed: Grade {grade}")
    
    async def _run_performance_benchmarks(self, config: Dict[str, Any]):
        """Run performance benchmarking"""
        logger.info("📈 Running performance benchmarks...")
        
        # This could include additional specialized performance tests
        # For now, we'll extract performance data from previous tests
        
        benchmarks = {
            'endpoint_response_times': {},
            'throughput_metrics': {},
            'resource_utilization': {},
            'scalability_assessment': {}
        }
        
        # Extract performance data from functional and stress tests
        if 'functional_tests' in self.report.results:
            ft = self.report.results['functional_tests']
            benchmarks['functional_performance'] = {
                'avg_response_time': ft.get('avg_response_time', 0),
                'max_response_time': ft.get('max_response_time', 0),
                'throughput': ft.get('total_tests', 0) / max(1, config.get('functional_test_duration', 300)) * 60
            }
        
        if 'stress_tests' in self.report.results:
            st = self.report.results['stress_tests']
            if 'scalability_analysis' in st:
                benchmarks['scalability_assessment'] = st['scalability_analysis']
        
        self.report.results['performance_benchmarks'] = benchmarks
        logger.info("📈 Performance benchmarks completed")

# CLI interface
async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="QNTI Full System Simulation")
    parser.add_argument('--url', default='http://localhost:5000', help='QNTI system URL')
    parser.add_argument('--quick', action='store_true', help='Run quick simulation (5 minutes)')
    parser.add_argument('--functional-only', action='store_true', help='Run functional tests only')
    parser.add_argument('--stress-only', action='store_true', help='Run stress tests only')
    parser.add_argument('--duration', type=int, help='Test duration in seconds')
    parser.add_argument('--users', type=int, help='Max concurrent users')
    parser.add_argument('--no-stress', action='store_true', help='Skip stress testing')
    
    args = parser.parse_args()
    
    # Create configuration based on arguments
    if args.quick:
        config = {
            'functional_test_duration': 300,  # 5 minutes
            'functional_test_users': 3,
            'stress_test_duration': 600,      # 10 minutes
            'stress_test_users': 10,
            'stress_rps': 20,
            'stress_ea_concurrent': 3,
            'include_stress_tests': not args.no_stress,
            'ea_strategies': ['Quick Test Strategy']
        }
    else:
        config = {
            'functional_test_duration': args.duration or 1800,  # 30 minutes
            'functional_test_users': args.users or 10,
            'stress_test_duration': args.duration or 3600,     # 1 hour
            'stress_test_users': args.users or 50,
            'stress_rps': 100,
            'stress_ea_concurrent': 10,
            'include_stress_tests': not args.no_stress,
            'ea_strategies': [
                'Comprehensive Test Strategy 1',
                'Comprehensive Test Strategy 2', 
                'Comprehensive Test Strategy 3'
            ]
        }
    
    # Override with specific test modes
    if args.functional_only:
        config['include_stress_tests'] = False
    
    if args.stress_only:
        config['functional_test_duration'] = 60  # Minimal functional testing
        config['functional_test_users'] = 1
    
    # Run simulation
    simulator = QNTIFullSimulationRunner(args.url)
    report = await simulator.run_full_simulation(config)
    
    return report

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("❌ Simulation interrupted by user")
        print("\n🛑 Simulation stopped by user")
    except Exception as e:
        logger.error(f"❌ Simulation failed: {e}")
        print(f"\n💥 Simulation failed: {e}")
        sys.exit(1)