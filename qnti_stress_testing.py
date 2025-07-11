#!/usr/bin/env python3
"""
QNTI Stress Testing Suite - Advanced Performance Testing
Specialized stress testing for EA Generation System and overall QNTI performance
"""

import asyncio
import logging
import time
import json
import random
import threading
import statistics
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import requests
import websocket
import queue
import multiprocessing

logger = logging.getLogger('QNTI_STRESS_TEST')

@dataclass
class StressTestConfig:
    """Configuration for stress testing"""
    # Basic settings
    qnti_url: str = "http://localhost:5000"
    test_duration: int = 1800  # 30 minutes
    
    # Load parameters
    max_concurrent_users: int = 50
    requests_per_second: int = 100
    ramp_up_time: int = 300  # 5 minutes
    
    # EA generation stress
    concurrent_ea_generations: int = 10
    ea_complexity_levels: List[str] = field(default_factory=lambda: ["simple", "medium", "complex"])
    
    # Memory and resource limits
    max_memory_usage_mb: int = 2048
    max_cpu_usage_percent: float = 80.0
    
    # Failure simulation
    simulate_network_issues: bool = True
    simulate_high_latency: bool = True
    error_injection_rate: float = 0.05  # 5% error injection

@dataclass
class StressTestMetrics:
    """Metrics collected during stress testing"""
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    
    # Resource metrics
    cpu_usage_samples: List[float] = field(default_factory=list)
    memory_usage_samples: List[float] = field(default_factory=list)
    
    # EA generation metrics
    ea_generations_started: int = 0
    ea_generations_completed: int = 0
    ea_generations_failed: int = 0
    
    # System stability metrics
    connection_drops: int = 0
    timeout_errors: int = 0
    server_errors: int = 0
    
    # Performance breakdown
    endpoint_performance: Dict[str, List[float]] = field(default_factory=dict)
    
    def add_response_time(self, endpoint: str, response_time: float):
        """Add response time measurement"""
        self.response_times.append(response_time)
        if endpoint not in self.endpoint_performance:
            self.endpoint_performance[endpoint] = []
        self.endpoint_performance[endpoint].append(response_time)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        stats = {
            'request_stats': {
                'total': self.total_requests,
                'successful': self.successful_requests,
                'failed': self.failed_requests,
                'success_rate': self.successful_requests / max(1, self.total_requests),
                'failure_rate': self.failed_requests / max(1, self.total_requests)
            },
            'response_time_stats': {},
            'resource_stats': {},
            'ea_generation_stats': {
                'started': self.ea_generations_started,
                'completed': self.ea_generations_completed,
                'failed': self.ea_generations_failed,
                'completion_rate': self.ea_generations_completed / max(1, self.ea_generations_started)
            }
        }
        
        if self.response_times:
            stats['response_time_stats'] = {
                'mean': statistics.mean(self.response_times),
                'median': statistics.median(self.response_times),
                'min': min(self.response_times),
                'max': max(self.response_times),
                'p95': self._percentile(self.response_times, 95),
                'p99': self._percentile(self.response_times, 99)
            }
        
        if self.cpu_usage_samples:
            stats['resource_stats']['cpu'] = {
                'mean': statistics.mean(self.cpu_usage_samples),
                'max': max(self.cpu_usage_samples),
                'min': min(self.cpu_usage_samples)
            }
        
        if self.memory_usage_samples:
            stats['resource_stats']['memory'] = {
                'mean': statistics.mean(self.memory_usage_samples),
                'max': max(self.memory_usage_samples),
                'min': min(self.memory_usage_samples)
            }
        
        return stats
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * percentile / 100
        f = int(k)
        c = k - f
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c

class SystemResourceMonitor:
    """Monitor system resources during stress testing"""
    
    def __init__(self, metrics: StressTestMetrics):
        self.metrics = metrics
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Resource monitoring loop"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics.cpu_usage_samples.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                self.metrics.memory_usage_samples.append(memory_mb)
                
                time.sleep(5)  # Sample every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")

class EAGenerationStressTester:
    """Stress test EA generation functionality"""
    
    def __init__(self, base_url: str, metrics: StressTestMetrics):
        self.base_url = base_url
        self.metrics = metrics
        self.session = requests.Session()
    
    async def run_concurrent_ea_generations(self, count: int, complexity: str = "medium"):
        """Run multiple concurrent EA generations"""
        tasks = []
        
        for i in range(count):
            task = self._generate_single_ea(f"StressTest_EA_{i}_{complexity}", complexity)
            tasks.append(task)
        
        # Run with limited concurrency to avoid overwhelming the system
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent EA generations
        
        async def limited_task(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[limited_task(task) for task in tasks])
        return results
    
    async def _generate_single_ea(self, ea_name: str, complexity: str) -> Dict[str, Any]:
        """Generate a single EA with specified complexity"""
        try:
            self.metrics.ea_generations_started += 1
            
            # Create EA configuration based on complexity
            ea_config = self._create_ea_config(ea_name, complexity)
            
            start_time = time.time()
            
            # Start EA generation workflow
            response = self.session.post(
                f"{self.base_url}/api/ea/workflow/start",
                json=ea_config,
                headers={'Content-Type': 'application/json'},
                timeout=60
            )
            
            response_time = time.time() - start_time
            self.metrics.add_response_time('ea_workflow_start', response_time)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    workflow_id = data.get('workflow_id')
                    
                    # Monitor workflow completion
                    completion_result = await self._monitor_workflow_completion(workflow_id)
                    
                    if completion_result['success']:
                        self.metrics.ea_generations_completed += 1
                    else:
                        self.metrics.ea_generations_failed += 1
                    
                    return completion_result
                else:
                    self.metrics.ea_generations_failed += 1
                    return {'success': False, 'error': data.get('error', 'Unknown error')}
            else:
                self.metrics.ea_generations_failed += 1
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            self.metrics.ea_generations_failed += 1
            return {'success': False, 'error': str(e)}
    
    def _create_ea_config(self, ea_name: str, complexity: str) -> Dict[str, Any]:
        """Create EA configuration based on complexity level"""
        base_config = {
            'ea_name': ea_name,
            'description': f'Stress test EA - {complexity} complexity',
            'symbols': ['EURUSD'],
            'timeframes': ['H1'],
            'method': 'genetic_algorithm',
            'auto_proceed': True
        }
        
        if complexity == "simple":
            base_config['indicators'] = [
                {'name': 'SMA', 'params': {}},
                {'name': 'RSI', 'params': {}}
            ]
        elif complexity == "medium":
            base_config['indicators'] = [
                {'name': 'SMA', 'params': {}},
                {'name': 'EMA', 'params': {}},
                {'name': 'RSI', 'params': {}},
                {'name': 'MACD', 'params': {}},
                {'name': 'Bollinger Bands', 'params': {}}
            ]
            base_config['symbols'] = ['EURUSD', 'GBPUSD']
        else:  # complex
            base_config['indicators'] = [
                {'name': 'SMA', 'params': {}},
                {'name': 'EMA', 'params': {}},
                {'name': 'RSI', 'params': {}},
                {'name': 'MACD', 'params': {}},
                {'name': 'Bollinger Bands', 'params': {}},
                {'name': 'Stochastic', 'params': {}},
                {'name': 'ATR', 'params': {}},
                {'name': 'CCI', 'params': {}}
            ]
            base_config['symbols'] = ['EURUSD', 'GBPUSD', 'USDJPY']
            base_config['timeframes'] = ['H1', 'H4']
        
        return base_config
    
    async def _monitor_workflow_completion(self, workflow_id: str, timeout: int = 600) -> Dict[str, Any]:
        """Monitor workflow until completion or timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(
                    f"{self.base_url}/api/ea/workflow/status/{workflow_id}",
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        status = data.get('status')
                        if status == 'completed':
                            return {'success': True, 'workflow_id': workflow_id}
                        elif status == 'failed':
                            return {'success': False, 'error': 'Workflow failed'}
                        # Still running, continue monitoring
                    else:
                        return {'success': False, 'error': 'Failed to get workflow status'}
                else:
                    return {'success': False, 'error': f'HTTP {response.status_code}'}
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': 'Workflow timeout'}

class APILoadTester:
    """Generate high load on API endpoints"""
    
    def __init__(self, base_url: str, metrics: StressTestMetrics):
        self.base_url = base_url
        self.metrics = metrics
        self.session = requests.Session()
    
    async def run_load_test(self, requests_per_second: int, duration: int):
        """Run sustained load test"""
        start_time = time.time()
        end_time = start_time + duration
        
        # List of endpoints to test
        endpoints = [
            ('GET', '/api/system/health'),
            ('GET', '/api/ea/indicators'),
            ('GET', '/api/ea/workflow/list'),
            ('GET', '/api/ai/insights/all'),
        ]
        
        request_interval = 1.0 / requests_per_second
        
        while time.time() < end_time:
            # Select random endpoint
            method, endpoint = random.choice(endpoints)
            
            # Make request
            asyncio.create_task(self._make_request(method, endpoint))
            
            # Wait for next request
            await asyncio.sleep(request_interval)
    
    async def _make_request(self, method: str, endpoint: str):
        """Make a single HTTP request"""
        try:
            start_time = time.time()
            
            if method == 'GET':
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=10)
            elif method == 'POST':
                response = self.session.post(f"{self.base_url}{endpoint}", timeout=10)
            
            response_time = time.time() - start_time
            
            self.metrics.total_requests += 1
            self.metrics.add_response_time(endpoint, response_time)
            
            if 200 <= response.status_code < 300:
                self.metrics.successful_requests += 1
            else:
                self.metrics.failed_requests += 1
                if response.status_code >= 500:
                    self.metrics.server_errors += 1
            
        except requests.exceptions.Timeout:
            self.metrics.timeout_errors += 1
            self.metrics.failed_requests += 1
        except requests.exceptions.ConnectionError:
            self.metrics.connection_drops += 1
            self.metrics.failed_requests += 1
        except Exception as e:
            self.metrics.failed_requests += 1
            logger.error(f"Request error: {e}")

class QNTIStressTestSuite:
    """Main stress testing orchestrator"""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.metrics = StressTestMetrics()
        self.resource_monitor = SystemResourceMonitor(self.metrics)
        self.ea_tester = EAGenerationStressTester(config.qnti_url, self.metrics)
        self.load_tester = APILoadTester(config.qnti_url, self.metrics)
    
    async def run_comprehensive_stress_test(self) -> Dict[str, Any]:
        """Run comprehensive stress test"""
        logger.info("🔥 Starting QNTI Comprehensive Stress Test")
        start_time = time.time()
        
        try:
            # Start resource monitoring
            self.resource_monitor.start_monitoring()
            
            # Run tests concurrently
            tasks = [
                self._run_ea_generation_stress(),
                self._run_api_load_stress(),
                self._run_endurance_test()
            ]
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
        
        finally:
            # Stop monitoring
            self.resource_monitor.stop_monitoring()
            
            # Generate results
            end_time = time.time()
            duration = end_time - start_time
            
            results = self._generate_stress_test_report(duration)
            
        return results
    
    async def _run_ea_generation_stress(self):
        """Run EA generation stress test"""
        logger.info("🤖 Running EA generation stress test")
        
        # Test different complexity levels
        for complexity in self.config.ea_complexity_levels:
            logger.info(f"Testing {complexity} EA generation")
            
            results = await self.ea_tester.run_concurrent_ea_generations(
                count=self.config.concurrent_ea_generations,
                complexity=complexity
            )
            
            logger.info(f"Completed {complexity} EA generation: {len([r for r in results if r.get('success')])} successful")
    
    async def _run_api_load_stress(self):
        """Run API load stress test"""
        logger.info("⚡ Running API load stress test")
        
        await self.load_tester.run_load_test(
            requests_per_second=self.config.requests_per_second,
            duration=self.config.test_duration // 2  # Run for half the test duration
        )
    
    async def _run_endurance_test(self):
        """Run endurance test with sustained load"""
        logger.info("🏃 Running endurance test")
        
        # Sustained moderate load for full duration
        endurance_rps = self.config.requests_per_second // 4  # 25% of max load
        
        await self.load_tester.run_load_test(
            requests_per_second=endurance_rps,
            duration=self.config.test_duration
        )
    
    def _generate_stress_test_report(self, duration: float) -> Dict[str, Any]:
        """Generate comprehensive stress test report"""
        stats = self.metrics.get_statistics()
        
        # Performance analysis
        performance_grade = self._calculate_performance_grade(stats)
        bottlenecks = self._identify_bottlenecks(stats)
        recommendations = self._generate_recommendations(stats)
        
        report = {
            'test_summary': {
                'duration': duration,
                'test_type': 'comprehensive_stress_test',
                'timestamp': datetime.now().isoformat(),
                'configuration': {
                    'max_concurrent_users': self.config.max_concurrent_users,
                    'requests_per_second': self.config.requests_per_second,
                    'concurrent_ea_generations': self.config.concurrent_ea_generations
                }
            },
            'performance_statistics': stats,
            'performance_grade': performance_grade,
            'identified_bottlenecks': bottlenecks,
            'recommendations': recommendations,
            'stability_assessment': self._assess_stability(),
            'scalability_analysis': self._analyze_scalability(stats)
        }
        
        # Save report
        report_file = f"qnti_stress_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Stress test report saved: {report_file}")
        
        # Print summary
        self._print_stress_test_summary(report)
        
        return report
    
    def _calculate_performance_grade(self, stats: Dict[str, Any]) -> str:
        """Calculate overall performance grade"""
        score = 100
        
        # Response time penalty
        if 'response_time_stats' in stats:
            avg_response = stats['response_time_stats'].get('mean', 0)
            if avg_response > 5:
                score -= 30
            elif avg_response > 2:
                score -= 15
            elif avg_response > 1:
                score -= 5
        
        # Success rate penalty
        success_rate = stats['request_stats'].get('success_rate', 0)
        if success_rate < 0.9:
            score -= 40
        elif success_rate < 0.95:
            score -= 20
        elif success_rate < 0.99:
            score -= 10
        
        # Resource usage penalty
        if 'resource_stats' in stats:
            cpu_usage = stats['resource_stats'].get('cpu', {}).get('mean', 0)
            if cpu_usage > 90:
                score -= 20
            elif cpu_usage > 80:
                score -= 10
        
        # EA generation penalty
        ea_completion_rate = stats['ea_generation_stats'].get('completion_rate', 0)
        if ea_completion_rate < 0.8:
            score -= 25
        elif ea_completion_rate < 0.9:
            score -= 15
        
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _identify_bottlenecks(self, stats: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # High response times
        if 'response_time_stats' in stats:
            if stats['response_time_stats'].get('p95', 0) > 10:
                bottlenecks.append("High 95th percentile response time indicates server bottleneck")
        
        # High CPU usage
        if 'resource_stats' in stats and 'cpu' in stats['resource_stats']:
            if stats['resource_stats']['cpu'].get('mean', 0) > 80:
                bottlenecks.append("High CPU usage indicates processing bottleneck")
        
        # High memory usage
        if 'resource_stats' in stats and 'memory' in stats['resource_stats']:
            if stats['resource_stats']['memory'].get('max', 0) > self.config.max_memory_usage_mb:
                bottlenecks.append("High memory usage indicates memory leak or excessive allocation")
        
        # Low EA completion rate
        if stats['ea_generation_stats'].get('completion_rate', 0) < 0.8:
            bottlenecks.append("Low EA generation completion rate indicates workflow bottleneck")
        
        # High error rates
        if stats['request_stats'].get('failure_rate', 0) > 0.1:
            bottlenecks.append("High request failure rate indicates system instability")
        
        return bottlenecks
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Response time recommendations
        if 'response_time_stats' in stats:
            avg_response = stats['response_time_stats'].get('mean', 0)
            if avg_response > 2:
                recommendations.append("Consider implementing response caching for frequently accessed endpoints")
                recommendations.append("Review database query optimization and indexing")
        
        # CPU usage recommendations
        if 'resource_stats' in stats and 'cpu' in stats['resource_stats']:
            cpu_usage = stats['resource_stats']['cpu'].get('mean', 0)
            if cpu_usage > 70:
                recommendations.append("Consider implementing horizontal scaling with load balancers")
                recommendations.append("Review CPU-intensive operations for optimization opportunities")
        
        # EA generation recommendations
        ea_completion_rate = stats['ea_generation_stats'].get('completion_rate', 0)
        if ea_completion_rate < 0.9:
            recommendations.append("Implement EA generation queueing system to manage concurrent requests")
            recommendations.append("Consider pre-computed optimization results for common indicator combinations")
        
        # Error handling recommendations
        if stats['request_stats'].get('failure_rate', 0) > 0.05:
            recommendations.append("Implement circuit breaker pattern for external service calls")
            recommendations.append("Add retry mechanisms with exponential backoff")
        
        return recommendations
    
    def _assess_stability(self) -> Dict[str, Any]:
        """Assess system stability"""
        return {
            'connection_drops': self.metrics.connection_drops,
            'timeout_errors': self.metrics.timeout_errors,
            'server_errors': self.metrics.server_errors,
            'stability_score': max(0, 100 - (self.metrics.connection_drops * 5 + 
                                           self.metrics.timeout_errors * 3 + 
                                           self.metrics.server_errors * 2))
        }
    
    def _analyze_scalability(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system scalability"""
        # Simple scalability analysis based on resource usage vs throughput
        throughput = stats['request_stats']['total'] / (self.config.test_duration / 60)  # requests per minute
        
        cpu_usage = 0
        if 'resource_stats' in stats and 'cpu' in stats['resource_stats']:
            cpu_usage = stats['resource_stats']['cpu'].get('mean', 0)
        
        # Rough scalability estimate
        if cpu_usage > 0:
            estimated_max_throughput = throughput * (80 / cpu_usage)  # Assume 80% CPU target
        else:
            estimated_max_throughput = throughput * 2  # Conservative estimate
        
        return {
            'current_throughput_rpm': throughput,
            'cpu_efficiency': throughput / max(1, cpu_usage),
            'estimated_max_throughput_rpm': estimated_max_throughput,
            'scalability_rating': 'High' if cpu_usage < 50 else 'Medium' if cpu_usage < 80 else 'Low'
        }
    
    def _print_stress_test_summary(self, report: Dict[str, Any]):
        """Print stress test summary"""
        print("\n" + "="*70)
        print("🔥 QNTI STRESS TEST RESULTS")
        print("="*70)
        
        stats = report['performance_statistics']
        
        print(f"🎯 Performance Grade: {report['performance_grade']}")
        print(f"⏱️  Test Duration: {report['test_summary']['duration']:.1f} seconds")
        print(f"📊 Total Requests: {stats['request_stats']['total']}")
        print(f"✅ Success Rate: {stats['request_stats']['success_rate']:.1%}")
        
        if 'response_time_stats' in stats:
            rt_stats = stats['response_time_stats']
            print(f"⚡ Avg Response Time: {rt_stats['mean']:.3f}s")
            print(f"📈 95th Percentile: {rt_stats['p95']:.3f}s")
        
        if 'resource_stats' in stats:
            if 'cpu' in stats['resource_stats']:
                print(f"💻 Avg CPU Usage: {stats['resource_stats']['cpu']['mean']:.1f}%")
            if 'memory' in stats['resource_stats']:
                print(f"🧠 Peak Memory: {stats['resource_stats']['memory']['max']:.1f} MB")
        
        ea_stats = stats['ea_generation_stats']
        print(f"🤖 EA Generations: {ea_stats['completed']}/{ea_stats['started']} completed")
        
        if report['identified_bottlenecks']:
            print(f"\n⚠️  Bottlenecks Identified ({len(report['identified_bottlenecks'])}):")
            for bottleneck in report['identified_bottlenecks'][:3]:
                print(f"   • {bottleneck}")
        
        print(f"\n🚀 Scalability: {report['scalability_analysis']['scalability_rating']}")
        print(f"📈 Estimated Max Throughput: {report['scalability_analysis']['estimated_max_throughput_rpm']:.0f} requests/min")
        print("="*70)

# CLI interface
async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QNTI Stress Testing Suite")
    parser.add_argument('--url', default='http://localhost:5000', help='QNTI system URL')
    parser.add_argument('--duration', type=int, default=1800, help='Test duration in seconds')
    parser.add_argument('--users', type=int, default=50, help='Max concurrent users')
    parser.add_argument('--rps', type=int, default=100, help='Requests per second')
    parser.add_argument('--ea-concurrent', type=int, default=10, help='Concurrent EA generations')
    parser.add_argument('--quick', action='store_true', help='Run quick stress test')
    
    args = parser.parse_args()
    
    # Create configuration
    config = StressTestConfig(
        qnti_url=args.url,
        test_duration=300 if args.quick else args.duration,
        max_concurrent_users=10 if args.quick else args.users,
        requests_per_second=20 if args.quick else args.rps,
        concurrent_ea_generations=3 if args.quick else args.ea_concurrent
    )
    
    if args.quick:
        config.ea_complexity_levels = ["simple", "medium"]
    
    # Run stress test
    stress_tester = QNTIStressTestSuite(config)
    results = await stress_tester.run_comprehensive_stress_test()
    
    return results

if __name__ == "__main__":
    asyncio.run(main())