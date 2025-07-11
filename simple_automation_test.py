#!/usr/bin/env python3
"""
Simple QNTI Automation Test - Basic API Testing
Tests core functionality without full system dependencies
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime
from typing import Dict, Any, List

class SimpleQNTITest:
    """Simple QNTI automation test"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.results = {
            'start_time': datetime.now().isoformat(),
            'tests': [],
            'summary': {}
        }
    
    async def test_basic_connectivity(self) -> Dict[str, Any]:
        """Test basic connectivity to QNTI system"""
        try:
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/", timeout=10) as response:
                    response_time = time.time() - start_time
                    return {
                        'test': 'basic_connectivity',
                        'success': response.status == 200,
                        'response_time': response_time,
                        'status_code': response.status,
                        'error': None
                    }
        except Exception as e:
            return {
                'test': 'basic_connectivity',
                'success': False,
                'response_time': time.time() - start_time,
                'status_code': None,
                'error': str(e)
            }
    
    async def test_system_health(self) -> Dict[str, Any]:
        """Test system health endpoint"""
        try:
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/system/health", timeout=10) as response:
                    response_time = time.time() - start_time
                    data = await response.json() if response.status == 200 else None
                    return {
                        'test': 'system_health',
                        'success': response.status == 200,
                        'response_time': response_time,
                        'status_code': response.status,
                        'data': data,
                        'error': None
                    }
        except Exception as e:
            return {
                'test': 'system_health',
                'success': False,
                'response_time': time.time() - start_time,
                'status_code': None,
                'data': None,
                'error': str(e)
            }
    
    async def test_ea_indicators(self) -> Dict[str, Any]:
        """Test EA indicators endpoint"""
        try:
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/ea/indicators", timeout=10) as response:
                    response_time = time.time() - start_time
                    data = await response.json() if response.status == 200 else None
                    return {
                        'test': 'ea_indicators',
                        'success': response.status == 200,
                        'response_time': response_time,
                        'status_code': response.status,
                        'data': data,
                        'error': None
                    }
        except Exception as e:
            return {
                'test': 'ea_indicators',
                'success': False,
                'response_time': time.time() - start_time,
                'status_code': None,
                'data': None,
                'error': str(e)
            }
    
    async def test_ea_workflow_creation(self) -> Dict[str, Any]:
        """Test EA workflow creation"""
        try:
            start_time = time.time()
            test_config = {
                'ea_name': 'Simple Test Strategy',
                'description': 'Automated test strategy',
                'symbols': ['EURUSD'],
                'timeframes': ['H1'],
                'indicators': [
                    {'name': 'SMA', 'params': {}},
                    {'name': 'RSI', 'params': {}}
                ],
                'method': 'genetic_algorithm',
                'auto_proceed': True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/ea/workflow/start",
                    json=test_config,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                ) as response:
                    response_time = time.time() - start_time
                    data = await response.json() if response.content_type == 'application/json' else None
                    return {
                        'test': 'ea_workflow_creation',
                        'success': response.status == 200,
                        'response_time': response_time,
                        'status_code': response.status,
                        'data': data,
                        'error': None
                    }
        except Exception as e:
            return {
                'test': 'ea_workflow_creation',
                'success': False,
                'response_time': time.time() - start_time,
                'status_code': None,
                'data': None,
                'error': str(e)
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests"""
        print("🚀 Starting Simple QNTI Automation Test")
        
        tests = [
            self.test_basic_connectivity(),
            self.test_system_health(),
            self.test_ea_indicators(),
            self.test_ea_workflow_creation()
        ]
        
        results = await asyncio.gather(*tests)
        
        # Calculate summary
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['success'])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        avg_response_time = sum(r['response_time'] for r in results) / total_tests if total_tests > 0 else 0
        
        self.results['tests'] = results
        self.results['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'end_time': datetime.now().isoformat()
        }
        
        # Save results
        report_file = f"simple_automation_test_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*50)
        print("🎯 SIMPLE QNTI AUTOMATION TEST RESULTS")
        print("="*50)
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Avg Response Time: {avg_response_time:.3f}s")
        
        if successful_tests < total_tests:
            print(f"\n❌ Failed Tests:")
            for result in results:
                if not result['success']:
                    print(f"  • {result['test']}: {result['error']}")
        
        print(f"\n📊 Full report saved: {report_file}")
        print("="*50)
        
        return self.results

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple QNTI Automation Test")
    parser.add_argument('--url', default='http://localhost:5000', help='QNTI system URL')
    args = parser.parse_args()
    
    tester = SimpleQNTITest(args.url)
    results = await tester.run_all_tests()
    
    return results

if __name__ == "__main__":
    asyncio.run(main())