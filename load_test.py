#!/usr/bin/env python3
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
