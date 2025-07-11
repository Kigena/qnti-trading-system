#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qnti_automation_suite import QNTIAutomationSuite, SimulationConfig

async def quick_test():
    config = SimulationConfig(
        qnti_url="http://localhost:5000",
        simulation_duration=60,
        max_concurrent_users=2,
        ea_strategies_to_test=["Quick Test Strategy"],
    )
    
    suite = QNTIAutomationSuite(config)
    results = await suite.run_comprehensive_simulation()
    
    print(f"Quick test completed: {results.success_rate:.1%} success rate")
    return results

if __name__ == "__main__":
    asyncio.run(quick_test())
