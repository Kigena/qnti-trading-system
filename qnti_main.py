#!/usr/bin/env python3
"""
Quantum Nexus Trading Intelligence (QNTI) - Main Entry Point
This is the main entry point for the QNTI system.
"""

import sys
import argparse
import logging
from qnti_main_system import QNTIMainSystem

def main():
    """Main entry point for QNTI system"""
    parser = argparse.ArgumentParser(description='QNTI Trading System')
    parser.add_argument('--port', type=int, default=5003, help='Port to run the web interface on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-auto-trading', action='store_true', help='Disable auto trading')
    parser.add_argument('--config', default='qnti_config.json', help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        # Create main system instance
        qnti_system = QNTIMainSystem(config_file=args.config)
        
        # Configure system based on arguments
        if args.no_auto_trading:
            qnti_system.auto_trading_enabled = False
        
        # Start the system
        qnti_system.start(
            host="0.0.0.0",
            port=args.port,
            debug=args.debug
        )
        
    except KeyboardInterrupt:
        print("\nShutting down QNTI system...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 