#!/bin/bash

echo "========================================"
echo "QNTI Trading System - Full Simulation"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    exit 1
fi

# Install required packages
echo "Installing required packages..."
pip3 install requests websocket-client Pillow selenium

# Check if QNTI server is running
echo "Checking QNTI server status..."
if ! curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
    echo "WARNING: QNTI server may not be running at http://localhost:5000"
    echo "Please start the QNTI server first"
    echo
    read -p "Continue anyway? (y/n): " choice
    if [[ ! "$choice" =~ ^[Yy]$ ]]; then
        echo "Simulation cancelled"
        exit 1
    fi
fi

echo
echo "Starting QNTI Full System Simulation..."
echo

# Run the full simulation
python3 run_qnti_full_simulation.py

echo
echo "Simulation completed!"
echo "Check the generated report files for detailed results."
echo 