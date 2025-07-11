#!/usr/bin/env python3
"""
QNTI Trading System - Simple Flask Application Entry Point
Basic web interface for deployment testing
"""

from flask import Flask, jsonify, render_template_string
import os
import json
from datetime import datetime

app = Flask(__name__)

# Basic HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>QNTI Trading System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f4f4f4; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
        .status { padding: 15px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .feature { margin: 15px 0; padding: 15px; background: #f8f9fa; border-left: 4px solid #007bff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 QNTI Trading System</h1>
            <h2>Quantum Nexus Trading Intelligence</h2>
        </div>
        
        <div class="status success">
            <strong>✅ System Status:</strong> Online and Running
        </div>
        
        <div class="status info">
            <strong>🕒 Deployment Time:</strong> {{ timestamp }}
        </div>
        
        <div class="feature">
            <h3>🔧 System Features</h3>
            <ul>
                <li>Advanced Trading Intelligence</li>
                <li>MT5 Integration Ready</li>
                <li>Real-time Market Analysis</li>
                <li>Automated Trading Strategies</li>
                <li>Risk Management System</li>
                <li>Multi-Asset Support</li>
            </ul>
        </div>
        
        <div class="feature">
            <h3>📊 API Endpoints</h3>
            <ul>
                <li><a href="/health">/health</a> - Health Check</li>
                <li><a href="/status">/status</a> - System Status</li>
                <li><a href="/config">/config</a> - Configuration</li>
            </ul>
        </div>
        
        <div class="status info">
            <strong>🏗️ Environment:</strong> {{ environment }}
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    """Main landing page"""
    return render_template_string(HTML_TEMPLATE, 
                                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
                                environment=os.environ.get('QNTI_ENV', 'production'))

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'QNTI Trading System',
        'version': '1.0.0'
    })

@app.route('/status')
def status():
    """System status endpoint"""
    return jsonify({
        'system': {
            'status': 'online',
            'uptime': 'running',
            'environment': os.environ.get('QNTI_ENV', 'production'),
            'python_path': os.environ.get('PYTHONPATH', '/app')
        },
        'features': {
            'trading': 'ready',
            'analysis': 'ready',
            'dashboard': 'ready',
            'api': 'active'
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/config')
def config():
    """Configuration information"""
    config_file = '/app/qnti_config.json'
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            return jsonify({
                'status': 'loaded',
                'config': config_data,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    else:
        return jsonify({
            'status': 'not_found',
            'message': 'Configuration file not found',
            'timestamp': datetime.now().isoformat()
        })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested endpoint does not exist',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An internal error occurred',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    # Get configuration
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('QNTI_DEBUG', 'false').lower() == 'true'
    
    print(f"🚀 Starting QNTI Trading System on port {port}")
    print(f"🌐 Environment: {os.environ.get('QNTI_ENV', 'production')}")
    print(f"🔧 Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug) 