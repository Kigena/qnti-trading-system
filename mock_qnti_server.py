#!/usr/bin/env python3
"""
Mock QNTI Server for Automation Testing
Simulates QNTI API endpoints for demonstration purposes
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from aiohttp import web, web_response
from typing import Dict, Any

class MockQNTIServer:
    """Mock QNTI server for testing automation"""
    
    def __init__(self):
        self.workflows = {}
        self.start_time = datetime.now()
    
    async def health_check(self, request) -> web_response.Response:
        """Mock health check endpoint"""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        health_data = {
            'status': 'healthy',
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'version': '1.0.0',
            'services': {
                'ea_generation': 'active',
                'backtesting': 'active',
                'ai_insights': 'active'
            }
        }
        
        return web.json_response(health_data)
    
    async def ea_indicators(self, request) -> web_response.Response:
        """Mock EA indicators endpoint"""
        await asyncio.sleep(0.2)  # Simulate processing time
        
        indicators = {
            'indicators': [
                {'name': 'SMA', 'category': 'trend', 'parameters': ['period']},
                {'name': 'EMA', 'category': 'trend', 'parameters': ['period']},
                {'name': 'RSI', 'category': 'momentum', 'parameters': ['period']},
                {'name': 'MACD', 'category': 'momentum', 'parameters': ['fast', 'slow', 'signal']},
                {'name': 'Bollinger Bands', 'category': 'volatility', 'parameters': ['period', 'std_dev']},
                {'name': 'Stochastic', 'category': 'momentum', 'parameters': ['k_period', 'd_period']},
                {'name': 'ATR', 'category': 'volatility', 'parameters': ['period']},
                {'name': 'CCI', 'category': 'momentum', 'parameters': ['period']},
            ],
            'count': 8,
            'timestamp': datetime.now().isoformat()
        }
        
        return web.json_response(indicators)
    
    async def ea_workflow_start(self, request) -> web_response.Response:
        """Mock EA workflow start endpoint"""
        await asyncio.sleep(0.5)  # Simulate processing time
        
        try:
            data = await request.json()
            
            # Validate required fields
            required_fields = ['ea_name', 'symbols', 'timeframes', 'indicators']
            for field in required_fields:
                if field not in data:
                    return web.json_response({
                        'success': False,
                        'error': f'Missing required field: {field}'
                    }, status=400)
            
            # Create workflow
            workflow_id = str(uuid.uuid4())
            workflow = {
                'id': workflow_id,
                'status': 'running',
                'config': data,
                'start_time': datetime.now().isoformat(),
                'progress': 0
            }
            
            self.workflows[workflow_id] = workflow
            
            # Simulate workflow completion after delay
            asyncio.create_task(self._simulate_workflow_completion(workflow_id))
            
            return web.json_response({
                'success': True,
                'workflow_id': workflow_id,
                'status': 'started',
                'message': f'EA generation workflow started for {data["ea_name"]}'
            })
            
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def ea_workflow_status(self, request) -> web_response.Response:
        """Mock EA workflow status endpoint"""
        workflow_id = request.match_info['workflow_id']
        
        await asyncio.sleep(0.1)  # Simulate processing time
        
        if workflow_id not in self.workflows:
            return web.json_response({
                'success': False,
                'error': 'Workflow not found'
            }, status=404)
        
        workflow = self.workflows[workflow_id]
        
        return web.json_response({
            'success': True,
            'workflow_id': workflow_id,
            'status': workflow['status'],
            'progress': workflow['progress'],
            'start_time': workflow['start_time']
        })
    
    async def ea_workflow_list(self, request) -> web_response.Response:
        """Mock EA workflow list endpoint"""
        await asyncio.sleep(0.1)
        
        workflows = []
        for wf_id, wf_data in self.workflows.items():
            workflows.append({
                'id': wf_id,
                'name': wf_data['config']['ea_name'],
                'status': wf_data['status'],
                'start_time': wf_data['start_time']
            })
        
        return web.json_response({
            'workflows': workflows,
            'count': len(workflows)
        })
    
    async def ai_insights(self, request) -> web_response.Response:
        """Mock AI insights endpoint"""
        await asyncio.sleep(0.3)
        
        insights = {
            'insights': [
                {
                    'type': 'market_analysis',
                    'message': 'EURUSD showing strong bullish momentum',
                    'confidence': 0.85,
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'type': 'strategy_recommendation',
                    'message': 'Consider trend-following strategies for current market conditions',
                    'confidence': 0.78,
                    'timestamp': datetime.now().isoformat()
                }
            ],
            'count': 2
        }
        
        return web.json_response(insights)
    
    async def dashboard(self, request) -> web_response.Response:
        """Mock dashboard endpoint"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>QNTI Dashboard - Mock</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
                .dashboard-card { background: #2d2d2d; padding: 20px; margin: 10px; border-radius: 8px; }
                .nav-links a { color: #4CAF50; margin: 0 10px; text-decoration: none; }
                .ai-insight-box { background: #333; padding: 15px; margin: 10px 0; border-left: 4px solid #4CAF50; }
            </style>
        </head>
        <body>
            <h1>🚀 QNTI Dashboard (Mock)</h1>
            <div class="nav-links">
                <a href="/ea-generation">EA Generation</a>
                <a href="/backtesting">Backtesting</a>
                <a href="/insights">AI Insights</a>
            </div>
            <div class="dashboard-card">
                <h2>System Status</h2>
                <p>✅ All systems operational</p>
            </div>
            <div class="ai-insight-box">
                <h3>AI Insight</h3>
                <p>Market conditions favor trend-following strategies</p>
            </div>
        </body>
        </html>
        """
        return web.Response(text=html_content, content_type='text/html')
    
    async def ea_generation_page(self, request) -> web_response.Response:
        """Mock EA generation page"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>EA Generation - QNTI</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
                .form-group { margin: 15px 0; }
                label { display: block; margin-bottom: 5px; }
                input, select, textarea { width: 300px; padding: 8px; background: #333; color: #fff; border: 1px solid #555; }
                button { background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
                .indicator-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0; }
                #workflowStatus { padding: 10px; margin: 10px 0; border-radius: 4px; }
                .status-running { background: #ff9800; }
                .status-completed { background: #4CAF50; }
                .status-failed { background: #f44336; }
            </style>
        </head>
        <body>
            <h1>🤖 EA Generation System</h1>
            <form id="eaForm">
                <div class="form-group">
                    <label for="eaName">EA Name:</label>
                    <input type="text" id="eaName" name="eaName" required>
                </div>
                <div class="form-group">
                    <label for="eaDescription">Description:</label>
                    <textarea id="eaDescription" name="eaDescription"></textarea>
                </div>
                <div class="form-group">
                    <label for="eaSymbols">Symbols:</label>
                    <select id="eaSymbols" multiple>
                        <option value="EURUSD">EURUSD</option>
                        <option value="GBPUSD">GBPUSD</option>
                        <option value="USDJPY">USDJPY</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="eaTimeframes">Timeframes:</label>
                    <select id="eaTimeframes" multiple>
                        <option value="M15">M15</option>
                        <option value="H1">H1</option>
                        <option value="H4">H4</option>
                        <option value="D1">D1</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Indicators:</label>
                    <div class="indicator-grid">
                        <label><input type="checkbox" id="indicator_SMA"> SMA</label>
                        <label><input type="checkbox" id="indicator_EMA"> EMA</label>
                        <label><input type="checkbox" id="indicator_RSI"> RSI</label>
                        <label><input type="checkbox" id="indicator_MACD"> MACD</label>
                    </div>
                </div>
                <div class="form-group">
                    <label for="optimizationMethod">Optimization Method:</label>
                    <select id="optimizationMethod">
                        <option value="genetic_algorithm">Genetic Algorithm</option>
                        <option value="grid_search">Grid Search</option>
                        <option value="bayesian">Bayesian Optimization</option>
                    </select>
                </div>
                <button type="submit">Generate EA</button>
            </form>
            <div id="workflowStatus" style="display: none;">
                Workflow Status: <span id="statusText">Ready</span>
            </div>
        </body>
        </html>
        """
        return web.Response(text=html_content, content_type='text/html')
    
    async def _simulate_workflow_completion(self, workflow_id: str):
        """Simulate workflow completion"""
        await asyncio.sleep(10)  # Simulate 10 seconds of processing
        
        if workflow_id in self.workflows:
            self.workflows[workflow_id]['status'] = 'completed'
            self.workflows[workflow_id]['progress'] = 100

def create_app() -> web.Application:
    """Create mock QNTI application"""
    app = web.Application()
    server = MockQNTIServer()
    
    # Dashboard routes
    app.router.add_get('/', server.dashboard)
    app.router.add_get('/ea-generation', server.ea_generation_page)
    
    # API routes
    app.router.add_get('/api/system/health', server.health_check)
    app.router.add_get('/api/ea/indicators', server.ea_indicators)
    app.router.add_post('/api/ea/workflow/start', server.ea_workflow_start)
    app.router.add_get('/api/ea/workflow/status/{workflow_id}', server.ea_workflow_status)
    app.router.add_get('/api/ea/workflow/list', server.ea_workflow_list)
    app.router.add_get('/api/ai/insights/all', server.ai_insights)
    
    return app

async def main():
    """Main entry point"""
    app = create_app()
    
    print("🚀 Starting Mock QNTI Server...")
    print("📍 Server will be available at: http://localhost:5000")
    print("🔗 Dashboard: http://localhost:5000")
    print("🤖 EA Generation: http://localhost:5000/ea-generation")
    print("📊 Health Check: http://localhost:5000/api/system/health")
    print("\nPress Ctrl+C to stop the server")
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 5000)
    await site.start()
    
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Mock QNTI Server...")
        await runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())