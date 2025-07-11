"""
QNTI API Integration
Flask routes integration for external API services
"""

import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from flask import Flask, jsonify, request
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from qnti_api_services import QNTIAPIManager, DataProvider, MarketData, NewsItem, SocialSentiment, VisionAnalysis

logger = logging.getLogger(__name__)

class QNTIAPIIntegration:
    """Main API integration class for Flask"""
    
    def __init__(self, app: Flask, config: Dict[str, Any]):
        self.app = app
        self.config = config
        self.api_manager = None
        self.background_tasks = {}
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Initialize API manager
        self._initialize_api_manager()
        
        # Setup API routes
        self._setup_api_routes()
        
        # Start background data collection
        self._start_background_tasks()
    
    def _initialize_api_manager(self):
        """Initialize the API manager"""
        try:
            self.api_manager = QNTIAPIManager(self.config)
            logger.info("API manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize API manager: {e}")
            self.api_manager = None
    
    def _setup_api_routes(self):
        """Setup all API routes"""
        
        @self.app.route('/api/external/status')
        def get_api_status():
            """Get status of all external API services"""
            try:
                if not self.api_manager:
                    return jsonify({"error": "API manager not initialized"}), 500
                
                status = self.api_manager.get_service_status()
                return jsonify(status)
                
            except Exception as e:
                logger.error(f"Error getting API status: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/external/market-data')
        def get_external_market_data():
            """Get market data from external providers"""
            try:
                if not self.api_manager:
                    return jsonify({"error": "API manager not initialized"}), 500
                
                symbols = request.args.getlist('symbols') or ['EURUSD', 'GBPUSD', 'USDJPY']
                
                # Get market data from all providers
                market_data = self.api_manager.get_all_market_data(symbols)
                
                # Format response
                formatted_data = {}
                for symbol, data_list in market_data.items():
                    formatted_data[symbol] = []
                    for data in data_list:
                        formatted_data[symbol].append({
                            'symbol': data.symbol,
                            'price': data.price,
                            'change': data.change,
                            'change_percent': data.change_percent,
                            'volume': data.volume,
                            'timestamp': data.timestamp.isoformat(),
                            'provider': data.provider.value
                        })
                
                return jsonify({
                    'success': True,
                    'data': formatted_data,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting external market data: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/external/news')
        def get_external_news():
            """Get news from external providers"""
            try:
                if not self.api_manager:
                    return jsonify({"error": "API manager not initialized"}), 500
                
                limit = int(request.args.get('limit', 20))
                
                # Get news from all providers
                news_items = self.api_manager.get_all_news(limit)
                
                # Format response
                formatted_news = []
                for news in news_items:
                    formatted_news.append({
                        'title': news.title,
                        'content': news.content,
                        'url': news.url,
                        'source': news.source,
                        'published_at': news.published_at.isoformat(),
                        'sentiment': news.sentiment,
                        'relevance': news.relevance
                    })
                
                return jsonify({
                    'success': True,
                    'news': formatted_news,
                    'count': len(formatted_news),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting external news: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/external/sentiment')
        def get_social_sentiment():
            """Get social sentiment from external providers"""
            try:
                if not self.api_manager:
                    return jsonify({"error": "API manager not initialized"}), 500
                
                symbols = request.args.getlist('symbols')
                limit = int(request.args.get('limit', 50))
                
                # Get sentiment from all providers
                sentiment_data = self.api_manager.get_social_sentiment(symbols, limit)
                
                # Format response
                formatted_sentiment = []
                for sentiment in sentiment_data:
                    formatted_sentiment.append({
                        'platform': sentiment.platform,
                        'symbol': sentiment.symbol,
                        'sentiment_score': sentiment.sentiment_score,
                        'mention_count': sentiment.mention_count,
                        'timestamp': sentiment.timestamp.isoformat(),
                        'raw_data': sentiment.raw_data
                    })
                
                # Calculate aggregate sentiment
                total_sentiment = sum(s.sentiment_score for s in sentiment_data)
                avg_sentiment = total_sentiment / len(sentiment_data) if sentiment_data else 0
                
                return jsonify({
                    'success': True,
                    'sentiment': formatted_sentiment,
                    'aggregate': {
                        'average_sentiment': avg_sentiment,
                        'total_mentions': sum(s.mention_count for s in sentiment_data),
                        'platforms': list(set(s.platform for s in sentiment_data))
                    },
                    'count': len(formatted_sentiment),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting social sentiment: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/external/combined-data')
        def get_combined_data():
            """Get combined market data, news, and sentiment"""
            try:
                if not self.api_manager:
                    return jsonify({"error": "API manager not initialized"}), 500
                
                symbols = request.args.getlist('symbols') or ['EURUSD', 'GBPUSD', 'USDJPY']
                news_limit = int(request.args.get('news_limit', 10))
                sentiment_limit = int(request.args.get('sentiment_limit', 20))
                
                # Execute all requests in parallel
                futures = []
                
                # Market data
                market_future = self.executor.submit(self.api_manager.get_all_market_data, symbols)
                futures.append(('market_data', market_future))
                
                # News
                news_future = self.executor.submit(self.api_manager.get_all_news, news_limit)
                futures.append(('news', news_future))
                
                # Sentiment
                sentiment_future = self.executor.submit(self.api_manager.get_social_sentiment, symbols, sentiment_limit)
                futures.append(('sentiment', sentiment_future))
                
                # Collect results
                results = {}
                for data_type, future in futures:
                    try:
                        results[data_type] = future.result(timeout=30)
                    except Exception as e:
                        logger.error(f"Error getting {data_type}: {e}")
                        results[data_type] = []
                
                # Format market data
                formatted_market = {}
                for symbol, data_list in results.get('market_data', {}).items():
                    formatted_market[symbol] = []
                    for data in data_list:
                        formatted_market[symbol].append({
                            'symbol': data.symbol,
                            'price': data.price,
                            'change': data.change,
                            'change_percent': data.change_percent,
                            'provider': data.provider.value,
                            'timestamp': data.timestamp.isoformat()
                        })
                
                # Format news
                formatted_news = []
                for news in results.get('news', []):
                    formatted_news.append({
                        'title': news.title,
                        'content': news.content,
                        'source': news.source,
                        'published_at': news.published_at.isoformat()
                    })
                
                # Format sentiment
                formatted_sentiment = []
                sentiment_data = results.get('sentiment', [])
                for sentiment in sentiment_data:
                    formatted_sentiment.append({
                        'platform': sentiment.platform,
                        'sentiment_score': sentiment.sentiment_score,
                        'mention_count': sentiment.mention_count,
                        'timestamp': sentiment.timestamp.isoformat()
                    })
                
                # Calculate aggregate sentiment
                total_sentiment = sum(s.sentiment_score for s in sentiment_data)
                avg_sentiment = total_sentiment / len(sentiment_data) if sentiment_data else 0
                
                return jsonify({
                    'success': True,
                    'market_data': formatted_market,
                    'news': formatted_news,
                    'sentiment': {
                        'items': formatted_sentiment,
                        'average_sentiment': avg_sentiment,
                        'total_mentions': sum(s.mention_count for s in sentiment_data)
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting combined data: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/external/providers')
        def get_available_providers():
            """Get list of available data providers"""
            try:
                if not self.api_manager:
                    return jsonify({"error": "API manager not initialized"}), 500
                
                providers = {}
                for service_name, service in self.api_manager.services.items():
                    providers[service_name] = {
                        'name': service_name,
                        'capabilities': {
                            'market_data': hasattr(service, 'get_forex_data'),
                            'news': hasattr(service, 'get_forex_news') or hasattr(service, 'get_market_news'),
                            'sentiment': hasattr(service, 'get_trading_sentiment')
                        },
                        'rate_limit': {
                            'max_calls': service.rate_limiter.max_calls,
                            'time_window': service.rate_limiter.time_window
                        }
                    }
                
                return jsonify({
                    'success': True,
                    'providers': providers,
                    'total_providers': len(providers)
                })
                
            except Exception as e:
                logger.error(f"Error getting providers: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/external/test-connection', methods=['POST'])
        def test_api_connection():
            """Test connection to specific API provider"""
            try:
                if not self.api_manager:
                    return jsonify({"error": "API manager not initialized"}), 500
                
                data = request.get_json()
                provider_name = data.get('provider')
                
                if not provider_name or provider_name not in self.api_manager.services:
                    return jsonify({"error": "Invalid provider"}), 400
                
                service = self.api_manager.services[provider_name]
                
                # Test basic functionality
                test_results = {}
                
                # Test market data if available
                if hasattr(service, 'get_forex_data'):
                    try:
                        result = service.get_forex_data('EURUSD')
                        test_results['market_data'] = result is not None
                    except Exception as e:
                        test_results['market_data'] = False
                        test_results['market_data_error'] = str(e)
                
                # Test news if available
                if hasattr(service, 'get_forex_news'):
                    try:
                        result = service.get_forex_news(limit=1)
                        test_results['news'] = len(result) > 0
                    except Exception as e:
                        test_results['news'] = False
                        test_results['news_error'] = str(e)
                
                # Test sentiment if available
                if hasattr(service, 'get_trading_sentiment'):
                    try:
                        result = service.get_trading_sentiment(limit=1)
                        test_results['sentiment'] = len(result) > 0
                    except Exception as e:
                        test_results['sentiment'] = False
                        test_results['sentiment_error'] = str(e)
                
                return jsonify({
                    'success': True,
                    'provider': provider_name,
                    'test_results': test_results,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error testing API connection: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/external/vision/analyze', methods=['POST'])
        def analyze_chart_image():
            """Analyze chart image using OpenAI Vision"""
            try:
                if not self.api_manager:
                    return jsonify({"error": "API manager not initialized"}), 500
                
                # Check if file is present
                if 'image' not in request.files:
                    return jsonify({"error": "No image file provided"}), 400
                
                file = request.files['image']
                if file.filename == '':
                    return jsonify({"error": "No file selected"}), 400
                
                # Get analysis parameters
                symbol = request.form.get('symbol', 'EURUSD')
                timeframe = request.form.get('timeframe', 'H1')
                
                # Read image data
                image_data = file.read()
                
                # Analyze the image
                analysis = self.api_manager.analyze_chart_image(image_data, symbol, timeframe)
                
                if analysis:
                    return jsonify({
                        'success': True,
                        'analysis': {
                            'analysis_id': analysis.analysis_id,
                            'symbol': analysis.symbol,
                            'timeframe': analysis.timeframe,
                            'signal': analysis.signal,
                            'confidence': analysis.confidence,
                            'analysis_text': analysis.analysis_text,
                            'key_levels': analysis.key_levels,
                            'risk_reward': analysis.risk_reward,
                            'timestamp': analysis.timestamp.isoformat()
                        }
                    })
                else:
                    return jsonify({"error": "Vision analysis failed"}), 500
                
            except Exception as e:
                logger.error(f"Error in vision analysis: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/external/vision/history')
        def get_vision_history():
            """Get vision analysis history"""
            try:
                if not self.api_manager:
                    return jsonify({"error": "API manager not initialized"}), 500
                
                limit = int(request.args.get('limit', 10))
                history = self.api_manager.get_vision_analysis_history(limit)
                
                formatted_history = []
                for analysis in history:
                    formatted_history.append({
                        'analysis_id': analysis.analysis_id,
                        'symbol': analysis.symbol,
                        'timeframe': analysis.timeframe,
                        'signal': analysis.signal,
                        'confidence': analysis.confidence,
                        'analysis_text': analysis.analysis_text,
                        'timestamp': analysis.timestamp.isoformat()
                    })
                
                return jsonify({
                    'success': True,
                    'history': formatted_history,
                    'count': len(formatted_history)
                })
                
            except Exception as e:
                logger.error(f"Error getting vision history: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/external/vision/test', methods=['POST'])
        def test_vision_connection():
            """Test OpenAI Vision connection"""
            try:
                if not self.api_manager:
                    return jsonify({"error": "API manager not initialized"}), 500
                
                connection_test = self.api_manager.test_vision_connection()
                
                return jsonify({
                    'success': True,
                    'connected': connection_test,
                    'service': 'openai_vision',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error testing vision connection: {e}")
                return jsonify({"error": str(e)}), 500
    
    def _start_background_tasks(self):
        """Start background data collection tasks"""
        try:
            # Start background news collection
            def collect_news():
                while True:
                    try:
                        if self.api_manager:
                            news = self.api_manager.get_all_news(20)
                            logger.info(f"Background: Collected {len(news)} news items")
                        threading.Event().wait(1800)  # 30 minutes
                    except Exception as e:
                        logger.error(f"Error in background news collection: {e}")
                        threading.Event().wait(600)  # 10 minutes on error
            
            # Start background sentiment collection
            def collect_sentiment():
                while True:
                    try:
                        if self.api_manager:
                            sentiment = self.api_manager.get_social_sentiment(limit=50)
                            logger.info(f"Background: Collected {len(sentiment)} sentiment items")
                        threading.Event().wait(900)  # 15 minutes
                    except Exception as e:
                        logger.error(f"Error in background sentiment collection: {e}")
                        threading.Event().wait(300)  # 5 minutes on error
            
            # Start background market data collection
            def collect_market_data():
                symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD']
                while True:
                    try:
                        if self.api_manager:
                            market_data = self.api_manager.get_all_market_data(symbols)
                            total_data = sum(len(data) for data in market_data.values())
                            logger.info(f"Background: Collected {total_data} market data points")
                        threading.Event().wait(300)  # 5 minutes
                    except Exception as e:
                        logger.error(f"Error in background market data collection: {e}")
                        threading.Event().wait(120)  # 2 minutes on error
            
            # Start background threads
            self.background_tasks['news'] = threading.Thread(target=collect_news, daemon=True)
            self.background_tasks['sentiment'] = threading.Thread(target=collect_sentiment, daemon=True)
            self.background_tasks['market_data'] = threading.Thread(target=collect_market_data, daemon=True)
            
            for task in self.background_tasks.values():
                task.start()
            
            logger.info("Background data collection tasks started")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    def shutdown(self):
        """Shutdown the API integration"""
        try:
            # Stop executor
            self.executor.shutdown(wait=True)
            logger.info("API integration shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

def integrate_external_apis(app: Flask, config: Dict[str, Any]) -> QNTIAPIIntegration:
    """Integrate external APIs with Flask app"""
    try:
        integration = QNTIAPIIntegration(app, config)
        logger.info("External API integration completed successfully")
        return integration
    except Exception as e:
        logger.error(f"Failed to integrate external APIs: {e}")
        return None 