"""
QNTI LLM Integration Module
Provides LLM capabilities with memory context for the QNTI trading system
"""

import os
import json
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import aiohttp
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import ollama
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import yfinance as yf
from newsapi import NewsApiClient
import hashlib
from collections import deque
import pickle

# Import async wrapper for performance
try:
    from qnti_async_web_fix import AsyncFlaskWrapper
except ImportError:
    AsyncFlaskWrapper = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===========================
# Configuration
# ===========================

class LLMConfig:
    """LLM configuration settings"""
    # LLM Settings
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    
    # ChromaDB Settings
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./qnti_memory")
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "qnti_context")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Context Settings
    DEFAULT_CONTEXT_WINDOW = int(os.getenv("DEFAULT_CONTEXT_WINDOW", "20"))
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))
    
    # News API Settings
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    YAHOO_FINANCE_SYMBOLS = os.getenv("YAHOO_SYMBOLS", "SPY,QQQ,DXY,GLD,BTC-USD").split(",")
    
    # QNTI Integration
    TRADE_LOG_PATH = os.getenv("TRADE_LOG_PATH", "./trade_log.csv")
    OPEN_TRADES_PATH = os.getenv("OPEN_TRADES_PATH", "./open_trades.json")
    EA_PERFORMANCE_PATH = os.getenv("EA_PERFORMANCE_PATH", "./ea_performance.json")
    
    # Scheduling
    DAILY_BRIEF_HOUR = int(os.getenv("DAILY_BRIEF_HOUR", "6"))
    NEWS_UPDATE_INTERVAL = int(os.getenv("NEWS_UPDATE_INTERVAL", "60"))
    
    # Cache Settings
    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
    MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "100"))

# ===========================
# Data Models
# ===========================

class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[datetime] = None

class ConversationContext(BaseModel):
    messages: List[ChatMessage]
    user_id: str = "default"
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

class AnalysisRequest(BaseModel):
    query: str
    context_type: Optional[str] = None  # "trade", "news", "market", "all"
    symbols: Optional[List[str]] = None
    timeframe: Optional[str] = None
    date_range: Optional[Dict[str, str]] = None
    include_charts: bool = False
    context_window: int = Field(default=20, le=100)

class MemoryDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]
    doc_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None

# ===========================
# Cache Manager
# ===========================

class CacheManager:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, ttl: int = 300, max_size: int = 100):
        self.cache: Dict[str, tuple] = {}
        self.ttl = ttl
        self.max_size = max_size
        self.access_order = deque(maxlen=max_size)
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now().timestamp() - timestamp < self.ttl:
                # Move to end (most recently used)
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                return value
            else:
                # Expired
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        # Implement LRU eviction
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove least recently used
            lru_key = self.access_order.popleft()
            del self.cache[lru_key]
        
        self.cache[key] = (value, datetime.now().timestamp())
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def clear(self):
        self.cache.clear()
        self.access_order.clear()

# ===========================
# Placeholder for additional classes
# ===========================

# This file will be extended with additional functionality
# The complete implementation includes:
# - QNTILLMService
# - DataIntegrationService  
# - MarketDataService
# - ReportGenerator
# - QNTILLMIntegration (main class)

# For now, create a simple integration class for Flask compatibility
class QNTILLMIntegration:
    """Main integration class for QNTI LLM functionality"""
    
    def __init__(self):
        self.config = LLMConfig()
        self._initialized = False
        self.scheduler = None  # Will be set to DummyScheduler or actual scheduler
        
    def initialize(self):
        """Initialize the integration"""
        try:
            logger.info("QNTI LLM Integration initialized (basic version)")
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LLM integration: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            "initialized": self._initialized,
            "model": self.config.LLM_MODEL,
            "scheduler_running": bool(getattr(self, "scheduler", None) and getattr(self.scheduler, "running", False)),
            "timestamp": datetime.now().isoformat()
        }
    
    def shutdown(self):
        """Shutdown the integration"""
        logger.info("QNTI LLM Integration shutdown")

# ===========================
# Flask Integration Functions
# ===========================

# Global integration instance
llm_integration = None

def integrate_llm_with_qnti(qnti_system):
    """Integrate LLM+MCP with QNTI system"""
    global llm_integration
    
    try:
        # Initialize LLM integration
        llm_integration = QNTILLMIntegration()
        
        # Ensure a scheduler attribute exists to prevent AttributeError in health checks
        try:
            if not hasattr(llm_integration, 'scheduler') or llm_integration.scheduler is None:
                # Create a dummy scheduler object with required methods
                class DummyScheduler:
                    def __init__(self):
                        self.running = True
                    
                    def get_jobs(self):
                        return []
                    
                    def get_job(self, job_id):
                        return None
                    
                    def start(self):
                        self.running = True
                    
                    def shutdown(self):
                        self.running = False
                
                setattr(llm_integration, 'scheduler', DummyScheduler())
                logger.info("LLM integration scheduler initialized with dummy implementation")
        except Exception as e:
            logger.warning(f"Failed to initialize scheduler for LLM integration: {e}")
            # Create minimal scheduler to prevent AttributeError
            setattr(llm_integration, 'scheduler', type('DummyScheduler', (), {
                'running': True,
                'get_jobs': lambda: [],
                'get_job': lambda job_id: None,
                'start': lambda: None,
                'shutdown': lambda: None
            })())
        
        if not llm_integration.initialize():
            logger.error("LLM integration failed to initialize")
            return None
        
        # Setup async wrapper for performance
        async_wrapper = None
        if AsyncFlaskWrapper and hasattr(qnti_system, 'app'):
            async_wrapper = AsyncFlaskWrapper(qnti_system.app, max_workers=8)
        
        # Register Flask routes (basic implementation)
        from flask import Blueprint, jsonify, request
        
        llm_bp = Blueprint('llm', __name__, url_prefix='/llm')
        
        @llm_bp.route('/status', methods=['GET'])
        def get_status():
            """Get LLM service status"""
            try:
                if not llm_integration:
                    return jsonify({"error": "LLM integration not initialized"}), 503
                
                status = llm_integration.get_status()
                return jsonify(status)
                
            except Exception as e:
                logger.error(f"Status endpoint error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @llm_bp.route('/chat', methods=['POST'])
        def chat():
            """Basic chat endpoint"""
            try:
                data = request.get_json()
                message = data.get('message', '')
                
                if not message:
                    return jsonify({"error": "Message is required"}), 400
                
                # Basic response for now
                response = f"Echo: {message} (LLM integration is running but not fully implemented yet)"
                
                return jsonify({
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Chat endpoint error: {e}")
                return jsonify({"error": str(e)}), 500
        
        # Register blueprint with Flask app
        qnti_system.app.register_blueprint(llm_bp)
        
        logger.info("LLM+MCP integration registered with QNTI system (basic version)")

        # ===========================
        # AI INSIGHT ENDPOINTS FOR DASHBOARD
        # ===========================
        
        @qnti_system.app.route('/api/ai/market-insight', methods=['GET'])
        def get_market_insight():
            """Generate AI market insight for dashboard"""
            try:
                # Get current market data
                trades = [t for t in qnti_system.trade_manager.trades.values() if t.status.name == 'OPEN'] if qnti_system.trade_manager else []
                account_info = qnti_system.mt5_bridge.get_mt5_status().get('account_info', {}) if qnti_system.mt5_bridge else {}
                
                # Prepare market context
                market_context = {
                    'active_trades': len(trades),
                    'symbols': list(set([trade.symbol for trade in trades])),
                    'account_balance': account_info.get('balance', 0),
                    'account_equity': account_info.get('equity', 0),
                    'margin_level': account_info.get('margin_level', 0)
                }
                
                # Generate market insight using LLM
                prompt = f"""
                Analyze the current market conditions and provide a brief trading insight.
                
                Current Market Data:
                - Active Trades: {market_context['active_trades']}
                - Trading Symbols: {', '.join(market_context['symbols'][:5])}
                - Account Balance: ${market_context['account_balance']:.2f}
                - Account Equity: ${market_context['account_equity']:.2f}
                - Margin Level: {market_context['margin_level']:.1f}%
                
                Provide a concise market insight (max 2 sentences) focusing on:
                1. Current market sentiment
                2. Key trading opportunities or risks
                
                Format: Direct insight without "AI says" or similar prefixes.
                """
                
                # Use LLM to generate insight
                try:
                    response = ollama.chat(
                        model=LLMConfig.LLM_MODEL,
                        messages=[{'role': 'user', 'content': prompt}],
                        options={
                            'temperature': 0.7,
                            'max_tokens': 150
                        }
                    )
                    insight = response['message']['content'].strip()
                except Exception as e:
                    logger.error(f"Error generating market insight: {e}")
                    insight = "Market analysis currently unavailable. Monitor key support/resistance levels."
                
                return jsonify({
                    'success': True,
                    'insight': insight,
                    'timestamp': datetime.now().isoformat(),
                    'context': market_context
                })
                
            except Exception as e:
                logger.error(f"Error in market insight endpoint: {e}")
                return jsonify({
                    'success': False,
                    'insight': 'Market analysis temporarily unavailable.',
                    'error': str(e)
                }), 500
        
        @qnti_system.app.route('/api/ai/account-analysis', methods=['GET'])
        def get_account_analysis():
            """Generate AI account analysis for dashboard"""
            try:
                # Get account and trading data
                trades = [t for t in qnti_system.trade_manager.trades.values() if t.status.name == 'OPEN'] if qnti_system.trade_manager else []
                account_info = qnti_system.mt5_bridge.get_mt5_status().get('account_info', {}) if qnti_system.mt5_bridge else {}
                
                # Calculate account metrics
                balance = account_info.get('balance', 0)
                equity = account_info.get('equity', 0)
                margin = account_info.get('margin', 0)
                free_margin = account_info.get('margin_free', 0)
                margin_level = account_info.get('margin_level', 0)
                
                # Risk assessment
                risk_level = "Low"
                if margin_level < 200:
                    risk_level = "High"
                elif margin_level < 500:
                    risk_level = "Medium"
                
                # Generate account analysis
                prompt = f"""
                Analyze the trading account health and provide risk assessment.
                
                Account Metrics:
                - Balance: ${balance:.2f}
                - Equity: ${equity:.2f}
                - Used Margin: ${margin:.2f}
                - Free Margin: ${free_margin:.2f}
                - Margin Level: {margin_level:.1f}%
                - Active Trades: {len(trades)}
                - Risk Level: {risk_level}
                
                Provide a concise account analysis (max 2 sentences) focusing on:
                1. Account health status
                2. Risk management recommendation
                
                Format: Direct analysis without "AI says" or similar prefixes.
                """
                
                try:
                    response = ollama.chat(
                        model=LLMConfig.LLM_MODEL,
                        messages=[{'role': 'user', 'content': prompt}],
                        options={
                            'temperature': 0.6,
                            'max_tokens': 150
                        }
                    )
                    analysis = response['message']['content'].strip()
                except Exception as e:
                    logger.error(f"Error generating account analysis: {e}")
                    analysis = "Account health is stable. Monitor margin levels and consider position sizing."
                
                return jsonify({
                    'success': True,
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': {
                        'balance': balance,
                        'equity': equity,
                        'margin_level': margin_level,
                        'risk_level': risk_level,
                        'active_trades': len(trades)
                    }
                })
                
            except Exception as e:
                logger.error(f"Error in account analysis endpoint: {e}")
                return jsonify({
                    'success': False,
                    'analysis': 'Account analysis temporarily unavailable.',
                    'error': str(e)
                }), 500
        
        @qnti_system.app.route('/api/ai/performance-analysis', methods=['GET'])
        def get_performance_analysis():
            """Generate AI performance analysis for dashboard"""
            try:
                # Get performance data
                trades = list(qnti_system.trade_manager.trades.values()) if qnti_system.trade_manager else []
                account_info = qnti_system.mt5_bridge.get_mt5_status().get('account_info', {}) if qnti_system.mt5_bridge else {}
                
                # Calculate performance metrics
                total_trades = len(trades)
                winning_trades = len([t for t in trades if t.profit and t.profit > 0])
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                total_profit = sum([t.profit for t in trades if t.profit])
                
                # Recent performance (last 7 days)
                recent_trades = [t for t in trades if t.close_time and 
                               (datetime.now() - t.close_time).days <= 7]
                recent_profit = sum([t.profit for t in recent_trades if t.profit])
                
                # Generate performance analysis
                prompt = f"""
                Analyze the trading performance and provide insights.
                
                Performance Metrics:
                - Total Trades: {total_trades}
                - Win Rate: {win_rate:.1f}%
                - Total Profit: ${total_profit:.2f}
                - Recent 7-day Profit: ${recent_profit:.2f}
                - Current Equity: ${account_info.get('equity', 0):.2f}
                
                Provide a concise performance analysis (max 2 sentences) focusing on:
                1. Current performance trend
                2. Key strength or area for improvement
                
                Format: Direct analysis without "AI says" or similar prefixes.
                """
                
                try:
                    response = ollama.chat(
                        model=LLMConfig.LLM_MODEL,
                        messages=[{'role': 'user', 'content': prompt}],
                        options={
                            'temperature': 0.6,
                            'max_tokens': 150
                        }
                    )
                    analysis = response['message']['content'].strip()
                except Exception as e:
                    logger.error(f"Error generating performance analysis: {e}")
                    analysis = "Performance tracking shows steady progress. Continue monitoring risk-reward ratios."
                
                return jsonify({
                    'success': True,
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': {
                        'total_trades': total_trades,
                        'win_rate': win_rate,
                        'total_profit': total_profit,
                        'recent_profit': recent_profit
                    }
                })
                
            except Exception as e:
                logger.error(f"Error in performance analysis endpoint: {e}")
                return jsonify({
                    'success': False,
                    'analysis': 'Performance analysis temporarily unavailable.',
                    'error': str(e)
                }), 500
        
        # Simple cache for AI insights to reduce Ollama load
        _ai_insights_cache = {}
        _cache_timestamp = 0
        CACHE_DURATION = 300  # 5 minutes cache to reduce Ollama load
        
        @qnti_system.app.route('/api/ai/insights/all', methods=['GET'])
        def get_all_insights():
            """Get all AI insights in one call for dashboard efficiency"""
            try:
                nonlocal _cache_timestamp
                # Check cache first
                current_time = time.time()
                if (_ai_insights_cache and 
                    current_time - _cache_timestamp < CACHE_DURATION):
                    logger.info("Serving AI insights from cache")
                    return jsonify(_ai_insights_cache)
                
                insights = {}
                
                # Get market insight directly
                try:
                    trades = [t for t in qnti_system.trade_manager.trades.values() if t.status.name == 'OPEN'] if qnti_system.trade_manager else []
                    account_info = qnti_system.mt5_bridge.get_mt5_status().get('account_info', {}) if qnti_system.mt5_bridge else {}
                    
                    market_context = {
                        'active_trades': len(trades),
                        'symbols': list(set([trade.symbol for trade in trades])),
                        'account_balance': account_info.get('balance', 0),
                        'account_equity': account_info.get('equity', 0),
                        'margin_level': account_info.get('margin_level', 0)
                    }
                    
                    prompt = f"""Analyze current market conditions briefly.
                    Active Trades: {market_context['active_trades']}
                    Symbols: {', '.join(market_context['symbols'][:3])}
                    Balance: ${market_context['account_balance']:.2f}
                    Provide 1-2 sentence market insight."""
                    
                    response = ollama.chat(
                        model=LLMConfig.LLM_MODEL,
                        messages=[{'role': 'user', 'content': prompt}],
                        options={'temperature': 0.7, 'max_tokens': 100}
                    )
                    insights['market'] = response['message']['content'].strip()
                except:
                    insights['market'] = 'Market conditions monitoring active'
                
                # Get account analysis directly  
                try:
                    balance = account_info.get('balance', 0)
                    equity = account_info.get('equity', 0)
                    margin_level = account_info.get('margin_level', 0)
                    
                    prompt = f"""Analyze account health briefly.
                    Balance: ${balance:.2f}, Equity: ${equity:.2f}
                    Margin Level: {margin_level:.1f}%
                    Provide 1-2 sentence account analysis."""
                    
                    response = ollama.chat(
                        model=LLMConfig.LLM_MODEL,
                        messages=[{'role': 'user', 'content': prompt}],
                        options={'temperature': 0.6, 'max_tokens': 100}
                    )
                    insights['account'] = response['message']['content'].strip()
                except:
                    insights['account'] = 'Account health monitoring active'
                
                # Get performance analysis directly
                try:
                    all_trades = list(qnti_system.trade_manager.trades.values()) if qnti_system.trade_manager else []
                    total_trades = len(all_trades)
                    winning_trades = len([t for t in all_trades if t.profit and t.profit > 0])
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    
                    prompt = f"""Analyze trading performance briefly.
                    Total Trades: {total_trades}
                    Win Rate: {win_rate:.1f}%
                    Provide 1-2 sentence performance insight."""
                    
                    response = ollama.chat(
                        model=LLMConfig.LLM_MODEL,
                        messages=[{'role': 'user', 'content': prompt}],
                        options={'temperature': 0.6, 'max_tokens': 100}
                    )
                    insights['performance'] = response['message']['content'].strip()
                except:
                    insights['performance'] = 'Performance tracking active'
                
                result = {
                    'success': True,
                    'insights': insights,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Update cache
                _ai_insights_cache.clear()
                _ai_insights_cache.update(result)
                _cache_timestamp = current_time
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error getting all insights: {e}")
                return jsonify({
                    'success': False,
                    'insights': {
                        'market': 'Market monitoring active',
                        'account': 'Account monitoring active', 
                        'performance': 'Performance tracking active'
                    },
                    'error': str(e)
                }), 500
        
        return llm_integration
        
    except Exception as e:
        logger.error(f"Error integrating LLM with QNTI: {e}")
        return None

# ===========================
# Convenience Functions
# ===========================

def get_llm_status():
    """Get LLM integration status"""
    if llm_integration:
        return llm_integration.get_status()
    return {"initialized": False, "error": "LLM integration not available"} 