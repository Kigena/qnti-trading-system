"""
QNTI API Services Integration
Comprehensive API service layer for external data providers
"""

import json
import asyncio
import aiohttp
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import threading
from dataclasses import dataclass
from enum import Enum
import requests
import pandas as pd
import praw
import tweepy
from newsapi import NewsApiClient
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.foreignexchange import ForeignExchange
import finnhub
from polygon import RESTClient
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import redis
from functools import wraps
import hashlib
import openai
import base64
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProvider(Enum):
    """Supported data providers"""
    NEWS_API = "news_api"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    TRADING_ECONOMICS = "trading_economics"
    REDDIT = "reddit"
    TWITTER = "twitter"
    POLYGON_IO = "polygon_io"
    TWELVE_DATA = "twelve_data"
    YAHOO_FINANCE = "yahoo_finance"

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    provider: DataProvider

@dataclass
class NewsItem:
    """News item structure"""
    title: str
    content: str
    url: str
    source: str
    published_at: datetime
    sentiment: Optional[float] = None
    relevance: Optional[float] = None

@dataclass
class SocialSentiment:
    """Social sentiment data"""
    platform: str
    symbol: str
    sentiment_score: float
    mention_count: int
    timestamp: datetime
    raw_data: Dict[str, Any]

class RateLimiter:
    """Rate limiter for API calls"""
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = time.time()
            # Remove old calls
            self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    # Clean up again after sleep
                    now = time.time()
                    self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            self.calls.append(now)

def with_rate_limit(rate_limiter: RateLimiter):
    """Decorator to add rate limiting to methods"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            rate_limiter.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper
    return decorator

class APICache:
    """Simple in-memory cache for API responses"""
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < 300:  # 5 minute cache
                    return value
                else:
                    del self.cache[key]
            return None
    
    def set(self, key: str, value: Any):
        """Set cached value"""
        with self.lock:
            self.cache[key] = (value, time.time())

class NewsAPIService:
    """News API service integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = NewsApiClient(api_key=api_key)
        self.rate_limiter = RateLimiter(max_calls=1000, time_window=86400)  # 1000 calls per day
        self.cache = APICache()
    
    def get_forex_news(self, symbols: List[str] = None, limit: int = 20) -> List[NewsItem]:
        """Get forex-related news"""
        cache_key = f"forex_news_{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            # Search for forex and trading news
            query = "forex OR trading OR currency OR EUR OR USD OR GBP OR JPY"
            articles = self.client.get_everything(
                q=query,
                language='en',
                sort_by='publishedAt',
                page_size=limit
            )
            
            news_items = []
            for article in articles.get('articles', []):
                news_item = NewsItem(
                    title=article.get('title', ''),
                    content=article.get('description', ''),
                    url=article.get('url', ''),
                    source=article.get('source', {}).get('name', ''),
                    published_at=datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00'))
                )
                news_items.append(news_item)
            
            self.cache.set(cache_key, news_items)
            logger.info(f"Retrieved {len(news_items)} forex news items")
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching forex news: {e}")
            return []

class AlphaVantageService:
    """Alpha Vantage API service"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        self.fd = FundamentalData(key=api_key, output_format='pandas')
        self.fx = ForeignExchange(key=api_key, output_format='pandas')
        self.rate_limiter = RateLimiter(max_calls=5, time_window=60)  # 5 calls per minute
        self.cache = APICache()
    
    def get_forex_data(self, from_symbol: str, to_symbol: str) -> Optional[MarketData]:
        """Get real-time forex data"""
        cache_key = f"forex_{from_symbol}_{to_symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            data, meta_data = self.fx.get_currency_exchange_rate(
                from_currency=from_symbol,
                to_currency=to_symbol
            )
            
            if not data.empty:
                price = float(data.iloc[0]['5. Exchange Rate'])
                
                market_data = MarketData(
                    symbol=f"{from_symbol}/{to_symbol}",
                    price=price,
                    change=0.0,  # Alpha Vantage doesn't provide change in this endpoint
                    change_percent=0.0,
                    volume=0,
                    timestamp=datetime.now(),
                    provider=DataProvider.ALPHA_VANTAGE
                )
                
                self.cache.set(cache_key, market_data)
                return market_data
                
        except Exception as e:
            logger.error(f"Error fetching forex data from Alpha Vantage: {e}")
            return None

class FinnhubService:
    """Finnhub API service"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = finnhub.Client(api_key=api_key)
        self.rate_limiter = RateLimiter(max_calls=60, time_window=60)  # 60 calls per minute
        self.cache = APICache()
    
    def get_forex_data(self, symbol: str) -> Optional[MarketData]:
        """Get forex data from Finnhub"""
        cache_key = f"finnhub_forex_{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            quote = self.client.quote(symbol)
            
            market_data = MarketData(
                symbol=symbol,
                price=quote['c'],  # Current price
                change=quote['d'],  # Change
                change_percent=quote['dp'],  # Change percent
                volume=0,
                timestamp=datetime.fromtimestamp(quote['t']),
                provider=DataProvider.FINNHUB
            )
            
            self.cache.set(cache_key, market_data)
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching data from Finnhub: {e}")
            return None
    
    def get_market_news(self, category: str = "forex", limit: int = 20) -> List[NewsItem]:
        """Get market news from Finnhub"""
        cache_key = f"finnhub_news_{category}_{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            news = self.client.general_news(category, min_id=0)
            
            news_items = []
            for article in news[:limit]:
                news_item = NewsItem(
                    title=article.get('headline', ''),
                    content=article.get('summary', ''),
                    url=article.get('url', ''),
                    source=article.get('source', ''),
                    published_at=datetime.fromtimestamp(article.get('datetime', 0))
                )
                news_items.append(news_item)
            
            self.cache.set(cache_key, news_items)
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching news from Finnhub: {e}")
            return []

class RedditService:
    """Reddit API service for sentiment analysis"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str, 
                 username: str, password: str):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            username=username,
            password=password
        )
        self.rate_limiter = RateLimiter(max_calls=100, time_window=60)  # Conservative limit
        self.cache = APICache()
    
    def get_trading_sentiment(self, symbols: List[str] = None, limit: int = 50) -> List[SocialSentiment]:
        """Get trading sentiment from Reddit"""
        cache_key = f"reddit_sentiment_{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            sentiments = []
            subreddits = ['forex', 'trading', 'investing', 'stocks']
            
            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                for submission in subreddit.hot(limit=limit//len(subreddits)):
                    # Simple sentiment scoring (could be enhanced with NLP)
                    sentiment_score = self._calculate_sentiment(submission.title + " " + submission.selftext)
                    
                    sentiment = SocialSentiment(
                        platform="Reddit",
                        symbol="GENERAL",
                        sentiment_score=sentiment_score,
                        mention_count=submission.num_comments,
                        timestamp=datetime.fromtimestamp(submission.created_utc),
                        raw_data={
                            'title': submission.title,
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'subreddit': subreddit_name
                        }
                    )
                    sentiments.append(sentiment)
            
            self.cache.set(cache_key, sentiments)
            return sentiments
            
        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment: {e}")
            return []
    
    def _calculate_sentiment(self, text: str) -> float:
        """Simple sentiment calculation (can be enhanced with proper NLP)"""
        positive_words = ['bullish', 'buy', 'long', 'profit', 'gain', 'up', 'rise', 'good', 'strong']
        negative_words = ['bearish', 'sell', 'short', 'loss', 'down', 'fall', 'bad', 'weak']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / max(total_words, 1)

class TwitterService:
    """Twitter API service for sentiment analysis"""
    
    def __init__(self, client_id: str, client_secret: str, bearer_token: str):
        self.client = tweepy.Client(bearer_token=bearer_token)
        self.rate_limiter = RateLimiter(max_calls=300, time_window=900)  # 300 per 15 min
        self.cache = APICache()
    
    def get_trading_sentiment(self, symbols: List[str] = None, limit: int = 100) -> List[SocialSentiment]:
        """Get trading sentiment from Twitter"""
        cache_key = f"twitter_sentiment_{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            sentiments = []
            query = "#forex OR #trading OR #EURUSD OR #GBPUSD OR #USDJPY -is:retweet lang:en"
            
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                max_results=min(limit, 100)
            ).flatten(limit=limit)
            
            for tweet in tweets:
                sentiment_score = self._calculate_sentiment(tweet.text)
                
                sentiment = SocialSentiment(
                    platform="Twitter",
                    symbol="GENERAL",
                    sentiment_score=sentiment_score,
                    mention_count=1,
                    timestamp=tweet.created_at,
                    raw_data={
                        'text': tweet.text,
                        'id': tweet.id,
                        'author_id': tweet.author_id
                    }
                )
                sentiments.append(sentiment)
            
            self.cache.set(cache_key, sentiments)
            return sentiments
            
        except Exception as e:
            logger.error(f"Error fetching Twitter sentiment: {e}")
            return []
    
    def _calculate_sentiment(self, text: str) -> float:
        """Simple sentiment calculation"""
        positive_words = ['bullish', 'buy', 'long', 'profit', 'gain', 'up', 'rise', 'good', 'strong', '🚀', '💰']
        negative_words = ['bearish', 'sell', 'short', 'loss', 'down', 'fall', 'bad', 'weak', '📉', '💸']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / max(total_words, 1)

class PolygonIOService:
    """Polygon.io API service"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = RESTClient(api_key)
        self.rate_limiter = RateLimiter(max_calls=5, time_window=60)  # 5 calls per minute
        self.cache = APICache()
    
    @with_rate_limit(rate_limiter)
    def get_forex_data(self, symbol: str) -> Optional[MarketData]:
        """Get forex data from Polygon.io"""
        cache_key = f"polygon_forex_{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            # Convert symbol format (e.g., EURUSD -> C:EURUSD)
            polygon_symbol = f"C:{symbol}"
            
            # Get current price
            ticker = self.client.get_real_time_currency_conversion(
                from_currency=symbol[:3],
                to_currency=symbol[3:],
                amount=1
            )
            
            market_data = MarketData(
                symbol=symbol,
                price=ticker.converted,
                change=0.0,  # Would need additional API call for change
                change_percent=0.0,
                volume=0,
                timestamp=datetime.now(),
                provider=DataProvider.POLYGON_IO
            )
            
            self.cache.set(cache_key, market_data)
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching data from Polygon.io: {e}")
            return None

class TwelveDataService:
    """Twelve Data API service"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.rate_limiter = RateLimiter(max_calls=60, time_window=60)  # 60 calls per minute
        self.cache = APICache()
    
    @with_rate_limit(rate_limiter)
    def get_forex_data(self, symbol: str) -> Optional[MarketData]:
        """Get forex data from Twelve Data"""
        cache_key = f"twelvedata_forex_{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            url = f"{self.base_url}/price"
            params = {
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            market_data = MarketData(
                symbol=symbol,
                price=float(data['price']),
                change=0.0,  # Would need additional API call
                change_percent=0.0,
                volume=0,
                timestamp=datetime.now(),
                provider=DataProvider.TWELVE_DATA
            )
            
            self.cache.set(cache_key, market_data)
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching data from Twelve Data: {e}")
            return None

@dataclass
class VisionAnalysis:
    """Vision analysis result structure"""
    analysis_id: str
    symbol: str
    timeframe: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    analysis_text: str
    key_levels: List[Dict[str, Any]]
    risk_reward: Optional[float]
    timestamp: datetime
    image_path: Optional[str] = None

class OpenAIVisionService:
    """OpenAI Vision API service for chart analysis"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
        self.rate_limiter = RateLimiter(max_calls=50, time_window=60)  # 50 calls per minute
        self.cache = APICache()
        
        # Analysis prompt template
        self.analysis_prompt = """
        You are an expert forex trader and technical analyst. Analyze this trading chart image and provide:
        
        1. **Trading Signal**: BUY, SELL, or HOLD
        2. **Confidence Level**: 0.0 to 1.0 (where 1.0 is highest confidence)
        3. **Key Technical Levels**: Support and resistance levels
        4. **Risk/Reward Ratio**: Potential risk vs reward
        5. **Detailed Analysis**: Technical reasoning for your recommendation
        
        Format your response as JSON with the following structure:
        {
            "signal": "BUY|SELL|HOLD",
            "confidence": 0.0-1.0,
            "key_levels": [
                {"type": "support", "price": 1.0850, "strength": "strong"},
                {"type": "resistance", "price": 1.0920, "strength": "medium"}
            ],
            "risk_reward": 2.5,
            "analysis": "Detailed technical analysis explaining the reasoning..."
        }
        
        Symbol: {symbol}
        Timeframe: {timeframe}
        """
    
    def analyze_chart_image(self, image_data: bytes, symbol: str, timeframe: str, 
                          analysis_id: str = None) -> Optional[VisionAnalysis]:
        """Analyze a chart image using OpenAI Vision"""
        if analysis_id is None:
            analysis_id = f"vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}"
        
        cache_key = f"vision_analysis_{analysis_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            self.rate_limiter.wait_if_needed()
            
            # Convert image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare the prompt
            prompt = self.analysis_prompt.format(symbol=symbol, timeframe=timeframe)
            
            # Call OpenAI Vision API
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            # Parse the response
            analysis_text = response.choices[0].message.content
            
            # Try to extract JSON from the response
            try:
                import re
                json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                if json_match:
                    analysis_json = json.loads(json_match.group())
                    
                    vision_analysis = VisionAnalysis(
                        analysis_id=analysis_id,
                        symbol=symbol,
                        timeframe=timeframe,
                        signal=analysis_json.get('signal', 'HOLD'),
                        confidence=float(analysis_json.get('confidence', 0.5)),
                        analysis_text=analysis_json.get('analysis', analysis_text),
                        key_levels=analysis_json.get('key_levels', []),
                        risk_reward=analysis_json.get('risk_reward'),
                        timestamp=datetime.now()
                    )
                else:
                    # Fallback if JSON parsing fails
                    vision_analysis = VisionAnalysis(
                        analysis_id=analysis_id,
                        symbol=symbol,
                        timeframe=timeframe,
                        signal='HOLD',
                        confidence=0.5,
                        analysis_text=analysis_text,
                        key_levels=[],
                        risk_reward=None,
                        timestamp=datetime.now()
                    )
                    
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to parse vision analysis JSON: {e}")
                # Create basic analysis result
                vision_analysis = VisionAnalysis(
                    analysis_id=analysis_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    signal='HOLD',
                    confidence=0.5,
                    analysis_text=analysis_text,
                    key_levels=[],
                    risk_reward=None,
                    timestamp=datetime.now()
                )
            
            # Cache the result
            self.cache.set(cache_key, vision_analysis)
            logger.info(f"Vision analysis completed for {symbol} - {vision_analysis.signal} (confidence: {vision_analysis.confidence})")
            
            return vision_analysis
            
        except Exception as e:
            logger.error(f"Error in OpenAI Vision analysis: {e}")
            return None
    
    def get_analysis_history(self, limit: int = 10) -> List[VisionAnalysis]:
        """Get recent analysis history (placeholder - would need database storage)"""
        # This would typically query a database
        # For now, return empty list
        return []
    
    def test_connection(self) -> bool:
        """Test OpenAI API connection"""
        try:
            # Simple test call
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI Vision connection test failed: {e}")
            return False

class QNTIAPIManager:
    """Main API manager that coordinates all services"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.services = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all API services"""
        api_keys = self.config.get('api_keys', {})
        
        # Initialize News API
        if api_keys.get('news_api'):
            try:
                self.services['news_api'] = NewsAPIService(api_keys['news_api'])
                logger.info("News API service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize News API: {e}")
        
        # Initialize Alpha Vantage
        if api_keys.get('alpha_vantage'):
            try:
                self.services['alpha_vantage'] = AlphaVantageService(api_keys['alpha_vantage'])
                logger.info("Alpha Vantage service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Alpha Vantage: {e}")
        
        # Initialize Finnhub
        if api_keys.get('finnhub'):
            try:
                self.services['finnhub'] = FinnhubService(api_keys['finnhub'])
                logger.info("Finnhub service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Finnhub: {e}")
        
        # Initialize Reddit
        reddit_keys = ['reddit_client_id', 'reddit_client_secret', 'reddit_user_agent', 
                      'reddit_username', 'reddit_password']
        if all(api_keys.get(key) for key in reddit_keys):
            try:
                self.services['reddit'] = RedditService(
                    api_keys['reddit_client_id'],
                    api_keys['reddit_client_secret'],
                    api_keys['reddit_user_agent'],
                    api_keys['reddit_username'],
                    api_keys['reddit_password']
                )
                logger.info("Reddit service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit: {e}")
        
        # Initialize Twitter
        twitter_keys = ['twitter_client_id', 'twitter_client_secret', 'twitter_bearer_token']
        if all(api_keys.get(key) for key in twitter_keys):
            try:
                self.services['twitter'] = TwitterService(
                    api_keys['twitter_client_id'],
                    api_keys['twitter_client_secret'],
                    api_keys['twitter_bearer_token']
                )
                logger.info("Twitter service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Twitter: {e}")
        
        # Initialize Polygon.io
        if api_keys.get('polygon_io'):
            try:
                self.services['polygon_io'] = PolygonIOService(api_keys['polygon_io'])
                logger.info("Polygon.io service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Polygon.io: {e}")
        
        # Initialize Twelve Data
        if api_keys.get('twelve_data'):
            try:
                self.services['twelve_data'] = TwelveDataService(api_keys['twelve_data'])
                logger.info("Twelve Data service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Twelve Data: {e}")
        
        # Initialize OpenAI Vision (using the vision config section)
        vision_config = self.config.get('vision', {})
        openai_api_key = vision_config.get('openai_api_key')
        if openai_api_key:
            try:
                self.services['openai_vision'] = OpenAIVisionService(openai_api_key)
                logger.info("OpenAI Vision service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI Vision: {e}")
    
    def get_all_market_data(self, symbols: List[str]) -> Dict[str, List[MarketData]]:
        """Get market data from all available providers"""
        results = {}
        
        futures = []
        for symbol in symbols:
            for service_name, service in self.services.items():
                if hasattr(service, 'get_forex_data'):
                    future = self.executor.submit(service.get_forex_data, symbol)
                    futures.append((service_name, symbol, future))
        
        for service_name, symbol, future in futures:
            try:
                result = future.result(timeout=30)
                if result:
                    if symbol not in results:
                        results[symbol] = []
                    results[symbol].append(result)
            except Exception as e:
                logger.error(f"Error getting data from {service_name} for {symbol}: {e}")
        
        return results
    
    def get_all_news(self, limit: int = 50) -> List[NewsItem]:
        """Get news from all available providers"""
        all_news = []
        
        futures = []
        for service_name, service in self.services.items():
            if hasattr(service, 'get_forex_news'):
                future = self.executor.submit(service.get_forex_news, None, limit)
                futures.append((service_name, future))
            elif hasattr(service, 'get_market_news'):
                future = self.executor.submit(service.get_market_news, "forex", limit)
                futures.append((service_name, future))
        
        for service_name, future in futures:
            try:
                result = future.result(timeout=30)
                all_news.extend(result)
            except Exception as e:
                logger.error(f"Error getting news from {service_name}: {e}")
        
        # Remove duplicates and sort by timestamp
        unique_news = {}
        for news in all_news:
            key = news.title + news.source
            if key not in unique_news:
                unique_news[key] = news
        
        sorted_news = sorted(unique_news.values(), key=lambda x: x.published_at, reverse=True)
        return sorted_news[:limit]
    
    def get_social_sentiment(self, symbols: List[str] = None, limit: int = 100) -> List[SocialSentiment]:
        """Get social sentiment from all available providers"""
        all_sentiment = []
        
        futures = []
        for service_name, service in self.services.items():
            if hasattr(service, 'get_trading_sentiment'):
                future = self.executor.submit(service.get_trading_sentiment, symbols, limit)
                futures.append((service_name, future))
        
        for service_name, future in futures:
            try:
                result = future.result(timeout=30)
                all_sentiment.extend(result)
            except Exception as e:
                logger.error(f"Error getting sentiment from {service_name}: {e}")
        
        return sorted(all_sentiment, key=lambda x: x.timestamp, reverse=True)
    
    def analyze_chart_image(self, image_data: bytes, symbol: str, timeframe: str) -> Optional[VisionAnalysis]:
        """Analyze chart image using OpenAI Vision"""
        vision_service = self.services.get('openai_vision')
        if vision_service:
            return vision_service.analyze_chart_image(image_data, symbol, timeframe)
        return None
    
    def get_vision_analysis_history(self, limit: int = 10) -> List[VisionAnalysis]:
        """Get vision analysis history"""
        vision_service = self.services.get('openai_vision')
        if vision_service:
            return vision_service.get_analysis_history(limit)
        return []
    
    def test_vision_connection(self) -> bool:
        """Test OpenAI Vision connection"""
        vision_service = self.services.get('openai_vision')
        if vision_service:
            return vision_service.test_connection()
        return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {}
        for service_name, service in self.services.items():
            service_status = {
                'initialized': True,
                'rate_limit_remaining': getattr(service, 'rate_limiter', None) is not None,
                'cache_size': len(getattr(service, 'cache', {}).cache) if hasattr(service, 'cache') else 0
            }
            
            # Add specific capabilities for each service
            if service_name == 'openai_vision':
                service_status['capabilities'] = ['chart_analysis', 'image_processing']
            elif hasattr(service, 'get_forex_data'):
                service_status['capabilities'] = ['market_data']
            elif hasattr(service, 'get_forex_news') or hasattr(service, 'get_market_news'):
                service_status['capabilities'] = ['news']
            elif hasattr(service, 'get_trading_sentiment'):
                service_status['capabilities'] = ['sentiment']
            
            status[service_name] = service_status
        
        return {
            'total_services': len(self.services),
            'services': status,
            'last_update': datetime.now().isoformat()
        } 