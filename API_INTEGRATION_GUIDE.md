# QNTI External API Integration Guide

## Overview

The QNTI External API Integration system provides comprehensive connectivity to multiple external data providers for enhanced market analysis, news sentiment, and real-time data aggregation. This system integrates seamlessly with your existing QNTI dashboard and provides background data collection with intelligent caching.

## Supported API Providers

### ✅ **Implemented Services:**

1. **News API** - Real-time news aggregation
2. **Alpha Vantage** - Financial market data
3. **Finnhub** - Stock and forex data with news
4. **Reddit** - Social sentiment analysis
5. **Twitter** - Social media sentiment tracking
6. **Polygon.io** - Financial market data
7. **Twelve Data** - Market data and analytics

### 🔧 **Configuration Required:**

All API keys are configured in `vision_config.json` under the `api_keys` section:

```json
{
  "api_keys": {
    "news_api": "your_news_api_key",
    "alpha_vantage": "your_alpha_vantage_key",
    "finnhub": "your_finnhub_key",
    "reddit_client_id": "your_reddit_client_id",
    "reddit_client_secret": "your_reddit_client_secret",
    "reddit_user_agent": "your_reddit_user_agent",
    "reddit_username": "your_reddit_username",
    "reddit_password": "your_reddit_password",
    "twitter_client_id": "your_twitter_client_id",
    "twitter_client_secret": "your_twitter_client_secret",
    "twitter_bearer_token": "your_twitter_bearer_token",
    "polygon_io": "your_polygon_io_key",
    "twelve_data": "your_twelve_data_key"
  }
}
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Configuration

Your API keys are already configured in `vision_config.json`. The system will automatically detect and initialize available services.

### 3. Start the System

```bash
python qnti_main_system.py
```

The system will automatically:
- Initialize all configured API services
- Start background data collection
- Integrate with the existing dashboard
- Set up rate limiting and caching

## API Endpoints

### Core Endpoints

#### 1. Service Status
```
GET /api/external/status
```
Returns the status of all configured API services.

**Example Response:**
```json
{
  "total_services": 6,
  "services": {
    "news_api": {
      "initialized": true,
      "rate_limit_remaining": true,
      "cache_size": 25
    },
    "alpha_vantage": {
      "initialized": true,
      "rate_limit_remaining": true,
      "cache_size": 12
    }
  },
  "last_update": "2024-01-07T10:30:00Z"
}
```

#### 2. Market Data
```
GET /api/external/market-data?symbols=EURUSD&symbols=GBPUSD&symbols=USDJPY
```
Aggregates market data from multiple providers.

**Example Response:**
```json
{
  "success": true,
  "data": {
    "EURUSD": [
      {
        "symbol": "EURUSD",
        "price": 1.08945,
        "change": -0.0012,
        "change_percent": -0.11,
        "provider": "alpha_vantage",
        "timestamp": "2024-01-07T10:30:00Z"
      }
    ]
  },
  "timestamp": "2024-01-07T10:30:00Z"
}
```

#### 3. News Data
```
GET /api/external/news?limit=20
```
Retrieves aggregated news from multiple sources.

**Example Response:**
```json
{
  "success": true,
  "news": [
    {
      "title": "EUR/USD Outlook: ECB Policy Decision Impact",
      "content": "The European Central Bank's latest decision...",
      "source": "Financial Times",
      "published_at": "2024-01-07T09:45:00Z",
      "sentiment": 0.2,
      "relevance": 0.85
    }
  ],
  "count": 20,
  "timestamp": "2024-01-07T10:30:00Z"
}
```

#### 4. Social Sentiment
```
GET /api/external/sentiment?limit=50
```
Analyzes social media sentiment from Reddit and Twitter.

**Example Response:**
```json
{
  "success": true,
  "sentiment": [
    {
      "platform": "Reddit",
      "symbol": "GENERAL",
      "sentiment_score": 0.15,
      "mention_count": 45,
      "timestamp": "2024-01-07T10:25:00Z"
    }
  ],
  "aggregate": {
    "average_sentiment": 0.12,
    "total_mentions": 234,
    "platforms": ["Reddit", "Twitter"]
  },
  "count": 50,
  "timestamp": "2024-01-07T10:30:00Z"
}
```

#### 5. Combined Data
```
GET /api/external/combined-data?symbols=EURUSD&news_limit=10&sentiment_limit=20
```
Retrieves market data, news, and sentiment in one request.

#### 6. Available Providers
```
GET /api/external/providers
```
Lists all available data providers and their capabilities.

#### 7. Test Connection
```
POST /api/external/test-connection
Content-Type: application/json

{
  "provider": "alpha_vantage"
}
```
Tests connection to a specific API provider.

#### 8. OpenAI Vision Analysis
```
POST /api/external/vision/analyze
Content-Type: multipart/form-data

Form Data:
- image: Chart image file (PNG, JPG, JPEG)
- symbol: Currency pair (default: "EURUSD")
- timeframe: Chart timeframe (default: "H1")
```
Analyzes a chart image using OpenAI Vision for trading signals.

**Example Response:**
```json
{
  "success": true,
  "analysis": {
    "analysis_id": "vision_20240115_103000_EURUSD",
    "symbol": "EURUSD",
    "timeframe": "H1",
    "signal": "BUY",
    "confidence": 0.85,
    "analysis_text": "Strong bullish momentum with clear support at 1.0850...",
    "key_levels": [
      {
        "type": "support",
        "price": 1.0850,
        "strength": "strong"
      },
      {
        "type": "resistance",
        "price": 1.0920,
        "strength": "medium"
      }
    ],
    "risk_reward": 2.5,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### 9. Vision Analysis History
```
GET /api/external/vision/history?limit=10
```
Retrieves previous vision analysis results.

#### 10. Test Vision Connection
```
POST /api/external/vision/test
Content-Type: application/json
```
Tests OpenAI Vision service connection and availability.

## Dashboard Integration

### New Dashboard Sections

1. **External Market Data** - Real-time aggregated market prices
2. **News & Social Sentiment** - Latest news with sentiment analysis
3. **API Services Status** - Health monitoring of all API services
4. **OpenAI Vision Analysis** - Integrated vision analysis with connection testing

### Features

- **Real-time Updates**: Data refreshes every 5 minutes
- **Manual Refresh**: Click refresh buttons for immediate updates
- **Visual Indicators**: Color-coded sentiment and change indicators
- **Multi-source Aggregation**: Combines data from multiple providers
- **Error Handling**: Graceful degradation when services are unavailable

## Background Data Collection

The system automatically runs background tasks to:

- **News Collection**: Every 30 minutes
- **Sentiment Analysis**: Every 15 minutes
- **Market Data**: Every 5 minutes

All data is cached to reduce API calls and improve performance.

## Rate Limiting

Each API service has built-in rate limiting:

- **News API**: 1,000 calls per day
- **Alpha Vantage**: 5 calls per minute
- **Finnhub**: 60 calls per minute
- **Reddit**: 100 calls per minute
- **Twitter**: 300 calls per 15 minutes
- **Polygon.io**: 5 calls per minute
- **Twelve Data**: 60 calls per minute
- **OpenAI Vision**: 50 calls per minute

## Caching System

- **Cache Duration**: 5 minutes for most data
- **Cache Size**: Automatically managed per service
- **Cache Invalidation**: Automatic cleanup of expired data

## Error Handling

### Service Failures
- Individual service failures don't affect other services
- Graceful degradation with fallback data
- Automatic retry mechanisms

### Rate Limit Handling
- Automatic delay when approaching limits
- Queue management for API calls
- Intelligent backoff strategies

### Network Issues
- Timeout handling (30 seconds)
- Connection retry logic
- Fallback to cached data

## Monitoring & Logging

### Service Health
Check service health via the dashboard or API endpoint:
```
GET /api/external/status
```

### Logs
Monitor system logs for:
- API initialization status
- Background task execution
- Error conditions
- Rate limit warnings

### Performance Metrics
- Response times
- Cache hit rates
- API call frequency
- Error rates

## Troubleshooting

### Common Issues

#### 1. API Service Not Initializing
**Symptoms**: Service shows as "Inactive" in dashboard
**Solutions**:
- Check API key validity
- Verify internet connectivity
- Review logs for specific error messages
- Test individual service connection

#### 2. No Data Appearing
**Symptoms**: Empty data sections in dashboard
**Solutions**:
- Check API service status
- Verify API key permissions
- Check rate limits
- Review browser console for errors

#### 3. Rate Limit Exceeded
**Symptoms**: API calls failing with rate limit errors
**Solutions**:
- Wait for rate limit reset
- Reduce polling frequency
- Implement additional caching
- Consider upgrading API plan

#### 4. Authentication Errors
**Symptoms**: API services failing to authenticate
**Solutions**:
- Verify API keys are correct
- Check API key permissions
- Ensure proper formatting (no extra spaces)
- Test keys with provider's documentation

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export QNTI_DEBUG=true
python qnti_main_system.py
```

### Manual Testing

Test individual API endpoints:
```bash
# Test service status
curl http://localhost:5000/api/external/status

# Test market data
curl "http://localhost:5000/api/external/market-data?symbols=EURUSD"

# Test news
curl "http://localhost:5000/api/external/news?limit=5"

# Test sentiment
curl "http://localhost:5000/api/external/sentiment?limit=10"
```

## API Key Management

### Security Best Practices

1. **Environment Variables**: Store sensitive keys in environment variables
2. **Key Rotation**: Regularly rotate API keys
3. **Access Control**: Limit API key permissions where possible
4. **Monitoring**: Monitor API key usage

### Key Validation

The system validates API keys on startup and provides detailed error messages for invalid keys.

## Performance Optimization

### Caching Strategy
- **Memory Caching**: Fast access to recent data
- **Intelligent Expiration**: Balanced between freshness and performance
- **Background Refresh**: Proactive data updates

### Parallel Processing
- **Concurrent API Calls**: Multiple services called simultaneously
- **Thread Pool**: Efficient resource utilization
- **Timeout Management**: Prevents hanging requests

### Data Aggregation
- **Multi-source Comparison**: Combines data from multiple providers
- **Weighted Averages**: Intelligent data synthesis
- **Outlier Detection**: Filters unreliable data points

## Extending the System

### Adding New API Providers

1. **Create Service Class**: Implement the provider interface
2. **Add Configuration**: Include API keys in config
3. **Register Service**: Add to API manager initialization
4. **Update Dashboard**: Add UI components if needed

### Custom Data Processing

1. **Data Transformation**: Modify data processing pipelines
2. **Custom Aggregation**: Implement specialized aggregation logic
3. **Enhanced Caching**: Add custom caching strategies

## Support

### Documentation
- API provider documentation links
- Configuration examples
- Troubleshooting guides

### Community Resources
- GitHub issues for bug reports
- Discussion forum for questions
- Example configurations

### Professional Support
- Custom integration services
- Performance optimization consulting
- Training and workshops

## Conclusion

The QNTI External API Integration system provides a robust, scalable solution for aggregating market data, news, and sentiment from multiple sources. With built-in rate limiting, caching, and error handling, it seamlessly integrates with your existing trading system to provide enhanced market intelligence.

For additional support or custom requirements, please refer to the troubleshooting section or contact the development team. 