# QNTI LLM+Memory Context Platform Integration Guide

## Overview

This guide provides step-by-step instructions for integrating the LLM+Memory Context Platform (MCP) module into your existing QNTI trading system infrastructure.

## Prerequisites

### Software Requirements
- Python 3.8+
- Windows 11 Pro (32GB RAM, RTX 4050)
- Ollama installed and running
- Existing QNTI Flask application

### Hardware Recommendations
- Minimum 16GB RAM (32GB recommended)
- GPU with 8GB+ VRAM for optimal LLM performance
- SSD storage for ChromaDB persistence

## Installation Steps

### 1. Install Ollama and Pull LLM Model

```bash
# Download and install Ollama from https://ollama.ai/download

# Pull the Llama 3 model
ollama pull llama3

# Verify installation
ollama list
```

### 2. Install Python Dependencies

```bash
# Install LLM+MCP specific dependencies
pip install -r requirements_llm.txt
```

### 3. Run Automated Setup

```bash
# Run the automated setup script
python setup_llm_integration.py

# Or install dependencies only
python setup_llm_integration.py deps

# Test the integration
python setup_llm_integration.py test
```

### 4. Manual Configuration

Edit `qnti_llm_config.json` to customize settings:

```json
{
  "llm": {
    "model": "llama3",
    "base_url": "http://localhost:11434"
  },
  "news": {
    "api_key": "your_newsapi_key_here"
  },
  "security": {
    "api_key": "your-secure-api-key"
  }
}
```

## Integration with Existing QNTI System

The LLM+MCP module has been designed to seamlessly integrate with your existing Flask-based QNTI system. The integration is automatically handled in `qnti_main_system.py`.

### Integration Points

1. **Flask Blueprint Registration**: LLM routes are added as a Flask blueprint
2. **Background Scheduler**: Automated news updates and daily brief generation
3. **Memory Service**: ChromaDB for persistent context storage
4. **Trade Integration**: Automatic indexing of trade history and performance

### System Architecture

```
QNTI Main System (Flask)
├── Core Trading Components
│   ├── Trade Manager
│   ├── MT5 Bridge
│   └── Vision Analyzer
├── Web Interface
│   ├── Dashboard
│   ├── API Routes
│   └── WebSocket Handlers
└── LLM+MCP Integration (NEW)
    ├── Memory Service (ChromaDB)
    ├── LLM Service (Ollama)
    ├── News Service (NewsAPI)
    ├── Market Data Service (Yahoo Finance)
    └── Background Scheduler
```

## API Endpoints

Once integrated, the following endpoints will be available:

### Chat and Analysis
- `POST /llm/chat` - General chat with LLM using memory context
- `POST /llm/analyze` - AI-powered trade/market analysis
- `GET /llm/daily-brief` - Get or generate daily market brief

### Context Management
- `POST /llm/context/upload` - Upload custom documents to context
- `GET /llm/context/search` - Search context from vector store

### Data Services
- `POST /llm/news/fetch` - Fetch news on demand
- `GET /llm/status` - Get LLM service status

## Usage Examples

### 1. Chat with Trading Context

```python
import requests

headers = {"Authorization": "Bearer qnti-secret-key"}

response = requests.post(
    "http://localhost:5000/llm/chat",
    headers=headers,
    json={
        "message": "What were my most profitable trades this week?",
        "context_window": 20,
        "user_id": "trader1"
    }
)
print(response.json())
```

### 2. Analyze Specific Trades

```python
response = requests.post(
    "http://localhost:5000/llm/analyze",
    headers=headers,
    json={
        "symbol": "EURUSD",
        "timeframe": "H1",
        "include_news": True,
        "context_window": 15
    }
)
print(response.json())
```

### 3. Get Daily Market Brief

```python
response = requests.get(
    "http://localhost:5000/llm/daily-brief",
    headers=headers
)
print(response.json())
```

### 4. Upload Custom Analysis

```python
response = requests.post(
    "http://localhost:5000/llm/context/upload",
    headers=headers,
    json={
        "document_type": "analysis",
        "content": "Market analysis: Strong bullish momentum in EURUSD...",
        "metadata": {
            "symbol": "EURUSD",
            "analyst": "trader1",
            "confidence": 0.85
        }
    }
)
```

## Features

### 1. Conversational AI
- Natural language queries about trading performance
- Context-aware responses using historical data
- Memory of previous conversations

### 2. Automated Market Intelligence
- Daily market briefs generated at 6 AM
- Real-time news monitoring and analysis
- Market sentiment analysis
- Economic indicator tracking

### 3. Trade Analysis
- Performance correlation analysis
- Risk assessment and recommendations
- Pattern recognition in trading behavior
- Strategy optimization suggestions

### 4. Memory Context Platform
- Persistent storage of trading context
- Automatic indexing of trades, news, and market data
- Semantic search across all stored information
- Context retention and cleanup

## Configuration Options

### LLM Settings
```json
{
  "llm": {
    "model": "llama3",
    "base_url": "http://localhost:11434",
    "temperature": 0.7,
    "max_tokens": 2000,
    "timeout": 30
  }
}
```

### Memory Settings
```json
{
  "chroma": {
    "path": "./qnti_memory",
    "collection_name": "qnti_context",
    "persist_directory": "./qnti_memory"
  }
}
```

### Scheduling Settings
```json
{
  "scheduling": {
    "daily_brief_hour": 6,
    "news_update_interval": 60,
    "market_data_interval": 30
  }
}
```

## Security

### API Authentication
All LLM endpoints require API key authentication:

```python
headers = {"Authorization": "Bearer your-api-key"}
```

### Configuration Security
- Store sensitive API keys in environment variables
- Use strong, unique API keys for production
- Regularly rotate authentication tokens

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Check if Ollama is running
   ollama list
   
   # Start Ollama service if needed
   ollama serve
   ```

2. **ChromaDB Permission Errors**
   ```bash
   # Ensure memory directory is writable
   chmod 755 ./qnti_memory
   ```

3. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements_llm.txt
   ```

4. **Memory Issues**
   - Increase system RAM allocation
   - Reduce context window size
   - Clean up old memory data

### Debug Mode

Enable debug logging by setting:
```python
import logging
logging.getLogger('QNTI').setLevel(logging.DEBUG)
```

## Performance Optimization

### Memory Management
- Set `max_context_documents` to limit memory usage
- Configure `context_retention_days` for automatic cleanup
- Monitor ChromaDB storage size

### LLM Performance
- Use GPU acceleration if available
- Adjust `temperature` and `max_tokens` for speed vs quality
- Consider using smaller models for faster responses

### Background Tasks
- Adjust update intervals based on system capacity
- Monitor scheduler performance
- Use background task queues for heavy operations

## Monitoring and Maintenance

### Health Checks
```python
# Check LLM service status
response = requests.get("http://localhost:5000/llm/status")
```

### Log Monitoring
- Monitor `qnti_main.log` for integration status
- Check Ollama logs for LLM performance
- Review ChromaDB logs for memory operations

### Backup and Recovery
- Backup `qnti_memory` directory regularly
- Export important context data
- Document custom configurations

## Advanced Features

### Custom Models
```bash
# Use different Ollama models
ollama pull mistral
ollama pull codellama
```

Update configuration:
```json
{
  "llm": {
    "model": "mistral"
  }
}
```

### External APIs
- Configure NewsAPI for real-time news
- Add custom data sources
- Integrate with external analysis tools

### Webhooks and Notifications
- Set up Discord/Telegram notifications
- Configure email alerts for important events
- Create custom webhook endpoints

## Support and Resources

### Documentation
- [Ollama Documentation](https://ollama.ai/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [NewsAPI Documentation](https://newsapi.org/docs)

### Community
- QNTI Discord Server
- GitHub Issues and Discussions
- Trading Community Forums

### Professional Support
- Custom integration services
- Performance optimization consulting
- Training and workshops

## Conclusion

The QNTI LLM+MCP integration provides powerful AI-driven insights and conversational capabilities to your trading system. With proper setup and configuration, it enhances decision-making through intelligent analysis of market data, trade performance, and news sentiment.

For additional support or custom requirements, please refer to the troubleshooting section or contact the development team. 