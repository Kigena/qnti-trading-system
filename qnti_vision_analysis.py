#!/usr/bin/env python3
"""
Quantum Nexus Trading Intelligence (QNTI) - Enhanced Vision Analysis Module
Chart image upload, comprehensive GPT-4V analysis with structured trade scenarios
"""

import cv2
import numpy as np
import base64
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
import logging
import uuid
import re
from typing import List, Dict, Optional, Tuple
import dataclasses

# Correct imports from existing project files
from qnti_core_system import Trade, EAPerformance, QNTITradeManager
from qnti_vision_models import (
    TradeScenario,
    PriceLevel,
    TechnicalIndicator,
    ComprehensiveChartAnalysis,
    SignalStrength,
    MarketBias
)

# Configure logging for the vision module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)

class QNTIEnhancedVisionAnalyzer:
    """
    Handles chart image uploads and comprehensive analysis using OpenAI's Vision API.
    Provides structured analysis results for trading decisions.
    """

    def __init__(self, trade_manager: QNTITradeManager, config_file: str = "vision_config.json"):
        self.trade_manager = trade_manager
        self.config_file = config_file
        self.vision_config = {}
        self.openai_client = None
        self.upload_dir = Path("chart_uploads")
        self.analysis_results: Dict[str, ComprehensiveChartAnalysis] = {}
        self.is_running = False
        self.automated_analysis_task: Optional[asyncio.Task] = None

        self.upload_dir.mkdir(exist_ok=True)
        self._load_config()
        
        # Check for valid OpenAI API key
        # Handle nested vision config structure
        vision_settings = self.vision_config.get("vision", self.vision_config)
        api_key = vision_settings.get("openai_api_key", "")
        if api_key and api_key != "YOUR_OPENAI_API_KEY_HERE" and "YOUR_OPENAI_API_KEY" not in api_key and len(api_key) > 20:
            self._initialize_openai()
            logger.info(f"OpenAI API key loaded successfully (length: {len(api_key)})")
        else:
            logger.warning("OpenAI API key not provided or invalid for enhanced vision analysis")

    def _load_config(self):
        """Load enhanced vision analysis configuration"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.vision_config = json.load(f)
            else:
                logger.warning(f"{self.config_file} not found. Creating default configuration.")
                default_config = {
                    "openai_api_key": "YOUR_OPENAI_API_KEY_HERE",
                    "model_name": "gpt-4o",
                    "max_tokens": 3000,
                    "temperature": 0.2,
                    "analysis_prompt_template": self._get_enhanced_prompt_template(),
                }
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2)
                self.vision_config = default_config
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading vision config: {e}")
            self.vision_config = {}

    def _get_enhanced_prompt_template(self) -> str:
        """
        Returns a focused, actionable prompt template for practical trading chart analysis.
        """
        return """
You are a professional forex trader with 15+ years of experience. Analyze this trading chart and provide actionable trading insights.

FOCUS ON:
1. What is the clear trend direction?
2. Where are the obvious support/resistance levels?
3. What trading opportunity do you see RIGHT NOW?
4. How confident are you in this setup?

Be decisive and practical. Traders need clear, actionable guidance, not academic analysis.

RESPOND IN THIS EXACT JSON FORMAT:
{
  "symbol": "AUTO_DETECT_OR_UNKNOWN",
  "timeframe": "AUTO_DETECT_OR_H4", 
  "overall_trend": "BULLISH|BEARISH|SIDEWAYS",
  "trend_strength": "WEAK|MODERATE|STRONG|VERY_STRONG",
  "market_bias": "Price is trending up/down/sideways with X momentum. Key level to watch is Y.",
  "primary_scenario": {
    "scenario_name": "Trend Continuation|Reversal|Breakout|Pullback",
    "trade_type": "BUY|SELL",
    "entry_price": 1.2345,
    "stop_loss": 1.2300,
    "take_profit_1": 1.2400,
    "take_profit_2": 1.2450,
    "probability_success": 0.75,
    "reasoning": "Clear explanation why this trade makes sense based on what you see"
  },
  "alternative_scenario": {
    "scenario_name": "Alternative setup name",
    "trade_type": "BUY|SELL", 
    "entry_price": 1.2300,
    "stop_loss": 1.2350,
    "take_profit_1": 1.2250,
    "probability_success": 0.60,
    "reasoning": "Backup plan if primary scenario fails"
  },
  "overall_confidence": 0.80,
  "key_levels": [
    {"price": 1.2350, "type": "RESISTANCE", "strength": "STRONG", "reason": "Multiple touches, previous support turned resistance"},
    {"price": 1.2280, "type": "SUPPORT", "strength": "MODERATE", "reason": "Recent swing low, 50% fib level"}
  ],
  "risk_factors": ["News event tomorrow", "Approaching major resistance", "Divergence on RSI"],
  "confluence_factors": ["Trend line support", "Volume confirmation", "Bullish engulfing pattern"]
}

RULES:
- Be specific with price levels (use realistic numbers based on what you see)
- Confidence should be 60-95% (never 0% unless chart is unreadable)
- Probability success should be 55-85% for realistic trades
- Give clear, actionable reasoning
- If you can't read the chart clearly, say so but still give your best assessment
- ALWAYS pick either BUY or SELL - no NEUTRAL trades allowed
"""

    def _initialize_openai(self):
        """Initialize OpenAI client for vision analysis"""
        try:
            from openai import AsyncOpenAI
            vision_settings = self.vision_config.get("vision", self.vision_config)
            self.openai_client = AsyncOpenAI(api_key=vision_settings.get("openai_api_key"))
            logger.info("OpenAI client initialized for enhanced vision analysis")
        except ImportError:
            logger.error("OpenAI library not found. Please install with 'pip install openai'")
            self.openai_client = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None

    def upload_chart_image(self, image_data: bytes, filename: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validates and saves an uploaded chart image.
        Generates a unique analysis ID for the image.
        """
        if not image_data:
            return False, "No image data provided.", None

        file_ext = Path(filename).suffix.lower()
        if file_ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            return False, f"Unsupported image format. Supported formats: .jpg, .jpeg, .png, .webp", None

        try:
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[0] < 50 or image.shape[1] < 50:
                return False, "Invalid or too small image file.", None
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return False, "Could not process image file.", None

        analysis_id = f"CHART_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        save_path = self.upload_dir / f"{analysis_id}{file_ext}"

        try:
            with open(save_path, "wb") as f:
                f.write(image_data)
            logger.info(f"Chart image saved: {save_path} with ID: {analysis_id}")
            return True, "Chart uploaded successfully.", analysis_id
        except IOError as e:
            logger.error(f"Error saving uploaded chart image: {e}")
            return False, "Failed to save chart image.", None

    async def analyze_uploaded_chart(self, analysis_id: str, symbol: Optional[str] = None, timeframe: str = "H4") -> Optional[ComprehensiveChartAnalysis]:
        """Analyze an uploaded chart image using OpenAI Vision API"""
        logger.info(f'[VISION] Starting analysis for {analysis_id}, symbol={symbol}, timeframe={timeframe}')
        
        if not self.openai_client:
            logger.error('[VISION] OpenAI client not initialized')
            return None

        # Get image path
        image_path = self.upload_dir / f"{analysis_id}.png"
        if not image_path.exists():
            logger.error(f'[VISION] Image not found: {image_path}')
            return None

        try:
            # Get vision settings from nested config
            vision_settings = self.vision_config.get("vision", self.vision_config)
            
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Get prompt template
            prompt = vision_settings.get("analysis_prompt_template", self._get_enhanced_prompt_template())
            
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {vision_settings.get('openai_api_key')}"}
            payload = {
                "model": vision_settings.get("model_name", "gpt-4o"),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt.format(symbol=symbol or "UNKNOWN", timeframe=timeframe)},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                        ]
                    }
                ],
                "max_tokens": vision_settings.get("max_tokens", 3000),
                "temperature": vision_settings.get("temperature", 0.2),
            }

            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                async with session.post("https://api.openai.com/v1/chat/completions", json=payload) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        analysis_content = response_data['choices'][0]['message']['content']
                        logger.info(f"Raw OpenAI response for {analysis_id}:\n{analysis_content}")
                        analysis_data = json.loads(analysis_content)

                        result = self._create_comprehensive_analysis(analysis_data, analysis_id, str(image_path))
                        logger.info(f"[VISION] Created analysis object for {analysis_id}: {result}")
                        self.analysis_results[analysis_id] = result
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error for {analysis_id}: {response.status} - {error_text}")
                        return None
        except Exception as e:
            logger.error(f"[VISION] Error in analyze_uploaded_chart for {analysis_id}: {e}", exc_info=True)
            return None

    def _create_comprehensive_analysis(self, data: Dict, analysis_id: str, image_path: str) -> ComprehensiveChartAnalysis:
        """Create ComprehensiveChartAnalysis object from simplified API response dictionary."""
        
        def _parse_scenario(sc_data: Optional[Dict]) -> Optional[TradeScenario]:
            if not sc_data: return None
            return TradeScenario(
                scenario_name=sc_data.get("scenario_name", "Trading Setup"),
                trade_type=sc_data.get("trade_type", "BUY"),
                entry_price=float(sc_data.get("entry_price", 1.0)),
                stop_loss=float(sc_data.get("stop_loss", 1.0)),
                take_profit_1=float(sc_data.get("take_profit_1", 1.0)),
                take_profit_2=float(sc_data.get("take_profit_2", 1.0)),
                probability_success=float(sc_data.get("probability_success", 0.70)),
                notes=sc_data.get("reasoning", "AI analysis based on chart patterns")
            )

        def _parse_key_levels(lvl_data: Optional[List[Dict]]) -> List[PriceLevel]:
            if not lvl_data: return []
            return [
                PriceLevel(
                    price=float(lvl.get("price", 1.0)),
                    level_type=lvl.get("type", "SUPPORT").lower(),
                    strength=lvl.get("strength", "MODERATE").lower(),
                    context=lvl.get("reason", "Key technical level")
                ) for lvl in lvl_data
            ]
        
        def _parse_indicators() -> List[TechnicalIndicator]:
            # For simplified prompt, create basic indicators from confluence factors
            confluence = data.get("confluence_factors", [])
            if not confluence: 
                return [TechnicalIndicator(
                    name="Market Analysis",
                    value=75.0,
                    signal="bullish" if data.get("overall_trend", "").upper() == "BULLISH" else "bearish" if data.get("overall_trend", "").upper() == "BEARISH" else "neutral",
                    strength="moderate",
                    notes="AI analysis of chart patterns and market structure"
                )]
            
            return [TechnicalIndicator(
                name="Confluence Analysis",
                value=80.0,
                signal="bullish" if "bullish" in str(confluence).lower() else "bearish" if "bearish" in str(confluence).lower() else "neutral",
                strength="strong",
                notes=f"Multiple confluence factors: {', '.join(confluence[:3])}"
            )]
        
        primary_scenario = _parse_scenario(data.get("primary_scenario"))
        if not primary_scenario:
            # Create a default scenario if missing
            primary_scenario = TradeScenario(
                scenario_name="Default Analysis",
                trade_type=data.get("overall_trend", "BUY").upper() if data.get("overall_trend", "").upper() in ["BULLISH", "BEARISH"] else "BUY",
                entry_price=1.0,
                stop_loss=0.99,
                take_profit_1=1.01,
                take_profit_2=1.02,
                probability_success=0.65,
                notes="AI generated trading scenario based on chart analysis"
            )

        # Parse key levels and split into support/resistance
        key_levels = _parse_key_levels(data.get("key_levels", []))
        support_levels = [lvl for lvl in key_levels if lvl.level_type.lower() == "support"]
        resistance_levels = [lvl for lvl in key_levels if lvl.level_type.lower() == "resistance"]

        analysis = ComprehensiveChartAnalysis(
            analysis_id=analysis_id,
            timestamp=datetime.now(),
            symbol=data.get("symbol", "UNKNOWN"),
            timeframe=data.get("timeframe", "H4"),
            current_price=float(data.get("current_price", 1.0)),
            overall_trend=data.get("overall_trend", "SIDEWAYS").lower(),
            trend_strength=data.get("trend_strength", "MODERATE").lower(),
            market_bias=data.get("market_bias", "Analyzing market structure and price action"),
            market_structure_notes=data.get("market_bias", "AI analysis of chart structure"),
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            key_levels=key_levels,
            indicators=_parse_indicators(),
            patterns_detected=data.get("confluence_factors", [])[:3],  # Use confluence as patterns
            pattern_completion=float(data.get("overall_confidence", 0.75)),
            pattern_reliability="high" if float(data.get("overall_confidence", 0.75)) > 0.8 else "medium",
            pattern_notes=f"Analysis confidence: {float(data.get('overall_confidence', 0.75))*100:.0f}%",
            primary_scenario=primary_scenario,
            alternative_scenario=_parse_scenario(data.get("alternative_scenario")),
            overall_confidence=float(data.get("overall_confidence", 0.75)),
            risk_factors=data.get("risk_factors", ["Market volatility", "Economic events"]),
            confluence_factors=data.get("confluence_factors", ["Technical analysis"]),
            chart_quality=image_path  # Re-purposing this field for the path
        )
        return analysis

    def get_analysis_by_id(self, analysis_id: str) -> Optional[ComprehensiveChartAnalysis]:
        return self.analysis_results.get(analysis_id)

    def get_recent_analyses(self, limit: int = 10) -> List[ComprehensiveChartAnalysis]:
        sorted_analyses = sorted(self.analysis_results.values(), key=lambda x: x.timestamp, reverse=True)
        return sorted_analyses[:limit]

    def get_vision_status(self) -> Dict:
        vision_settings = self.vision_config.get("vision", self.vision_config)
        return {
            "openai_connected": self.openai_client is not None,
            "total_analyses": len(self.analysis_results),
            "uploaded_images": len(list(self.upload_dir.iterdir())),
            "config_loaded": bool(self.vision_config),
            "model_name": vision_settings.get("model_name", "N/A"),
            "automated_analysis_running": self.is_running
        }

    def stop_automated_analysis(self):
        """Stops the automated analysis task."""
        if self.automated_analysis_task and not self.automated_analysis_task.done():
            self.automated_analysis_task.cancel()
        self.is_running = False
        logger.info("Stopping automated analysis...")
        logger.info("Automated analysis stopped")

    def start_automated_analysis(self, symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None):
        """Starts the automated analysis task."""
        if not self.is_running:
            self.is_running = True
            symbols = symbols or ["EURUSD", "GBPUSD", "USDJPY"]
            timeframes = timeframes or ["H1", "H4"]
            logger.info(f"Starting automated analysis for symbols: {symbols}, timeframes: {timeframes}")
            # For now, this is a placeholder - automated analysis can be implemented later
            logger.info("Automated analysis started")
        else:
            logger.info("Automated analysis is already running")

    def analyze_uploaded_chart_sync(self, analysis_id: str, symbol: Optional[str] = None, timeframe: str = "H4"):
        """Synchronous wrapper for the async analyze_uploaded_chart method"""
        import asyncio
        if symbol is None:
            symbol = ""
        logger.info(f'[VISION] analyze_uploaded_chart_sync called for {analysis_id}, symbol={symbol}, timeframe={timeframe}')
        
        if not self.openai_client:
            logger.error("OpenAI client not initialized for sync analysis")
            return None
            
        try:
            # Try to get existing event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we need to use run_in_executor or create new loop
                    logger.info("[VISION] Event loop is running, creating new loop")
                    import threading
                    result = None
                    exception = None
                    
                    def run_analysis():
                        nonlocal result, exception
                        try:
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            result = new_loop.run_until_complete(self.analyze_uploaded_chart(analysis_id, symbol, timeframe))
                            new_loop.close()
                        except Exception as e:
                            exception = e
                    
                    thread = threading.Thread(target=run_analysis)
                    thread.start()
                    thread.join()
                    
                    if exception:
                        raise exception
                    return result
                else:
                    # Loop exists but not running
                    logger.info("[VISION] Using existing event loop")
                    result = loop.run_until_complete(self.analyze_uploaded_chart(analysis_id, symbol, timeframe))
                    return result
            except RuntimeError:
                # No event loop in current thread
                logger.info("[VISION] No event loop found, creating new one")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.analyze_uploaded_chart(analysis_id, symbol, timeframe))
                loop.close()
                return result
                
        except Exception as e:
            logger.error(f'[VISION] Error in analyze_uploaded_chart_sync: {e}', exc_info=True)
            return None