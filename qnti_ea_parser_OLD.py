"""
QNTI EA Parser - MQL4/MQL5 Expert Advisor Code Parser
Handles parsing of EA source code and extraction of trading parameters and logic
"""

import re
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import uuid
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cool Trading Bot Name Generator
class TradingBotNameGenerator:
    """Generates cool names for trading strategies and bots"""
    
    PREFIXES = [
        "Alpha", "Beta", "Gamma", "Delta", "Sigma", "Omega", "Prime", "Apex", "Titan", "Vortex",
        "Quantum", "Nexus", "Phoenix", "Storm", "Blade", "Shadow", "Lightning", "Thunder", "Cyber",
        "Nano", "Mega", "Ultra", "Hyper", "Super", "Turbo", "Nitro", "Rocket", "Laser", "Plasma",
        "Crystal", "Diamond", "Gold", "Silver", "Steel", "Iron", "Fire", "Ice", "Wind", "Earth",
        "Dark", "Light", "Void", "Star", "Nova", "Comet", "Meteor", "Galaxy", "Cosmic", "Solar"
    ]
    
    CORES = [
        "Hunter", "Striker", "Warrior", "Guardian", "Sentinel", "Phantom", "Ghost", "Spirit",
        "Falcon", "Eagle", "Hawk", "Wolf", "Tiger", "Lion", "Dragon", "Serpent", "Viper", "Cobra",
        "Bullet", "Arrow", "Spear", "Sword", "Dagger", "Axe", "Hammer", "Shield", "Armor", "Crown",
        "Engine", "Core", "Matrix", "Circuit", "Pulse", "Wave", "Beam", "Ray", "Force", "Power",
        "Mind", "Brain", "Eye", "Vision", "Sight", "Focus", "Target", "Lock", "Strike", "Hit",
        "Flow", "Stream", "Current", "Tide", "Storm", "Blitz", "Rush", "Dash", "Speed", "Swift"
    ]
    
    SUFFIXES = [
        "X", "Pro", "Elite", "Prime", "Max", "Ultra", "Turbo", "Plus", "Advanced", "Supreme",
        "Master", "Expert", "Ace", "Champion", "Legend", "Hero", "King", "Emperor", "Lord", "Chief",
        "Alpha", "Beta", "Gamma", "One", "Zero", "Neo", "Ex", "Xtreme", "Force", "Power",
        "2000", "3000", "V2", "V3", "MK2", "MK3", "Gen2", "Gen3", "Next", "Future"
    ]
    
    THEMES = {
        "aggressive": {
            "prefixes": ["Alpha", "Apex", "Storm", "Blade", "Lightning", "Thunder", "Fire", "Dark", "Void"],
            "cores": ["Hunter", "Striker", "Warrior", "Falcon", "Eagle", "Wolf", "Tiger", "Dragon", "Bullet", "Strike"],
            "suffixes": ["X", "Elite", "Max", "Turbo", "Xtreme", "Force", "Alpha", "Master", "Champion"]
        },
        "defensive": {
            "prefixes": ["Shield", "Guardian", "Fortress", "Steel", "Iron", "Crystal", "Diamond"],
            "cores": ["Guardian", "Sentinel", "Shield", "Armor", "Wall", "Barrier", "Defender"],
            "suffixes": ["Pro", "Elite", "Supreme", "Master", "Expert", "Prime", "Advanced"]
        },
        "technical": {
            "prefixes": ["Quantum", "Cyber", "Nano", "Digital", "Binary", "Neural", "Logic"],
            "cores": ["Engine", "Core", "Matrix", "Circuit", "Brain", "Mind", "Algorithm"],
            "suffixes": ["2000", "3000", "V2", "V3", "MK2", "Gen2", "Next", "Future", "AI"]
        },
        "mystical": {
            "prefixes": ["Shadow", "Phantom", "Spirit", "Mystic", "Oracle", "Sage", "Wizard"],
            "cores": ["Ghost", "Spirit", "Vision", "Eye", "Sight", "Oracle", "Prophet"],
            "suffixes": ["Prime", "Ancient", "Eternal", "Mystic", "Sacred", "Divine"]
        }
    }
    
    @classmethod
    def generate_random_name(cls, theme: Optional[str] = None) -> str:
        """Generate a random trading bot name"""
        if theme and theme in cls.THEMES:
            theme_data = cls.THEMES[theme]
            prefix = random.choice(theme_data["prefixes"])
            core = random.choice(theme_data["cores"])
            suffix = random.choice(theme_data["suffixes"]) if random.random() > 0.3 else ""
        else:
            prefix = random.choice(cls.PREFIXES)
            core = random.choice(cls.CORES)
            suffix = random.choice(cls.SUFFIXES) if random.random() > 0.4 else ""
        
        if suffix:
            return f"{prefix}{core} {suffix}"
        else:
            return f"{prefix}{core}"
    
    @classmethod
    def generate_ea_name(cls, indicators: List[str], timeframes: List[str], magic_number: Optional[int] = None) -> str:
        """Generate an epic EA name based on indicators, timeframes, and characteristics"""
        # Determine theme based on indicators
        theme = cls._determine_theme_from_indicators(indicators)
        
        # Generate base name
        base_name = cls.generate_random_name(theme)
        
        # Add timeframe if available and not generic
        if timeframes and timeframes[0] not in ["CURRENT", "0"]:
            base_name = f"{base_name} {timeframes[0]}"
        
        # Add magic number suffix for uniqueness (last 3 digits)
        if magic_number:
            magic_suffix = str(magic_number)[-3:]
            if magic_suffix != "000":
                base_name = f"{base_name}-{magic_suffix}"
        
        return base_name
    
    @classmethod
    def generate_strategy_name(cls, strategy_id: int, indicators: List[str]) -> str:
        """Generate a strategy name based on indicators and strategy characteristics"""
        # Determine theme based on indicators
        theme = cls._determine_theme_from_indicators(indicators)
        
        # Generate base name
        base_name = cls.generate_random_name(theme)
        
        # Add strategy identifier
        strategy_suffix = ["Alpha", "Beta", "Gamma", "Delta", "Sigma", "Omega"][strategy_id % 6]
        
        return f"{base_name} {strategy_suffix}"
    
    @classmethod
    def _determine_theme_from_indicators(cls, indicators: List[str]) -> str:
        """Determine naming theme based on indicators used"""
        indicator_names = " ".join(indicators).lower()
        
        # Aggressive indicators
        if any(word in indicator_names for word in ["macd", "rsi", "stochastic", "momentum", "oscillator"]):
            return "aggressive"
        
        # Defensive indicators
        elif any(word in indicator_names for word in ["bollinger", "moving average", "sma", "ema", "bands"]):
            return "defensive"
        
        # Technical indicators
        elif any(word in indicator_names for word in ["cci", "atr", "adx", "williams", "ultimate"]):
            return "technical"
        
        # Default to mystical for unique indicators
        else:
            return "mystical"

@dataclass
class EAParameter:
    """Represents an EA input parameter"""
    name: str
    type: str
    default_value: Any
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None

@dataclass
class TradingRule:
    """Represents a trading rule extracted from EA code"""
    type: str  # 'entry', 'exit', 'condition'
    direction: str  # 'buy', 'sell', 'both'
    conditions: List[str]
    actions: List[str]
    indicators_used: List[str]
    line_number: int

@dataclass
class StrategyInfo:
    """Represents a single strategy within a Portfolio EA"""
    strategy_id: int
    name: str
    magic_number: int
    indicators: List[str]
    entry_conditions: List[str]
    exit_conditions: List[str]
    parameters: Dict[str, Any]
    stop_loss: Optional[int] = None
    take_profit: Optional[int] = None
    is_trailing_stop: bool = False
    opposite_reverse: bool = False

@dataclass
class EAProfile:
    """Complete EA profile with all parsed information"""
    id: str
    name: str
    description: str
    parameters: List[EAParameter]
    trading_rules: List[TradingRule]
    indicators: List[str]
    symbols: List[str]
    timeframes: List[str]
    magic_numbers: List[int]
    created_at: str
    source_code: str
    execution_status: str = "stopped"
    performance_stats: Optional[Dict[str, Any]] = None
    # Portfolio EA specific fields
    is_portfolio: bool = False
    strategies: Optional[List[StrategyInfo]] = None
    base_magic_number: Optional[int] = None
    strategies_count: int = 1

class MQLCodeParser:
    """Advanced MQL4/MQL5 code parser for Expert Advisors"""
    
    def __init__(self):
        self.indicators_map = {
            # Standard MT4/MT5 indicators
            'iMA': 'Moving Average',
            'iRSI': 'RSI',
            'iMACD': 'MACD',
            'iBands': 'Bollinger Bands',
            'iStochastic': 'Stochastic',
            'iCCI': 'CCI',
            'iADX': 'ADX',
            'iATR': 'ATR',
            'iCustom': 'Custom Indicator',
            'iSAR': 'Parabolic SAR',
            'iMomentum': 'Momentum',
            'iDeMarker': 'DeMarker',
            'iRVI': 'RVI',
            'iWPR': 'Williams %R',
            'iAC': 'Accelerator',
            'iAO': 'Awesome Oscillator',
            'iFractals': 'Fractals',
            'iGator': 'Gator',
            'iIchimoku': 'Ichimoku',
            'iBWMFI': 'Market Facilitation Index',
            'iEnvelopes': 'Envelopes',
            'iForce': 'Force Index',
            'iOsMA': 'OsMA',
            'iStdDev': 'Standard Deviation',
            'iVariance': 'Variance',
            'iVolumes': 'Volumes',
            'iBearsPower': 'Bears Power',
            'iBullsPower': 'Bulls Power',
            'iMFI': 'Money Flow Index',
            'iOBV': 'On Balance Volume',
            'iROC': 'Rate of Change',
            'iTEMA': 'Triple Exponential Moving Average',
            'iVIDyA': 'Variable Index Dynamic Average',
            'iWAD': 'Williams Accumulation/Distribution',
            'iAlligator': 'Alligator',
            # Additional common patterns
            'MA': 'Moving Average',
            'EMA': 'Exponential Moving Average',
            'SMA': 'Simple Moving Average',
            'RSI': 'RSI',
            'MACD': 'MACD',
            'Bollinger': 'Bollinger Bands',
            'Stoch': 'Stochastic',
            'CCI': 'CCI',
            'ADX': 'ADX',
            'ATR': 'ATR',
            'SAR': 'Parabolic SAR',
            'Momentum': 'Momentum',
            'WPR': 'Williams %R',
            'AC': 'Accelerator',
            'AO': 'Awesome Oscillator',
            'Fractals': 'Fractals',
            'Ichimoku': 'Ichimoku',
            'Envelopes': 'Envelopes',
            'Force': 'Force Index',
            'OsMA': 'OsMA',
            'StdDev': 'Standard Deviation',
            'MFI': 'Money Flow Index',
            'OBV': 'On Balance Volume',
            'ROC': 'Rate of Change',
            'TEMA': 'Triple Exponential Moving Average',
            'VIDyA': 'Variable Index Dynamic Average',
            'WAD': 'Williams Accumulation/Distribution',
            'Alligator': 'Alligator'
        }
        
        self.order_types = {
            'OP_BUY': 'Buy',
            'OP_SELL': 'Sell',
            'OP_BUYLIMIT': 'Buy Limit',
            'OP_SELLLIMIT': 'Sell Limit',
            'OP_BUYSTOP': 'Buy Stop',
            'OP_SELLSTOP': 'Sell Stop',
            'ORDER_TYPE_BUY': 'Buy',
            'ORDER_TYPE_SELL': 'Sell'
        }
        
    def parse_ea_code(self, code: str, ea_name: Optional[str] = None) -> EAProfile:
        """Parse complete EA code and return structured profile"""
        try:
            logger.info(f"🔍 Starting EA code parsing for {ea_name or 'Unnamed EA'}")
            
            # Clean and prepare code
            code = self._clean_code(code)
            
            # Generate unique ID
            ea_id = str(uuid.uuid4())[:8]
            
            # Parse components first to determine characteristics for name generation
            parameters = self._extract_parameters(code)
            trading_rules = self._extract_trading_rules(code)
            indicators = self._extract_indicators(code)
            symbols = self._extract_symbols(code)
            timeframes = self._extract_timeframes(code)
            magic_numbers = self._extract_magic_numbers(code)
            
            # Check if this is a Portfolio EA
            is_portfolio = self._is_portfolio_ea(code)
            
            # Generate cool autogenerated name based on EA characteristics
            if ea_name:
                # If a specific name was provided, use it
                name = ea_name
            else:
                # Generate epic name based on indicators and characteristics
                name = TradingBotNameGenerator.generate_ea_name(indicators, timeframes, magic_numbers[0] if magic_numbers else None)
                
                logger.info(f"🎯 Generated epic EA name: {name}")
            
            # Extract description
            description = self._extract_description(code)
            
            # Portfolio-specific parsing
            strategies = []
            base_magic_number = None
            strategies_count = 1
            
            if is_portfolio:
                logger.info(f"📊 Detected Portfolio EA with multiple strategies")
                strategies = self._extract_portfolio_strategies(code)
                base_magic_number = self._extract_base_magic_number(code)
                strategies_count = self._extract_strategies_count(code)
                
                # Update magic numbers to include all strategy magic numbers
                if base_magic_number:
                    for i in range(strategies_count):
                        strategy_magic = base_magic_number * 1000 + i
                        if strategy_magic not in magic_numbers:
                            magic_numbers.append(strategy_magic)
            
            # Create profile
            profile = EAProfile(
                id=ea_id,
                name=name,
                description=description,
                parameters=parameters,
                trading_rules=trading_rules,
                indicators=indicators,
                symbols=symbols,
                timeframes=timeframes,
                magic_numbers=magic_numbers,
                created_at=datetime.now().isoformat(),
                source_code=code,
                performance_stats={},
                is_portfolio=is_portfolio,
                strategies=strategies if is_portfolio else [],
                base_magic_number=base_magic_number,
                strategies_count=strategies_count
            )
            
            if is_portfolio:
                logger.info(f"✅ Portfolio EA parsing completed: {strategies_count} strategies, {len(parameters)} parameters, {len(trading_rules)} rules")
            else:
                logger.info(f"✅ EA parsing completed: {len(parameters)} parameters, {len(trading_rules)} rules")
            
            return profile
            
        except Exception as e:
            logger.error(f"❌ EA parsing failed: {e}")
            raise Exception(f"Failed to parse EA code: {str(e)}")
    
    def _is_portfolio_ea(self, code: str) -> bool:
        """Detect if this is a Portfolio EA"""
        portfolio_indicators = [
            'EA Studio Portfolio',
            'Portfolio Expert',
            'strategiesCount',
            'GetMagicNumber',
            'Base_Magic_Number',
            'SetSignals',
            'GetEntrySignal_',
            'GetExitSignal_',
            'indHandlers\[',
            'STRATEGY CODE'
        ]
        
        return any(indicator in code for indicator in portfolio_indicators)
    
    def _extract_portfolio_strategies(self, code: str) -> List[StrategyInfo]:
        """Extract individual strategies from Portfolio EA"""
        strategies = []
        
        try:
            # Extract strategies count
            strategies_count = self._extract_strategies_count(code)
            base_magic = self._extract_base_magic_number(code)
            
            # Look for strategy functions
            strategy_functions = []
            
            # Find GetEntrySignal_ and GetExitSignal_ functions
            entry_pattern = r'Signal\s+GetEntrySignal_(\d+)\(\)'
            exit_pattern = r'Signal\s+GetExitSignal_(\d+)\(\)'
            
            entry_matches = re.findall(entry_pattern, code)
            exit_matches = re.findall(exit_pattern, code)
            
            # Get unique strategy indices
            strategy_indices = set()
            for match in entry_matches + exit_matches:
                strategy_indices.add(int(match))
            
            # Extract each strategy
            for strategy_id in sorted(strategy_indices):
                strategy = self._extract_single_strategy(code, strategy_id, base_magic)
                if strategy:
                    strategies.append(strategy)
            
            # If no specific strategies found, create default based on count
            if not strategies and strategies_count > 1:
                for i in range(strategies_count):
                    magic_number = base_magic * 1000 + i if base_magic else 100 + i
                    # Generate cool name for default strategy
                    strategy_name = TradingBotNameGenerator.generate_strategy_name(i, [])
                    strategy = StrategyInfo(
                        strategy_id=i,
                        name=strategy_name,
                        magic_number=magic_number,
                        indicators=[],
                        entry_conditions=[],
                        exit_conditions=[],
                        parameters={}
                    )
                    strategies.append(strategy)
            
            logger.info(f"📈 Extracted {len(strategies)} strategies from Portfolio EA")
            return strategies
            
        except Exception as e:
            logger.error(f"❌ Failed to extract portfolio strategies: {e}")
            return []
    
    def _extract_single_strategy(self, code: str, strategy_id: int, base_magic: Optional[int]) -> Optional[StrategyInfo]:
        """Extract a single strategy from Portfolio EA"""
        try:
            # Calculate magic number
            magic_number = base_magic * 1000 + strategy_id if base_magic else 100 + strategy_id
            
            # Extract strategy indicators from indHandlers
            indicators = self._extract_strategy_indicators(code, strategy_id)
            
            # Extract entry conditions
            entry_conditions = self._extract_strategy_entry_conditions(code, strategy_id)
            
            # Extract exit conditions  
            exit_conditions = self._extract_strategy_exit_conditions(code, strategy_id)
            
            # Extract strategy parameters from STRATEGY CODE comments
            parameters = self._extract_strategy_parameters(code, strategy_id)
            
            # Generate cool strategy name
            strategy_name = TradingBotNameGenerator.generate_strategy_name(strategy_id, indicators)
            
            strategy = StrategyInfo(
                strategy_id=strategy_id,
                name=strategy_name,
                magic_number=magic_number,
                indicators=indicators,
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                parameters=parameters,
                stop_loss=parameters.get('stopLoss'),
                take_profit=parameters.get('takeProfit'),
                is_trailing_stop=parameters.get('isTrailingStop', False),
                opposite_reverse=parameters.get('oppositeEntrySignal', False)
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"❌ Failed to extract strategy {strategy_id}: {e}")
            return None
    
    def _extract_strategy_indicators(self, code: str, strategy_id: int) -> List[str]:
        """Extract indicators for a specific strategy"""
        indicators = []
        
        # Look for indHandlers[strategy_id] patterns
        handler_pattern = f'indHandlers\[{strategy_id}\]\[\d+\]\[0\]\s*=\s*i(\w+)\('
        matches = re.findall(handler_pattern, code)
        
        for match in matches:
            indicator_name = self.indicators_map.get(f'i{match}', match)
            if indicator_name not in indicators:
                indicators.append(indicator_name)
        
        return indicators
    
    def _extract_strategy_entry_conditions(self, code: str, strategy_id: int) -> List[str]:
        """Extract entry conditions for a specific strategy"""
        conditions = []
        
        # Look for GetEntrySignal_XXX function
        function_pattern = f'Signal\s+GetEntrySignal_{strategy_id:03d}\(\).*?return\s+CreateEntrySignal'
        function_match = re.search(function_pattern, code, re.DOTALL)
        
        if function_match:
            function_body = function_match.group(0)
            
            # Extract conditions from the function body
            condition_patterns = [
                r'bool\s+\w+long\s*=\s*([^;]+);',
                r'bool\s+\w+short\s*=\s*([^;]+);',
            ]
            
            for pattern in condition_patterns:
                matches = re.findall(pattern, function_body)
                conditions.extend(matches)
        
        return conditions
    
    def _extract_strategy_exit_conditions(self, code: str, strategy_id: int) -> List[str]:
        """Extract exit conditions for a specific strategy"""
        conditions = []
        
        # Look for GetExitSignal_XXX function
        function_pattern = f'Signal\s+GetExitSignal_{strategy_id:03d}\(\).*?return\s+CreateExitSignal'
        function_match = re.search(function_pattern, code, re.DOTALL)
        
        if function_match:
            function_body = function_match.group(0)
            
            # Extract conditions from the function body
            condition_patterns = [
                r'bool\s+\w+long\s*=\s*([^;]+);',
                r'bool\s+\w+short\s*=\s*([^;]+);',
            ]
            
            for pattern in condition_patterns:
                matches = re.findall(pattern, function_body)
                conditions.extend(matches)
        
        return conditions
    
    def _extract_strategy_parameters(self, code: str, strategy_id: int) -> Dict[str, Any]:
        """Extract parameters for a specific strategy from STRATEGY CODE comments"""
        parameters = {}
        
        # Look for STRATEGY CODE comments
        strategy_pattern = r'/\*STRATEGY CODE\s+(\{.*?\})\*/'
        matches = re.findall(strategy_pattern, code, re.DOTALL)
        
        if strategy_id < len(matches):
            try:
                strategy_json = matches[strategy_id]
                strategy_data = json.loads(strategy_json)
                
                if 'properties' in strategy_data:
                    properties = strategy_data['properties']
                    parameters.update({
                        'entryLots': properties.get('entryLots', 0.01),
                        'stopLoss': properties.get('stopLoss', 0),
                        'takeProfit': properties.get('takeProfit', 0),
                        'useStopLoss': properties.get('useStopLoss', False),
                        'useTakeProfit': properties.get('useTakeProfit', False),
                        'isTrailingStop': properties.get('isTrailingStop', False),
                        'oppositeEntrySignal': properties.get('oppositeEntrySignal', 0),
                        'tradeDirectionMode': properties.get('tradeDirectionMode', 0)
                    })
                
                # Extract indicators from openFilters and closeFilters
                indicators = []
                for filter_type in ['openFilters', 'closeFilters']:
                    if filter_type in strategy_data:
                        for filter_item in strategy_data[filter_type]:
                            if 'name' in filter_item:
                                indicators.append(filter_item['name'])
                
                parameters['indicators'] = indicators
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON in EA code: {e}")
                # Attempt to extract basic indicators manually as fallback
                try:
                    # Look for common indicator patterns
                    import re
                    patterns = [
                        r'i(MA|RSI|MACD|Stochastic|CCI|ADX|Bollinger|ATR|Williams|Momentum|DeMarker|WPR|AC|AO|Alligator|Fractals|Gator|Ichimoku)\s*\(',
                        r'(MovingAverage|RSI|MACD|StochasticOscillator)\s*\(',
                        r'(SMA|EMA|WMA|LWMA)\s*\('
                    ]
                    
                    found_indicators = set()
                    for pattern in patterns:
                        matches = re.finditer(pattern, code, re.IGNORECASE)
                        for match in matches:
                            indicator_name = match.group(1) if match.group(1) else match.group(0)
                            found_indicators.add(indicator_name.strip('('))
                    
                    if found_indicators:
                        parameters['indicators'] = list(found_indicators)
                        logger.info(f"Extracted {len(found_indicators)} indicators using pattern matching")
                        
                except Exception as fallback_error:
                    logger.error(f"Fallback indicator extraction failed: {fallback_error}")
                    parameters['indicators'] = []
        
        return parameters
    
    def _extract_base_magic_number(self, code: str) -> Optional[int]:
        """Extract base magic number for Portfolio EA"""
        # Look for Base_Magic_Number
        base_magic_patterns = [
            r'Base_Magic_Number\s*=\s*(\d+)',
            r'input\s+int\s+Base_Magic_Number\s*=\s*(\d+)',
            r'static\s+input\s+int\s+Base_Magic_Number\s*=\s*(\d+)',
        ]
        
        for pattern in base_magic_patterns:
            match = re.search(pattern, code, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def _extract_strategies_count(self, code: str) -> int:
        """Extract number of strategies in Portfolio EA"""
        # Look for strategiesCount constant
        count_pattern = r'strategiesCount\s*=\s*(\d+)'
        match = re.search(count_pattern, code)
        
        if match:
            return int(match.group(1))
        
        # Count GetEntrySignal functions as fallback
        entry_functions = re.findall(r'GetEntrySignal_(\d+)', code)
        if entry_functions:
            return len(set(entry_functions))
        
        return 1
    
    def _clean_code(self, code: str) -> str:
        """Clean and normalize MQL code"""
        # Remove comments (but preserve STRATEGY CODE comments for portfolio parsing)
        # First preserve STRATEGY CODE comments
        strategy_comments = re.findall(r'/\*STRATEGY CODE.*?\*/', code, re.DOTALL)
        
        # Remove other comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*(?!STRATEGY CODE).*?\*/', '', code, flags=re.DOTALL)
        
        # Restore STRATEGY CODE comments
        for i, comment in enumerate(strategy_comments):
            code += f'\n{comment}'
        
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        code = re.sub(r'\s*([{}();,])\s*', r'\1', code)
        
        return code.strip()
    
    def _extract_ea_name(self, code: str) -> str:
        """Extract EA name from code"""
        # Look for property name
        name_match = re.search(r'#property\s+name\s+"([^"]+)"', code, re.IGNORECASE)
        if name_match:
            return name_match.group(1)
        
        # Look for file name comment
        file_match = re.search(r'//\s*([^/\n]+\.mq[45])', code, re.IGNORECASE)
        if file_match:
            file_name = file_match.group(1).replace('.mq4', '').replace('.mq5', '')
            # Clean up comment formatting characters
            file_name = re.sub(r'^[\|\s]+', '', file_name)  # Remove leading | and spaces
            file_name = re.sub(r'[\|\s]+$', '', file_name)  # Remove trailing | and spaces
            file_name = re.sub(r'\s+', ' ', file_name)      # Normalize whitespace
            return file_name.strip()
        
        return "Parsed EA"
    
    def _extract_description(self, code: str) -> str:
        """Extract EA description"""
        desc_match = re.search(r'#property\s+description\s+"([^"]+)"', code, re.IGNORECASE)
        if desc_match:
            return desc_match.group(1)
        
        # Look for initial comment block
        comment_match = re.search(r'/\*\*(.*?)\*/', code, re.DOTALL)
        if comment_match:
            return comment_match.group(1).strip()[:200]
        
        return "Expert Advisor imported from MQL code"
    
    def _extract_parameters(self, code: str) -> List[EAParameter]:
        """Extract input/extern parameters"""
        parameters = []
        
        # Pattern for extern/input parameters
        param_patterns = [
            r'extern\s+(\w+)\s+(\w+)\s*=\s*([^;]+);',
            r'input\s+(\w+)\s+(\w+)\s*=\s*([^;]+);',
            r'sinput\s+(\w+)\s+(\w+)\s*=\s*([^;]+);'
        ]
        
        for pattern in param_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for match in matches:
                param_type, param_name, default_value = match
                
                # Clean default value
                default_value = default_value.strip().rstrip(',')
                
                # Convert to appropriate type
                converted_value = self._convert_parameter_value(param_type, default_value)
                
                parameter = EAParameter(
                    name=param_name,
                    type=param_type,
                    default_value=converted_value,
                    description=f"{param_type} parameter"
                )
                
                parameters.append(parameter)
        
        logger.info(f"📊 Extracted {len(parameters)} parameters")
        return parameters
    
    def _convert_parameter_value(self, param_type: str, value: str) -> Any:
        """Convert parameter value to appropriate Python type"""
        value = value.strip().strip('"\'')
        
        if param_type.lower() in ['int', 'long', 'short', 'char', 'uchar']:
            try:
                return int(float(value))
            except:
                return 0
        elif param_type.lower() in ['double', 'float']:
            try:
                return float(value)
            except:
                return 0.0
        elif param_type.lower() == 'bool':
            return value.lower() in ['true', '1', 'yes']
        elif param_type.lower() == 'string':
            return str(value)
        else:
            return value
    
    def _extract_trading_rules(self, code: str) -> List[TradingRule]:
        """Extract trading rules and logic"""
        rules = []
        
        # Find OrderSend calls
        order_patterns = [
            r'OrderSend\s*\([^)]+\)',
            r'PositionOpen\s*\([^)]+\)',
            r'trade\.Buy\s*\([^)]+\)',
            r'trade\.Sell\s*\([^)]+\)',
            r'OrderSendAsync\s*\([^)]+\)'
        ]
        
        for order_pattern in order_patterns:
            order_matches = re.finditer(order_pattern, code, re.IGNORECASE)
            
            for match in order_matches:
                line_num = code[:match.start()].count('\n') + 1
                order_call = match.group(0)
                
                # Extract order type
                direction = 'both'
                if any(keyword in order_call.upper() for keyword in ['OP_BUY', 'ORDER_TYPE_BUY', '.BUY']):
                    direction = 'buy'
                elif any(keyword in order_call.upper() for keyword in ['OP_SELL', 'ORDER_TYPE_SELL', '.SELL']):
                    direction = 'sell'
                
                # Find surrounding conditions with better context
                conditions = self._extract_conditions_around_line(code, line_num, context_lines=15)
                indicators_used = self._extract_indicators_from_conditions(conditions)
                
                # Extract entry conditions from function context
                function_context = self._extract_function_context(code, line_num)
                if function_context:
                    conditions.extend(function_context)
                
                rule = TradingRule(
                    type='entry',
                    direction=direction,
                    conditions=conditions,
                    actions=[order_call],
                    indicators_used=indicators_used,
                    line_number=line_num
                )
                
                rules.append(rule)
        
        # Find OrderClose calls
        close_patterns = [
            r'OrderClose\s*\([^)]+\)',
            r'PositionClose\s*\([^)]+\)',
            r'trade\.PositionClose\s*\([^)]+\)',
            r'OrderCloseBy\s*\([^)]+\)'
        ]
        
        for close_pattern in close_patterns:
            close_matches = re.finditer(close_pattern, code, re.IGNORECASE)
            
            for match in close_matches:
                line_num = code[:match.start()].count('\n') + 1
                close_call = match.group(0)
                
                conditions = self._extract_conditions_around_line(code, line_num, context_lines=10)
                indicators_used = self._extract_indicators_from_conditions(conditions)
                
                rule = TradingRule(
                    type='exit',
                    direction='both',
                    conditions=conditions,
                    actions=[close_call],
                    indicators_used=indicators_used,
                    line_number=line_num
                )
                
                rules.append(rule)
        
        # Look for stop loss and take profit modifications
        modify_patterns = [
            r'OrderModify\s*\([^)]+\)',
            r'PositionModify\s*\([^)]+\)',
            r'trade\.PositionModify\s*\([^)]+\)'
        ]
        
        for modify_pattern in modify_patterns:
            modify_matches = re.finditer(modify_pattern, code, re.IGNORECASE)
            
            for match in modify_matches:
                line_num = code[:match.start()].count('\n') + 1
                modify_call = match.group(0)
                
                conditions = self._extract_conditions_around_line(code, line_num, context_lines=8)
                indicators_used = self._extract_indicators_from_conditions(conditions)
                
                rule = TradingRule(
                    type='modification',
                    direction='both',
                    conditions=conditions,
                    actions=[modify_call],
                    indicators_used=indicators_used,
                    line_number=line_num
                )
                
                rules.append(rule)
        
        # Extract signal generation logic
        signal_rules = self._extract_signal_logic(code)
        rules.extend(signal_rules)
        
        logger.info(f"📋 Extracted {len(rules)} trading rules")
        return rules
    
    def _extract_conditions_around_line(self, code: str, line_num: int, context_lines: int = 10) -> List[str]:
        """Extract conditions around a specific line"""
        lines = code.split('\n')
        conditions = []
        
        # Look for if statements in nearby lines
        start_line = max(0, line_num - context_lines)
        end_line = min(len(lines), line_num + 5)
        
        for i in range(start_line, end_line):
            if i < len(lines):
                line = lines[i].strip()
                if line.startswith('if') and '(' in line:
                    # Extract condition from if statement
                    condition_match = re.search(r'if\s*\(([^)]+)\)', line)
                    if condition_match:
                        conditions.append(condition_match.group(1))
                
                # Look for while loops
                if line.startswith('while') and '(' in line:
                    condition_match = re.search(r'while\s*\(([^)]+)\)', line)
                    if condition_match:
                        conditions.append(condition_match.group(1))
                
                # Look for logical operators and comparisons
                if any(op in line for op in ['>', '<', '>=', '<=', '==', '!=', '&&', '||']):
                    # Clean up the condition
                    cleaned_condition = line.replace('{', '').replace('}', '').strip()
                    if cleaned_condition and not cleaned_condition.startswith('//'):
                        conditions.append(cleaned_condition)
        
        return conditions
    
    def _extract_indicators_from_conditions(self, conditions: List[str]) -> List[str]:
        """Extract indicators mentioned in conditions"""
        indicators = []
        
        for condition in conditions:
            for indicator_func, indicator_name in self.indicators_map.items():
                if indicator_func in condition:
                    if indicator_name not in indicators:
                        indicators.append(indicator_name)
        
        return indicators
    
    def _extract_indicators(self, code: str) -> List[str]:
        """Extract all indicators used in the EA"""
        indicators = []
        
        # EA Studio specific indicator extraction from comments
        ea_studio_indicators = re.findall(r'//\s*---\s*([^-\n]+?)\s*---', code, re.IGNORECASE)
        for indicator in ea_studio_indicators:
            indicator = indicator.strip()
            if indicator and indicator != "Expert Properties" and len(indicator) > 2:
                # Clean up EA Studio indicator names
                if any(word in indicator for word in ["Oscillator", "Power", "Color", "Average", "RSI", "MACD", "Bands", "Stochastic"]):
                    indicators.append(indicator)
        
        # EA Studio specific indicator handlers
        handler_patterns = [
            (r'indHandlers\[\d+\]\[\d+\]\[\d+\]\s*=\s*iAC\s*\(', 'Accelerator Oscillator'),
            (r'indHandlers\[\d+\]\[\d+\]\[\d+\]\s*=\s*iBullsPower\s*\(', 'Bulls Power'),
            (r'indHandlers\[\d+\]\[\d+\]\[\d+\]\s*=\s*iBearsPower\s*\(', 'Bears Power'),
            (r'indHandlers\[\d+\]\[\d+\]\[\d+\]\s*=\s*iMA\s*\(', 'Moving Average'),
            (r'indHandlers\[\d+\]\[\d+\]\[\d+\]\s*=\s*iRSI\s*\(', 'RSI'),
            (r'indHandlers\[\d+\]\[\d+\]\[\d+\]\s*=\s*iMACD\s*\(', 'MACD'),
            (r'indHandlers\[\d+\]\[\d+\]\[\d+\]\s*=\s*iBands\s*\(', 'Bollinger Bands'),
            (r'indHandlers\[\d+\]\[\d+\]\[\d+\]\s*=\s*iStochastic\s*\(', 'Stochastic'),
        ]
        
        for pattern, name in handler_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                if name not in indicators:
                    indicators.append(name)
        
        # Look for Candle Color pattern (EA Studio specific)
        if re.search(r'//\s*Candle\s*Color', code, re.IGNORECASE):
            if "Candle Color" not in indicators:
                indicators.append("Candle Color")
        
        # Look for strategy code JSON (EA Studio specific)
        strategy_match = re.search(r'STRATEGY\s+CODE\s+(\{.*?\})', code, re.DOTALL)
        if strategy_match:
            try:
                import json
                strategy_data = json.loads(strategy_match.group(1))
                # Extract indicators from openFilters and closeFilters
                for filter_group in ['openFilters', 'closeFilters']:
                    if filter_group in strategy_data:
                        for filter_item in strategy_data[filter_group]:
                            if 'name' in filter_item:
                                indicator_name = filter_item['name']
                                if indicator_name not in indicators:
                                    indicators.append(indicator_name)
            except:
                pass  # Ignore JSON parsing errors
        
        # Check for standard indicator function calls
        for indicator_func, indicator_name in self.indicators_map.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(indicator_func) + r'\b'
            if re.search(pattern, code, re.IGNORECASE):
                if indicator_name not in indicators:
                    indicators.append(indicator_name)
        
        # Additional pattern matching for common indicator patterns
        additional_patterns = {
            r'\b(?:simple|sma).*moving.*average\b': 'Simple Moving Average',
            r'\b(?:exponential|ema).*moving.*average\b': 'Exponential Moving Average',
            r'\b(?:weighted|wma).*moving.*average\b': 'Weighted Moving Average',
            r'\brelative.*strength.*index\b': 'RSI',
            r'\bstochastic.*oscillator\b': 'Stochastic',
            r'\bbollinger.*bands?\b': 'Bollinger Bands',
            r'\bmoving.*average.*convergence.*divergence\b': 'MACD',
            r'\bcommodity.*channel.*index\b': 'CCI',
            r'\baverage.*directional.*index\b': 'ADX',
            r'\baverage.*true.*range\b': 'ATR',
            r'\bparabolic.*sar\b': 'Parabolic SAR',
            r'\bwilliams.*%?r\b': 'Williams %R',
            r'\baccelerator.*oscillator\b': 'Accelerator',
            r'\bawesome.*oscillator\b': 'Awesome Oscillator',
            r'\bmoney.*flow.*index\b': 'Money Flow Index',
            r'\bon.*balance.*volume\b': 'On Balance Volume',
            r'\bichimoku.*kinko.*hyo\b': 'Ichimoku',
            r'\bforce.*index\b': 'Force Index',
            r'\bdetrended.*price.*oscillator\b': 'Detrended Price Oscillator',
            r'\brate.*of.*change\b': 'Rate of Change',
            r'\bstandard.*deviation\b': 'Standard Deviation',
            r'\blinear.*regression\b': 'Linear Regression',
            r'\bzig.*zag\b': 'ZigZag',
            r'\bfibonacci.*retracement\b': 'Fibonacci Retracement',
            r'\bpivot.*points?\b': 'Pivot Points',
            r'\bsupport.*resistance\b': 'Support/Resistance',
            r'\btrend.*lines?\b': 'Trend Lines',
            r'\bvolume.*weighted.*average.*price\b': 'VWAP',
            r'\baccumulation.*distribution\b': 'Accumulation/Distribution',
            r'\bchaikin.*oscillator\b': 'Chaikin Oscillator',
            r'\belders?.*ray\b': 'Elder Ray',
            r'\bkeltner.*channels?\b': 'Keltner Channels',
            r'\bprice.*channels?\b': 'Price Channels',
            r'\bdonchian.*channels?\b': 'Donchian Channels',
            r'\baroon.*oscillator\b': 'Aroon Oscillator',
            r'\btrix\b': 'TRIX',
            r'\bultimate.*oscillator\b': 'Ultimate Oscillator',
            r'\bchande.*momentum.*oscillator\b': 'Chande Momentum Oscillator',
            r'\bmass.*index\b': 'Mass Index',
            r'\bvortex.*indicator\b': 'Vortex Indicator',
            r'\bknow.*sure.*thing\b': 'Know Sure Thing',
            r'\bpretty.*good.*oscillator\b': 'Pretty Good Oscillator',
            r'\bschiff.*pitchfork\b': 'Schiff Pitchfork',
            r'\bandrews?.*pitchfork\b': 'Andrews Pitchfork'
        }
        
        code_lower = code.lower()
        for pattern, indicator_name in additional_patterns.items():
            if re.search(pattern, code_lower):
                if indicator_name not in indicators:
                    indicators.append(indicator_name)
        
        # Look for custom indicator files (.ex4, .ex5)
        custom_indicator_pattern = r'["\']([^"\']+\.ex[45])["\']'
        custom_matches = re.findall(custom_indicator_pattern, code, re.IGNORECASE)
        for match in custom_matches:
            indicator_name = f"Custom: {match.replace('.ex4', '').replace('.ex5', '')}"
            if indicator_name not in indicators:
                indicators.append(indicator_name)
        
        # Look for iCustom calls with indicator names
        icustom_pattern = r'iCustom\s*\([^,]+,\s*[^,]+,\s*["\']([^"\']+)["\']'
        icustom_matches = re.findall(icustom_pattern, code)
        for match in icustom_matches:
            indicator_name = f"Custom: {match}"
            if indicator_name not in indicators:
                indicators.append(indicator_name)
        
        # If no indicators found, check for basic price action patterns
        if not indicators:
            price_patterns = [
                (r'\b(?:high|low|open|close)\s*\[\s*\d+\s*\]', 'Price Action'),
                (r'\b(?:ask|bid)\b', 'Price Action'),
                (r'\bcandle.*pattern\b', 'Candlestick Patterns'),
                (r'\bpin.*bar\b', 'Pin Bar Pattern'),
                (r'\bdoji\b', 'Doji Pattern'),
                (r'\bhammer\b', 'Hammer Pattern'),
                (r'\bengulfing\b', 'Engulfing Pattern'),
                (r'\bmorning.*star\b', 'Morning Star Pattern'),
                (r'\bevening.*star\b', 'Evening Star Pattern'),
                (r'\bshooting.*star\b', 'Shooting Star Pattern'),
                (r'\bhanging.*man\b', 'Hanging Man Pattern')
            ]
            
            for pattern, indicator_name in price_patterns:
                if re.search(pattern, code_lower):
                    if indicator_name not in indicators:
                        indicators.append(indicator_name)
        
        logger.info(f"🔍 Extracted {len(indicators)} indicators: {indicators}")
        return indicators
    
    def _extract_function_context(self, code: str, line_num: int) -> List[str]:
        """Extract function context around a line"""
        lines = code.split('\n')
        conditions = []
        
        # Find the function that contains this line
        function_start = line_num
        for i in range(line_num - 1, max(0, line_num - 50), -1):
            if i < len(lines):
                line = lines[i].strip()
                if any(keyword in line.lower() for keyword in ['void ', 'int ', 'double ', 'bool ', 'string ']):
                    if '(' in line and ')' in line:
                        function_start = i
                        break
        
        # Extract conditions from the function
        for i in range(function_start, min(len(lines), line_num + 10)):
            if i < len(lines):
                line = lines[i].strip()
                # Look for variable assignments that might be conditions
                if '=' in line and not line.startswith('//') and '{' not in line:
                    conditions.append(line)
        
        return conditions
    
    def _extract_signal_logic(self, code: str) -> List[TradingRule]:
        """Extract signal generation logic"""
        rules = []
        
        # Look for common signal patterns
        signal_patterns = [
            (r'buy.*signal', 'entry', 'buy'),
            (r'sell.*signal', 'entry', 'sell'),
            (r'long.*signal', 'entry', 'buy'),
            (r'short.*signal', 'entry', 'sell'),
            (r'close.*signal', 'exit', 'both'),
            (r'exit.*signal', 'exit', 'both')
        ]
        
        for pattern, rule_type, direction in signal_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                signal_text = match.group(0)
                
                # Get surrounding context
                conditions = self._extract_conditions_around_line(code, line_num, context_lines=5)
                indicators_used = self._extract_indicators_from_conditions(conditions)
                
                rule = TradingRule(
                    type=rule_type,
                    direction=direction,
                    conditions=conditions + [f"Signal: {signal_text}"],
                    actions=[f"Generated signal: {signal_text}"],
                    indicators_used=indicators_used,
                    line_number=line_num
                )
                
                rules.append(rule)
        
        return rules
    
    def _extract_symbols(self, code: str) -> List[str]:
        """Extract trading symbols"""
        symbols = []
        
        # Look for explicit symbol definitions
        symbol_patterns = [
            r'"([A-Z]{6})"',  # Standard forex pairs
            r'"([A-Z]{3,6})"',  # Other symbols
            r'Symbol\(\)'  # Current symbol
        ]
        
        for pattern in symbol_patterns:
            matches = re.findall(pattern, code)
            symbols.extend(matches)
        
        # If no symbols found, assume current symbol
        if not symbols:
            symbols = ['CURRENT']
        
        return list(set(symbols))
    
    def _extract_timeframes(self, code: str) -> List[str]:
        """Extract timeframes used"""
        timeframes = []
        
        # Standard MQL timeframe constants
        tf_map = {
            'PERIOD_M1': 'M1', 'PERIOD_M5': 'M5', 'PERIOD_M15': 'M15',
            'PERIOD_M30': 'M30', 'PERIOD_H1': 'H1', 'PERIOD_H4': 'H4',
            'PERIOD_D1': 'D1', 'PERIOD_W1': 'W1', 'PERIOD_MN1': 'MN1',
            'Period()': 'CURRENT'
        }
        
        for tf_const, tf_name in tf_map.items():
            if tf_const in code:
                timeframes.append(tf_name)
        
        # Extract from EA name first (EA Studio often puts timeframe in name)
        ea_name = self._extract_ea_name(code)
        if ea_name:
            # Look for timeframe patterns in EA name
            name_tf_patterns = [
                r'\b(M\d+|H\d+|D\d+|W\d+|MN\d+)\b',  # Standard TF format
                r'\b(\d+)M\b',  # 15M format
                r'\b(\d+)H\b',  # 1H format  
                r'\b(\d+)D\b',  # 1D format
            ]
            
            for pattern in name_tf_patterns:
                matches = re.findall(pattern, ea_name, re.IGNORECASE)
                for match in matches:
                    tf_clean = match.upper().strip()
                    # Convert numeric formats to standard
                    if re.match(r'^\d+$', tf_clean):
                        # Numeric only, determine if minutes or hours
                        num = int(tf_clean)
                        if num <= 60:  # Assume minutes for values <= 60
                            tf_clean = f"M{num}"
                        else:  # Assume hours for larger values
                            tf_clean = f"H{num}"
                    
                    if tf_clean and tf_clean not in timeframes:
                        timeframes.append(tf_clean)
        
        # EA Studio Portfolio specific timeframe extraction
        # Look for timeframes in EA Studio comments
        ea_studio_tf_patterns = [
            r'//\s*TimeFrame[:\s]*([MHD]\d+|CURRENT)',
            r'//\s*Period[:\s]*([MHD]\d+|CURRENT)',
            r'//\s*Chart[:\s]*([MHD]\d+|CURRENT)',
            r'Timeframe[:\s]*([MHD]\d+|CURRENT)',
            r'Period[:\s]*([MHD]\d+|CURRENT)',
        ]
        
        for pattern in ea_studio_tf_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for match in matches:
                tf_clean = match.upper().strip()
                if tf_clean and tf_clean not in timeframes:
                    timeframes.append(tf_clean)
        
        # Look for numeric timeframe values (EA Studio format)
        numeric_tf_patterns = [
            r'timeframe\s*=\s*(\d+)',
            r'Period\s*=\s*(\d+)',
            r'ChartPeriod\s*=\s*(\d+)',
        ]
        
        tf_numeric_map = {
            '1': 'M1', '5': 'M5', '15': 'M15', '30': 'M30',
            '60': 'H1', '240': 'H4', '1440': 'D1', '10080': 'W1', '43200': 'MN1'
        }
        
        for pattern in numeric_tf_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for match in matches:
                if match in tf_numeric_map:
                    tf_name = tf_numeric_map[match]
                    if tf_name not in timeframes:
                        timeframes.append(tf_name)
        
        # Portfolio EA specific: Look for timeframes in strategy code sections
        strategy_tf_pattern = r'STRATEGY\s+CODE.*?"timeframe"\s*:\s*"?([^",}]+)"?'
        strategy_matches = re.findall(strategy_tf_pattern, code, re.DOTALL | re.IGNORECASE)
        for match in strategy_matches:
            tf_clean = match.strip().upper()
            if tf_clean and tf_clean not in timeframes:
                timeframes.append(tf_clean)
        
        # Look for timeframes in EA Studio generated comments
        comment_tf_pattern = r'//.*?([MH]\d+|D1|W1|MN1)(?:\s|$|,|;)'
        comment_matches = re.findall(comment_tf_pattern, code, re.IGNORECASE)
        for match in comment_matches:
            tf_clean = match.upper().strip()
            if tf_clean and len(tf_clean) <= 4 and tf_clean not in timeframes:
                timeframes.append(tf_clean)
        
        # Portfolio EA: Extract from strategy metadata
        if self._is_portfolio_ea(code):
            # Look for timeframes in Portfolio EA metadata comments
            portfolio_tf_patterns = [
                r'//\s*Chart\s*Period[:\s]*([^,\n]+)',
                r'//\s*Timeframe[:\s]*([^,\n]+)',
                r'//\s*Symbol\s*Period[:\s]*([^,\n]+)',
                r'/\*.*?Period[:\s]*([MHD]\d+|CURRENT).*?\*/',
            ]
            
            for pattern in portfolio_tf_patterns:
                matches = re.findall(pattern, code, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    tf_clean = match.strip().upper()
                    # Clean up common formats
                    tf_clean = re.sub(r'[^\w\d]', '', tf_clean)
                    if tf_clean and len(tf_clean) <= 4 and tf_clean not in timeframes:
                        timeframes.append(tf_clean)
        
        # Enhanced EA Studio specific patterns
        # Look for indicator calls with timeframe parameters
        indicator_tf_patterns = [
            r'i[A-Z][a-zA-Z]*\s*\(\s*NULL\s*,\s*(\d+)',  # iMA(NULL,15,...) format
            r'i[A-Z][a-zA-Z]*\s*\(\s*[^,]+\s*,\s*(\d+)',  # iMA(Symbol(),15,...) format
        ]
        
        for pattern in indicator_tf_patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                if match != "0" and match in tf_numeric_map:  # 0 means current timeframe
                    tf_name = tf_numeric_map[match]
                    if tf_name not in timeframes:
                        timeframes.append(tf_name)
        
        # Special handling for EA Studio: if no timeframes found but EA name contains timeframe
        if not timeframes and ea_name:
            # Extract timeframe from EA name as last resort
            name_lower = ea_name.lower()
            # Order from longest to shortest to avoid false matches (M15 before M1)
            for tf_pattern in ['m15', 'm30', 'mn1', 'm5', 'm1', 'h4', 'h1', 'd1', 'w1']:
                if re.search(rf'\b{tf_pattern}\b', name_lower):  # Use word boundaries
                    timeframes.append(tf_pattern.upper())
                    break
        
        # If no timeframes found, assume current
        if not timeframes:
            timeframes = ['CURRENT']
        
        # Clean and validate timeframes
        valid_timeframes = []
        for tf in timeframes:
            tf_clean = tf.upper().strip()
            # Validate timeframe format
            if tf_clean in ['CURRENT'] or re.match(r'^[MHD]\d+$|^W1$|^MN1$', tf_clean):
                if tf_clean not in valid_timeframes:
                    valid_timeframes.append(tf_clean)
        
        return valid_timeframes if valid_timeframes else ['CURRENT']
    
    def _extract_magic_numbers(self, code: str) -> List[int]:
        """Extract magic numbers"""
        magic_numbers = []
        
        # Look for magic number assignments (EA Studio and standard formats)
        magic_patterns = [
            r'Magic_Number\s*=\s*(\d+)',  # EA Studio format
            r'MagicNumber\s*=\s*(\d+)',   # Standard format
            r'Magic\s*=\s*(\d+)',         # Short format
            r'magic\s*=\s*(\d+)',         # Lowercase
            r'input\s+int\s+Magic_Number\s*=\s*(\d+)',  # EA Studio input format
            r'extern\s+int\s+Magic_Number\s*=\s*(\d+)', # Extern format
        ]
        
        for pattern in magic_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for match in matches:
                magic_numbers.append(int(match))
        
        # Also extract from parameter declarations
        param_magic_pattern = r'input\s+int\s+(\w*[Mm]agic\w*)\s*=\s*(\d+)'
        param_matches = re.findall(param_magic_pattern, code, re.IGNORECASE)
        for param_name, magic_value in param_matches:
            magic_numbers.append(int(magic_value))
        
        return list(set(magic_numbers))

class EAExecutionEngine:
    """Execution engine for parsed EAs"""
    
    def __init__(self, mt5_bridge=None):
        self.mt5_bridge = mt5_bridge
        self.running_eas = {}
        self.profiles_dir = "ea_profiles"
        self._ensure_profiles_dir()
    
    def _ensure_profiles_dir(self):
        """Ensure profiles directory exists"""
        if not os.path.exists(self.profiles_dir):
            os.makedirs(self.profiles_dir)
    
    def save_profile(self, profile: EAProfile) -> str:
        """Save EA profile to file"""
        try:
            filename = f"{self.profiles_dir}/profile_{profile.id}.json"
            
            # Convert to dict for JSON serialization
            profile_dict = asdict(profile)
            
            with open(filename, 'w') as f:
                json.dump(profile_dict, f, indent=2, default=str)
            
            logger.info(f"💾 EA profile saved: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Failed to save EA profile: {e}")
            raise
    
    def load_profile(self, profile_id: str) -> EAProfile:
        """Load EA profile from file"""
        try:
            filename = f"{self.profiles_dir}/profile_{profile_id}.json"
            
            with open(filename, 'r') as f:
                profile_dict = json.load(f)
            
            # Convert back to EAProfile object
            profile = EAProfile(**profile_dict)
            return profile
            
        except Exception as e:
            logger.error(f"❌ Failed to load EA profile: {e}")
            raise
    
    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all saved EA profiles"""
        profiles = []
        
        try:
            for filename in os.listdir(self.profiles_dir):
                if filename.startswith('profile_') and filename.endswith('.json'):
                    profile_id = filename.replace('profile_', '').replace('.json', '')
                    
                    try:
                        profile = self.load_profile(profile_id)
                        profiles.append({
                            'id': profile.id,
                            'name': profile.name,
                            'description': profile.description,
                            'parameters_count': len(profile.parameters),
                            'rules_count': len(profile.trading_rules),
                            'status': profile.execution_status,
                            'created_at': profile.created_at
                        })
                    except:
                        continue
        
        except Exception as e:
            logger.error(f"❌ Failed to list profiles: {e}")
        
        return profiles
    
    def start_ea(self, profile_id: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """Start EA execution"""
        try:
            profile = self.load_profile(profile_id)
            
            # Update parameters if provided
            if parameters:
                for param in profile.parameters:
                    if param.name in parameters:
                        param.default_value = parameters[param.name]
            
            # Mark as running
            profile.execution_status = "running"
            self.save_profile(profile)
            
            # Store in running EAs
            self.running_eas[profile_id] = {
                'profile': profile,
                'start_time': datetime.now(),
                'trades_count': 0,
                'last_signal': None
            }
            
            logger.info(f"▶️ EA started: {profile.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start EA: {e}")
            return False
    
    def stop_ea(self, profile_id: str) -> bool:
        """Stop EA execution"""
        try:
            if profile_id in self.running_eas:
                profile = self.running_eas[profile_id]['profile']
                profile.execution_status = "stopped"
                self.save_profile(profile)
                
                del self.running_eas[profile_id]
                logger.info(f"⏹️ EA stopped: {profile.name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to stop EA: {e}")
            return False
    
    def get_running_status(self) -> Dict[str, Any]:
        """Get status of all running EAs"""
        status = {}
        
        for profile_id, ea_info in self.running_eas.items():
            status[profile_id] = {
                'name': ea_info['profile'].name,
                'start_time': ea_info['start_time'].isoformat(),
                'trades_count': ea_info['trades_count'],
                'last_signal': ea_info['last_signal'],
                'status': 'running'
            }
        
        return status

# Sample EA code for testing
SAMPLE_EA_CODE = """
//+------------------------------------------------------------------+
//| Sample Moving Average EA                                          |
//+------------------------------------------------------------------+
#property copyright "QNTI System"
#property name "Sample MA EA"
#property description "Simple Moving Average Expert Advisor"

// Input parameters
extern int Magic = 12345;
extern double Lots = 0.1;
extern int StopLoss = 50;
extern int TakeProfit = 100;
extern bool UseMA = true;
extern int MAPeriod = 14;
extern int MAShift = 0;
extern int MAMethod = 0;
extern int MAPrice = 0;
extern bool UseTrailingStop = false;
extern int TrailingStop = 30;
extern int MaxTrades = 1;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    if(!UseMA) return;
    
    double ma_current = iMA(Symbol(), Period(), MAPeriod, MAShift, MAMethod, MAPrice, 0);
    double ma_previous = iMA(Symbol(), Period(), MAPeriod, MAShift, MAMethod, MAPrice, 1);
    
    double price_current = Close[0];
    double price_previous = Close[1];
    
    // Buy signal
    if(price_current > ma_current && price_previous <= ma_previous && CountTrades() < MaxTrades)
    {
        OrderSend(Symbol(), OP_BUY, Lots, Ask, 3, 
                 Ask - StopLoss * Point, 
                 Ask + TakeProfit * Point, 
                 "MA Buy", Magic, 0, Blue);
    }
    
    // Sell signal
    if(price_current < ma_current && price_previous >= ma_previous && CountTrades() < MaxTrades)
    {
        OrderSend(Symbol(), OP_SELL, Lots, Bid, 3, 
                 Bid + StopLoss * Point, 
                 Bid - TakeProfit * Point, 
                 "MA Sell", Magic, 0, Red);
    }
    
    // Trailing stop
    if(UseTrailingStop)
    {
        ManageTrailingStop();
    }
}

//+------------------------------------------------------------------+
//| Count open trades                                                |
//+------------------------------------------------------------------+
int CountTrades()
{
    int count = 0;
    for(int i = 0; i < OrdersTotal(); i++)
    {
        if(OrderSelect(i, SELECT_BY_POS) && OrderMagicNumber() == Magic)
        {
            count++;
        }
    }
    return count;
}

//+------------------------------------------------------------------+
//| Manage trailing stop                                             |
//+------------------------------------------------------------------+
void ManageTrailingStop()
{
    for(int i = 0; i < OrdersTotal(); i++)
    {
        if(OrderSelect(i, SELECT_BY_POS) && OrderMagicNumber() == Magic)
        {
            if(OrderType() == OP_BUY)
            {
                double newSL = Bid - TrailingStop * Point;
                if(newSL > OrderStopLoss())
                {
                    OrderModify(OrderTicket(), OrderOpenPrice(), newSL, OrderTakeProfit(), 0);
                }
            }
            else if(OrderType() == OP_SELL)
            {
                double newSL = Ask + TrailingStop * Point;
                if(newSL < OrderStopLoss())
                {
                    OrderModify(OrderTicket(), OrderOpenPrice(), newSL, OrderTakeProfit(), 0);
                }
            }
        }
    }
}
"""

def get_sample_ea_code() -> str:
    """Return sample EA code for testing"""
    return SAMPLE_EA_CODE

# Initialize global instances
parser = MQLCodeParser()
execution_engine = EAExecutionEngine()

def parse_ea_code(code: str, ea_name: Optional[str] = None) -> EAProfile:
    """Parse EA code and return profile"""
    return parser.parse_ea_code(code, ea_name)

def save_ea_profile(profile: EAProfile) -> str:
    """Save EA profile"""
    return execution_engine.save_profile(profile)

def get_ea_profiles() -> List[Dict[str, Any]]:
    """Get list of all EA profiles"""
    return execution_engine.list_profiles()

def start_ea_execution(profile_id: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
    """Start EA execution"""
    return execution_engine.start_ea(profile_id, parameters)

def stop_ea_execution(profile_id: str) -> bool:
    """Stop EA execution"""
    return execution_engine.stop_ea(profile_id)

def get_execution_status() -> Dict[str, Any]:
    """Get execution status of all EAs"""
    return execution_engine.get_running_status()

if __name__ == "__main__":
    # Test the parser
    print("🧪 Testing EA Parser...")
    
    sample_code = get_sample_ea_code()
    profile = parse_ea_code(sample_code, "Sample MA EA")
    
    print(f"✅ Parsed EA: {profile.name}")
    print(f"📊 Parameters: {len(profile.parameters)}")
    print(f"📋 Rules: {len(profile.trading_rules)}")
    print(f"📈 Indicators: {profile.indicators}")
    
    # Save profile
    filename = save_ea_profile(profile)
    print(f"💾 Saved to: {filename}") 