#!/usr/bin/env python3
"""
QNTI Advanced Trading Module
Implements advanced order types including trailing stops, OCO orders, and bracket orders
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from uuid import uuid4

logger = logging.getLogger('QNTI_ADVANCED_TRADING')

class OrderType(Enum):
    """Advanced order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"  # One-Cancels-Other
    BRACKET = "bracket"
    ICEBERG = "iceberg"
    TWAP = "twap"  # Time-Weighted Average Price

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    ACTIVE = "active"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"

@dataclass
class AdvancedOrder:
    """Advanced order data structure"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    trigger_price: Optional[float] = None
    
    # Advanced parameters
    trail_amount: Optional[float] = None  # For trailing stops
    trail_percent: Optional[float] = None  # For trailing stops
    take_profit: Optional[float] = None  # For bracket orders
    stop_loss: Optional[float] = None  # For bracket orders
    
    # OCO parameters
    oco_partner_id: Optional[str] = None  # For OCO orders
    
    # Iceberg parameters
    visible_quantity: Optional[float] = None  # For iceberg orders
    
    # TWAP parameters
    twap_duration: Optional[int] = None  # Duration in seconds
    twap_intervals: Optional[int] = None  # Number of intervals
    
    # Order state
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: Optional[float] = None
    average_fill_price: Optional[float] = None
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    expires_at: Optional[datetime] = None
    
    # Metadata
    ea_name: Optional[str] = None
    magic_number: Optional[int] = None
    comment: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity

@dataclass
class TrailingStopOrder(AdvancedOrder):
    """Specialized trailing stop order"""
    high_water_mark: Optional[float] = None
    low_water_mark: Optional[float] = None
    
    def update_water_marks(self, current_price: float):
        """Update high/low water marks for trailing stop"""
        if self.side == OrderSide.BUY:
            # For buy orders, track lowest price
            if self.low_water_mark is None or current_price < self.low_water_mark:
                self.low_water_mark = current_price
        else:
            # For sell orders, track highest price
            if self.high_water_mark is None or current_price > self.high_water_mark:
                self.high_water_mark = current_price

class QNTIAdvancedTrading:
    """Advanced Trading System for QNTI"""
    
    def __init__(self, trade_manager, mt5_bridge=None):
        self.trade_manager = trade_manager
        self.mt5_bridge = mt5_bridge
        
        # Order storage
        self.orders: Dict[str, AdvancedOrder] = {}
        self.order_history: List[AdvancedOrder] = []
        
        # Order monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 0.1  # 100ms monitoring
        
        # Event callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'order_filled': [],
            'order_cancelled': [],
            'order_rejected': [],
            'order_updated': [],
            'bracket_triggered': [],
            'trailing_stop_updated': []
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'average_fill_time': 0.0,
            'slippage_stats': []
        }
        
        # Risk management
        self.risk_settings = {
            'max_order_size': 10.0,
            'max_daily_orders': 100,
            'max_concurrent_orders': 20,
            'allowed_symbols': None,  # None means all symbols allowed
            'position_size_limits': {}
        }
        
        # Start monitoring
        self.start_monitoring()
        
        logger.info("Advanced Trading System initialized")
    
    def start_monitoring(self):
        """Start order monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Order monitoring started")
    
    def stop_monitoring(self):
        """Stop order monitoring thread"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        logger.info("Order monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for orders"""
        while self.monitoring_active:
            try:
                self._process_active_orders()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _process_active_orders(self):
        """Process all active orders"""
        current_time = datetime.now()
        
        for order_id, order in list(self.orders.items()):
            if order.status not in [OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED]:
                continue
            
            try:
                # Check for expiration
                if order.expires_at and current_time > order.expires_at:
                    self._expire_order(order)
                    continue
                
                # Get current market price
                current_price = self._get_current_price(order.symbol)
                if current_price is None:
                    continue
                
                # Process order based on type
                if order.order_type == OrderType.TRAILING_STOP:
                    self._process_trailing_stop(order, current_price)
                elif order.order_type == OrderType.OCO:
                    self._process_oco_order(order, current_price)
                elif order.order_type == OrderType.BRACKET:
                    self._process_bracket_order(order, current_price)
                elif order.order_type == OrderType.ICEBERG:
                    self._process_iceberg_order(order, current_price)
                elif order.order_type == OrderType.TWAP:
                    self._process_twap_order(order, current_price)
                else:
                    # Standard order processing
                    self._process_standard_order(order, current_price)
                    
            except Exception as e:
                logger.error(f"Error processing order {order_id}: {e}")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        try:
            if self.mt5_bridge:
                symbol_info = self.mt5_bridge.get_symbol_info(symbol)
                if symbol_info:
                    return symbol_info.get('bid', 0.0)
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _process_trailing_stop(self, order: TrailingStopOrder, current_price: float):
        """Process trailing stop order"""
        try:
            # Update water marks
            order.update_water_marks(current_price)
            
            # Calculate new stop price
            new_stop_price = None
            
            if order.side == OrderSide.SELL:
                # For sell orders, trail below the high water mark
                if order.high_water_mark:
                    if order.trail_amount:
                        new_stop_price = order.high_water_mark - order.trail_amount
                    elif order.trail_percent:
                        new_stop_price = order.high_water_mark * (1 - order.trail_percent / 100)
            else:
                # For buy orders, trail above the low water mark
                if order.low_water_mark:
                    if order.trail_amount:
                        new_stop_price = order.low_water_mark + order.trail_amount
                    elif order.trail_percent:
                        new_stop_price = order.low_water_mark * (1 + order.trail_percent / 100)
            
            # Update stop price if calculated
            if new_stop_price and new_stop_price != order.stop_price:
                order.stop_price = new_stop_price
                order.updated_at = datetime.now()
                self._trigger_callback('trailing_stop_updated', order)
            
            # Check if stop price is triggered
            if order.stop_price:
                triggered = False
                if order.side == OrderSide.SELL and current_price <= order.stop_price:
                    triggered = True
                elif order.side == OrderSide.BUY and current_price >= order.stop_price:
                    triggered = True
                
                if triggered:
                    self._execute_order(order, current_price)
                    
        except Exception as e:
            logger.error(f"Error processing trailing stop {order.id}: {e}")
    
    def _process_oco_order(self, order: AdvancedOrder, current_price: float):
        """Process One-Cancels-Other order"""
        try:
            # Check if order should be triggered
            triggered = False
            
            if order.order_type == OrderType.LIMIT:
                if (order.side == OrderSide.BUY and current_price <= order.price) or \
                   (order.side == OrderSide.SELL and current_price >= order.price):
                    triggered = True
            elif order.order_type == OrderType.STOP:
                if (order.side == OrderSide.BUY and current_price >= order.stop_price) or \
                   (order.side == OrderSide.SELL and current_price <= order.stop_price):
                    triggered = True
            
            if triggered:
                # Execute this order
                self._execute_order(order, current_price)
                
                # Cancel partner order
                if order.oco_partner_id and order.oco_partner_id in self.orders:
                    partner_order = self.orders[order.oco_partner_id]
                    self._cancel_order(partner_order, "OCO partner filled")
                    
        except Exception as e:
            logger.error(f"Error processing OCO order {order.id}: {e}")
    
    def _process_bracket_order(self, order: AdvancedOrder, current_price: float):
        """Process bracket order (parent order with take profit and stop loss)"""
        try:
            # First execute the parent order if not filled
            if order.filled_quantity == 0:
                self._execute_order(order, current_price)
                return
            
            # If parent is filled, monitor take profit and stop loss
            if order.filled_quantity > 0:
                # Check take profit
                if order.take_profit:
                    tp_triggered = False
                    if order.side == OrderSide.BUY and current_price >= order.take_profit:
                        tp_triggered = True
                    elif order.side == OrderSide.SELL and current_price <= order.take_profit:
                        tp_triggered = True
                    
                    if tp_triggered:
                        self._close_position(order, current_price, "Take Profit")
                        return
                
                # Check stop loss
                if order.stop_loss:
                    sl_triggered = False
                    if order.side == OrderSide.BUY and current_price <= order.stop_loss:
                        sl_triggered = True
                    elif order.side == OrderSide.SELL and current_price >= order.stop_loss:
                        sl_triggered = True
                    
                    if sl_triggered:
                        self._close_position(order, current_price, "Stop Loss")
                        return
                        
        except Exception as e:
            logger.error(f"Error processing bracket order {order.id}: {e}")
    
    def _process_iceberg_order(self, order: AdvancedOrder, current_price: float):
        """Process iceberg order (large order split into smaller visible portions)"""
        try:
            # Calculate visible quantity
            visible_qty = order.visible_quantity or min(order.remaining_quantity, order.quantity * 0.1)
            
            # Create market order for visible quantity
            if visible_qty > 0:
                visible_order = AdvancedOrder(
                    id=f"{order.id}_iceberg_{int(time.time())}",
                    symbol=order.symbol,
                    side=order.side,
                    order_type=OrderType.MARKET,
                    quantity=visible_qty,
                    price=order.price,
                    ea_name=order.ea_name,
                    magic_number=order.magic_number,
                    comment=f"Iceberg slice of {order.id}"
                )
                
                # Execute the slice
                if self._execute_order(visible_order, current_price):
                    order.filled_quantity += visible_qty
                    order.remaining_quantity -= visible_qty
                    order.updated_at = datetime.now()
                    
                    # Check if order is fully filled
                    if order.remaining_quantity <= 0:
                        order.status = OrderStatus.FILLED
                        self._trigger_callback('order_filled', order)
                        
        except Exception as e:
            logger.error(f"Error processing iceberg order {order.id}: {e}")
    
    def _process_twap_order(self, order: AdvancedOrder, current_price: float):
        """Process TWAP (Time-Weighted Average Price) order"""
        try:
            if not order.twap_duration or not order.twap_intervals:
                return
            
            # Calculate interval size
            interval_duration = order.twap_duration / order.twap_intervals
            interval_quantity = order.quantity / order.twap_intervals
            
            # Check if it's time for next interval
            elapsed = (datetime.now() - order.created_at).total_seconds()
            intervals_passed = int(elapsed / interval_duration)
            
            expected_filled = min(intervals_passed * interval_quantity, order.quantity)
            
            if order.filled_quantity < expected_filled:
                # Execute next interval
                quantity_to_fill = expected_filled - order.filled_quantity
                
                interval_order = AdvancedOrder(
                    id=f"{order.id}_twap_{intervals_passed}",
                    symbol=order.symbol,
                    side=order.side,
                    order_type=OrderType.MARKET,
                    quantity=quantity_to_fill,
                    ea_name=order.ea_name,
                    magic_number=order.magic_number,
                    comment=f"TWAP interval {intervals_passed} of {order.id}"
                )
                
                if self._execute_order(interval_order, current_price):
                    order.filled_quantity += quantity_to_fill
                    order.remaining_quantity -= quantity_to_fill
                    order.updated_at = datetime.now()
                    
                    # Update average fill price
                    if order.average_fill_price is None:
                        order.average_fill_price = current_price
                    else:
                        total_value = (order.average_fill_price * (order.filled_quantity - quantity_to_fill) + 
                                     current_price * quantity_to_fill)
                        order.average_fill_price = total_value / order.filled_quantity
                    
                    # Check if order is fully filled
                    if order.filled_quantity >= order.quantity:
                        order.status = OrderStatus.FILLED
                        self._trigger_callback('order_filled', order)
                        
        except Exception as e:
            logger.error(f"Error processing TWAP order {order.id}: {e}")
    
    def _process_standard_order(self, order: AdvancedOrder, current_price: float):
        """Process standard order types"""
        try:
            triggered = False
            
            if order.order_type == OrderType.LIMIT:
                if (order.side == OrderSide.BUY and current_price <= order.price) or \
                   (order.side == OrderSide.SELL and current_price >= order.price):
                    triggered = True
            elif order.order_type == OrderType.STOP:
                if (order.side == OrderSide.BUY and current_price >= order.stop_price) or \
                   (order.side == OrderSide.SELL and current_price <= order.stop_price):
                    triggered = True
            elif order.order_type == OrderType.MARKET:
                triggered = True
            
            if triggered:
                self._execute_order(order, current_price)
                
        except Exception as e:
            logger.error(f"Error processing standard order {order.id}: {e}")
    
    def _execute_order(self, order: AdvancedOrder, price: float) -> bool:
        """Execute an order"""
        try:
            # Validate order
            if not self._validate_order(order):
                order.status = OrderStatus.REJECTED
                self._trigger_callback('order_rejected', order)
                return False
            
            # Execute via MT5 bridge if available
            if self.mt5_bridge:
                result = self._execute_via_mt5(order, price)
                if result:
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.remaining_quantity = 0
                    order.average_fill_price = price
                    order.updated_at = datetime.now()
                    self._trigger_callback('order_filled', order)
                    
                    # Update performance stats
                    self.performance_stats['filled_orders'] += 1
                    fill_time = (order.updated_at - order.created_at).total_seconds()
                    self._update_average_fill_time(fill_time)
                    
                    return True
            
            # Fallback to trade manager
            return self._execute_via_trade_manager(order, price)
            
        except Exception as e:
            logger.error(f"Error executing order {order.id}: {e}")
            order.status = OrderStatus.REJECTED
            self._trigger_callback('order_rejected', order)
            return False
    
    def _execute_via_mt5(self, order: AdvancedOrder, price: float) -> bool:
        """Execute order via MT5 bridge"""
        try:
            # Convert to MT5 order format
            mt5_order = {
                'symbol': order.symbol,
                'volume': order.quantity,
                'type': 'BUY' if order.side == OrderSide.BUY else 'SELL',
                'price': price,
                'sl': order.stop_loss,
                'tp': order.take_profit,
                'comment': order.comment or f"Advanced Order {order.id}",
                'magic': order.magic_number or 0
            }
            
            # Place order
            result = self.mt5_bridge.place_order(**mt5_order)
            return result.get('success', False)
            
        except Exception as e:
            logger.error(f"Error executing via MT5: {e}")
            return False
    
    def _execute_via_trade_manager(self, order: AdvancedOrder, price: float) -> bool:
        """Execute order via trade manager"""
        try:
            # Create trade object
            from qnti_core_system import Trade, TradeSource, TradeStatus
            
            trade = Trade(
                id=f"ADV_{order.id}",
                symbol=order.symbol,
                side=order.side.value,
                volume=order.quantity,
                open_price=price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                source=TradeSource.ADVANCED_ORDER,
                status=TradeStatus.ACTIVE,
                ea_name=order.ea_name,
                magic_number=order.magic_number
            )
            
            # Add to trade manager
            self.trade_manager.add_trade(trade)
            return True
            
        except Exception as e:
            logger.error(f"Error executing via trade manager: {e}")
            return False
    
    def _validate_order(self, order: AdvancedOrder) -> bool:
        """Validate order parameters"""
        try:
            # Check basic parameters
            if order.quantity <= 0:
                return False
            
            # Check risk limits
            if order.quantity > self.risk_settings['max_order_size']:
                return False
            
            # Check symbol restrictions
            if (self.risk_settings['allowed_symbols'] and 
                order.symbol not in self.risk_settings['allowed_symbols']):
                return False
            
            # Check daily order limit
            today_orders = sum(1 for o in self.order_history 
                             if o.created_at.date() == datetime.now().date())
            if today_orders >= self.risk_settings['max_daily_orders']:
                return False
            
            # Check concurrent orders
            active_orders = sum(1 for o in self.orders.values() 
                              if o.status in [OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED])
            if active_orders >= self.risk_settings['max_concurrent_orders']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False
    
    def _cancel_order(self, order: AdvancedOrder, reason: str = ""):
        """Cancel an order"""
        try:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            order.comment = f"{order.comment or ''} - Cancelled: {reason}".strip()
            
            self._trigger_callback('order_cancelled', order)
            self.performance_stats['cancelled_orders'] += 1
            
            logger.info(f"Order {order.id} cancelled: {reason}")
            
        except Exception as e:
            logger.error(f"Error cancelling order {order.id}: {e}")
    
    def _expire_order(self, order: AdvancedOrder):
        """Expire an order"""
        try:
            order.status = OrderStatus.EXPIRED
            order.updated_at = datetime.now()
            
            self._trigger_callback('order_cancelled', order)
            logger.info(f"Order {order.id} expired")
            
        except Exception as e:
            logger.error(f"Error expiring order {order.id}: {e}")
    
    def _close_position(self, order: AdvancedOrder, price: float, reason: str):
        """Close a position"""
        try:
            # Close via MT5 if available
            if self.mt5_bridge:
                result = self.mt5_bridge.close_position(
                    symbol=order.symbol,
                    volume=order.filled_quantity,
                    price=price
                )
                
                if result.get('success', False):
                    order.status = OrderStatus.FILLED
                    order.updated_at = datetime.now()
                    order.comment = f"{order.comment or ''} - Closed: {reason}".strip()
                    
                    self._trigger_callback('bracket_triggered', order)
                    logger.info(f"Position closed for order {order.id}: {reason}")
                    
        except Exception as e:
            logger.error(f"Error closing position for order {order.id}: {e}")
    
    def _trigger_callback(self, event_type: str, order: AdvancedOrder):
        """Trigger event callbacks"""
        try:
            if event_type in self.callbacks:
                for callback in self.callbacks[event_type]:
                    try:
                        callback(order)
                    except Exception as e:
                        logger.error(f"Error in callback {callback}: {e}")
        except Exception as e:
            logger.error(f"Error triggering callback {event_type}: {e}")
    
    def _update_average_fill_time(self, fill_time: float):
        """Update average fill time statistic"""
        try:
            current_avg = self.performance_stats['average_fill_time']
            filled_count = self.performance_stats['filled_orders']
            
            if filled_count == 1:
                self.performance_stats['average_fill_time'] = fill_time
            else:
                self.performance_stats['average_fill_time'] = (
                    (current_avg * (filled_count - 1) + fill_time) / filled_count
                )
        except Exception as e:
            logger.error(f"Error updating average fill time: {e}")
    
    # Public API methods
    
    def place_trailing_stop(self, symbol: str, side: OrderSide, quantity: float, 
                           trail_amount: float = None, trail_percent: float = None,
                           **kwargs) -> str:
        """Place a trailing stop order"""
        try:
            order = TrailingStopOrder(
                id=str(uuid4()),
                symbol=symbol,
                side=side,
                order_type=OrderType.TRAILING_STOP,
                quantity=quantity,
                trail_amount=trail_amount,
                trail_percent=trail_percent,
                **kwargs
            )
            
            order.status = OrderStatus.ACTIVE
            self.orders[order.id] = order
            self.performance_stats['total_orders'] += 1
            
            logger.info(f"Trailing stop order placed: {order.id}")
            return order.id
            
        except Exception as e:
            logger.error(f"Error placing trailing stop: {e}")
            return None
    
    def place_oco_order(self, symbol: str, side: OrderSide, quantity: float,
                       limit_price: float, stop_price: float, **kwargs) -> tuple:
        """Place an OCO (One-Cancels-Other) order"""
        try:
            # Create limit order
            limit_order = AdvancedOrder(
                id=str(uuid4()),
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=limit_price,
                **kwargs
            )
            
            # Create stop order
            stop_order = AdvancedOrder(
                id=str(uuid4()),
                symbol=symbol,
                side=side,
                order_type=OrderType.STOP,
                quantity=quantity,
                stop_price=stop_price,
                **kwargs
            )
            
            # Link orders
            limit_order.oco_partner_id = stop_order.id
            stop_order.oco_partner_id = limit_order.id
            
            # Activate orders
            limit_order.status = OrderStatus.ACTIVE
            stop_order.status = OrderStatus.ACTIVE
            
            self.orders[limit_order.id] = limit_order
            self.orders[stop_order.id] = stop_order
            self.performance_stats['total_orders'] += 2
            
            logger.info(f"OCO order placed: {limit_order.id} / {stop_order.id}")
            return limit_order.id, stop_order.id
            
        except Exception as e:
            logger.error(f"Error placing OCO order: {e}")
            return None, None
    
    def place_bracket_order(self, symbol: str, side: OrderSide, quantity: float,
                           price: float, take_profit: float, stop_loss: float,
                           **kwargs) -> str:
        """Place a bracket order"""
        try:
            order = AdvancedOrder(
                id=str(uuid4()),
                symbol=symbol,
                side=side,
                order_type=OrderType.BRACKET,
                quantity=quantity,
                price=price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                **kwargs
            )
            
            order.status = OrderStatus.ACTIVE
            self.orders[order.id] = order
            self.performance_stats['total_orders'] += 1
            
            logger.info(f"Bracket order placed: {order.id}")
            return order.id
            
        except Exception as e:
            logger.error(f"Error placing bracket order: {e}")
            return None
    
    def place_iceberg_order(self, symbol: str, side: OrderSide, quantity: float,
                           visible_quantity: float, price: float = None, **kwargs) -> str:
        """Place an iceberg order"""
        try:
            order = AdvancedOrder(
                id=str(uuid4()),
                symbol=symbol,
                side=side,
                order_type=OrderType.ICEBERG,
                quantity=quantity,
                visible_quantity=visible_quantity,
                price=price,
                **kwargs
            )
            
            order.status = OrderStatus.ACTIVE
            self.orders[order.id] = order
            self.performance_stats['total_orders'] += 1
            
            logger.info(f"Iceberg order placed: {order.id}")
            return order.id
            
        except Exception as e:
            logger.error(f"Error placing iceberg order: {e}")
            return None
    
    def place_twap_order(self, symbol: str, side: OrderSide, quantity: float,
                        duration: int, intervals: int, **kwargs) -> str:
        """Place a TWAP order"""
        try:
            order = AdvancedOrder(
                id=str(uuid4()),
                symbol=symbol,
                side=side,
                order_type=OrderType.TWAP,
                quantity=quantity,
                twap_duration=duration,
                twap_intervals=intervals,
                **kwargs
            )
            
            order.status = OrderStatus.ACTIVE
            self.orders[order.id] = order
            self.performance_stats['total_orders'] += 1
            
            logger.info(f"TWAP order placed: {order.id}")
            return order.id
            
        except Exception as e:
            logger.error(f"Error placing TWAP order: {e}")
            return None
    
    def cancel_order(self, order_id: str, reason: str = "") -> bool:
        """Cancel an order"""
        try:
            if order_id in self.orders:
                order = self.orders[order_id]
                self._cancel_order(order, reason)
                return True
            return False
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status"""
        try:
            if order_id in self.orders:
                order = self.orders[order_id]
                return {
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'type': order.order_type.value,
                    'status': order.status.value,
                    'quantity': order.quantity,
                    'filled_quantity': order.filled_quantity,
                    'remaining_quantity': order.remaining_quantity,
                    'price': order.price,
                    'average_fill_price': order.average_fill_price,
                    'created_at': order.created_at.isoformat(),
                    'updated_at': order.updated_at.isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Error getting order status {order_id}: {e}")
            return None
    
    def get_active_orders(self) -> List[Dict]:
        """Get all active orders"""
        try:
            active_orders = []
            for order in self.orders.values():
                if order.status in [OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED]:
                    active_orders.append(self.get_order_status(order.id))
            return active_orders
        except Exception as e:
            logger.error(f"Error getting active orders: {e}")
            return []
    
    def get_order_history(self, limit: int = 100) -> List[Dict]:
        """Get order history"""
        try:
            history = []
            for order in sorted(self.order_history, 
                              key=lambda x: x.updated_at, reverse=True)[:limit]:
                history.append(self.get_order_status(order.id))
            return history
        except Exception as e:
            logger.error(f"Error getting order history: {e}")
            return []
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        try:
            stats = self.performance_stats.copy()
            stats['active_orders'] = len([o for o in self.orders.values() 
                                        if o.status in [OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED]])
            return stats
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add event callback"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def remove_callback(self, event_type: str, callback: Callable):
        """Remove event callback"""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
    
    def update_risk_settings(self, settings: Dict):
        """Update risk management settings"""
        try:
            self.risk_settings.update(settings)
            logger.info("Risk settings updated")
        except Exception as e:
            logger.error(f"Error updating risk settings: {e}")
    
    def shutdown(self):
        """Shutdown the advanced trading system"""
        try:
            self.stop_monitoring()
            
            # Cancel all active orders
            for order in list(self.orders.values()):
                if order.status in [OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED]:
                    self._cancel_order(order, "System shutdown")
            
            # Move all orders to history
            self.order_history.extend(self.orders.values())
            self.orders.clear()
            
            logger.info("Advanced trading system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")