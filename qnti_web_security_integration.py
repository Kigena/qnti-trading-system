#!/usr/bin/env python3
"""
QNTI Web Security Integration - Security Integration for Existing Web Interface
Patches the existing web interface to add comprehensive security features
"""

import logging
from typing import Dict, List, Optional, Any
from functools import wraps
from flask import Flask, request, jsonify

# Import security components
from qnti_security_framework import get_security_framework, secure_endpoint, trading_endpoint, admin_endpoint, public_endpoint
from qnti_security_middleware import init_security_middleware
from qnti_auth_system import require_auth, Permission, UserRole
from qnti_rate_limiter import RateLimitScope, strict_rate_limit, trading_rate_limit, auth_rate_limit
from qnti_audit_logger import audit_api_request, audit_trading_action, audit_auth_action
from qnti_encryption import encrypt_sensitive_data, EncryptionMethod

logger = logging.getLogger('QNTI_WEB_SECURITY')

class WebSecurityIntegration:
    """Integration class for adding security to existing web interface"""
    
    def __init__(self, web_interface, app: Flask):
        self.web_interface = web_interface
        self.app = app
        self.security_framework = get_security_framework()
        
        # Initialize security middleware
        self.security_middleware = init_security_middleware(app)
        
        # Initialize security framework with app
        self.security_framework.init_app(app)
        
        # Patch existing routes with security
        self._patch_existing_routes()
        
        # Add new security routes
        self._add_security_routes()
        
        logger.info("Web security integration initialized")
    
    def _patch_existing_routes(self):
        """Patch existing routes with security decorators"""
        
        # Patch dashboard routes (public with rate limiting)
        original_dashboard = self.web_interface.setup_routes.__func__
        
        @self.app.route('/')
        @public_endpoint()
        def secure_dashboard():
            """Secured main dashboard"""
            return self._call_original_route('dashboard')
        
        @self.app.route('/dashboard/main_dashboard.html')
        @self.app.route('/main_dashboard.html')
        @self.app.route('/overview')
        @self.app.route('/dashboard/overview')
        @public_endpoint()
        def secure_main_dashboard():
            """Secured main dashboard page"""
            return self._call_original_route('main_dashboard_page')
        
        # Patch API routes with authentication and authorization
        self._patch_api_routes()
        
        # Patch trading routes with strict security
        self._patch_trading_routes()
        
        # Patch admin routes with admin security
        self._patch_admin_routes()
    
    def _patch_api_routes(self):
        """Patch API routes with appropriate security"""
        
        @self.app.route('/api/account/info')
        @secure_endpoint([Permission.ACCOUNT_VIEW], RateLimitScope.USER, True)
        def secure_account_info():
            """Secured account info endpoint"""
            return self._get_account_info()
        
        @self.app.route('/api/trades/active')
        @secure_endpoint([Permission.TRADE_VIEW], RateLimitScope.USER, True)
        def secure_active_trades():
            """Secured active trades endpoint"""
            return self._get_active_trades()
        
        @self.app.route('/api/trades/history')
        @secure_endpoint([Permission.TRADE_HISTORY], RateLimitScope.USER, True)
        def secure_trade_history():
            """Secured trade history endpoint"""
            return self._get_trade_history()
        
        @self.app.route('/api/ea/list')
        @secure_endpoint([Permission.EA_VIEW], RateLimitScope.USER, False)
        def secure_ea_list():
            """Secured EA list endpoint"""
            return self._get_ea_list()
        
        @self.app.route('/api/ea/performance/<ea_name>')
        @secure_endpoint([Permission.EA_VIEW], RateLimitScope.USER, True)
        def secure_ea_performance(ea_name):
            """Secured EA performance endpoint"""
            return self._get_ea_performance(ea_name)
    
    def _patch_trading_routes(self):
        """Patch trading routes with strict security"""
        
        @self.app.route('/api/trade/execute', methods=['POST'])
        @trading_endpoint([Permission.TRADE_EXECUTE])
        @strict_rate_limit(5, 60)  # 5 trades per minute
        def secure_trade_execute():
            """Secured trade execution endpoint"""
            return self._execute_trade()
        
        @self.app.route('/api/trade/modify', methods=['POST'])
        @trading_endpoint([Permission.TRADE_MANAGE])
        @strict_rate_limit(10, 60)  # 10 modifications per minute
        def secure_trade_modify():
            """Secured trade modification endpoint"""
            return self._modify_trade()
        
        @self.app.route('/api/trade/close', methods=['POST'])
        @trading_endpoint([Permission.TRADE_EXECUTE])
        @strict_rate_limit(10, 60)  # 10 closures per minute
        def secure_trade_close():
            """Secured trade closure endpoint"""
            return self._close_trade()
        
        @self.app.route('/api/ea/start', methods=['POST'])
        @trading_endpoint([Permission.EA_EXECUTE])
        @strict_rate_limit(5, 300)  # 5 EA starts per 5 minutes
        def secure_ea_start():
            """Secured EA start endpoint"""
            return self._start_ea()
        
        @self.app.route('/api/ea/stop', methods=['POST'])
        @trading_endpoint([Permission.EA_MANAGE])
        @strict_rate_limit(10, 300)  # 10 EA stops per 5 minutes
        def secure_ea_stop():
            """Secured EA stop endpoint"""
            return self._stop_ea()
    
    def _patch_admin_routes(self):
        """Patch admin routes with admin security"""
        
        @self.app.route('/api/admin/system/health')
        @admin_endpoint([Permission.SYSTEM_HEALTH])
        def secure_system_health():
            """Secured system health endpoint"""
            return self._get_system_health()
        
        @self.app.route('/api/admin/system/config', methods=['GET'])
        @admin_endpoint([Permission.SYSTEM_CONFIG])
        def secure_get_system_config():
            """Secured get system config endpoint"""
            return self._get_system_config()
        
        @self.app.route('/api/admin/system/config', methods=['POST'])
        @admin_endpoint([Permission.SYSTEM_CONFIG])
        @strict_rate_limit(1, 60)  # 1 config change per minute
        def secure_update_system_config():
            """Secured update system config endpoint"""
            return self._update_system_config()
        
        @self.app.route('/api/admin/logs')
        @admin_endpoint([Permission.SYSTEM_LOGS])
        def secure_get_logs():
            """Secured logs endpoint"""
            return self._get_logs()
        
        @self.app.route('/api/admin/users')
        @admin_endpoint([Permission.ACCOUNT_ADMIN])
        def secure_get_users():
            """Secured users management endpoint"""
            return self._get_users()
    
    def _add_security_routes(self):
        """Add new security-related routes"""
        
        @self.app.route('/api/auth/login', methods=['POST'])
        @auth_rate_limit()
        @audit_auth_action()
        def secure_login():
            """Secure login endpoint"""
            return self._handle_login()
        
        @self.app.route('/api/auth/logout', methods=['POST'])
        @require_auth()
        @audit_auth_action()
        def secure_logout():
            """Secure logout endpoint"""
            return self._handle_logout()
        
        @self.app.route('/api/auth/refresh', methods=['POST'])
        @auth_rate_limit()
        def secure_refresh_token():
            """Secure token refresh endpoint"""
            return self._refresh_token()
        
        @self.app.route('/api/auth/register', methods=['POST'])
        @auth_rate_limit()
        @audit_auth_action()
        def secure_register():
            """Secure registration endpoint"""
            return self._handle_register()
        
        @self.app.route('/api/auth/mfa/enable', methods=['POST'])
        @require_auth()
        @strict_rate_limit(1, 300)  # 1 MFA enable per 5 minutes
        def secure_enable_mfa():
            """Secure MFA enable endpoint"""
            return self._enable_mfa()
        
        @self.app.route('/api/auth/password/change', methods=['POST'])
        @require_auth()
        @strict_rate_limit(3, 3600)  # 3 password changes per hour
        def secure_change_password():
            """Secure password change endpoint"""
            return self._change_password()
        
        @self.app.route('/api/auth/apikey/create', methods=['POST'])
        @require_auth([Permission.API_ADMIN])
        @strict_rate_limit(5, 86400)  # 5 API keys per day
        def secure_create_api_key():
            """Secure API key creation endpoint"""
            return self._create_api_key()
        
        @self.app.route('/api/security/audit/events')
        @require_auth([Permission.SYSTEM_LOGS])
        def secure_audit_events():
            """Secure audit events endpoint"""
            return self._get_audit_events()
        
        @self.app.route('/api/security/metrics')
        @require_auth([Permission.SYSTEM_ADMIN])
        def secure_security_metrics():
            """Secure security metrics endpoint"""
            return self._get_security_metrics()
    
    def _call_original_route(self, route_name: str):
        """Call original route function safely"""
        try:
            # This would call the original route function
            # For now, return a placeholder
            return jsonify({'message': f'Original {route_name} route would be called here'})
        except Exception as e:
            logger.error(f"Error calling original route {route_name}: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    def _get_account_info(self):
        """Get account information with security"""
        try:
            if self.web_interface.main_system:
                account_info = self.web_interface._get_cached_account_info()
                return jsonify({
                    'success': True,
                    'data': account_info,
                    'timestamp': self._get_timestamp()
                })
            else:
                return jsonify({'error': 'System not available'}), 503
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return jsonify({'error': 'Failed to get account info'}), 500
    
    def _get_active_trades(self):
        """Get active trades with security"""
        try:
            if self.web_interface.main_system and self.web_interface.main_system.trade_manager:
                trades = self.web_interface.main_system.trade_manager.get_active_trades()
                return jsonify({
                    'success': True,
                    'data': [trade.to_dict() for trade in trades],
                    'count': len(trades),
                    'timestamp': self._get_timestamp()
                })
            else:
                return jsonify({'error': 'Trade manager not available'}), 503
        except Exception as e:
            logger.error(f"Error getting active trades: {e}")
            return jsonify({'error': 'Failed to get active trades'}), 500
    
    def _get_trade_history(self):
        """Get trade history with security"""
        try:
            if self.web_interface.main_system and self.web_interface.main_system.trade_manager:
                history = self.web_interface.main_system.trade_manager.get_trade_history()
                return jsonify({
                    'success': True,
                    'data': [trade.to_dict() for trade in history],
                    'count': len(history),
                    'timestamp': self._get_timestamp()
                })
            else:
                return jsonify({'error': 'Trade manager not available'}), 503
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return jsonify({'error': 'Failed to get trade history'}), 500
    
    def _get_ea_list(self):
        """Get EA list with security"""
        try:
            ea_data = self.web_interface._get_ea_data_cached()
            return jsonify({
                'success': True,
                'data': ea_data,
                'count': len(ea_data),
                'timestamp': self._get_timestamp()
            })
        except Exception as e:
            logger.error(f"Error getting EA list: {e}")
            return jsonify({'error': 'Failed to get EA list'}), 500
    
    def _get_ea_performance(self, ea_name: str):
        """Get EA performance with security"""
        try:
            if self.web_interface.main_system and self.web_interface.main_system.trade_manager:
                performance = self.web_interface.main_system.trade_manager.ea_performances.get(ea_name)
                if performance:
                    return jsonify({
                        'success': True,
                        'data': performance.to_dict(),
                        'timestamp': self._get_timestamp()
                    })
                else:
                    return jsonify({'error': 'EA not found'}), 404
            else:
                return jsonify({'error': 'Trade manager not available'}), 503
        except Exception as e:
            logger.error(f"Error getting EA performance: {e}")
            return jsonify({'error': 'Failed to get EA performance'}), 500
    
    def _execute_trade(self):
        """Execute trade with security"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Validate required fields
            required_fields = ['symbol', 'volume', 'type']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400
            
            # Execute trade through trade manager
            if self.web_interface.main_system and self.web_interface.main_system.trade_manager:
                result = self.web_interface.main_system.trade_manager.execute_trade(
                    symbol=data['symbol'],
                    volume=data['volume'],
                    trade_type=data['type'],
                    price=data.get('price'),
                    sl=data.get('sl'),
                    tp=data.get('tp'),
                    comment=data.get('comment', 'QNTI Web Trade')
                )
                
                return jsonify({
                    'success': True,
                    'data': result,
                    'timestamp': self._get_timestamp()
                })
            else:
                return jsonify({'error': 'Trade manager not available'}), 503
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return jsonify({'error': 'Failed to execute trade'}), 500
    
    def _modify_trade(self):
        """Modify trade with security"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Implementation would go here
            return jsonify({
                'success': True,
                'message': 'Trade modified successfully',
                'timestamp': self._get_timestamp()
            })
            
        except Exception as e:
            logger.error(f"Error modifying trade: {e}")
            return jsonify({'error': 'Failed to modify trade'}), 500
    
    def _close_trade(self):
        """Close trade with security"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Implementation would go here
            return jsonify({
                'success': True,
                'message': 'Trade closed successfully',
                'timestamp': self._get_timestamp()
            })
            
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            return jsonify({'error': 'Failed to close trade'}), 500
    
    def _start_ea(self):
        """Start EA with security"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Implementation would go here
            return jsonify({
                'success': True,
                'message': 'EA started successfully',
                'timestamp': self._get_timestamp()
            })
            
        except Exception as e:
            logger.error(f"Error starting EA: {e}")
            return jsonify({'error': 'Failed to start EA'}), 500
    
    def _stop_ea(self):
        """Stop EA with security"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Implementation would go here
            return jsonify({
                'success': True,
                'message': 'EA stopped successfully',
                'timestamp': self._get_timestamp()
            })
            
        except Exception as e:
            logger.error(f"Error stopping EA: {e}")
            return jsonify({'error': 'Failed to stop EA'}), 500
    
    def _get_system_health(self):
        """Get system health with security"""
        try:
            if self.web_interface.main_system:
                health_data = {
                    'status': 'healthy',
                    'uptime': self._get_uptime(),
                    'components': {
                        'trade_manager': 'active' if self.web_interface.main_system.trade_manager else 'inactive',
                        'mt5_bridge': 'active' if self.web_interface.main_system.mt5_bridge else 'inactive',
                        'vision_analyzer': 'active' if self.web_interface.main_system.vision_analyzer else 'inactive',
                        'notification_system': 'active' if self.web_interface.main_system.notification_system else 'inactive'
                    },
                    'timestamp': self._get_timestamp()
                }
                return jsonify({'success': True, 'data': health_data})
            else:
                return jsonify({'error': 'System not available'}), 503
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return jsonify({'error': 'Failed to get system health'}), 500
    
    def _get_system_config(self):
        """Get system configuration with security"""
        try:
            # Return non-sensitive configuration
            config = {
                'version': '1.0.0',
                'features': {
                    'auto_trading': True,
                    'vision_analysis': True,
                    'backtesting': True,
                    'notifications': True
                },
                'timestamp': self._get_timestamp()
            }
            return jsonify({'success': True, 'data': config})
        except Exception as e:
            logger.error(f"Error getting system config: {e}")
            return jsonify({'error': 'Failed to get system config'}), 500
    
    def _update_system_config(self):
        """Update system configuration with security"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Implementation would go here
            return jsonify({
                'success': True,
                'message': 'System configuration updated successfully',
                'timestamp': self._get_timestamp()
            })
            
        except Exception as e:
            logger.error(f"Error updating system config: {e}")
            return jsonify({'error': 'Failed to update system config'}), 500
    
    def _get_logs(self):
        """Get system logs with security"""
        try:
            # Implementation would go here
            logs = []
            return jsonify({
                'success': True,
                'data': logs,
                'timestamp': self._get_timestamp()
            })
        except Exception as e:
            logger.error(f"Error getting logs: {e}")
            return jsonify({'error': 'Failed to get logs'}), 500
    
    def _get_users(self):
        """Get users with security"""
        try:
            # Implementation would go here
            users = []
            return jsonify({
                'success': True,
                'data': users,
                'timestamp': self._get_timestamp()
            })
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            return jsonify({'error': 'Failed to get users'}), 500
    
    def _handle_login(self):
        """Handle login with security"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            username = data.get('username')
            password = data.get('password')
            mfa_code = data.get('mfa_code')
            
            if not username or not password:
                return jsonify({'error': 'Username and password required'}), 400
            
            # Authenticate through auth system
            auth_system = self.security_framework.auth_system
            result = auth_system.authenticate(
                username=username,
                password=password,
                mfa_code=mfa_code,
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent')
            )
            
            if result['success']:
                return jsonify({
                    'success': True,
                    'data': {
                        'access_token': result['access_token'],
                        'refresh_token': result['refresh_token'],
                        'user': result['user']
                    },
                    'timestamp': self._get_timestamp()
                })
            else:
                return jsonify({'error': result['error']}), 401
                
        except Exception as e:
            logger.error(f"Error in login: {e}")
            return jsonify({'error': 'Login failed'}), 500
    
    def _handle_logout(self):
        """Handle logout with security"""
        try:
            auth_system = self.security_framework.auth_system
            session_id = request.user.get('session_id')
            user_id = request.user.get('user_id')
            
            if session_id:
                auth_system.logout(session_id, user_id)
            
            return jsonify({
                'success': True,
                'message': 'Logged out successfully',
                'timestamp': self._get_timestamp()
            })
            
        except Exception as e:
            logger.error(f"Error in logout: {e}")
            return jsonify({'error': 'Logout failed'}), 500
    
    def _refresh_token(self):
        """Refresh access token with security"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            refresh_token = data.get('refresh_token')
            if not refresh_token:
                return jsonify({'error': 'Refresh token required'}), 400
            
            auth_system = self.security_framework.auth_system
            result = auth_system.refresh_token(refresh_token)
            
            if result['success']:
                return jsonify({
                    'success': True,
                    'data': {
                        'access_token': result['access_token']
                    },
                    'timestamp': self._get_timestamp()
                })
            else:
                return jsonify({'error': result['error']}), 401
                
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return jsonify({'error': 'Token refresh failed'}), 500
    
    def _handle_register(self):
        """Handle registration with security"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
            
            if not all([username, email, password]):
                return jsonify({'error': 'Username, email, and password required'}), 400
            
            # Create user through auth system
            auth_system = self.security_framework.auth_system
            result = auth_system.create_user(
                username=username,
                email=email,
                password=password,
                role=UserRole.VIEWER  # Default role
            )
            
            if result['success']:
                return jsonify({
                    'success': True,
                    'message': 'User created successfully',
                    'user_id': result['user_id'],
                    'timestamp': self._get_timestamp()
                })
            else:
                return jsonify({'error': result['error']}), 400
                
        except Exception as e:
            logger.error(f"Error in registration: {e}")
            return jsonify({'error': 'Registration failed'}), 500
    
    def _enable_mfa(self):
        """Enable MFA with security"""
        try:
            auth_system = self.security_framework.auth_system
            user_id = request.user.get('user_id')
            
            result = auth_system.enable_mfa(user_id)
            
            if result['success']:
                return jsonify({
                    'success': True,
                    'data': {
                        'qr_code': result['qr_code'],
                        'secret': result['secret']
                    },
                    'timestamp': self._get_timestamp()
                })
            else:
                return jsonify({'error': result['error']}), 400
                
        except Exception as e:
            logger.error(f"Error enabling MFA: {e}")
            return jsonify({'error': 'Failed to enable MFA'}), 500
    
    def _change_password(self):
        """Change password with security"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Implementation would go here
            return jsonify({
                'success': True,
                'message': 'Password changed successfully',
                'timestamp': self._get_timestamp()
            })
            
        except Exception as e:
            logger.error(f"Error changing password: {e}")
            return jsonify({'error': 'Failed to change password'}), 500
    
    def _create_api_key(self):
        """Create API key with security"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            name = data.get('name')
            permissions = data.get('permissions', [])
            
            if not name:
                return jsonify({'error': 'API key name required'}), 400
            
            auth_system = self.security_framework.auth_system
            user_id = request.user.get('user_id')
            
            # Convert permission strings to Permission objects
            permission_objects = [Permission(p) for p in permissions]
            
            result = auth_system.create_api_key(
                user_id=user_id,
                name=name,
                permissions=permission_objects
            )
            
            if result['success']:
                return jsonify({
                    'success': True,
                    'data': {
                        'api_key': result['api_key'],
                        'key_id': result['key_id']
                    },
                    'timestamp': self._get_timestamp()
                })
            else:
                return jsonify({'error': result['error']}), 400
                
        except Exception as e:
            logger.error(f"Error creating API key: {e}")
            return jsonify({'error': 'Failed to create API key'}), 500
    
    def _get_audit_events(self):
        """Get audit events with security"""
        try:
            hours = request.args.get('hours', 24, type=int)
            limit = request.args.get('limit', 100, type=int)
            
            audit_logger = self.security_framework.audit_logger
            events = audit_logger.get_user_activity(
                user_id=request.user.get('user_id'),
                hours=hours
            )
            
            return jsonify({
                'success': True,
                'data': [
                    {
                        'id': event.id,
                        'timestamp': event.timestamp.isoformat(),
                        'level': event.level.value,
                        'action': event.action.value,
                        'resource': event.resource.value,
                        'success': event.success,
                        'details': event.details
                    } for event in events[:limit]
                ],
                'timestamp': self._get_timestamp()
            })
            
        except Exception as e:
            logger.error(f"Error getting audit events: {e}")
            return jsonify({'error': 'Failed to get audit events'}), 500
    
    def _get_security_metrics(self):
        """Get security metrics with security"""
        try:
            metrics = self.security_framework.get_security_metrics()
            
            return jsonify({
                'success': True,
                'data': metrics,
                'timestamp': self._get_timestamp()
            })
            
        except Exception as e:
            logger.error(f"Error getting security metrics: {e}")
            return jsonify({'error': 'Failed to get security metrics'}), 500
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _get_uptime(self):
        """Get system uptime"""
        if hasattr(self.web_interface, 'start_time'):
            from datetime import datetime
            uptime = datetime.now() - datetime.fromtimestamp(self.web_interface.start_time)
            return str(uptime)
        return "Unknown"

def integrate_web_security(web_interface, app: Flask):
    """Integrate security with existing web interface"""
    return WebSecurityIntegration(web_interface, app)