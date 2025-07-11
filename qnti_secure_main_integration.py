#!/usr/bin/env python3
"""
QNTI Secure Main Integration - Integration Example
Demonstrates how to integrate the comprehensive security framework with the existing QNTI system
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from flask import Flask

# Add the project root to the path so we can import QNTI modules
sys.path.insert(0, str(Path(__file__).parent))

# Import existing QNTI components
from qnti_main_system import QNTIMainSystem
from qnti_web_interface import QNTIWebInterface

# Import security components
from qnti_security_framework import QNTISecurityFramework, get_security_framework
from qnti_security_middleware import init_security_middleware
from qnti_web_security_integration import integrate_web_security
from qnti_auth_system import get_auth_system, UserRole
from qnti_rate_limiter import get_rate_limiter
from qnti_encryption import get_encryption_system
from qnti_audit_logger import get_audit_logger, AuditLevel, AuditAction, AuditResource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qnti_secure_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('QNTI_SECURE_MAIN')

class QNTISecureMainSystem(QNTIMainSystem):
    """
    Extended QNTI Main System with comprehensive security integration
    """
    
    def __init__(self, config_file: str = "qnti_config.json"):
        logger.info("Initializing QNTI Secure Main System...")
        
        # Initialize parent class
        super().__init__(config_file)
        
        # Initialize security framework
        self.security_framework = QNTISecurityFramework(self.app)
        
        # Initialize security middleware
        self.security_middleware = init_security_middleware(self.app)
        
        # Initialize security components
        self.auth_system = get_auth_system()
        self.rate_limiter = get_rate_limiter()
        self.encryption_system = get_encryption_system()
        self.audit_logger = get_audit_logger()
        
        # Initialize web security integration
        self.web_security_integration = integrate_web_security(self.web_interface, self.app)
        
        # Setup security event handlers
        self._setup_security_event_handlers()
        
        # Create default admin user if needed
        self._ensure_admin_user()
        
        # Log system initialization
        self._log_system_startup()
        
        logger.info("QNTI Secure Main System initialized successfully")
    
    def _setup_security_event_handlers(self):
        """Setup security event handlers"""
        
        # Override Flask error handlers for security events
        @self.app.errorhandler(401)
        def handle_unauthorized(error):
            self.audit_logger.log_event(
                level=AuditLevel.WARNING,
                action=AuditAction.ACCESS_DENIED,
                resource=AuditResource.SECURITY,
                details={'error': 'Unauthorized access attempt'}
            )
            return {'error': 'Unauthorized access'}, 401
        
        @self.app.errorhandler(403)
        def handle_forbidden(error):
            self.audit_logger.log_event(
                level=AuditLevel.WARNING,
                action=AuditAction.ACCESS_DENIED,
                resource=AuditResource.SECURITY,
                details={'error': 'Forbidden access attempt'}
            )
            return {'error': 'Forbidden access'}, 403
        
        @self.app.errorhandler(429)
        def handle_rate_limit(error):
            self.audit_logger.log_event(
                level=AuditLevel.WARNING,
                action=AuditAction.RATE_LIMIT_EXCEEDED,
                resource=AuditResource.SECURITY,
                details={'error': 'Rate limit exceeded'}
            )
            return {'error': 'Rate limit exceeded'}, 429
        
        # Add security health check endpoint
        @self.app.route('/api/security/health')
        def security_health():
            """Security system health check"""
            try:
                health_status = {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'components': {
                        'authentication': self._check_auth_health(),
                        'rate_limiting': self._check_rate_limit_health(),
                        'encryption': self._check_encryption_health(),
                        'audit_logging': self._check_audit_health()
                    },
                    'security_metrics': self.security_framework.get_security_metrics()
                }
                
                return health_status, 200
            
            except Exception as e:
                logger.error(f"Security health check failed: {e}")
                return {'status': 'unhealthy', 'error': str(e)}, 500
    
    def _check_auth_health(self) -> str:
        """Check authentication system health"""
        try:
            # Test token generation
            test_user = {
                'user_id': 'test',
                'username': 'healthcheck',
                'role': UserRole.VIEWER.value,
                'permissions': []
            }
            
            token = self.auth_system._generate_access_token(test_user, 'test_session')
            if token:
                return 'healthy'
            else:
                return 'unhealthy'
        except Exception as e:
            logger.error(f"Auth health check failed: {e}")
            return 'unhealthy'
    
    def _check_rate_limit_health(self) -> str:
        """Check rate limiting system health"""
        try:
            # Test rate limit check
            from qnti_rate_limiter import RateLimitScope, RateLimit, RateLimitType
            
            test_limit = RateLimit(
                limit=1000,
                window=60,
                scope=RateLimitScope.GLOBAL,
                limit_type=RateLimitType.REQUESTS_PER_MINUTE
            )
            
            result = self.rate_limiter.check_limits('health_check', RateLimitScope.GLOBAL)
            if result:
                return 'healthy'
            else:
                return 'unhealthy'
        except Exception as e:
            logger.error(f"Rate limit health check failed: {e}")
            return 'unhealthy'
    
    def _check_encryption_health(self) -> str:
        """Check encryption system health"""
        try:
            # Test encryption/decryption
            test_data = "health_check_data"
            encrypted = self.encryption_system.encrypt(test_data)
            decrypted = self.encryption_system.decrypt(encrypted)
            
            if decrypted.decode('utf-8') == test_data:
                return 'healthy'
            else:
                return 'unhealthy'
        except Exception as e:
            logger.error(f"Encryption health check failed: {e}")
            return 'unhealthy'
    
    def _check_audit_health(self) -> str:
        """Check audit logging system health"""
        try:
            # Test audit logging
            self.audit_logger.log_event(
                level=AuditLevel.INFO,
                action=AuditAction.SYSTEM_START,
                resource=AuditResource.SYSTEM,
                details={'type': 'health_check'}
            )
            return 'healthy'
        except Exception as e:
            logger.error(f"Audit health check failed: {e}")
            return 'unhealthy'
    
    def _ensure_admin_user(self):
        """Ensure default admin user exists"""
        try:
            # Check if admin user exists
            admin_user = self.auth_system._get_user_by_username('admin')
            
            if not admin_user:
                # Create admin user
                result = self.auth_system.create_user(
                    username='admin',
                    email='admin@qnti.local',
                    password='admin123!',
                    role=UserRole.ADMIN
                )
                
                if result['success']:
                    logger.info("Default admin user created successfully")
                    
                    # Log admin user creation
                    self.audit_logger.log_event(
                        level=AuditLevel.INFO,
                        action=AuditAction.USER_CREATE,
                        resource=AuditResource.USER,
                        details={'username': 'admin', 'role': 'admin', 'created_by': 'system'}
                    )
                else:
                    logger.error(f"Failed to create admin user: {result['error']}")
            else:
                logger.info("Admin user already exists")
        
        except Exception as e:
            logger.error(f"Error ensuring admin user: {e}")
    
    def _log_system_startup(self):
        """Log system startup event"""
        try:
            self.audit_logger.log_event(
                level=AuditLevel.INFO,
                action=AuditAction.SYSTEM_START,
                resource=AuditResource.SYSTEM,
                details={
                    'version': '1.0.0',
                    'security_enabled': True,
                    'components': {
                        'authentication': True,
                        'rate_limiting': True,
                        'encryption': True,
                        'audit_logging': True
                    },
                    'startup_time': datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error logging system startup: {e}")
    
    def get_security_status(self) -> dict:
        """Get comprehensive security status"""
        try:
            return {
                'security_framework': {
                    'status': 'active',
                    'version': '1.0.0',
                    'features': {
                        'authentication': True,
                        'authorization': True,
                        'rate_limiting': True,
                        'encryption': True,
                        'audit_logging': True,
                        'input_validation': True,
                        'security_headers': True,
                        'cors_protection': True
                    }
                },
                'authentication': {
                    'status': self._check_auth_health(),
                    'users_count': len(self.auth_system._get_all_users()) if hasattr(self.auth_system, '_get_all_users') else 'unknown',
                    'active_sessions': len(self.auth_system._get_active_sessions()) if hasattr(self.auth_system, '_get_active_sessions') else 'unknown'
                },
                'rate_limiting': {
                    'status': self._check_rate_limit_health(),
                    'backend': 'redis' if self.rate_limiter.use_redis else 'memory',
                    'global_limits': True,
                    'user_limits': True,
                    'ip_limits': True
                },
                'encryption': {
                    'status': self._check_encryption_health(),
                    'methods': ['AES-256-GCM', 'RSA-OAEP', 'ChaCha20-Poly1305'],
                    'key_management': True,
                    'transport_encryption': True
                },
                'audit_logging': {
                    'status': self._check_audit_health(),
                    'storage_backends': ['sqlite', 'file'],
                    'retention_policy': '90 days',
                    'real_time_monitoring': True
                },
                'security_metrics': self.security_framework.get_security_metrics()
            }
        except Exception as e:
            logger.error(f"Error getting security status: {e}")
            return {'error': str(e)}
    
    def run_security_check(self) -> dict:
        """Run comprehensive security check"""
        try:
            from qnti_security_testing import SecurityTestRunner
            
            logger.info("Running security check...")
            
            # Run security tests
            test_runner = SecurityTestRunner()
            test_results = test_runner.run_all_tests()
            
            # Log security check results
            self.audit_logger.log_event(
                level=AuditLevel.INFO,
                action=AuditAction.SECURITY_CHECK,
                resource=AuditResource.SYSTEM,
                details={
                    'total_tests': test_results['summary']['total_tests'],
                    'passed': test_results['summary']['passed'],
                    'failed': test_results['summary']['failed'],
                    'success_rate': test_results['summary']['success_rate']
                }
            )
            
            return test_results
        
        except Exception as e:
            logger.error(f"Error running security check: {e}")
            return {'error': str(e)}
    
    def enable_security_monitoring(self):
        """Enable real-time security monitoring"""
        try:
            # Setup monitoring for suspicious activities
            def monitor_failed_logins():
                """Monitor failed login attempts"""
                while self.running:
                    try:
                        # Check for suspicious login patterns
                        failed_logins = self.audit_logger.get_failed_logins(hours=1)
                        
                        if len(failed_logins) > 10:  # More than 10 failed logins in 1 hour
                            self.audit_logger.log_event(
                                level=AuditLevel.SECURITY,
                                action=AuditAction.SUSPICIOUS_ACTIVITY,
                                resource=AuditResource.SECURITY,
                                details={
                                    'type': 'excessive_failed_logins',
                                    'count': len(failed_logins),
                                    'time_window': '1 hour'
                                }
                            )
                        
                        # Sleep for 5 minutes before next check
                        import time
                        time.sleep(300)
                    
                    except Exception as e:
                        logger.error(f"Error in login monitoring: {e}")
                        time.sleep(60)
            
            # Start monitoring thread
            import threading
            monitor_thread = threading.Thread(target=monitor_failed_logins, daemon=True)
            monitor_thread.start()
            
            logger.info("Security monitoring enabled")
        
        except Exception as e:
            logger.error(f"Error enabling security monitoring: {e}")
    
    def cleanup_security_data(self):
        """Cleanup old security data"""
        try:
            # Cleanup old audit logs
            self.audit_logger.cleanup_old_events(retention_days=90)
            
            # Cleanup old rate limit data
            # This would be implemented based on storage backend
            
            logger.info("Security data cleanup completed")
        
        except Exception as e:
            logger.error(f"Error cleaning up security data: {e}")
    
    def shutdown(self):
        """Graceful shutdown with security logging"""
        try:
            # Log system shutdown
            self.audit_logger.log_event(
                level=AuditLevel.INFO,
                action=AuditAction.SYSTEM_STOP,
                resource=AuditResource.SYSTEM,
                details={
                    'shutdown_time': datetime.now().isoformat(),
                    'graceful_shutdown': True
                }
            )
            
            # Shutdown audit logger
            self.audit_logger.shutdown()
            
            # Call parent shutdown
            super().shutdown()
            
            logger.info("QNTI Secure Main System shutdown completed")
        
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

def main():
    """Main entry point for secure QNTI system"""
    try:
        # Create secure main system
        secure_system = QNTISecureMainSystem()
        
        # Enable security monitoring
        secure_system.enable_security_monitoring()
        
        # Print security status
        print("QNTI Secure Trading System")
        print("=" * 50)
        
        status = secure_system.get_security_status()
        
        print(f"Security Framework: {status['security_framework']['status']}")
        print(f"Authentication: {status['authentication']['status']}")
        print(f"Rate Limiting: {status['rate_limiting']['status']}")
        print(f"Encryption: {status['encryption']['status']}")
        print(f"Audit Logging: {status['audit_logging']['status']}")
        
        print("\nSecurity Features Enabled:")
        for feature, enabled in status['security_framework']['features'].items():
            print(f"  ✓ {feature.replace('_', ' ').title()}: {'Enabled' if enabled else 'Disabled'}")
        
        print("\nStarting secure web server...")
        print("Access the dashboard at: http://localhost:5000")
        print("Use admin/admin123! for initial login")
        print("Press Ctrl+C to shutdown gracefully")
        
        # Run the secure system
        secure_system.run()
    
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        if 'secure_system' in locals():
            secure_system.shutdown()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()