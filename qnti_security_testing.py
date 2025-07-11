#!/usr/bin/env python3
"""
QNTI Security Testing and Validation Framework
Comprehensive security testing suite for the QNTI trading system
"""

import os
import json
import time
import logging
import hashlib
import secrets
import requests
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import unittest
from unittest.mock import Mock, patch
import concurrent.futures
from urllib.parse import urljoin

# Import security components for testing
from qnti_security_framework import QNTISecurityFramework, SecurityConfig
from qnti_auth_system import QNTIAuthSystem, UserRole, Permission
from qnti_rate_limiter import QNTIRateLimiter, RateLimitScope, RateLimit, RateLimitType
from qnti_encryption import QNTIEncryption, EncryptionMethod
from qnti_audit_logger import QNTIAuditLogger, AuditLevel, AuditAction, AuditResource

logger = logging.getLogger('QNTI_SECURITY_TEST')

class SecurityTestResult(Enum):
    """Security test result status"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIP = "SKIP"

@dataclass
class TestResult:
    """Test result data structure"""
    name: str
    status: SecurityTestResult
    message: str
    details: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class SecurityTestSuite:
    """Base class for security test suites"""
    
    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []
        self.setup_completed = False
    
    def setup(self):
        """Setup test environment"""
        try:
            # Initialize test environment
            self.test_start_time = time.time()
            
            # Create test data directory
            os.makedirs('test_data', exist_ok=True)
            
            # Initialize test database/files if needed
            self.test_files = []
            
            # Setup test server endpoints
            self.test_endpoints = {
                'local': 'http://localhost:5000',
                'test_api': 'http://localhost:5001'
            }
            
            # Setup test user credentials
            self.test_credentials = {
                'username': 'test_user',
                'password': 'test_pass_123!',
                'email': 'test@example.com'
            }
            
            # Initialize security test parameters
            self.security_config = {
                'max_attempts': 5,
                'lockout_duration': 300,  # 5 minutes
                'session_timeout': 1800,  # 30 minutes
                'password_min_length': 8
            }
            
            # Create test log file
            self.test_log_file = f'test_data/security_test_{int(time.time())}.log'
            logging.basicConfig(
                filename=self.test_log_file,
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            
            logger.info("Security test environment setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up test environment: {e}")
            raise
    
    def teardown(self):
        """Cleanup test environment"""
        try:
            # Clean up test files
            for test_file in getattr(self, 'test_files', []):
                try:
                    if os.path.exists(test_file):
                        os.remove(test_file)
                        logger.info(f"Removed test file: {test_file}")
                except Exception as e:
                    logger.warning(f"Could not remove test file {test_file}: {e}")
            
            # Clean up test data directory if empty
            try:
                if os.path.exists('test_data') and not os.listdir('test_data'):
                    os.rmdir('test_data')
                    logger.info("Removed empty test_data directory")
            except Exception as e:
                logger.warning(f"Could not remove test_data directory: {e}")
            
            # Reset test state
            self.setup_completed = False
            
            # Log test completion
            if hasattr(self, 'test_start_time'):
                test_duration = time.time() - self.test_start_time
                logger.info(f"Security testing completed in {test_duration:.2f} seconds")
            
            # Close any open connections
            if hasattr(self, 'test_session'):
                try:
                    self.test_session.close()
                except:
                    pass
            
            logger.info("Security test environment cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during test environment teardown: {e}")
    
    def run_test(self, test_func, test_name: str) -> TestResult:
        """Run a single test with timing and error handling"""
        start_time = time.time()
        
        try:
            if not self.setup_completed:
                self.setup()
                self.setup_completed = True
            
            result = test_func()
            execution_time = time.time() - start_time
            
            if isinstance(result, TestResult):
                result.execution_time = execution_time
                return result
            else:
                return TestResult(
                    name=test_name,
                    status=SecurityTestResult.PASS,
                    message=str(result) if result else "Test passed",
                    execution_time=execution_time
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                name=test_name,
                status=SecurityTestResult.FAIL,
                message=f"Test failed: {str(e)}",
                execution_time=execution_time
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all tests in the suite"""
        test_methods = [method for method in dir(self) if method.startswith('test_')]
        
        for method_name in test_methods:
            test_func = getattr(self, method_name)
            test_name = method_name.replace('test_', '').replace('_', ' ').title()
            
            result = self.run_test(test_func, test_name)
            self.results.append(result)
            
            logger.info(f"Test '{test_name}': {result.status.value} - {result.message}")
        
        self.teardown()
        return self.results

class AuthenticationTestSuite(SecurityTestSuite):
    """Authentication system test suite"""
    
    def __init__(self):
        super().__init__("Authentication")
        self.auth_system = None
        self.test_user_id = None
    
    def setup(self):
        """Setup authentication test environment"""
        self.auth_system = QNTIAuthSystem(db_path="test_auth.db")
        
        # Create test user
        result = self.auth_system.create_user(
            username="testuser",
            email="test@example.com",
            password="TestPass123!",
            role=UserRole.TRADER
        )
        
        if result['success']:
            self.test_user_id = result['user_id']
        else:
            raise Exception(f"Failed to create test user: {result['error']}")
    
    def teardown(self):
        """Cleanup authentication test environment"""
        if os.path.exists("test_auth.db"):
            os.remove("test_auth.db")
    
    def test_user_creation(self) -> TestResult:
        """Test user creation functionality"""
        result = self.auth_system.create_user(
            username="newuser",
            email="newuser@example.com",
            password="NewPass123!",
            role=UserRole.VIEWER
        )
        
        if result['success']:
            return TestResult(
                name="User Creation",
                status=SecurityTestResult.PASS,
                message="User created successfully",
                details={"user_id": result['user_id']}
            )
        else:
            return TestResult(
                name="User Creation",
                status=SecurityTestResult.FAIL,
                message=f"Failed to create user: {result['error']}"
            )
    
    def test_authentication_success(self) -> TestResult:
        """Test successful authentication"""
        result = self.auth_system.authenticate(
            username="testuser",
            password="TestPass123!",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0"
        )
        
        if result['success']:
            return TestResult(
                name="Authentication Success",
                status=SecurityTestResult.PASS,
                message="Authentication successful",
                details={
                    "access_token_length": len(result['access_token']),
                    "refresh_token_length": len(result['refresh_token']),
                    "user_role": result['user']['role']
                }
            )
        else:
            return TestResult(
                name="Authentication Success",
                status=SecurityTestResult.FAIL,
                message=f"Authentication failed: {result['error']}"
            )
    
    def test_authentication_failure(self) -> TestResult:
        """Test authentication failure with wrong password"""
        result = self.auth_system.authenticate(
            username="testuser",
            password="WrongPassword",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0"
        )
        
        if not result['success']:
            return TestResult(
                name="Authentication Failure",
                status=SecurityTestResult.PASS,
                message="Authentication correctly failed with wrong password"
            )
        else:
            return TestResult(
                name="Authentication Failure",
                status=SecurityTestResult.FAIL,
                message="Authentication should have failed with wrong password"
            )
    
    def test_brute_force_protection(self) -> TestResult:
        """Test brute force protection"""
        # Attempt multiple failed logins
        for i in range(6):  # Exceed max attempts
            self.auth_system.authenticate(
                username="testuser",
                password="WrongPassword",
                ip_address="192.168.1.100",
                user_agent="BruteForceAgent/1.0"
            )
        
        # Try legitimate login after lockout
        result = self.auth_system.authenticate(
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.100",
            user_agent="BruteForceAgent/1.0"
        )
        
        if not result['success'] and "locked" in result['error'].lower():
            return TestResult(
                name="Brute Force Protection",
                status=SecurityTestResult.PASS,
                message="Account correctly locked after multiple failed attempts"
            )
        else:
            return TestResult(
                name="Brute Force Protection",
                status=SecurityTestResult.FAIL,
                message="Brute force protection not working properly"
            )
    
    def test_jwt_token_validation(self) -> TestResult:
        """Test JWT token validation"""
        # Get valid token
        auth_result = self.auth_system.authenticate(
            username="testuser",
            password="TestPass123!",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0"
        )
        
        if not auth_result['success']:
            return TestResult(
                name="JWT Token Validation",
                status=SecurityTestResult.FAIL,
                message="Failed to get authentication token"
            )
        
        # Validate token
        token = auth_result['access_token']
        validation_result = self.auth_system.verify_token(token)
        
        if validation_result['valid']:
            return TestResult(
                name="JWT Token Validation",
                status=SecurityTestResult.PASS,
                message="JWT token validation successful",
                details={
                    "user_id": validation_result['payload']['user_id'],
                    "role": validation_result['payload']['role']
                }
            )
        else:
            return TestResult(
                name="JWT Token Validation",
                status=SecurityTestResult.FAIL,
                message=f"JWT token validation failed: {validation_result['error']}"
            )
    
    def test_mfa_functionality(self) -> TestResult:
        """Test multi-factor authentication"""
        # Enable MFA
        mfa_result = self.auth_system.enable_mfa(self.test_user_id)
        
        if not mfa_result['success']:
            return TestResult(
                name="MFA Functionality",
                status=SecurityTestResult.FAIL,
                message=f"Failed to enable MFA: {mfa_result['error']}"
            )
        
        # Test MFA secret generation
        secret = mfa_result['secret']
        qr_code = mfa_result['qr_code']
        
        if secret and qr_code:
            return TestResult(
                name="MFA Functionality",
                status=SecurityTestResult.PASS,
                message="MFA enabled successfully",
                details={
                    "secret_length": len(secret),
                    "qr_code_length": len(qr_code)
                }
            )
        else:
            return TestResult(
                name="MFA Functionality",
                status=SecurityTestResult.FAIL,
                message="MFA secret or QR code generation failed"
            )

class RateLimitingTestSuite(SecurityTestSuite):
    """Rate limiting test suite"""
    
    def __init__(self):
        super().__init__("Rate Limiting")
        self.rate_limiter = None
    
    def setup(self):
        """Setup rate limiting test environment"""
        self.rate_limiter = QNTIRateLimiter(use_redis=False)  # Use memory backend for testing
    
    def test_rate_limit_basic(self) -> TestResult:
        """Test basic rate limiting functionality"""
        # Create a strict rate limit
        rate_limit = RateLimit(
            limit=3,
            window=60,
            scope=RateLimitScope.IP,
            limit_type=RateLimitType.REQUESTS_PER_MINUTE
        )
        
        identifier = "192.168.1.100"
        
        # Test requests within limit
        results = []
        for i in range(3):
            result = self.rate_limiter.memory_limiter.check_rate_limit(identifier, rate_limit)
            results.append(result.allowed)
        
        # Test request exceeding limit
        result = self.rate_limiter.memory_limiter.check_rate_limit(identifier, rate_limit)
        limit_exceeded = not result.allowed
        
        if all(results) and limit_exceeded:
            return TestResult(
                name="Rate Limit Basic",
                status=SecurityTestResult.PASS,
                message="Basic rate limiting working correctly",
                details={
                    "allowed_requests": 3,
                    "limit_exceeded": limit_exceeded,
                    "remaining": result.remaining
                }
            )
        else:
            return TestResult(
                name="Rate Limit Basic",
                status=SecurityTestResult.FAIL,
                message="Basic rate limiting not working correctly"
            )
    
    def test_rate_limit_scopes(self) -> TestResult:
        """Test different rate limit scopes"""
        rate_limit = RateLimit(
            limit=2,
            window=60,
            scope=RateLimitScope.USER,
            limit_type=RateLimitType.REQUESTS_PER_MINUTE
        )
        
        # Test different users
        user1_result = self.rate_limiter.memory_limiter.check_rate_limit("user1", rate_limit)
        user2_result = self.rate_limiter.memory_limiter.check_rate_limit("user2", rate_limit)
        
        if user1_result.allowed and user2_result.allowed:
            return TestResult(
                name="Rate Limit Scopes",
                status=SecurityTestResult.PASS,
                message="Rate limit scopes working correctly",
                details={
                    "user1_allowed": user1_result.allowed,
                    "user2_allowed": user2_result.allowed
                }
            )
        else:
            return TestResult(
                name="Rate Limit Scopes",
                status=SecurityTestResult.FAIL,
                message="Rate limit scopes not working correctly"
            )
    
    def test_rate_limit_reset(self) -> TestResult:
        """Test rate limit reset functionality"""
        rate_limit = RateLimit(
            limit=1,
            window=1,  # 1 second window
            scope=RateLimitScope.IP,
            limit_type=RateLimitType.REQUESTS_PER_MINUTE
        )
        
        identifier = "192.168.1.101"
        
        # Use up the limit
        result1 = self.rate_limiter.memory_limiter.check_rate_limit(identifier, rate_limit)
        result2 = self.rate_limiter.memory_limiter.check_rate_limit(identifier, rate_limit)
        
        # Wait for reset
        time.sleep(2)
        
        # Try again after reset
        result3 = self.rate_limiter.memory_limiter.check_rate_limit(identifier, rate_limit)
        
        if result1.allowed and not result2.allowed and result3.allowed:
            return TestResult(
                name="Rate Limit Reset",
                status=SecurityTestResult.PASS,
                message="Rate limit reset working correctly"
            )
        else:
            return TestResult(
                name="Rate Limit Reset",
                status=SecurityTestResult.FAIL,
                message="Rate limit reset not working correctly"
            )

class EncryptionTestSuite(SecurityTestSuite):
    """Encryption test suite"""
    
    def __init__(self):
        super().__init__("Encryption")
        self.encryption_system = None
    
    def setup(self):
        """Setup encryption test environment"""
        self.encryption_system = QNTIEncryption()
    
    def test_aes_encryption(self) -> TestResult:
        """Test AES encryption/decryption"""
        test_data = "This is sensitive trading data"
        
        # Encrypt data
        encrypted = self.encryption_system.encrypt(test_data, EncryptionMethod.AES_256_GCM)
        
        # Decrypt data
        decrypted = self.encryption_system.decrypt(encrypted)
        decrypted_text = decrypted.decode('utf-8')
        
        if decrypted_text == test_data:
            return TestResult(
                name="AES Encryption",
                status=SecurityTestResult.PASS,
                message="AES encryption/decryption successful",
                details={
                    "original_length": len(test_data),
                    "encrypted_length": len(encrypted.ciphertext),
                    "method": encrypted.method.value
                }
            )
        else:
            return TestResult(
                name="AES Encryption",
                status=SecurityTestResult.FAIL,
                message="AES encryption/decryption failed"
            )
    
    def test_rsa_encryption(self) -> TestResult:
        """Test RSA encryption/decryption"""
        test_data = "API Key: qnti_test_key_123456"
        
        # Encrypt data
        encrypted = self.encryption_system.encrypt(test_data, EncryptionMethod.RSA_OAEP)
        
        # Decrypt data
        decrypted = self.encryption_system.decrypt(encrypted)
        decrypted_text = decrypted.decode('utf-8')
        
        if decrypted_text == test_data:
            return TestResult(
                name="RSA Encryption",
                status=SecurityTestResult.PASS,
                message="RSA encryption/decryption successful",
                details={
                    "original_length": len(test_data),
                    "encrypted_length": len(encrypted.ciphertext),
                    "method": encrypted.method.value
                }
            )
        else:
            return TestResult(
                name="RSA Encryption",
                status=SecurityTestResult.FAIL,
                message="RSA encryption/decryption failed"
            )
    
    def test_key_derivation(self) -> TestResult:
        """Test key derivation functionality"""
        password = "user_password_123"
        salt = secrets.token_bytes(32)
        
        # Derive key
        key1 = self.encryption_system.derive_key(password, salt)
        key2 = self.encryption_system.derive_key(password, salt)
        
        # Different salt should produce different key
        different_salt = secrets.token_bytes(32)
        key3 = self.encryption_system.derive_key(password, different_salt)
        
        if key1 == key2 and key1 != key3:
            return TestResult(
                name="Key Derivation",
                status=SecurityTestResult.PASS,
                message="Key derivation working correctly",
                details={
                    "key_length": len(key1),
                    "same_salt_same_key": key1 == key2,
                    "different_salt_different_key": key1 != key3
                }
            )
        else:
            return TestResult(
                name="Key Derivation",
                status=SecurityTestResult.FAIL,
                message="Key derivation not working correctly"
            )
    
    def test_data_serialization(self) -> TestResult:
        """Test encrypted data serialization"""
        test_data = {"account_balance": 50000.00, "equity": 49850.00}
        
        # Encrypt data
        encrypted = self.encryption_system.encrypt(test_data, EncryptionMethod.AES_256_GCM)
        
        # Serialize encrypted data
        serialized = self.encryption_system.serialize_encrypted_data(encrypted)
        
        # Deserialize encrypted data
        deserialized = self.encryption_system.deserialize_encrypted_data(serialized)
        
        # Decrypt data
        decrypted = self.encryption_system.decrypt(deserialized)
        decrypted_data = json.loads(decrypted.decode('utf-8'))
        
        if decrypted_data == test_data:
            return TestResult(
                name="Data Serialization",
                status=SecurityTestResult.PASS,
                message="Data serialization working correctly",
                details={
                    "serialized_length": len(serialized),
                    "deserialized_method": deserialized.method.value
                }
            )
        else:
            return TestResult(
                name="Data Serialization",
                status=SecurityTestResult.FAIL,
                message="Data serialization not working correctly"
            )

class InputValidationTestSuite(SecurityTestSuite):
    """Input validation test suite"""
    
    def __init__(self):
        super().__init__("Input Validation")
        self.security_framework = None
    
    def setup(self):
        """Setup input validation test environment"""
        self.security_framework = QNTISecurityFramework()
    
    def test_sql_injection_detection(self) -> TestResult:
        """Test SQL injection detection"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM passwords",
            "admin'/*",
            "1; INSERT INTO users VALUES('hacker', 'password')"
        ]
        
        detected_count = 0
        for malicious_input in malicious_inputs:
            try:
                self.security_framework.input_validator._sanitize_string(malicious_input)
            except Exception:
                detected_count += 1
        
        if detected_count == len(malicious_inputs):
            return TestResult(
                name="SQL Injection Detection",
                status=SecurityTestResult.PASS,
                message="All SQL injection attempts detected",
                details={"detected": detected_count, "total": len(malicious_inputs)}
            )
        else:
            return TestResult(
                name="SQL Injection Detection",
                status=SecurityTestResult.FAIL,
                message=f"Only {detected_count}/{len(malicious_inputs)} SQL injection attempts detected"
            )
    
    def test_xss_detection(self) -> TestResult:
        """Test XSS detection"""
        malicious_inputs = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<iframe src='javascript:alert(\"XSS\")'></iframe>",
            "onload=alert('XSS')"
        ]
        
        detected_count = 0
        for malicious_input in malicious_inputs:
            try:
                self.security_framework.input_validator._sanitize_string(malicious_input)
            except Exception:
                detected_count += 1
        
        if detected_count == len(malicious_inputs):
            return TestResult(
                name="XSS Detection",
                status=SecurityTestResult.PASS,
                message="All XSS attempts detected",
                details={"detected": detected_count, "total": len(malicious_inputs)}
            )
        else:
            return TestResult(
                name="XSS Detection",
                status=SecurityTestResult.FAIL,
                message=f"Only {detected_count}/{len(malicious_inputs)} XSS attempts detected"
            )
    
    def test_input_sanitization(self) -> TestResult:
        """Test input sanitization"""
        test_input = "<script>alert('test')</script>Hello World"
        
        sanitized = self.security_framework.input_validator._sanitize_string(test_input)
        
        if "<script>" not in sanitized and "Hello World" in sanitized:
            return TestResult(
                name="Input Sanitization",
                status=SecurityTestResult.PASS,
                message="Input sanitization working correctly",
                details={"original": test_input, "sanitized": sanitized}
            )
        else:
            return TestResult(
                name="Input Sanitization",
                status=SecurityTestResult.FAIL,
                message="Input sanitization not working correctly"
            )

class SecurityTestRunner:
    """Main security test runner"""
    
    def __init__(self):
        self.test_suites = [
            AuthenticationTestSuite(),
            RateLimitingTestSuite(),
            EncryptionTestSuite(),
            InputValidationTestSuite()
        ]
        self.results: List[TestResult] = []
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all security test suites"""
        start_time = time.time()
        
        print("=" * 60)
        print("QNTI Security Test Suite")
        print("=" * 60)
        
        suite_results = {}
        
        for suite in self.test_suites:
            print(f"\nRunning {suite.name} Tests...")
            print("-" * 40)
            
            suite_results[suite.name] = suite.run_all_tests()
            self.results.extend(suite_results[suite.name])
            
            # Print suite summary
            passed = sum(1 for r in suite_results[suite.name] if r.status == SecurityTestResult.PASS)
            failed = sum(1 for r in suite_results[suite.name] if r.status == SecurityTestResult.FAIL)
            warnings = sum(1 for r in suite_results[suite.name] if r.status == SecurityTestResult.WARNING)
            
            print(f"Suite Summary: {passed} passed, {failed} failed, {warnings} warnings")
        
        total_time = time.time() - start_time
        
        # Generate final report
        report = self._generate_report(suite_results, total_time)
        
        print("\n" + "=" * 60)
        print("FINAL REPORT")
        print("=" * 60)
        print(f"Total Tests: {len(self.results)}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Warnings: {report['summary']['warnings']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"Total Time: {total_time:.2f} seconds")
        
        if report['summary']['failed'] > 0:
            print("\nFAILED TESTS:")
            for result in self.results:
                if result.status == SecurityTestResult.FAIL:
                    print(f"  ❌ {result.name}: {result.message}")
        
        return report
    
    def _generate_report(self, suite_results: Dict[str, List[TestResult]], total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.results)
        passed = sum(1 for r in self.results if r.status == SecurityTestResult.PASS)
        failed = sum(1 for r in self.results if r.status == SecurityTestResult.FAIL)
        warnings = sum(1 for r in self.results if r.status == SecurityTestResult.WARNING)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "success_rate": (passed / total_tests * 100) if total_tests > 0 else 0,
                "total_time": total_time
            },
            "suite_results": {},
            "recommendations": []
        }
        
        # Add suite-specific results
        for suite_name, results in suite_results.items():
            suite_passed = sum(1 for r in results if r.status == SecurityTestResult.PASS)
            suite_failed = sum(1 for r in results if r.status == SecurityTestResult.FAIL)
            suite_warnings = sum(1 for r in results if r.status == SecurityTestResult.WARNING)
            
            report["suite_results"][suite_name] = {
                "passed": suite_passed,
                "failed": suite_failed,
                "warnings": suite_warnings,
                "success_rate": (suite_passed / len(results) * 100) if results else 0,
                "tests": [
                    {
                        "name": r.name,
                        "status": r.status.value,
                        "message": r.message,
                        "execution_time": r.execution_time,
                        "details": r.details
                    } for r in results
                ]
            }
        
        # Add security recommendations
        if failed > 0:
            report["recommendations"].append(
                "Address failed security tests immediately before deploying to production"
            )
        
        if warnings > 0:
            report["recommendations"].append(
                "Review warning messages and consider implementing additional security measures"
            )
        
        report["recommendations"].extend([
            "Regularly run security tests as part of CI/CD pipeline",
            "Keep security dependencies updated",
            "Monitor security logs for suspicious activity",
            "Conduct periodic security audits"
        ])
        
        return report
    
    def save_report(self, filename: str = "security_test_report.json"):
        """Save test report to file"""
        report = self._generate_report({}, 0)
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Test report saved to {filename}")

class PenetrationTestSuite(SecurityTestSuite):
    """Penetration testing suite for API endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        super().__init__("Penetration Testing")
        self.base_url = base_url
        self.session = requests.Session()
        self.access_token = None
    
    def setup(self):
        """Setup penetration testing environment"""
        # Try to get authentication token
        try:
            response = self.session.post(
                urljoin(self.base_url, "/api/auth/login"),
                json={"username": "admin", "password": "admin123!"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    self.access_token = data['data']['access_token']
                    self.session.headers.update({
                        'Authorization': f'Bearer {self.access_token}'
                    })
        except Exception as e:
            logger.warning(f"Could not authenticate for penetration testing: {e}")
    
    def test_unauthorized_access(self) -> TestResult:
        """Test unauthorized access to protected endpoints"""
        protected_endpoints = [
            "/api/account/info",
            "/api/trades/active",
            "/api/ea/list",
            "/api/admin/system/health"
        ]
        
        unauthorized_count = 0
        
        # Create session without authentication
        unauth_session = requests.Session()
        
        for endpoint in protected_endpoints:
            try:
                response = unauth_session.get(urljoin(self.base_url, endpoint))
                if response.status_code == 401:
                    unauthorized_count += 1
            except Exception:
                pass
        
        if unauthorized_count == len(protected_endpoints):
            return TestResult(
                name="Unauthorized Access",
                status=SecurityTestResult.PASS,
                message="All protected endpoints correctly deny unauthorized access",
                details={"protected_endpoints": len(protected_endpoints)}
            )
        else:
            return TestResult(
                name="Unauthorized Access",
                status=SecurityTestResult.FAIL,
                message=f"Only {unauthorized_count}/{len(protected_endpoints)} endpoints deny unauthorized access"
            )
    
    def test_rate_limiting_enforcement(self) -> TestResult:
        """Test rate limiting enforcement"""
        endpoint = "/api/auth/login"
        
        # Send multiple requests rapidly
        responses = []
        for i in range(10):
            try:
                response = self.session.post(
                    urljoin(self.base_url, endpoint),
                    json={"username": "test", "password": "test"}
                )
                responses.append(response.status_code)
            except Exception:
                pass
        
        # Check if any requests were rate limited (429 status)
        rate_limited = any(status == 429 for status in responses)
        
        if rate_limited:
            return TestResult(
                name="Rate Limiting Enforcement",
                status=SecurityTestResult.PASS,
                message="Rate limiting is enforced",
                details={"responses": responses}
            )
        else:
            return TestResult(
                name="Rate Limiting Enforcement",
                status=SecurityTestResult.WARNING,
                message="Rate limiting may not be properly enforced",
                details={"responses": responses}
            )
    
    def test_sql_injection_endpoints(self) -> TestResult:
        """Test SQL injection on API endpoints"""
        sql_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/*",
            "1; SELECT * FROM passwords"
        ]
        
        vulnerable_endpoints = 0
        
        for payload in sql_payloads:
            try:
                # Test on various endpoints
                endpoints = [
                    f"/api/ea/performance/{payload}",
                    f"/api/trades/history?symbol={payload}",
                ]
                
                for endpoint in endpoints:
                    response = self.session.get(urljoin(self.base_url, endpoint))
                    
                    # Check for SQL error messages in response
                    if response.status_code == 500:
                        response_text = response.text.lower()
                        if any(error in response_text for error in ['sql', 'database', 'syntax']):
                            vulnerable_endpoints += 1
                            
            except Exception:
                pass
        
        if vulnerable_endpoints == 0:
            return TestResult(
                name="SQL Injection Endpoints",
                status=SecurityTestResult.PASS,
                message="No SQL injection vulnerabilities found",
                details={"payloads_tested": len(sql_payloads)}
            )
        else:
            return TestResult(
                name="SQL Injection Endpoints",
                status=SecurityTestResult.FAIL,
                message=f"Found {vulnerable_endpoints} potentially vulnerable endpoints",
                details={"vulnerable_endpoints": vulnerable_endpoints}
            )

# Main execution
if __name__ == "__main__":
    runner = SecurityTestRunner()
    report = runner.run_all_tests()
    runner.save_report()
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if report['summary']['failed'] == 0 else 1)