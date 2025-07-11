#!/usr/bin/env python3
"""
QNTI Security API Documentation Generator
Generates comprehensive API documentation with security specifications
"""

import json
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from qnti_auth_system import Permission, UserRole
from qnti_rate_limiter import RateLimitScope

@dataclass
class SecurityRequirement:
    """Security requirement specification"""
    authentication: bool = True
    permissions: List[str] = None
    rate_limit: Dict[str, Any] = None
    encryption: bool = False
    audit_logging: bool = True
    input_validation: bool = True

@dataclass
class APIEndpoint:
    """API endpoint specification"""
    path: str
    method: str
    summary: str
    description: str
    security: SecurityRequirement
    parameters: List[Dict[str, Any]] = None
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = None
    examples: List[Dict[str, Any]] = None

class QNTISecurityAPIDocumentationGenerator:
    """Generate comprehensive API documentation with security specifications"""
    
    def __init__(self):
        self.api_version = "1.0.0"
        self.api_title = "QNTI Trading System API"
        self.api_description = "Comprehensive API for the Quantum Nexus Trading Intelligence system with enterprise-grade security"
        self.server_url = "https://api.qnti.trading"
        
        # Define all API endpoints
        self.endpoints = self._define_endpoints()
    
    def _define_endpoints(self) -> List[APIEndpoint]:
        """Define all API endpoints with their security specifications"""
        return [
            # Authentication endpoints
            APIEndpoint(
                path="/api/auth/login",
                method="POST",
                summary="User authentication",
                description="Authenticate user with username/password and optional MFA",
                security=SecurityRequirement(
                    authentication=False,
                    rate_limit={"scope": "IP", "limit": 5, "window": 300},
                    audit_logging=True
                ),
                request_body={
                    "type": "object",
                    "properties": {
                        "username": {"type": "string", "description": "Username"},
                        "password": {"type": "string", "description": "Password", "format": "password"},
                        "mfa_code": {"type": "string", "description": "MFA code (if enabled)", "pattern": "^[0-9]{6}$"}
                    },
                    "required": ["username", "password"]
                },
                responses={
                    "200": {
                        "description": "Login successful",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {"type": "boolean"},
                                        "data": {
                                            "type": "object",
                                            "properties": {
                                                "access_token": {"type": "string"},
                                                "refresh_token": {"type": "string"},
                                                "user": {"$ref": "#/components/schemas/User"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "401": {"description": "Invalid credentials"},
                    "429": {"description": "Rate limit exceeded"}
                }
            ),
            
            # Account endpoints
            APIEndpoint(
                path="/api/account/info",
                method="GET",
                summary="Get account information",
                description="Retrieve account balance, equity, and margin information",
                security=SecurityRequirement(
                    authentication=True,
                    permissions=[Permission.ACCOUNT_VIEW.value],
                    rate_limit={"scope": "USER", "limit": 100, "window": 60},
                    encryption=True,
                    audit_logging=True
                ),
                responses={
                    "200": {
                        "description": "Account information retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {"type": "boolean"},
                                        "data": {"$ref": "#/components/schemas/AccountInfo"}
                                    }
                                }
                            }
                        }
                    },
                    "401": {"description": "Authentication required"},
                    "403": {"description": "Insufficient permissions"}
                }
            ),
            
            # Trading endpoints
            APIEndpoint(
                path="/api/trade/execute",
                method="POST",
                summary="Execute trade",
                description="Execute a new trade order",
                security=SecurityRequirement(
                    authentication=True,
                    permissions=[Permission.TRADE_EXECUTE.value],
                    rate_limit={"scope": "USER", "limit": 5, "window": 60},
                    encryption=True,
                    audit_logging=True,
                    input_validation=True
                ),
                request_body={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Trading symbol", "pattern": "^[A-Z]{6}$"},
                        "volume": {"type": "number", "description": "Trade volume", "minimum": 0.01, "maximum": 100},
                        "type": {"type": "string", "enum": ["BUY", "SELL"], "description": "Trade type"},
                        "price": {"type": "number", "description": "Entry price (for pending orders)"},
                        "sl": {"type": "number", "description": "Stop loss price"},
                        "tp": {"type": "number", "description": "Take profit price"},
                        "comment": {"type": "string", "description": "Trade comment", "maxLength": 50}
                    },
                    "required": ["symbol", "volume", "type"]
                },
                responses={
                    "200": {
                        "description": "Trade executed successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {"type": "boolean"},
                                        "data": {"$ref": "#/components/schemas/TradeResult"}
                                    }
                                }
                            }
                        }
                    },
                    "400": {"description": "Invalid trade parameters"},
                    "401": {"description": "Authentication required"},
                    "403": {"description": "Insufficient permissions"},
                    "429": {"description": "Rate limit exceeded"}
                }
            ),
            
            # EA Management endpoints
            APIEndpoint(
                path="/api/ea/list",
                method="GET",
                summary="List Expert Advisors",
                description="Get list of all Expert Advisors with their status",
                security=SecurityRequirement(
                    authentication=True,
                    permissions=[Permission.EA_VIEW.value],
                    rate_limit={"scope": "USER", "limit": 60, "window": 60},
                    audit_logging=True
                ),
                responses={
                    "200": {
                        "description": "EA list retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {"type": "boolean"},
                                        "data": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/ExpertAdvisor"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            ),
            
            # Admin endpoints
            APIEndpoint(
                path="/api/admin/system/health",
                method="GET",
                summary="System health check",
                description="Get system health status and component information",
                security=SecurityRequirement(
                    authentication=True,
                    permissions=[Permission.SYSTEM_HEALTH.value],
                    rate_limit={"scope": "USER", "limit": 30, "window": 60},
                    audit_logging=True
                ),
                responses={
                    "200": {
                        "description": "System health information",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {"type": "boolean"},
                                        "data": {"$ref": "#/components/schemas/SystemHealth"}
                                    }
                                }
                            }
                        }
                    }
                }
            ),
            
            # Security endpoints
            APIEndpoint(
                path="/api/security/audit/events",
                method="GET",
                summary="Get audit events",
                description="Retrieve security audit events",
                security=SecurityRequirement(
                    authentication=True,
                    permissions=[Permission.SYSTEM_LOGS.value],
                    rate_limit={"scope": "USER", "limit": 20, "window": 60},
                    audit_logging=True
                ),
                parameters=[
                    {
                        "name": "hours",
                        "in": "query",
                        "description": "Hours to look back",
                        "schema": {"type": "integer", "minimum": 1, "maximum": 168, "default": 24}
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "description": "Maximum number of events",
                        "schema": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100}
                    }
                ],
                responses={
                    "200": {
                        "description": "Audit events retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {"type": "boolean"},
                                        "data": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/AuditEvent"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            )
        ]
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI 3.0 specification"""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": self.api_title,
                "description": self.api_description,
                "version": self.api_version,
                "contact": {
                    "name": "QNTI Support",
                    "email": "support@qnti.trading"
                },
                "license": {
                    "name": "Proprietary",
                    "url": "https://qnti.trading/license"
                }
            },
            "servers": [
                {
                    "url": self.server_url,
                    "description": "Production server"
                },
                {
                    "url": "https://api-dev.qnti.trading",
                    "description": "Development server"
                }
            ],
            "security": [
                {"BearerAuth": []},
                {"ApiKeyAuth": []}
            ],
            "paths": {},
            "components": {
                "securitySchemes": self._get_security_schemes(),
                "schemas": self._get_schemas(),
                "responses": self._get_common_responses(),
                "parameters": self._get_common_parameters(),
                "headers": self._get_security_headers()
            }
        }
        
        # Add paths
        for endpoint in self.endpoints:
            if endpoint.path not in spec["paths"]:
                spec["paths"][endpoint.path] = {}
            
            spec["paths"][endpoint.path][endpoint.method.lower()] = self._generate_path_item(endpoint)
        
        return spec
    
    def _generate_path_item(self, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Generate OpenAPI path item for endpoint"""
        path_item = {
            "summary": endpoint.summary,
            "description": endpoint.description,
            "operationId": f"{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}",
            "tags": [self._get_tag_from_path(endpoint.path)],
            "security": self._get_endpoint_security(endpoint.security),
            "responses": endpoint.responses or {"200": {"description": "Success"}},
            "x-security-requirements": self._get_security_requirements_extension(endpoint.security)
        }
        
        # Add parameters
        if endpoint.parameters:
            path_item["parameters"] = endpoint.parameters
        
        # Add request body
        if endpoint.request_body:
            path_item["requestBody"] = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": endpoint.request_body
                    }
                }
            }
        
        return path_item
    
    def _get_security_schemes(self) -> Dict[str, Any]:
        """Get security schemes for OpenAPI"""
        return {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token authentication"
            },
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key authentication"
            }
        }
    
    def _get_schemas(self) -> Dict[str, Any]:
        """Get common schemas for OpenAPI"""
        return {
            "User": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "User ID"},
                    "username": {"type": "string", "description": "Username"},
                    "email": {"type": "string", "description": "Email address"},
                    "role": {"type": "string", "enum": [role.value for role in UserRole]},
                    "permissions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "User permissions"
                    }
                }
            },
            "AccountInfo": {
                "type": "object",
                "properties": {
                    "balance": {"type": "number", "description": "Account balance"},
                    "equity": {"type": "number", "description": "Account equity"},
                    "margin": {"type": "number", "description": "Used margin"},
                    "free_margin": {"type": "number", "description": "Free margin"},
                    "margin_level": {"type": "number", "description": "Margin level percentage"},
                    "profit": {"type": "number", "description": "Current profit/loss"}
                }
            },
            "TradeResult": {
                "type": "object",
                "properties": {
                    "ticket": {"type": "integer", "description": "Trade ticket number"},
                    "symbol": {"type": "string", "description": "Trading symbol"},
                    "volume": {"type": "number", "description": "Trade volume"},
                    "type": {"type": "string", "description": "Trade type"},
                    "open_price": {"type": "number", "description": "Opening price"},
                    "open_time": {"type": "string", "format": "date-time", "description": "Opening time"},
                    "profit": {"type": "number", "description": "Current profit/loss"}
                }
            },
            "ExpertAdvisor": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "EA name"},
                    "magic_number": {"type": "integer", "description": "Magic number"},
                    "symbol": {"type": "string", "description": "Trading symbol"},
                    "status": {"type": "string", "enum": ["active", "inactive"], "description": "EA status"},
                    "total_trades": {"type": "integer", "description": "Total number of trades"},
                    "win_rate": {"type": "number", "description": "Win rate percentage"},
                    "total_profit": {"type": "number", "description": "Total profit/loss"},
                    "profit_factor": {"type": "number", "description": "Profit factor"}
                }
            },
            "SystemHealth": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
                    "uptime": {"type": "string", "description": "System uptime"},
                    "components": {
                        "type": "object",
                        "properties": {
                            "trade_manager": {"type": "string", "enum": ["active", "inactive"]},
                            "mt5_bridge": {"type": "string", "enum": ["active", "inactive"]},
                            "vision_analyzer": {"type": "string", "enum": ["active", "inactive"]},
                            "notification_system": {"type": "string", "enum": ["active", "inactive"]}
                        }
                    }
                }
            },
            "AuditEvent": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Event ID"},
                    "timestamp": {"type": "string", "format": "date-time", "description": "Event timestamp"},
                    "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "SECURITY"]},
                    "action": {"type": "string", "description": "Action performed"},
                    "resource": {"type": "string", "description": "Resource affected"},
                    "user_id": {"type": "string", "description": "User ID"},
                    "username": {"type": "string", "description": "Username"},
                    "ip_address": {"type": "string", "description": "IP address"},
                    "success": {"type": "boolean", "description": "Whether action was successful"},
                    "details": {"type": "object", "description": "Additional details"}
                }
            },
            "Error": {
                "type": "object",
                "properties": {
                    "error": {"type": "string", "description": "Error message"},
                    "code": {"type": "string", "description": "Error code"},
                    "timestamp": {"type": "string", "format": "date-time", "description": "Error timestamp"}
                }
            }
        }
    
    def _get_common_responses(self) -> Dict[str, Any]:
        """Get common responses for OpenAPI"""
        return {
            "BadRequest": {
                "description": "Bad request",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "Unauthorized": {
                "description": "Authentication required",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "Forbidden": {
                "description": "Insufficient permissions",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "RateLimitExceeded": {
                "description": "Rate limit exceeded",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                },
                "headers": {
                    "X-RateLimit-Limit": {"$ref": "#/components/headers/X-RateLimit-Limit"},
                    "X-RateLimit-Remaining": {"$ref": "#/components/headers/X-RateLimit-Remaining"},
                    "X-RateLimit-Reset": {"$ref": "#/components/headers/X-RateLimit-Reset"},
                    "Retry-After": {"$ref": "#/components/headers/Retry-After"}
                }
            }
        }
    
    def _get_common_parameters(self) -> Dict[str, Any]:
        """Get common parameters for OpenAPI"""
        return {
            "LimitParam": {
                "name": "limit",
                "in": "query",
                "description": "Maximum number of results",
                "schema": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100}
            },
            "OffsetParam": {
                "name": "offset",
                "in": "query",
                "description": "Number of results to skip",
                "schema": {"type": "integer", "minimum": 0, "default": 0}
            }
        }
    
    def _get_security_headers(self) -> Dict[str, Any]:
        """Get security headers for OpenAPI"""
        return {
            "X-RateLimit-Limit": {
                "description": "Request limit per time window",
                "schema": {"type": "integer"}
            },
            "X-RateLimit-Remaining": {
                "description": "Remaining requests in current window",
                "schema": {"type": "integer"}
            },
            "X-RateLimit-Reset": {
                "description": "Time when rate limit resets",
                "schema": {"type": "integer"}
            },
            "Retry-After": {
                "description": "Number of seconds to wait before retrying",
                "schema": {"type": "integer"}
            },
            "X-Security-Context": {
                "description": "Security context information",
                "schema": {"type": "string"}
            }
        }
    
    def _get_tag_from_path(self, path: str) -> str:
        """Get tag from API path"""
        if path.startswith("/api/auth"):
            return "Authentication"
        elif path.startswith("/api/account"):
            return "Account"
        elif path.startswith("/api/trade"):
            return "Trading"
        elif path.startswith("/api/ea"):
            return "Expert Advisors"
        elif path.startswith("/api/admin"):
            return "Administration"
        elif path.startswith("/api/security"):
            return "Security"
        else:
            return "General"
    
    def _get_endpoint_security(self, security: SecurityRequirement) -> List[Dict[str, List[str]]]:
        """Get security requirements for endpoint"""
        if not security.authentication:
            return []
        
        return [
            {"BearerAuth": security.permissions or []},
            {"ApiKeyAuth": security.permissions or []}
        ]
    
    def _get_security_requirements_extension(self, security: SecurityRequirement) -> Dict[str, Any]:
        """Get security requirements extension"""
        return {
            "authentication": security.authentication,
            "permissions": security.permissions or [],
            "rate_limit": security.rate_limit,
            "encryption": security.encryption,
            "audit_logging": security.audit_logging,
            "input_validation": security.input_validation
        }
    
    def generate_security_guide(self) -> str:
        """Generate security implementation guide"""
        guide = f"""
# QNTI API Security Guide

## Overview
The QNTI Trading System API implements enterprise-grade security measures to protect your trading operations and sensitive data.

## Authentication

### JWT Token Authentication
- **Token Type**: Bearer JWT
- **Header**: `Authorization: Bearer <token>`
- **Expiration**: 1 hour (configurable)
- **Refresh**: Use refresh token to get new access token

### API Key Authentication
- **Header**: `X-API-Key: <api-key>`
- **Format**: `qnti_<32-character-string>`
- **Permissions**: Configurable per key
- **Expiration**: 90 days (configurable)

## Authorization

### Role-Based Access Control (RBAC)
- **Admin**: Full system access
- **Trader**: Trading operations access
- **Viewer**: Read-only access
- **API User**: Programmatic access
- **Guest**: Limited public access

### Permission System
- Fine-grained permissions for each API endpoint
- Permissions are enforced at multiple levels
- Automatic permission checking in decorators

## Rate Limiting

### Rate Limit Scopes
- **Global**: System-wide limits
- **User**: Per-user limits
- **IP**: Per-IP address limits
- **API Key**: Per-API key limits

### Rate Limit Headers
- `X-RateLimit-Limit`: Request limit per time window
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset time (Unix timestamp)
- `Retry-After`: Seconds to wait before retry

### Common Rate Limits
- **Authentication**: 5 requests per 5 minutes per IP
- **Trading**: 5 trades per minute per user
- **General API**: 100 requests per minute per user

## Security Features

### Input Validation
- All inputs are validated and sanitized
- SQL injection prevention
- XSS protection
- File type validation
- Size limits enforcement

### Encryption
- **In Transit**: TLS 1.3 encryption
- **At Rest**: AES-256 encryption
- **Sensitive Data**: Additional encryption layer
- **Key Management**: Automatic key rotation

### Security Headers
- **HSTS**: Strict transport security
- **CSP**: Content security policy
- **X-Frame-Options**: Clickjacking protection
- **X-Content-Type-Options**: MIME type sniffing prevention

### Audit Logging
- All API requests are logged
- Security events are tracked
- Suspicious activity detection
- Compliance reporting

## Best Practices

### Client Implementation
1. **Always use HTTPS** in production
2. **Store tokens securely** (never in localStorage)
3. **Implement token refresh** logic
4. **Handle rate limiting** gracefully
5. **Validate server certificates**

### Security Considerations
1. **API Keys**: Keep them secret and rotate regularly
2. **Permissions**: Request only necessary permissions
3. **Rate Limits**: Implement client-side rate limiting
4. **Error Handling**: Don't expose sensitive information
5. **Logging**: Log security events on client side

### Error Handling
```json
{{
  "error": "Rate limit exceeded",
  "code": "RATE_LIMIT_EXCEEDED",
  "timestamp": "2024-01-01T00:00:00Z",
  "retry_after": 60
}}
```

## Integration Examples

### Authentication Flow
```python
import requests

# Login
response = requests.post('/api/auth/login', json={{
    'username': 'trader',
    'password': 'secure_password'
}})

tokens = response.json()['data']
access_token = tokens['access_token']

# Use token for API calls
headers = {{'Authorization': f'Bearer {{access_token}}'}}
response = requests.get('/api/account/info', headers=headers)
```

### Rate Limit Handling
```python
import time

def make_request_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            time.sleep(retry_after)
            continue
        
        return response
    
    raise Exception("Max retries exceeded")
```

## Monitoring and Alerting

### Security Metrics
- Failed authentication attempts
- Rate limit violations
- Suspicious activity patterns
- API key usage patterns

### Alerts
- Brute force attacks
- Unusual trading patterns
- API abuse detection
- Security policy violations

Generated on: {datetime.now().isoformat()}
"""
        return guide.strip()
    
    def save_documentation(self, output_dir: str = "docs"):
        """Save all documentation files"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save OpenAPI spec
        openapi_spec = self.generate_openapi_spec()
        with open(f"{output_dir}/openapi.json", "w") as f:
            json.dump(openapi_spec, f, indent=2)
        
        with open(f"{output_dir}/openapi.yaml", "w") as f:
            yaml.dump(openapi_spec, f, default_flow_style=False)
        
        # Save security guide
        security_guide = self.generate_security_guide()
        with open(f"{output_dir}/security_guide.md", "w") as f:
            f.write(security_guide)
        
        # Save endpoint summary
        self._save_endpoint_summary(output_dir)
        
        print(f"Documentation saved to {output_dir}/")
    
    def _save_endpoint_summary(self, output_dir: str):
        """Save endpoint summary table"""
        summary = "# QNTI API Endpoint Summary\n\n"
        summary += "| Endpoint | Method | Authentication | Permissions | Rate Limit | Encryption |\n"
        summary += "|----------|---------|---------------|-------------|------------|------------|\n"
        
        for endpoint in self.endpoints:
            auth = "✓" if endpoint.security.authentication else "✗"
            perms = ", ".join(endpoint.security.permissions) if endpoint.security.permissions else "None"
            rate_limit = f"{endpoint.security.rate_limit['limit']}/{endpoint.security.rate_limit['window']}s ({endpoint.security.rate_limit['scope']})" if endpoint.security.rate_limit else "Default"
            encryption = "✓" if endpoint.security.encryption else "✗"
            
            summary += f"| {endpoint.path} | {endpoint.method} | {auth} | {perms} | {rate_limit} | {encryption} |\n"
        
        with open(f"{output_dir}/endpoint_summary.md", "w") as f:
            f.write(summary)

# Usage example
if __name__ == "__main__":
    generator = QNTISecurityAPIDocumentationGenerator()
    generator.save_documentation("qnti_api_docs")