#!/usr/bin/env python3
"""
QNTI Authentication System - JWT-based Authentication with RBAC
Comprehensive security system for the QNTI trading platform
"""

import jwt
import uuid
import bcrypt
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from functools import wraps
from flask import request, jsonify, current_app
from enum import Enum
import sqlite3
import json
import os
from dataclasses import dataclass, asdict
import pyotp
import qrcode
from io import BytesIO
import base64

logger = logging.getLogger('QNTI_AUTH')

class UserRole(Enum):
    """User roles for RBAC system"""
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"
    API_USER = "api_user"
    GUEST = "guest"

class Permission(Enum):
    """System permissions"""
    # Trading permissions
    TRADE_EXECUTE = "trade:execute"
    TRADE_VIEW = "trade:view"
    TRADE_MANAGE = "trade:manage"
    TRADE_HISTORY = "trade:history"
    
    # Account permissions
    ACCOUNT_VIEW = "account:view"
    ACCOUNT_MANAGE = "account:manage"
    ACCOUNT_ADMIN = "account:admin"
    
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_LOGS = "system:logs"
    SYSTEM_HEALTH = "system:health"
    
    # API permissions
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"
    
    # EA permissions
    EA_UPLOAD = "ea:upload"
    EA_MANAGE = "ea:manage"
    EA_EXECUTE = "ea:execute"
    EA_VIEW = "ea:view"

@dataclass
class User:
    """User model"""
    id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    permissions: List[Permission]
    is_active: bool = True
    is_verified: bool = False
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    api_key: Optional[str] = None
    created_at: datetime = None
    last_login: Optional[datetime] = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class Session:
    """Session model"""
    id: str
    user_id: str
    ip_address: str
    user_agent: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    is_active: bool = True

class RolePermissionManager:
    """Manages role-based permissions"""
    
    ROLE_PERMISSIONS = {
        UserRole.ADMIN: [
            Permission.TRADE_EXECUTE, Permission.TRADE_VIEW, Permission.TRADE_MANAGE, Permission.TRADE_HISTORY,
            Permission.ACCOUNT_VIEW, Permission.ACCOUNT_MANAGE, Permission.ACCOUNT_ADMIN,
            Permission.SYSTEM_ADMIN, Permission.SYSTEM_CONFIG, Permission.SYSTEM_LOGS, Permission.SYSTEM_HEALTH,
            Permission.API_READ, Permission.API_WRITE, Permission.API_ADMIN,
            Permission.EA_UPLOAD, Permission.EA_MANAGE, Permission.EA_EXECUTE, Permission.EA_VIEW
        ],
        UserRole.TRADER: [
            Permission.TRADE_EXECUTE, Permission.TRADE_VIEW, Permission.TRADE_MANAGE, Permission.TRADE_HISTORY,
            Permission.ACCOUNT_VIEW,
            Permission.SYSTEM_HEALTH,
            Permission.API_READ, Permission.API_WRITE,
            Permission.EA_UPLOAD, Permission.EA_MANAGE, Permission.EA_EXECUTE, Permission.EA_VIEW
        ],
        UserRole.VIEWER: [
            Permission.TRADE_VIEW, Permission.TRADE_HISTORY,
            Permission.ACCOUNT_VIEW,
            Permission.SYSTEM_HEALTH,
            Permission.API_READ,
            Permission.EA_VIEW
        ],
        UserRole.API_USER: [
            Permission.API_READ, Permission.API_WRITE,
            Permission.TRADE_VIEW, Permission.TRADE_HISTORY,
            Permission.ACCOUNT_VIEW
        ],
        UserRole.GUEST: [
            Permission.SYSTEM_HEALTH
        ]
    }
    
    @classmethod
    def get_permissions_for_role(cls, role: UserRole) -> List[Permission]:
        """Get permissions for a specific role"""
        return cls.ROLE_PERMISSIONS.get(role, [])
    
    @classmethod
    def has_permission(cls, user_role: UserRole, permission: Permission) -> bool:
        """Check if role has specific permission"""
        return permission in cls.get_permissions_for_role(user_role)

class QNTIAuthSystem:
    """Main authentication system"""
    
    def __init__(self, db_path: str = "qnti_data/auth.db", secret_key: str = None):
        self.db_path = db_path
        self.secret_key = secret_key or os.environ.get('QNTI_SECRET_KEY', secrets.token_hex(32))
        self.algorithm = 'HS256'
        self.access_token_expire = timedelta(hours=1)
        self.refresh_token_expire = timedelta(days=7)
        self.session_timeout = timedelta(hours=24)
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        
        # Initialize database
        self._init_database()
        
        # Create default admin user if none exists
        self._create_default_admin()
    
    def _init_database(self):
        """Initialize SQLite database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    is_verified BOOLEAN DEFAULT FALSE,
                    mfa_enabled BOOLEAN DEFAULT FALSE,
                    mfa_secret TEXT,
                    api_key TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP
                )
            ''')
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Audit log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    action TEXT NOT NULL,
                    resource TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT TRUE
                )
            ''')
            
            # API keys table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    name TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    last_used TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.commit()
    
    def _create_default_admin(self):
        """Create default admin user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users WHERE role = ?", (UserRole.ADMIN.value,))
            admin_count = cursor.fetchone()[0]
            
            if admin_count == 0:
                admin_user = User(
                    id=str(uuid.uuid4()),
                    username="admin",
                    email="admin@qnti.local",
                    password_hash=self._hash_password("admin123!"),
                    role=UserRole.ADMIN,
                    permissions=RolePermissionManager.get_permissions_for_role(UserRole.ADMIN),
                    is_active=True,
                    is_verified=True
                )
                
                self._save_user(admin_user)
                logger.info("Default admin user created: admin/admin123!")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _save_user(self, user: User):
        """Save user to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO users 
                (id, username, email, password_hash, role, permissions, is_active, is_verified, 
                 mfa_enabled, mfa_secret, api_key, created_at, last_login, login_attempts, locked_until)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user.id, user.username, user.email, user.password_hash,
                user.role.value, json.dumps([p.value for p in user.permissions]),
                user.is_active, user.is_verified, user.mfa_enabled, user.mfa_secret,
                user.api_key, user.created_at, user.last_login, user.login_attempts,
                user.locked_until
            ))
            conn.commit()
    
    def _get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            
            if row:
                return User(
                    id=row[0], username=row[1], email=row[2], password_hash=row[3],
                    role=UserRole(row[4]), permissions=[Permission(p) for p in json.loads(row[5])],
                    is_active=bool(row[6]), is_verified=bool(row[7]), mfa_enabled=bool(row[8]),
                    mfa_secret=row[9], api_key=row[10], created_at=datetime.fromisoformat(row[11]) if row[11] else None,
                    last_login=datetime.fromisoformat(row[12]) if row[12] else None,
                    login_attempts=row[13], locked_until=datetime.fromisoformat(row[14]) if row[14] else None
                )
        return None
    
    def _get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            
            if row:
                return User(
                    id=row[0], username=row[1], email=row[2], password_hash=row[3],
                    role=UserRole(row[4]), permissions=[Permission(p) for p in json.loads(row[5])],
                    is_active=bool(row[6]), is_verified=bool(row[7]), mfa_enabled=bool(row[8]),
                    mfa_secret=row[9], api_key=row[10], created_at=datetime.fromisoformat(row[11]) if row[11] else None,
                    last_login=datetime.fromisoformat(row[12]) if row[12] else None,
                    login_attempts=row[13], locked_until=datetime.fromisoformat(row[14]) if row[14] else None
                )
        return None
    
    def create_user(self, username: str, email: str, password: str, role: UserRole) -> Dict[str, Any]:
        """Create new user"""
        try:
            # Check if user exists
            if self._get_user_by_username(username):
                return {"success": False, "error": "Username already exists"}
            
            # Create user
            user = User(
                id=str(uuid.uuid4()),
                username=username,
                email=email,
                password_hash=self._hash_password(password),
                role=role,
                permissions=RolePermissionManager.get_permissions_for_role(role)
            )
            
            self._save_user(user)
            self._audit_log(user.id, "USER_CREATE", f"user:{user.id}", {"username": username, "role": role.value})
            
            return {"success": True, "user_id": user.id}
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return {"success": False, "error": str(e)}
    
    def authenticate(self, username: str, password: str, mfa_code: str = None, 
                    ip_address: str = None, user_agent: str = None) -> Dict[str, Any]:
        """Authenticate user and return tokens"""
        try:
            user = self._get_user_by_username(username)
            if not user:
                self._audit_log(None, "LOGIN_FAILED", f"user:{username}", {"reason": "user_not_found"}, ip_address, user_agent, False)
                return {"success": False, "error": "Invalid credentials"}
            
            # Check if user is locked
            if user.locked_until and datetime.utcnow() < user.locked_until:
                self._audit_log(user.id, "LOGIN_FAILED", f"user:{user.id}", {"reason": "account_locked"}, ip_address, user_agent, False)
                return {"success": False, "error": "Account is locked"}
            
            # Check if user is active
            if not user.is_active:
                self._audit_log(user.id, "LOGIN_FAILED", f"user:{user.id}", {"reason": "account_inactive"}, ip_address, user_agent, False)
                return {"success": False, "error": "Account is inactive"}
            
            # Verify password
            if not self._verify_password(password, user.password_hash):
                user.login_attempts += 1
                if user.login_attempts >= self.max_login_attempts:
                    user.locked_until = datetime.utcnow() + self.lockout_duration
                self._save_user(user)
                self._audit_log(user.id, "LOGIN_FAILED", f"user:{user.id}", {"reason": "invalid_password"}, ip_address, user_agent, False)
                return {"success": False, "error": "Invalid credentials"}
            
            # Check MFA if enabled
            if user.mfa_enabled:
                if not mfa_code or not self._verify_mfa_code(user.mfa_secret, mfa_code):
                    self._audit_log(user.id, "LOGIN_FAILED", f"user:{user.id}", {"reason": "invalid_mfa"}, ip_address, user_agent, False)
                    return {"success": False, "error": "Invalid MFA code"}
            
            # Reset login attempts
            user.login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.utcnow()
            self._save_user(user)
            
            # Create session
            session = self._create_session(user.id, ip_address, user_agent)
            
            # Generate tokens
            access_token = self._generate_access_token(user, session.id)
            refresh_token = self._generate_refresh_token(user, session.id)
            
            self._audit_log(user.id, "LOGIN_SUCCESS", f"user:{user.id}", {"session_id": session.id}, ip_address, user_agent)
            
            return {
                "success": True,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "role": user.role.value,
                    "permissions": [p.value for p in user.permissions]
                }
            }
        
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {"success": False, "error": "Authentication failed"}
    
    def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> Session:
        """Create new session"""
        session = Session(
            id=str(uuid.uuid4()),
            user_id=user_id,
            ip_address=ip_address or "unknown",
            user_agent=user_agent or "unknown",
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.session_timeout
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sessions (id, user_id, ip_address, user_agent, created_at, last_activity, expires_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session.id, session.user_id, session.ip_address, session.user_agent,
                  session.created_at, session.last_activity, session.expires_at, session.is_active))
            conn.commit()
        
        return session
    
    def _generate_access_token(self, user: User, session_id: str) -> str:
        """Generate JWT access token"""
        payload = {
            'user_id': user.id,
            'username': user.username,
            'role': user.role.value,
            'permissions': [p.value for p in user.permissions],
            'session_id': session_id,
            'type': 'access',
            'exp': datetime.utcnow() + self.access_token_expire,
            'iat': datetime.utcnow(),
            'jti': str(uuid.uuid4())
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def _generate_refresh_token(self, user: User, session_id: str) -> str:
        """Generate JWT refresh token"""
        payload = {
            'user_id': user.id,
            'session_id': session_id,
            'type': 'refresh',
            'exp': datetime.utcnow() + self.refresh_token_expire,
            'iat': datetime.utcnow(),
            'jti': str(uuid.uuid4())
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get('type') != 'access':
                return {"valid": False, "error": "Invalid token type"}
            
            # Check if session is active
            session_id = payload.get('session_id')
            if not self._is_session_active(session_id):
                return {"valid": False, "error": "Session expired"}
            
            # Update session activity
            self._update_session_activity(session_id)
            
            return {"valid": True, "payload": payload}
        
        except jwt.ExpiredSignatureError:
            return {"valid": False, "error": "Token expired"}
        except jwt.InvalidTokenError:
            return {"valid": False, "error": "Invalid token"}
    
    def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token"""
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get('type') != 'refresh':
                return {"success": False, "error": "Invalid token type"}
            
            user_id = payload.get('user_id')
            session_id = payload.get('session_id')
            
            # Check if session is active
            if not self._is_session_active(session_id):
                return {"success": False, "error": "Session expired"}
            
            # Get user
            user = self._get_user_by_id(user_id)
            if not user or not user.is_active:
                return {"success": False, "error": "User not found or inactive"}
            
            # Generate new access token
            new_access_token = self._generate_access_token(user, session_id)
            
            return {"success": True, "access_token": new_access_token}
        
        except jwt.ExpiredSignatureError:
            return {"success": False, "error": "Refresh token expired"}
        except jwt.InvalidTokenError:
            return {"success": False, "error": "Invalid refresh token"}
    
    def _is_session_active(self, session_id: str) -> bool:
        """Check if session is active"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT is_active, expires_at FROM sessions 
                WHERE id = ? AND is_active = TRUE
            ''', (session_id,))
            row = cursor.fetchone()
            
            if row:
                expires_at = datetime.fromisoformat(row[1])
                return datetime.utcnow() < expires_at
        return False
    
    def _update_session_activity(self, session_id: str):
        """Update session last activity"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions SET last_activity = ? WHERE id = ?
            ''', (datetime.utcnow(), session_id))
            conn.commit()
    
    def logout(self, session_id: str, user_id: str = None):
        """Logout user by deactivating session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions SET is_active = FALSE WHERE id = ?
            ''', (session_id,))
            conn.commit()
        
        if user_id:
            self._audit_log(user_id, "LOGOUT", f"session:{session_id}", {"session_id": session_id})
    
    def _audit_log(self, user_id: str, action: str, resource: str, details: Dict = None, 
                   ip_address: str = None, user_agent: str = None, success: bool = True):
        """Add audit log entry"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO audit_log (user_id, action, resource, details, ip_address, user_agent, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, action, resource, json.dumps(details) if details else None,
                  ip_address, user_agent, success))
            conn.commit()
    
    def enable_mfa(self, user_id: str) -> Dict[str, Any]:
        """Enable MFA for user"""
        try:
            user = self._get_user_by_id(user_id)
            if not user:
                return {"success": False, "error": "User not found"}
            
            # Generate secret
            secret = pyotp.random_base32()
            user.mfa_secret = secret
            user.mfa_enabled = True
            self._save_user(user)
            
            # Generate QR code
            totp = pyotp.TOTP(secret)
            provisioning_uri = totp.provisioning_uri(
                name=user.email,
                issuer_name="QNTI Trading System"
            )
            
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            buffer = BytesIO()
            img.save(buffer)
            buffer.seek(0)
            
            qr_code_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            self._audit_log(user_id, "MFA_ENABLE", f"user:{user_id}", {"enabled": True})
            
            return {
                "success": True,
                "secret": secret,
                "qr_code": qr_code_base64,
                "provisioning_uri": provisioning_uri
            }
        
        except Exception as e:
            logger.error(f"Error enabling MFA: {e}")
            return {"success": False, "error": str(e)}
    
    def _verify_mfa_code(self, secret: str, code: str) -> bool:
        """Verify MFA code"""
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=1)
    
    def create_api_key(self, user_id: str, name: str, permissions: List[Permission], 
                      expires_at: datetime = None) -> Dict[str, Any]:
        """Create API key for user"""
        try:
            user = self._get_user_by_id(user_id)
            if not user:
                return {"success": False, "error": "User not found"}
            
            # Generate API key
            api_key = f"qnti_{secrets.token_urlsafe(32)}"
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            key_id = str(uuid.uuid4())
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO api_keys (id, user_id, key_hash, name, permissions, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (key_id, user_id, key_hash, name, json.dumps([p.value for p in permissions]), expires_at))
                conn.commit()
            
            self._audit_log(user_id, "API_KEY_CREATE", f"api_key:{key_id}", {"name": name, "permissions": [p.value for p in permissions]})
            
            return {"success": True, "api_key": api_key, "key_id": key_id}
        
        except Exception as e:
            logger.error(f"Error creating API key: {e}")
            return {"success": False, "error": str(e)}
    
    def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify API key"""
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT ak.id, ak.user_id, ak.permissions, ak.expires_at, u.username, u.role, u.is_active
                    FROM api_keys ak
                    JOIN users u ON ak.user_id = u.id
                    WHERE ak.key_hash = ? AND ak.is_active = TRUE
                ''', (key_hash,))
                row = cursor.fetchone()
                
                if row:
                    expires_at = datetime.fromisoformat(row[3]) if row[3] else None
                    if expires_at and datetime.utcnow() > expires_at:
                        return {"valid": False, "error": "API key expired"}
                    
                    if not row[6]:  # user is_active
                        return {"valid": False, "error": "User account inactive"}
                    
                    # Update last used
                    cursor.execute('''
                        UPDATE api_keys SET last_used = ? WHERE id = ?
                    ''', (datetime.utcnow(), row[0]))
                    conn.commit()
                    
                    return {
                        "valid": True,
                        "user_id": row[1],
                        "username": row[4],
                        "role": row[5],
                        "permissions": json.loads(row[2]),
                        "key_id": row[0]
                    }
            
            return {"valid": False, "error": "Invalid API key"}
        
        except Exception as e:
            logger.error(f"Error verifying API key: {e}")
            return {"valid": False, "error": str(e)}

# Global auth instance
auth_system = None

def get_auth_system() -> QNTIAuthSystem:
    """Get global auth system instance"""
    global auth_system
    if auth_system is None:
        auth_system = QNTIAuthSystem()
    return auth_system

def require_auth(permissions: List[Permission] = None):
    """Decorator to require authentication"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth = get_auth_system()
            
            # Check for JWT token
            token = None
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
            
            # Check for API key
            api_key = request.headers.get('X-API-Key')
            
            if token:
                # JWT authentication
                result = auth.verify_token(token)
                if not result["valid"]:
                    return jsonify({"error": result["error"]}), 401
                
                user_data = result["payload"]
                user_permissions = [Permission(p) for p in user_data.get("permissions", [])]
                
            elif api_key:
                # API key authentication
                result = auth.verify_api_key(api_key)
                if not result["valid"]:
                    return jsonify({"error": result["error"]}), 401
                
                user_data = {
                    "user_id": result["user_id"],
                    "username": result["username"],
                    "role": result["role"]
                }
                user_permissions = [Permission(p) for p in result["permissions"]]
                
            else:
                return jsonify({"error": "Authentication required"}), 401
            
            # Check permissions
            if permissions:
                for permission in permissions:
                    if permission not in user_permissions:
                        return jsonify({"error": "Insufficient permissions"}), 403
            
            # Add user data to request context
            request.user = user_data
            request.user_permissions = user_permissions
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_role(required_role: UserRole):
    """Decorator to require specific role"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(request, 'user'):
                return jsonify({"error": "Authentication required"}), 401
            
            user_role = UserRole(request.user.get('role'))
            if user_role != required_role and user_role != UserRole.ADMIN:
                return jsonify({"error": "Insufficient role"}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator