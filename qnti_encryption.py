#!/usr/bin/env python3
"""
QNTI Encryption System - Request/Response Encryption for Sensitive Data
Advanced encryption system with multiple cipher support and key management
"""

import os
import json
import base64
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from functools import wraps
from flask import request, jsonify, g
from dataclasses import dataclass
from enum import Enum
import struct
import hmac

# Cryptography imports
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet

logger = logging.getLogger('QNTI_ENCRYPTION')

class EncryptionMethod(Enum):
    """Encryption methods supported"""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_OAEP = "rsa_oaep"
    FERNET = "fernet"

class KeyDerivationMethod(Enum):
    """Key derivation methods"""
    PBKDF2 = "pbkdf2"
    HKDF = "hkdf"
    SCRYPT = "scrypt"

@dataclass
class EncryptionConfig:
    """Encryption configuration"""
    method: EncryptionMethod
    key_size: int
    iv_size: int
    tag_size: int
    key_derivation: KeyDerivationMethod
    iterations: int = 100000
    salt_size: int = 32

@dataclass
class EncryptedData:
    """Encrypted data container"""
    ciphertext: bytes
    iv: bytes
    tag: Optional[bytes] = None
    salt: Optional[bytes] = None
    method: EncryptionMethod = EncryptionMethod.AES_256_GCM

class QNTIEncryption:
    """Main encryption system"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.master_key = master_key or self._generate_master_key()
        self.backend = default_backend()
        
        # Encryption configurations
        self.configs = {
            EncryptionMethod.AES_256_GCM: EncryptionConfig(
                method=EncryptionMethod.AES_256_GCM,
                key_size=32,
                iv_size=12,
                tag_size=16,
                key_derivation=KeyDerivationMethod.HKDF
            ),
            EncryptionMethod.AES_256_CBC: EncryptionConfig(
                method=EncryptionMethod.AES_256_CBC,
                key_size=32,
                iv_size=16,
                tag_size=0,
                key_derivation=KeyDerivationMethod.PBKDF2
            ),
            EncryptionMethod.CHACHA20_POLY1305: EncryptionConfig(
                method=EncryptionMethod.CHACHA20_POLY1305,
                key_size=32,
                iv_size=12,
                tag_size=16,
                key_derivation=KeyDerivationMethod.HKDF
            ),
            EncryptionMethod.FERNET: EncryptionConfig(
                method=EncryptionMethod.FERNET,
                key_size=32,
                iv_size=0,
                tag_size=0,
                key_derivation=KeyDerivationMethod.HKDF
            )
        }
        
        # RSA key pair for asymmetric encryption
        self.rsa_private_key = None
        self.rsa_public_key = None
        self._generate_rsa_keys()
    
    def _generate_master_key(self) -> bytes:
        """Generate master key"""
        return secrets.token_bytes(32)
    
    def _generate_rsa_keys(self, key_size: int = 2048):
        """Generate RSA key pair"""
        self.rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=self.backend
        )
        self.rsa_public_key = self.rsa_private_key.public_key()
    
    def derive_key(self, password: Union[str, bytes], salt: bytes, 
                   method: KeyDerivationMethod = KeyDerivationMethod.PBKDF2,
                   key_length: int = 32, iterations: int = 100000) -> bytes:
        """Derive key from password"""
        if isinstance(password, str):
            password = password.encode('utf-8')
        
        if method == KeyDerivationMethod.PBKDF2:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                iterations=iterations,
                backend=self.backend
            )
            return kdf.derive(password)
        
        elif method == KeyDerivationMethod.HKDF:
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                info=b'QNTI encryption key',
                backend=self.backend
            )
            return hkdf.derive(password)
        
        else:
            raise ValueError(f"Unsupported key derivation method: {method}")
    
    def encrypt_aes_gcm(self, data: bytes, key: bytes) -> EncryptedData:
        """Encrypt data using AES-256-GCM"""
        config = self.configs[EncryptionMethod.AES_256_GCM]
        
        # Generate IV
        iv = secrets.token_bytes(config.iv_size)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return EncryptedData(
            ciphertext=ciphertext,
            iv=iv,
            tag=encryptor.tag,
            method=EncryptionMethod.AES_256_GCM
        )
    
    def decrypt_aes_gcm(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt data using AES-256-GCM"""
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(encrypted_data.iv, encrypted_data.tag),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        return decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()
    
    def encrypt_aes_cbc(self, data: bytes, key: bytes) -> EncryptedData:
        """Encrypt data using AES-256-CBC"""
        config = self.configs[EncryptionMethod.AES_256_CBC]
        
        # Generate IV
        iv = secrets.token_bytes(config.iv_size)
        
        # Pad data to block size
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padded_data = data + bytes([padding_length] * padding_length)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        return EncryptedData(
            ciphertext=ciphertext,
            iv=iv,
            method=EncryptionMethod.AES_256_CBC
        )
    
    def decrypt_aes_cbc(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt data using AES-256-CBC"""
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(encrypted_data.iv),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()
        
        # Remove padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def encrypt_chacha20_poly1305(self, data: bytes, key: bytes) -> EncryptedData:
        """Encrypt data using ChaCha20-Poly1305"""
        config = self.configs[EncryptionMethod.CHACHA20_POLY1305]
        
        # Generate nonce
        nonce = secrets.token_bytes(config.iv_size)
        
        # Create cipher
        cipher = Cipher(
            algorithms.ChaCha20(key, nonce),
            modes.GCM(nonce),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return EncryptedData(
            ciphertext=ciphertext,
            iv=nonce,
            tag=encryptor.tag,
            method=EncryptionMethod.CHACHA20_POLY1305
        )
    
    def decrypt_chacha20_poly1305(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt data using ChaCha20-Poly1305"""
        cipher = Cipher(
            algorithms.ChaCha20(key, encrypted_data.iv),
            modes.GCM(encrypted_data.iv, encrypted_data.tag),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        return decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()
    
    def encrypt_fernet(self, data: bytes, key: bytes) -> EncryptedData:
        """Encrypt data using Fernet (symmetric encryption)"""
        # Derive Fernet key
        fernet_key = base64.urlsafe_b64encode(key)
        f = Fernet(fernet_key)
        
        # Encrypt data
        ciphertext = f.encrypt(data)
        
        return EncryptedData(
            ciphertext=ciphertext,
            iv=b'',  # Fernet handles IV internally
            method=EncryptionMethod.FERNET
        )
    
    def decrypt_fernet(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt data using Fernet"""
        fernet_key = base64.urlsafe_b64encode(key)
        f = Fernet(fernet_key)
        
        return f.decrypt(encrypted_data.ciphertext)
    
    def encrypt_rsa(self, data: bytes, public_key=None) -> EncryptedData:
        """Encrypt data using RSA-OAEP"""
        if public_key is None:
            public_key = self.rsa_public_key
        
        # RSA has size limitations, so we encrypt in chunks
        key_size = public_key.key_size // 8
        max_chunk_size = key_size - 2 * hashes.SHA256().digest_size - 2
        
        if len(data) > max_chunk_size:
            # For large data, use hybrid encryption
            # Generate symmetric key and encrypt data with it
            symmetric_key = secrets.token_bytes(32)
            
            # Encrypt data with AES
            encrypted_data = self.encrypt_aes_gcm(data, symmetric_key)
            
            # Encrypt symmetric key with RSA
            encrypted_key = public_key.encrypt(
                symmetric_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Combine encrypted key and encrypted data
            combined = encrypted_key + encrypted_data.iv + encrypted_data.tag + encrypted_data.ciphertext
            
            return EncryptedData(
                ciphertext=combined,
                iv=b'',
                method=EncryptionMethod.RSA_OAEP
            )
        else:
            # Direct RSA encryption for small data
            ciphertext = public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return EncryptedData(
                ciphertext=ciphertext,
                iv=b'',
                method=EncryptionMethod.RSA_OAEP
            )
    
    def decrypt_rsa(self, encrypted_data: EncryptedData, private_key=None) -> bytes:
        """Decrypt data using RSA-OAEP"""
        if private_key is None:
            private_key = self.rsa_private_key
        
        key_size = private_key.key_size // 8
        
        if len(encrypted_data.ciphertext) > key_size:
            # Hybrid decryption
            # Extract encrypted symmetric key
            encrypted_key = encrypted_data.ciphertext[:key_size]
            
            # Decrypt symmetric key
            symmetric_key = private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Extract IV, tag, and ciphertext
            remaining = encrypted_data.ciphertext[key_size:]
            iv = remaining[:12]  # AES-GCM IV size
            tag = remaining[12:28]  # AES-GCM tag size
            ciphertext = remaining[28:]
            
            # Decrypt data with AES
            aes_data = EncryptedData(
                ciphertext=ciphertext,
                iv=iv,
                tag=tag,
                method=EncryptionMethod.AES_256_GCM
            )
            
            return self.decrypt_aes_gcm(aes_data, symmetric_key)
        else:
            # Direct RSA decryption
            return private_key.decrypt(
                encrypted_data.ciphertext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
    
    def encrypt(self, data: Union[str, bytes, dict], method: EncryptionMethod = EncryptionMethod.AES_256_GCM,
                key: Optional[bytes] = None, password: Optional[str] = None) -> EncryptedData:
        """Main encryption method"""
        # Convert data to bytes
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, dict):
            data_bytes = json.dumps(data).encode('utf-8')
        else:
            data_bytes = data
        
        # Generate or derive key
        if key is None:
            if password:
                config = self.configs[method]
                salt = secrets.token_bytes(config.salt_size)
                key = self.derive_key(password, salt, config.key_derivation, config.key_size)
            else:
                key = self.master_key
        
        # Encrypt based on method
        if method == EncryptionMethod.AES_256_GCM:
            return self.encrypt_aes_gcm(data_bytes, key)
        elif method == EncryptionMethod.AES_256_CBC:
            return self.encrypt_aes_cbc(data_bytes, key)
        elif method == EncryptionMethod.CHACHA20_POLY1305:
            return self.encrypt_chacha20_poly1305(data_bytes, key)
        elif method == EncryptionMethod.FERNET:
            return self.encrypt_fernet(data_bytes, key)
        elif method == EncryptionMethod.RSA_OAEP:
            return self.encrypt_rsa(data_bytes)
        else:
            raise ValueError(f"Unsupported encryption method: {method}")
    
    def decrypt(self, encrypted_data: EncryptedData, key: Optional[bytes] = None,
                password: Optional[str] = None) -> bytes:
        """Main decryption method"""
        # Generate or derive key
        if key is None:
            if password and encrypted_data.salt:
                config = self.configs[encrypted_data.method]
                key = self.derive_key(password, encrypted_data.salt, config.key_derivation, config.key_size)
            else:
                key = self.master_key
        
        # Decrypt based on method
        if encrypted_data.method == EncryptionMethod.AES_256_GCM:
            return self.decrypt_aes_gcm(encrypted_data, key)
        elif encrypted_data.method == EncryptionMethod.AES_256_CBC:
            return self.decrypt_aes_cbc(encrypted_data, key)
        elif encrypted_data.method == EncryptionMethod.CHACHA20_POLY1305:
            return self.decrypt_chacha20_poly1305(encrypted_data, key)
        elif encrypted_data.method == EncryptionMethod.FERNET:
            return self.decrypt_fernet(encrypted_data, key)
        elif encrypted_data.method == EncryptionMethod.RSA_OAEP:
            return self.decrypt_rsa(encrypted_data)
        else:
            raise ValueError(f"Unsupported encryption method: {encrypted_data.method}")
    
    def serialize_encrypted_data(self, encrypted_data: EncryptedData) -> str:
        """Serialize encrypted data to base64 string"""
        data = {
            'method': encrypted_data.method.value,
            'ciphertext': base64.b64encode(encrypted_data.ciphertext).decode('utf-8'),
            'iv': base64.b64encode(encrypted_data.iv).decode('utf-8') if encrypted_data.iv else None,
            'tag': base64.b64encode(encrypted_data.tag).decode('utf-8') if encrypted_data.tag else None,
            'salt': base64.b64encode(encrypted_data.salt).decode('utf-8') if encrypted_data.salt else None
        }
        return base64.b64encode(json.dumps(data).encode('utf-8')).decode('utf-8')
    
    def deserialize_encrypted_data(self, serialized_data: str) -> EncryptedData:
        """Deserialize encrypted data from base64 string"""
        try:
            data = json.loads(base64.b64decode(serialized_data).decode('utf-8'))
            
            return EncryptedData(
                method=EncryptionMethod(data['method']),
                ciphertext=base64.b64decode(data['ciphertext']),
                iv=base64.b64decode(data['iv']) if data['iv'] else b'',
                tag=base64.b64decode(data['tag']) if data['tag'] else None,
                salt=base64.b64decode(data['salt']) if data['salt'] else None
            )
        except Exception as e:
            raise ValueError(f"Invalid encrypted data format: {e}")
    
    def get_public_key_pem(self) -> str:
        """Get RSA public key in PEM format"""
        return self.rsa_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
    
    def verify_integrity(self, data: bytes, signature: bytes) -> bool:
        """Verify data integrity using HMAC"""
        expected_signature = hmac.new(
            self.master_key,
            data,
            hashlib.sha256
        ).digest()
        return hmac.compare_digest(signature, expected_signature)
    
    def sign_data(self, data: bytes) -> bytes:
        """Sign data using HMAC"""
        return hmac.new(
            self.master_key,
            data,
            hashlib.sha256
        ).digest()

# Global encryption instance
encryption_system = None

def get_encryption_system() -> QNTIEncryption:
    """Get global encryption system instance"""
    global encryption_system
    if encryption_system is None:
        encryption_system = QNTIEncryption()
    return encryption_system

def encrypt_sensitive_data(methods: List[EncryptionMethod] = None):
    """Decorator to encrypt sensitive API responses"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if client supports encryption
            encryption_header = request.headers.get('X-Encryption-Support')
            if not encryption_header:
                return f(*args, **kwargs)
            
            # Get response
            response = f(*args, **kwargs)
            
            # Extract data from response
            if hasattr(response, 'get_json'):
                data = response.get_json()
            else:
                data = response
            
            # Encrypt sensitive fields
            if isinstance(data, dict):
                encrypted_data = encrypt_response_data(data, methods or [EncryptionMethod.AES_256_GCM])
                return jsonify(encrypted_data)
            
            return response
        return decorated_function
    return decorator

def encrypt_response_data(data: dict, methods: List[EncryptionMethod]) -> dict:
    """Encrypt sensitive fields in response data"""
    encryption = get_encryption_system()
    
    # Fields that should be encrypted
    sensitive_fields = [
        'password', 'token', 'secret', 'key', 'private_key',
        'account_number', 'balance', 'equity', 'credit',
        'trade_data', 'position_data', 'order_data'
    ]
    
    def encrypt_field(value, field_name):
        if field_name.lower() in sensitive_fields:
            method = methods[0] if methods else EncryptionMethod.AES_256_GCM
            encrypted = encryption.encrypt(value, method)
            return {
                'encrypted': True,
                'method': method.value,
                'data': encryption.serialize_encrypted_data(encrypted)
            }
        return value
    
    def process_dict(obj):
        if isinstance(obj, dict):
            return {k: process_dict(encrypt_field(v, k)) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [process_dict(item) for item in obj]
        else:
            return obj
    
    return process_dict(data)

def decrypt_request_data():
    """Decorator to decrypt encrypted request data"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if request contains encrypted data
            if request.is_json:
                data = request.get_json()
                if isinstance(data, dict):
                    decrypted_data = decrypt_request_dict(data)
                    # Replace request data
                    request._cached_json = decrypted_data
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def decrypt_request_dict(data: dict) -> dict:
    """Decrypt encrypted fields in request data"""
    encryption = get_encryption_system()
    
    def decrypt_field(value):
        if isinstance(value, dict) and value.get('encrypted'):
            try:
                encrypted_data = encryption.deserialize_encrypted_data(value['data'])
                decrypted_bytes = encryption.decrypt(encrypted_data)
                return json.loads(decrypted_bytes.decode('utf-8'))
            except Exception as e:
                logger.error(f"Failed to decrypt field: {e}")
                return value
        return value
    
    def process_dict(obj):
        if isinstance(obj, dict):
            return {k: process_dict(decrypt_field(v)) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [process_dict(item) for item in obj]
        else:
            return obj
    
    return process_dict(data)

class EncryptionMiddleware:
    """Flask middleware for automatic encryption/decryption"""
    
    def __init__(self, app=None, encryption_system=None):
        self.app = app
        self.encryption = encryption_system or get_encryption_system()
        self.sensitive_endpoints = [
            '/api/auth/login',
            '/api/auth/register',
            '/api/account/info',
            '/api/trade/execute',
            '/api/trade/history'
        ]
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize middleware with Flask app"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
    
    def before_request(self):
        """Process request before handling"""
        # Check if endpoint requires encryption
        if request.endpoint in self.sensitive_endpoints:
            # Decrypt request data if encrypted
            if request.is_json:
                data = request.get_json()
                if isinstance(data, dict):
                    decrypted_data = decrypt_request_dict(data)
                    request._cached_json = decrypted_data
            
            # Store encryption info for response
            g.encryption_required = True
            g.encryption_methods = [EncryptionMethod.AES_256_GCM]
    
    def after_request(self, response):
        """Process response after handling"""
        try:
            # Encrypt response if required
            if hasattr(g, 'encryption_required') and g.encryption_required:
                encryption_support = request.headers.get('X-Encryption-Support')
                if encryption_support:
                    if response.is_json:
                        data = response.get_json()
                        if isinstance(data, dict):
                            encrypted_data = encrypt_response_data(data, g.encryption_methods)
                            response.data = json.dumps(encrypted_data)
                            response.headers['X-Encryption-Used'] = 'true'
        except Exception as e:
            logger.error(f"Error in encryption middleware: {e}")
        
        return response