#!/usr/bin/env python3
"""
QNTI Database Configuration Management
PostgreSQL connection pooling and configuration management
"""

import os
import json
import logging
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration structure"""
    host: str
    port: int
    database: str
    username: str
    password: str
    schema: str = "qnti"
    ssl_mode: str = "require"
    connect_timeout: int = 30
    command_timeout: int = 60
    pool_min_conn: int = 5
    pool_max_conn: int = 20
    pool_max_overflow: int = 30
    pool_recycle: int = 3600  # 1 hour
    pool_pre_ping: bool = True
    
    @property
    def dsn(self) -> str:
        """Generate PostgreSQL DSN string"""
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}&connect_timeout={self.connect_timeout}"
        )
    
    @property
    def connection_params(self) -> Dict[str, Any]:
        """Generate connection parameters dict"""
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.username,
            'password': self.password,
            'sslmode': self.ssl_mode,
            'connect_timeout': self.connect_timeout,
            'options': f'-c search_path={self.schema}'
        }

class QNTIDatabaseManager:
    """Advanced database connection manager with pooling"""
    
    def __init__(self, config_path: str = "db_config.json"):
        self.config_path = Path(config_path)
        self.config: Optional[DatabaseConfig] = None
        self.connection_pool: Optional[pool.ThreadedConnectionPool] = None
        self._pool_lock = threading.Lock()
        self._health_check_thread = None
        self._health_check_interval = 60  # seconds
        self._running = False
        
        # Load configuration
        self._load_config()
        
        # Initialize connection pool
        self._initialize_pool()
        
        # Start health monitoring
        self._start_health_monitoring()
        
        logger.info("QNTI Database Manager initialized")
    
    def _load_config(self):
        """Load database configuration from file or environment"""
        try:
            # Try to load from file first
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                    self.config = DatabaseConfig(**config_data)
                logger.info(f"Database config loaded from {self.config_path}")
            else:
                # Load from environment variables
                self.config = DatabaseConfig(
                    host=os.getenv('QNTI_DB_HOST', 'localhost'),
                    port=int(os.getenv('QNTI_DB_PORT', '5432')),
                    database=os.getenv('QNTI_DB_NAME', 'qnti_trading'),
                    username=os.getenv('QNTI_DB_USER', 'qnti_user'),
                    password=os.getenv('QNTI_DB_PASSWORD', 'qnti_password'),
                    schema=os.getenv('QNTI_DB_SCHEMA', 'qnti'),
                    ssl_mode=os.getenv('QNTI_DB_SSL_MODE', 'require'),
                    pool_min_conn=int(os.getenv('QNTI_DB_POOL_MIN', '5')),
                    pool_max_conn=int(os.getenv('QNTI_DB_POOL_MAX', '20'))
                )
                logger.info("Database config loaded from environment variables")
                
                # Save config to file for future use
                self._save_config()
                
        except Exception as e:
            logger.error(f"Error loading database config: {e}")
            raise
    
    def _save_config(self):
        """Save current configuration to file"""
        try:
            config_data = {
                'host': self.config.host,
                'port': self.config.port,
                'database': self.config.database,
                'username': self.config.username,
                'password': self.config.password,
                'schema': self.config.schema,
                'ssl_mode': self.config.ssl_mode,
                'connect_timeout': self.config.connect_timeout,
                'command_timeout': self.config.command_timeout,
                'pool_min_conn': self.config.pool_min_conn,
                'pool_max_conn': self.config.pool_max_conn,
                'pool_max_overflow': self.config.pool_max_overflow,
                'pool_recycle': self.config.pool_recycle,
                'pool_pre_ping': self.config.pool_pre_ping
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.info(f"Database config saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving database config: {e}")
    
    def _initialize_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            with self._pool_lock:
                if self.connection_pool:
                    self.connection_pool.closeall()
                
                self.connection_pool = pool.ThreadedConnectionPool(
                    minconn=self.config.pool_min_conn,
                    maxconn=self.config.pool_max_conn,
                    **self.config.connection_params
                )
                
                logger.info(f"Connection pool initialized with {self.config.pool_min_conn}-{self.config.pool_max_conn} connections")
                
        except Exception as e:
            logger.error(f"Error initializing connection pool: {e}")
            raise
    
    def _start_health_monitoring(self):
        """Start background health monitoring thread"""
        if self._health_check_thread is None or not self._health_check_thread.is_alive():
            self._running = True
            self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
            self._health_check_thread.start()
            logger.info("Database health monitoring started")
    
    def _health_check_loop(self):
        """Background health check loop"""
        while self._running:
            try:
                # Check pool health
                with self.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT 1")
                        result = cursor.fetchone()
                        if result[0] != 1:
                            logger.warning("Database health check failed")
                        else:
                            logger.debug("Database health check passed")
                
                # Check pool statistics
                if self.connection_pool:
                    pool_size = self.connection_pool.minconn + len(self.connection_pool._used)
                    logger.debug(f"Connection pool size: {pool_size}/{self.config.pool_max_conn}")
                
                time.sleep(self._health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(self._health_check_interval)
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        connection = None
        try:
            with self._pool_lock:
                if not self.connection_pool:
                    self._initialize_pool()
                
                connection = self.connection_pool.getconn()
                
                if connection:
                    # Test connection
                    if self.config.pool_pre_ping:
                        connection.ping()
                    
                    yield connection
                else:
                    raise Exception("Failed to get connection from pool")
                    
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection and self.connection_pool:
                self.connection_pool.putconn(connection)
    
    @contextmanager
    def get_cursor(self, dictionary=True):
        """Get database cursor with automatic connection management"""
        with self.get_connection() as connection:
            cursor_factory = RealDictCursor if dictionary else None
            cursor = connection.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
                connection.commit()
            except Exception as e:
                connection.rollback()
                raise
            finally:
                cursor.close()
    
    def execute_query(self, query: str, params: tuple = None, fetch_all: bool = True) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, params)
                
                if fetch_all:
                    return cursor.fetchall()
                else:
                    return cursor.fetchone()
                    
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
    
    def execute_command(self, command: str, params: tuple = None) -> int:
        """Execute INSERT, UPDATE, DELETE command"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute(command, params)
                return cursor.rowcount
                
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            logger.error(f"Command: {command}")
            logger.error(f"Params: {params}")
            raise
    
    def execute_many(self, command: str, params_list: List[tuple]) -> int:
        """Execute command with multiple parameter sets"""
        try:
            with self.get_cursor() as cursor:
                cursor.executemany(command, params_list)
                return cursor.rowcount
                
        except Exception as e:
            logger.error(f"Batch execution error: {e}")
            logger.error(f"Command: {command}")
            raise
    
    def call_function(self, function_name: str, params: tuple = None) -> Any:
        """Call a PostgreSQL function"""
        try:
            with self.get_cursor() as cursor:
                cursor.callproc(function_name, params)
                return cursor.fetchone()
                
        except Exception as e:
            logger.error(f"Function call error: {e}")
            logger.error(f"Function: {function_name}")
            logger.error(f"Params: {params}")
            raise
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version()")
                    version = cursor.fetchone()[0]
                    logger.info(f"Database connection test successful: {version}")
                    return True
                    
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        try:
            with self._pool_lock:
                if not self.connection_pool:
                    return {"error": "Connection pool not initialized"}
                
                return {
                    "min_connections": self.connection_pool.minconn,
                    "max_connections": self.connection_pool.maxconn,
                    "used_connections": len(self.connection_pool._used),
                    "free_connections": len(self.connection_pool._pool),
                    "total_connections": len(self.connection_pool._used) + len(self.connection_pool._pool)
                }
                
        except Exception as e:
            logger.error(f"Error getting pool stats: {e}")
            return {"error": str(e)}
    
    def close_pool(self):
        """Close connection pool"""
        try:
            self._running = False
            
            if self._health_check_thread and self._health_check_thread.is_alive():
                self._health_check_thread.join(timeout=5)
            
            with self._pool_lock:
                if self.connection_pool:
                    self.connection_pool.closeall()
                    self.connection_pool = None
                    logger.info("Connection pool closed")
                    
        except Exception as e:
            logger.error(f"Error closing connection pool: {e}")
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.close_pool()

# Global database manager instance
_db_manager: Optional[QNTIDatabaseManager] = None

def get_database_manager(config_path: str = "db_config.json") -> QNTIDatabaseManager:
    """Get or create database manager singleton"""
    global _db_manager
    if _db_manager is None:
        _db_manager = QNTIDatabaseManager(config_path)
    return _db_manager

def close_database_manager():
    """Close the global database manager"""
    global _db_manager
    if _db_manager:
        _db_manager.close_pool()
        _db_manager = None

# Convenience functions for common database operations
def execute_query(query: str, params: tuple = None, fetch_all: bool = True) -> List[Dict[str, Any]]:
    """Execute query using global database manager"""
    return get_database_manager().execute_query(query, params, fetch_all)

def execute_command(command: str, params: tuple = None) -> int:
    """Execute command using global database manager"""
    return get_database_manager().execute_command(command, params)

def execute_many(command: str, params_list: List[tuple]) -> int:
    """Execute batch command using global database manager"""
    return get_database_manager().execute_many(command, params_list)

def get_cursor(dictionary: bool = True):
    """Get cursor using global database manager"""
    return get_database_manager().get_cursor(dictionary)

def get_connection():
    """Get connection using global database manager"""
    return get_database_manager().get_connection()

# Configuration management functions
def create_default_config(config_path: str = "db_config.json"):
    """Create default database configuration file"""
    default_config = {
        "host": "localhost",
        "port": 5432,
        "database": "qnti_trading",
        "username": "qnti_user",
        "password": "qnti_password",
        "schema": "qnti",
        "ssl_mode": "require",
        "connect_timeout": 30,
        "command_timeout": 60,
        "pool_min_conn": 5,
        "pool_max_conn": 20,
        "pool_max_overflow": 30,
        "pool_recycle": 3600,
        "pool_pre_ping": True
    }
    
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    logger.info(f"Default database config created at {config_path}")

def validate_config(config_path: str = "db_config.json") -> bool:
    """Validate database configuration"""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        required_fields = ['host', 'port', 'database', 'username', 'password']
        for field in required_fields:
            if field not in config_data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Test connection
        temp_config = DatabaseConfig(**config_data)
        temp_manager = QNTIDatabaseManager(config_path)
        result = temp_manager.test_connection()
        temp_manager.close_pool()
        
        return result
        
    except Exception as e:
        logger.error(f"Config validation error: {e}")
        return False

if __name__ == "__main__":
    # Test database configuration
    logging.basicConfig(level=logging.INFO)
    
    # Create default config if it doesn't exist
    if not Path("db_config.json").exists():
        create_default_config()
        print("Default config created. Please update db_config.json with your database credentials.")
    else:
        # Test connection
        db_manager = get_database_manager()
        
        if db_manager.test_connection():
            print("Database connection successful!")
            
            # Show pool stats
            stats = db_manager.get_pool_stats()
            print(f"Pool stats: {stats}")
            
            # Test query
            try:
                result = db_manager.execute_query("SELECT key, value FROM system_config LIMIT 5")
                print(f"Sample config entries: {result}")
            except Exception as e:
                print(f"Query test error: {e}")
        else:
            print("Database connection failed!")
        
        close_database_manager()
