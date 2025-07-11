#!/usr/bin/env python3
"""
QNTI Data Migration Utility
Migrates data from SQLite to PostgreSQL with data integrity validation
"""

import sqlite3
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import hashlib
import time

# Import database configuration
from database_config import get_database_manager, DatabaseConfig

logger = logging.getLogger(__name__)

@dataclass
class MigrationStats:
    """Migration statistics"""
    table_name: str
    total_records: int
    migrated_records: int
    failed_records: int
    start_time: datetime
    end_time: Optional[datetime] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def duration(self) -> timedelta:
        if self.end_time:
            return self.end_time - self.start_time
        return timedelta(0)
    
    @property
    def success_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return (self.migrated_records / self.total_records) * 100

class QNTIDataMigrator:
    """Data migration utility for QNTI system"""
    
    def __init__(self, sqlite_db_path: str = "qnti_data/qnti.db", 
                 portfolio_db_path: str = "qnti_portfolio.db",
                 ea_profiling_db_path: str = "qnti_data/ea_profiling.db"):
        self.sqlite_db_path = Path(sqlite_db_path)
        self.portfolio_db_path = Path(portfolio_db_path)
        self.ea_profiling_db_path = Path(ea_profiling_db_path)
        
        # PostgreSQL connection
        self.pg_manager = get_database_manager()
        
        # Migration statistics
        self.migration_stats: List[MigrationStats] = []
        
        # UUID mapping for ID conversion
        self.id_mapping: Dict[str, Dict[str, str]] = {
            'accounts': {},
            'portfolios': {},
            'positions': {},
            'trades': {},
            'ea_profiles': {}
        }
        
        logger.info("QNTI Data Migrator initialized")
    
    def generate_uuid_mapping(self, table_name: str, old_ids: List[str]):
        """Generate UUID mapping for old string IDs"""
        for old_id in old_ids:
            if old_id not in self.id_mapping[table_name]:
                self.id_mapping[table_name][old_id] = str(uuid.uuid4())
    
    def get_sqlite_connection(self, db_path: Path) -> sqlite3.Connection:
        """Get SQLite connection"""
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {db_path}")
        
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def validate_postgresql_schema(self) -> bool:
        """Validate PostgreSQL schema exists"""
        try:
            query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'qnti' 
            ORDER BY table_name
            """
            
            tables = self.pg_manager.execute_query(query)
            table_names = [t['table_name'] for t in tables]
            
            required_tables = [
                'accounts', 'portfolios', 'positions', 'portfolio_snapshots',
                'trades', 'ea_performance', 'ea_profiles', 'ea_indicators',
                'ea_analysis', 'system_config', 'system_logs'
            ]
            
            missing_tables = [t for t in required_tables if t not in table_names]
            
            if missing_tables:
                logger.error(f"Missing PostgreSQL tables: {missing_tables}")
                return False
            
            logger.info(f"PostgreSQL schema validation successful. Found {len(table_names)} tables")
            return True
            
        except Exception as e:
            logger.error(f"PostgreSQL schema validation failed: {e}")
            return False
    
    def migrate_accounts(self) -> MigrationStats:
        """Migrate accounts from portfolio database"""
        stats = MigrationStats(table_name="accounts", total_records=0, migrated_records=0, failed_records=0, start_time=datetime.now())
        
        try:
            # Connect to portfolio SQLite database
            sqlite_conn = self.get_sqlite_connection(self.portfolio_db_path)
            cursor = sqlite_conn.cursor()
            
            # Get all accounts
            cursor.execute("SELECT * FROM accounts")
            accounts = cursor.fetchall()
            stats.total_records = len(accounts)
            
            if stats.total_records == 0:
                logger.warning("No accounts found in SQLite database")
                stats.end_time = datetime.now()
                return stats
            
            # Generate UUID mapping
            old_ids = [acc['id'] for acc in accounts]
            self.generate_uuid_mapping('accounts', old_ids)
            
            # Migrate accounts
            for account in accounts:
                try:
                    new_id = self.id_mapping['accounts'][account['id']]
                    
                    # Parse metadata
                    metadata = json.loads(account['metadata']) if account['metadata'] else {}
                    
                    # Insert into PostgreSQL
                    insert_query = """
                    INSERT INTO accounts (
                        id, name, account_type, broker, login, server, currency,
                        balance, equity, margin, free_margin, margin_level, profit,
                        leverage, is_active, created_at, last_updated, metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """
                    
                    params = (
                        new_id, account['name'], account['account_type'], account['broker'],
                        account['login'], account['server'], account['currency'],
                        float(account['balance']), float(account['equity']), float(account['margin']),
                        float(account['free_margin']), float(account['margin_level']), float(account['profit']),
                        int(account['leverage']), bool(account['is_active']),
                        datetime.fromisoformat(account['created_at']),
                        datetime.fromisoformat(account['last_updated']),
                        json.dumps(metadata)
                    )
                    
                    self.pg_manager.execute_command(insert_query, params)
                    stats.migrated_records += 1
                    
                except Exception as e:
                    error_msg = f"Failed to migrate account {account['id']}: {str(e)}"
                    logger.error(error_msg)
                    stats.errors.append(error_msg)
                    stats.failed_records += 1
            
            sqlite_conn.close()
            
        except Exception as e:
            error_msg = f"Error migrating accounts: {str(e)}"
            logger.error(error_msg)
            stats.errors.append(error_msg)
        
        stats.end_time = datetime.now()
        logger.info(f"Accounts migration completed: {stats.migrated_records}/{stats.total_records} successful")
        return stats
    
    def migrate_portfolios(self) -> MigrationStats:
        """Migrate portfolios from portfolio database"""
        stats = MigrationStats(table_name="portfolios", total_records=0, migrated_records=0, failed_records=0, start_time=datetime.now())
        
        try:
            sqlite_conn = self.get_sqlite_connection(self.portfolio_db_path)
            cursor = sqlite_conn.cursor()
            
            cursor.execute("SELECT * FROM portfolios")
            portfolios = cursor.fetchall()
            stats.total_records = len(portfolios)
            
            if stats.total_records == 0:
                logger.warning("No portfolios found in SQLite database")
                stats.end_time = datetime.now()
                return stats
            
            # Generate UUID mapping
            old_ids = [port['id'] for port in portfolios]
            self.generate_uuid_mapping('portfolios', old_ids)
            
            for portfolio in portfolios:
                try:
                    new_id = self.id_mapping['portfolios'][portfolio['id']]
                    
                    # Parse JSON fields
                    account_ids = json.loads(portfolio['account_ids'])
                    allocation_weights = json.loads(portfolio['allocation_weights'])
                    performance_metrics = json.loads(portfolio['performance_metrics']) if portfolio['performance_metrics'] else {}
                    
                    # Convert account IDs to UUIDs
                    new_account_ids = [self.id_mapping['accounts'].get(acc_id, acc_id) for acc_id in account_ids]
                    new_allocation_weights = {}
                    for acc_id, weight in allocation_weights.items():
                        new_acc_id = self.id_mapping['accounts'].get(acc_id, acc_id)
                        new_allocation_weights[new_acc_id] = weight
                    
                    insert_query = """
                    INSERT INTO portfolios (
                        id, name, description, account_ids, allocation_method,
                        allocation_weights, total_value, total_profit, daily_pnl,
                        status, created_at, last_updated, performance_metrics
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """
                    
                    params = (
                        new_id, portfolio['name'], portfolio['description'],
                        json.dumps(new_account_ids), portfolio['allocation_method'],
                        json.dumps(new_allocation_weights), float(portfolio['total_value']),
                        float(portfolio['total_profit']), float(portfolio['daily_pnl']),
                        portfolio['status'], datetime.fromisoformat(portfolio['created_at']),
                        datetime.fromisoformat(portfolio['last_updated']),
                        json.dumps(performance_metrics)
                    )
                    
                    self.pg_manager.execute_command(insert_query, params)
                    stats.migrated_records += 1
                    
                except Exception as e:
                    error_msg = f"Failed to migrate portfolio {portfolio['id']}: {str(e)}"
                    logger.error(error_msg)
                    stats.errors.append(error_msg)
                    stats.failed_records += 1
            
            sqlite_conn.close()
            
        except Exception as e:
            error_msg = f"Error migrating portfolios: {str(e)}"
            logger.error(error_msg)
            stats.errors.append(error_msg)
        
        stats.end_time = datetime.now()
        logger.info(f"Portfolios migration completed: {stats.migrated_records}/{stats.total_records} successful")
        return stats
    
    def migrate_positions(self) -> MigrationStats:
        """Migrate positions from portfolio database"""
        stats = MigrationStats(table_name="positions", total_records=0, migrated_records=0, failed_records=0, start_time=datetime.now())
        
        try:
            sqlite_conn = self.get_sqlite_connection(self.portfolio_db_path)
            cursor = sqlite_conn.cursor()
            
            cursor.execute("SELECT * FROM positions")
            positions = cursor.fetchall()
            stats.total_records = len(positions)
            
            if stats.total_records == 0:
                logger.warning("No positions found in SQLite database")
                stats.end_time = datetime.now()
                return stats
            
            # Generate UUID mapping
            old_ids = [pos['id'] for pos in positions]
            self.generate_uuid_mapping('positions', old_ids)
            
            for position in positions:
                try:
                    new_id = self.id_mapping['positions'][position['id']]
                    new_account_id = self.id_mapping['accounts'].get(position['account_id'], position['account_id'])
                    
                    insert_query = """
                    INSERT INTO positions (
                        id, account_id, symbol, side, volume, open_price,
                        current_price, profit_loss, swap, commission,
                        open_time, magic_number, comment
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """
                    
                    params = (
                        new_id, new_account_id, position['symbol'], position['side'],
                        float(position['volume']), float(position['open_price']),
                        float(position['current_price']), float(position['profit_loss']),
                        float(position['swap']), float(position['commission']),
                        datetime.fromisoformat(position['open_time']),
                        int(position['magic_number']) if position['magic_number'] else None,
                        position['comment']
                    )
                    
                    self.pg_manager.execute_command(insert_query, params)
                    stats.migrated_records += 1
                    
                except Exception as e:
                    error_msg = f"Failed to migrate position {position['id']}: {str(e)}"
                    logger.error(error_msg)
                    stats.errors.append(error_msg)
                    stats.failed_records += 1
            
            sqlite_conn.close()
            
        except Exception as e:
            error_msg = f"Error migrating positions: {str(e)}"
            logger.error(error_msg)
            stats.errors.append(error_msg)
        
        stats.end_time = datetime.now()
        logger.info(f"Positions migration completed: {stats.migrated_records}/{stats.total_records} successful")
        return stats
    
    def migrate_portfolio_snapshots(self) -> MigrationStats:
        """Migrate portfolio snapshots from portfolio database"""
        stats = MigrationStats(table_name="portfolio_snapshots", total_records=0, migrated_records=0, failed_records=0, start_time=datetime.now())
        
        try:
            sqlite_conn = self.get_sqlite_connection(self.portfolio_db_path)
            cursor = sqlite_conn.cursor()
            
            cursor.execute("SELECT * FROM portfolio_snapshots")
            snapshots = cursor.fetchall()
            stats.total_records = len(snapshots)
            
            if stats.total_records == 0:
                logger.warning("No portfolio snapshots found in SQLite database")
                stats.end_time = datetime.now()
                return stats
            
            for snapshot in snapshots:
                try:
                    new_portfolio_id = self.id_mapping['portfolios'].get(snapshot['portfolio_id'], snapshot['portfolio_id'])
                    
                    # Parse JSON fields
                    account_values = json.loads(snapshot['account_values'])
                    metrics = json.loads(snapshot['metrics']) if snapshot['metrics'] else {}
                    
                    # Convert account IDs in account_values
                    new_account_values = {}
                    for acc_id, value in account_values.items():
                        new_acc_id = self.id_mapping['accounts'].get(acc_id, acc_id)
                        new_account_values[new_acc_id] = value
                    
                    insert_query = """
                    INSERT INTO portfolio_snapshots (
                        portfolio_id, timestamp, total_value, total_profit,
                        daily_pnl, account_values, positions_count, metrics
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """
                    
                    params = (
                        new_portfolio_id, datetime.fromisoformat(snapshot['timestamp']),
                        float(snapshot['total_value']), float(snapshot['total_profit']),
                        float(snapshot['daily_pnl']), json.dumps(new_account_values),
                        int(snapshot['positions_count']), json.dumps(metrics)
                    )
                    
                    self.pg_manager.execute_command(insert_query, params)
                    stats.migrated_records += 1
                    
                except Exception as e:
                    error_msg = f"Failed to migrate portfolio snapshot {snapshot['id']}: {str(e)}"
                    logger.error(error_msg)
                    stats.errors.append(error_msg)
                    stats.failed_records += 1
            
            sqlite_conn.close()
            
        except Exception as e:
            error_msg = f"Error migrating portfolio snapshots: {str(e)}"
            logger.error(error_msg)
            stats.errors.append(error_msg)
        
        stats.end_time = datetime.now()
        logger.info(f"Portfolio snapshots migration completed: {stats.migrated_records}/{stats.total_records} successful")
        return stats
    
    def migrate_trades(self) -> MigrationStats:
        """Migrate trades from core system database"""
        stats = MigrationStats(table_name="trades", total_records=0, migrated_records=0, failed_records=0, start_time=datetime.now())
        
        try:
            sqlite_conn = self.get_sqlite_connection(self.sqlite_db_path)
            cursor = sqlite_conn.cursor()
            
            cursor.execute("SELECT * FROM trades")
            trades = cursor.fetchall()
            stats.total_records = len(trades)
            
            if stats.total_records == 0:
                logger.warning("No trades found in SQLite database")
                stats.end_time = datetime.now()
                return stats
            
            # Generate UUID mapping
            old_ids = [trade['trade_id'] for trade in trades]
            self.generate_uuid_mapping('trades', old_ids)
            
            for trade in trades:
                try:
                    # Parse strategy tags
                    strategy_tags = json.loads(trade['strategy_tags']) if trade['strategy_tags'] else []
                    
                    insert_query = """
                    INSERT INTO trades (
                        trade_id, magic_number, symbol, trade_type, lot_size,
                        open_price, close_price, stop_loss, take_profit,
                        open_time, close_time, profit, commission, swap,
                        source, status, ea_name, ai_confidence, strategy_tags, notes
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """
                    
                    params = (
                        trade['trade_id'], int(trade['magic_number']) if trade['magic_number'] else None,
                        trade['symbol'], trade['trade_type'], float(trade['lot_size']),
                        float(trade['open_price']), float(trade['close_price']) if trade['close_price'] else None,
                        float(trade['stop_loss']) if trade['stop_loss'] else None,
                        float(trade['take_profit']) if trade['take_profit'] else None,
                        datetime.fromisoformat(trade['open_time']) if trade['open_time'] else None,
                        datetime.fromisoformat(trade['close_time']) if trade['close_time'] else None,
                        float(trade['profit']) if trade['profit'] else None,
                        float(trade['commission']) if trade['commission'] else None,
                        float(trade['swap']) if trade['swap'] else None,
                        trade['source'], trade['status'], trade['ea_name'],
                        float(trade['ai_confidence']) if trade['ai_confidence'] else None,
                        json.dumps(strategy_tags), trade['notes']
                    )
                    
                    self.pg_manager.execute_command(insert_query, params)
                    stats.migrated_records += 1
                    
                except Exception as e:
                    error_msg = f"Failed to migrate trade {trade['trade_id']}: {str(e)}"
                    logger.error(error_msg)
                    stats.errors.append(error_msg)
                    stats.failed_records += 1
            
            sqlite_conn.close()
            
        except Exception as e:
            error_msg = f"Error migrating trades: {str(e)}"
            logger.error(error_msg)
            stats.errors.append(error_msg)
        
        stats.end_time = datetime.now()
        logger.info(f"Trades migration completed: {stats.migrated_records}/{stats.total_records} successful")
        return stats
    
    def migrate_ea_performance(self) -> MigrationStats:
        """Migrate EA performance from core system database"""
        stats = MigrationStats(table_name="ea_performance", total_records=0, migrated_records=0, failed_records=0, start_time=datetime.now())
        
        try:
            sqlite_conn = self.get_sqlite_connection(self.sqlite_db_path)
            cursor = sqlite_conn.cursor()
            
            cursor.execute("SELECT * FROM ea_performance")
            ea_performances = cursor.fetchall()
            stats.total_records = len(ea_performances)
            
            if stats.total_records == 0:
                logger.warning("No EA performance records found in SQLite database")
                stats.end_time = datetime.now()
                return stats
            
            for ea_perf in ea_performances:
                try:
                    # Convert avg_trade_duration from seconds to interval
                    avg_duration = None
                    if ea_perf['avg_trade_duration']:
                        avg_duration = timedelta(seconds=float(ea_perf['avg_trade_duration']))
                    
                    insert_query = """
                    INSERT INTO ea_performance (
                        ea_name, magic_number, symbol, total_trades, winning_trades,
                        losing_trades, total_profit, total_loss, win_rate,
                        profit_factor, max_drawdown, avg_trade_duration,
                        last_trade_time, status, risk_score, confidence_level
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """
                    
                    params = (
                        ea_perf['ea_name'], int(ea_perf['magic_number']) if ea_perf['magic_number'] else None,
                        ea_perf['symbol'], int(ea_perf['total_trades']), int(ea_perf['winning_trades']),
                        int(ea_perf['losing_trades']), float(ea_perf['total_profit']),
                        float(ea_perf['total_loss']), float(ea_perf['win_rate']),
                        float(ea_perf['profit_factor']), float(ea_perf['max_drawdown']),
                        avg_duration,
                        datetime.fromisoformat(ea_perf['last_trade_time']) if ea_perf['last_trade_time'] else None,
                        ea_perf['status'], float(ea_perf['risk_score']), float(ea_perf['confidence_level'])
                    )
                    
                    self.pg_manager.execute_command(insert_query, params)
                    stats.migrated_records += 1
                    
                except Exception as e:
                    error_msg = f"Failed to migrate EA performance {ea_perf['ea_name']}: {str(e)}"
                    logger.error(error_msg)
                    stats.errors.append(error_msg)
                    stats.failed_records += 1
            
            sqlite_conn.close()
            
        except Exception as e:
            error_msg = f"Error migrating EA performance: {str(e)}"
            logger.error(error_msg)
            stats.errors.append(error_msg)
        
        stats.end_time = datetime.now()
        logger.info(f"EA performance migration completed: {stats.migrated_records}/{stats.total_records} successful")
        return stats
    
    def migrate_ea_profiles(self) -> MigrationStats:
        """Migrate EA profiles from EA profiling database"""
        stats = MigrationStats(table_name="ea_profiles", total_records=0, migrated_records=0, failed_records=0, start_time=datetime.now())
        
        try:
            if not self.ea_profiling_db_path.exists():
                logger.warning(f"EA profiling database not found: {self.ea_profiling_db_path}")
                stats.end_time = datetime.now()
                return stats
            
            sqlite_conn = self.get_sqlite_connection(self.ea_profiling_db_path)
            cursor = sqlite_conn.cursor()
            
            cursor.execute("SELECT * FROM ea_profiles")
            ea_profiles = cursor.fetchall()
            stats.total_records = len(ea_profiles)
            
            if stats.total_records == 0:
                logger.warning("No EA profiles found in SQLite database")
                stats.end_time = datetime.now()
                return stats
            
            for ea_profile in ea_profiles:
                try:
                    insert_query = """
                    INSERT INTO ea_profiles (
                        ea_name, magic_number, symbol, strategy_type,
                        description, profile_data, confidence_score,
                        created_at, last_updated
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """
                    
                    params = (
                        ea_profile['ea_name'], int(ea_profile['magic_number']) if ea_profile['magic_number'] else None,
                        ea_profile['symbol'], ea_profile['strategy_type'],
                        ea_profile['description'], ea_profile['profile_data'],
                        float(ea_profile['confidence_score']),
                        datetime.fromisoformat(ea_profile['created_date']) if ea_profile['created_date'] else datetime.now(),
                        datetime.fromisoformat(ea_profile['last_updated']) if ea_profile['last_updated'] else datetime.now()
                    )
                    
                    self.pg_manager.execute_command(insert_query, params)
                    stats.migrated_records += 1
                    
                except Exception as e:
                    error_msg = f"Failed to migrate EA profile {ea_profile['ea_name']}: {str(e)}"
                    logger.error(error_msg)
                    stats.errors.append(error_msg)
                    stats.failed_records += 1
            
            sqlite_conn.close()
            
        except Exception as e:
            error_msg = f"Error migrating EA profiles: {str(e)}"
            logger.error(error_msg)
            stats.errors.append(error_msg)
        
        stats.end_time = datetime.now()
        logger.info(f"EA profiles migration completed: {stats.migrated_records}/{stats.total_records} successful")
        return stats
    
    def migrate_ea_indicators(self) -> MigrationStats:
        """Migrate EA indicators from EA profiling database"""
        stats = MigrationStats(table_name="ea_indicators", total_records=0, migrated_records=0, failed_records=0, start_time=datetime.now())
        
        try:
            if not self.ea_profiling_db_path.exists():
                logger.warning(f"EA profiling database not found: {self.ea_profiling_db_path}")
                stats.end_time = datetime.now()
                return stats
            
            sqlite_conn = self.get_sqlite_connection(self.ea_profiling_db_path)
            cursor = sqlite_conn.cursor()
            
            cursor.execute("SELECT * FROM ea_indicators")
            ea_indicators = cursor.fetchall()
            stats.total_records = len(ea_indicators)
            
            if stats.total_records == 0:
                logger.warning("No EA indicators found in SQLite database")
                stats.end_time = datetime.now()
                return stats
            
            for ea_indicator in ea_indicators:
                try:
                    insert_query = """
                    INSERT INTO ea_indicators (
                        ea_name, indicator_name, indicator_type,
                        parameters, timeframe, weight
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s
                    )
                    """
                    
                    params = (
                        ea_indicator['ea_name'], ea_indicator['indicator_name'],
                        ea_indicator['indicator_type'], ea_indicator['parameters'],
                        ea_indicator['timeframe'], float(ea_indicator['weight'])
                    )
                    
                    self.pg_manager.execute_command(insert_query, params)
                    stats.migrated_records += 1
                    
                except Exception as e:
                    error_msg = f"Failed to migrate EA indicator {ea_indicator['id']}: {str(e)}"
                    logger.error(error_msg)
                    stats.errors.append(error_msg)
                    stats.failed_records += 1
            
            sqlite_conn.close()
            
        except Exception as e:
            error_msg = f"Error migrating EA indicators: {str(e)}"
            logger.error(error_msg)
            stats.errors.append(error_msg)
        
        stats.end_time = datetime.now()
        logger.info(f"EA indicators migration completed: {stats.migrated_records}/{stats.total_records} successful")
        return stats
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity after migration"""
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'overall_status': 'PASSED'
        }
        
        try:
            # Check record counts
            for table_name in ['accounts', 'portfolios', 'positions', 'trades', 'ea_performance', 'ea_profiles']:
                try:
                    count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                    result = self.pg_manager.execute_query(count_query, fetch_all=False)
                    count = result['count'] if result else 0
                    
                    validation_results['checks'][f'{table_name}_count'] = {
                        'status': 'PASSED',
                        'count': count,
                        'message': f'{table_name} has {count} records'
                    }
                    
                except Exception as e:
                    validation_results['checks'][f'{table_name}_count'] = {
                        'status': 'FAILED',
                        'error': str(e),
                        'message': f'Failed to count {table_name} records'
                    }
                    validation_results['overall_status'] = 'FAILED'
            
            # Check foreign key constraints
            fk_checks = [
                ('positions', 'account_id', 'accounts', 'id'),
                ('portfolio_snapshots', 'portfolio_id', 'portfolios', 'id')
            ]
            
            for child_table, fk_column, parent_table, pk_column in fk_checks:
                try:
                    fk_query = f"""
                    SELECT COUNT(*) as orphan_count 
                    FROM {child_table} c 
                    LEFT JOIN {parent_table} p ON c.{fk_column} = p.{pk_column}
                    WHERE p.{pk_column} IS NULL AND c.{fk_column} IS NOT NULL
                    """
                    
                    result = self.pg_manager.execute_query(fk_query, fetch_all=False)
                    orphan_count = result['orphan_count'] if result else 0
                    
                    if orphan_count == 0:
                        validation_results['checks'][f'{child_table}_fk_{fk_column}'] = {
                            'status': 'PASSED',
                            'message': f'No orphaned records in {child_table}.{fk_column}'
                        }
                    else:
                        validation_results['checks'][f'{child_table}_fk_{fk_column}'] = {
                            'status': 'FAILED',
                            'orphan_count': orphan_count,
                            'message': f'Found {orphan_count} orphaned records in {child_table}.{fk_column}'
                        }
                        validation_results['overall_status'] = 'FAILED'
                        
                except Exception as e:
                    validation_results['checks'][f'{child_table}_fk_{fk_column}'] = {
                        'status': 'ERROR',
                        'error': str(e),
                        'message': f'Error checking FK constraint {child_table}.{fk_column}'
                    }
                    validation_results['overall_status'] = 'FAILED'
            
            # Check data consistency
            try:
                # Check if all EA names in trades exist in ea_performance
                ea_consistency_query = """
                SELECT COUNT(*) as missing_ea_count
                FROM trades t
                LEFT JOIN ea_performance ep ON t.ea_name = ep.ea_name
                WHERE t.ea_name IS NOT NULL AND ep.ea_name IS NULL
                """
                
                result = self.pg_manager.execute_query(ea_consistency_query, fetch_all=False)
                missing_ea_count = result['missing_ea_count'] if result else 0
                
                if missing_ea_count == 0:
                    validation_results['checks']['ea_consistency'] = {
                        'status': 'PASSED',
                        'message': 'All EA names in trades have corresponding performance records'
                    }
                else:
                    validation_results['checks']['ea_consistency'] = {
                        'status': 'WARNING',
                        'missing_count': missing_ea_count,
                        'message': f'Found {missing_ea_count} trades with EA names not in ea_performance'
                    }
                    
            except Exception as e:
                validation_results['checks']['ea_consistency'] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'message': 'Error checking EA consistency'
                }
                validation_results['overall_status'] = 'FAILED'
            
            logger.info(f"Data integrity validation completed: {validation_results['overall_status']}")
            
        except Exception as e:
            logger.error(f"Data integrity validation failed: {e}")
            validation_results['overall_status'] = 'FAILED'
            validation_results['error'] = str(e)
        
        return validation_results
    
    def run_full_migration(self) -> Dict[str, Any]:
        """Run complete migration process"""
        migration_report = {
            'start_time': datetime.now().isoformat(),
            'status': 'STARTED',
            'stats': [],
            'validation': None,
            'id_mapping': self.id_mapping
        }
        
        try:
            # Validate PostgreSQL schema
            if not self.validate_postgresql_schema():
                migration_report['status'] = 'FAILED'
                migration_report['error'] = 'PostgreSQL schema validation failed'
                return migration_report
            
            # Run migrations in order
            migration_order = [
                ('accounts', self.migrate_accounts),
                ('portfolios', self.migrate_portfolios),
                ('positions', self.migrate_positions),
                ('portfolio_snapshots', self.migrate_portfolio_snapshots),
                ('trades', self.migrate_trades),
                ('ea_performance', self.migrate_ea_performance),
                ('ea_profiles', self.migrate_ea_profiles),
                ('ea_indicators', self.migrate_ea_indicators)
            ]
            
            for table_name, migration_func in migration_order:
                logger.info(f"Starting migration of {table_name}...")
                stats = migration_func()
                self.migration_stats.append(stats)
                migration_report['stats'].append({
                    'table': stats.table_name,
                    'total_records': stats.total_records,
                    'migrated_records': stats.migrated_records,
                    'failed_records': stats.failed_records,
                    'success_rate': stats.success_rate,
                    'duration': str(stats.duration),
                    'errors': stats.errors
                })
            
            # Validate data integrity
            logger.info("Validating data integrity...")
            validation_results = self.validate_data_integrity()
            migration_report['validation'] = validation_results
            
            # Determine overall status
            failed_migrations = [s for s in self.migration_stats if s.failed_records > 0]
            if failed_migrations or validation_results['overall_status'] == 'FAILED':
                migration_report['status'] = 'COMPLETED_WITH_ERRORS'
            else:
                migration_report['status'] = 'COMPLETED_SUCCESSFULLY'
            
            migration_report['end_time'] = datetime.now().isoformat()
            
            # Save migration report
            self.save_migration_report(migration_report)
            
            logger.info(f"Migration completed with status: {migration_report['status']}")
            
        except Exception as e:
            migration_report['status'] = 'FAILED'
            migration_report['error'] = str(e)
            migration_report['end_time'] = datetime.now().isoformat()
            logger.error(f"Migration failed: {e}")
        
        return migration_report
    
    def save_migration_report(self, report: Dict[str, Any]):
        """Save migration report to file"""
        try:
            report_path = Path("migration_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Migration report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving migration report: {e}")
    
    def cleanup_sqlite_data(self, backup_first: bool = True):
        """Cleanup SQLite data after successful migration"""
        if backup_first:
            # Create backup before cleanup
            backup_dir = Path("migration_backup")
            backup_dir.mkdir(exist_ok=True)
            
            for db_path in [self.sqlite_db_path, self.portfolio_db_path, self.ea_profiling_db_path]:
                if db_path.exists():
                    backup_path = backup_dir / f"{db_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                    db_path.rename(backup_path)
                    logger.info(f"Backed up {db_path} to {backup_path}")
        
        logger.info("SQLite data cleanup completed")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('migration.log'),
            logging.StreamHandler()
        ]
    )
    
    # Run migration
    migrator = QNTIDataMigrator()
    
    # Test PostgreSQL connection first
    if not migrator.pg_manager.test_connection():
        print("PostgreSQL connection failed. Please check your database configuration.")
        exit(1)
    
    print("Starting QNTI data migration...")
    
    # Run full migration
    report = migrator.run_full_migration()
    
    # Print summary
    print(f"\nMigration Status: {report['status']}")
    print(f"\nMigration Summary:")
    for stat in report['stats']:
        print(f"  {stat['table']}: {stat['migrated_records']}/{stat['total_records']} records migrated ({stat['success_rate']:.1f}%)")
    
    if report['validation']:
        print(f"\nData Validation: {report['validation']['overall_status']}")
    
    if report['status'] == 'COMPLETED_SUCCESSFULLY':
        print("\nMigration completed successfully!")
        
        # Ask user if they want to cleanup SQLite files
        response = input("\nDo you want to backup and remove SQLite files? (y/n): ")
        if response.lower() == 'y':
            migrator.cleanup_sqlite_data(backup_first=True)
            print("SQLite files backed up and cleaned.")
    else:
        print("\nMigration completed with errors. Please check the migration report.")
    
    print(f"\nDetailed migration report saved to: migration_report.json")
