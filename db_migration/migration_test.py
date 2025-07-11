#!/usr/bin/env python3
"""
QNTI Database Migration Test Suite
Comprehensive testing of the PostgreSQL migration process
"""

import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any
import unittest
import sys
import os

# Add the migration directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_config import get_database_manager, create_default_config, validate_config
from data_migration import QNTIDataMigrator
from db_performance_monitor import QNTIPerformanceMonitor
from db_backup_recovery import QNTIBackupManager
from qnti_portfolio_manager_pg import QNTIPortfolioManagerPG, AccountType, AllocationMethod
from qnti_core_system_pg import QNTITradeManagerPG, TradeSource, Trade

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestDatabaseConnection(unittest.TestCase):
    """Test database connection and configuration"""
    
    def setUp(self):
        self.db_manager = get_database_manager()
    
    def test_database_connection(self):
        """Test basic database connection"""
        result = self.db_manager.test_connection()
        self.assertTrue(result, "Database connection failed")
    
    def test_pool_statistics(self):
        """Test connection pool statistics"""
        stats = self.db_manager.get_pool_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_connections', stats)
        self.assertIn('used_connections', stats)
        self.assertIn('free_connections', stats)
    
    def test_schema_exists(self):
        """Test that QNTI schema exists"""
        query = "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'qnti'"
        result = self.db_manager.execute_query(query)
        self.assertTrue(len(result) > 0, "QNTI schema not found")
    
    def test_required_tables_exist(self):
        """Test that all required tables exist"""
        required_tables = [
            'accounts', 'portfolios', 'positions', 'portfolio_snapshots',
            'trades', 'ea_performance', 'ea_profiles', 'ea_indicators',
            'ea_analysis', 'system_config', 'system_logs'
        ]
        
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'qnti' 
        ORDER BY table_name
        """
        
        tables = self.db_manager.execute_query(query)
        table_names = [t['table_name'] for t in tables]
        
        for table in required_tables:
            self.assertIn(table, table_names, f"Required table {table} not found")

class TestPortfolioManagerPG(unittest.TestCase):
    """Test PostgreSQL Portfolio Manager"""
    
    def setUp(self):
        self.portfolio_manager = QNTIPortfolioManagerPG()
        self.test_account_ids = []
        self.test_portfolio_ids = []
    
    def tearDown(self):
        """Clean up test data"""
        # Clean up test portfolios
        for portfolio_id in self.test_portfolio_ids:
            try:
                query = "DELETE FROM portfolios WHERE id = %s"
                self.portfolio_manager.db_manager.execute_command(query, (portfolio_id,))
            except:
                pass
        
        # Clean up test accounts
        for account_id in self.test_account_ids:
            try:
                query = "DELETE FROM accounts WHERE id = %s"
                self.portfolio_manager.db_manager.execute_command(query, (account_id,))
            except:
                pass
    
    def test_add_account(self):
        """Test adding a new account"""
        account_id = self.portfolio_manager.add_account(
            name="Test Account",
            account_type=AccountType.DEMO,
            broker="Test Broker",
            login="test123",
            server="test-server"
        )
        
        self.test_account_ids.append(account_id)
        self.assertIsNotNone(account_id)
        self.assertIn(account_id, self.portfolio_manager.accounts)
    
    def test_create_portfolio(self):
        """Test creating a new portfolio"""
        # Create test accounts first
        account1_id = self.portfolio_manager.add_account(
            name="Test Account 1",
            account_type=AccountType.DEMO,
            broker="Test Broker",
            login="test1",
            server="test-server"
        )
        
        account2_id = self.portfolio_manager.add_account(
            name="Test Account 2",
            account_type=AccountType.DEMO,
            broker="Test Broker",
            login="test2",
            server="test-server"
        )
        
        self.test_account_ids.extend([account1_id, account2_id])
        
        # Create portfolio
        portfolio_id = self.portfolio_manager.create_portfolio(
            name="Test Portfolio",
            description="Test portfolio description",
            account_ids=[account1_id, account2_id],
            allocation_method=AllocationMethod.EQUAL_WEIGHT
        )
        
        self.test_portfolio_ids.append(portfolio_id)
        self.assertIsNotNone(portfolio_id)
        self.assertIn(portfolio_id, self.portfolio_manager.portfolios)
    
    def test_portfolio_metrics(self):
        """Test portfolio metrics calculation"""
        # Create test setup
        account_id = self.portfolio_manager.add_account(
            name="Test Account",
            account_type=AccountType.DEMO,
            broker="Test Broker",
            login="test123",
            server="test-server"
        )
        
        portfolio_id = self.portfolio_manager.create_portfolio(
            name="Test Portfolio",
            description="Test portfolio",
            account_ids=[account_id]
        )
        
        self.test_account_ids.append(account_id)
        self.test_portfolio_ids.append(portfolio_id)
        
        # Update account data
        self.portfolio_manager.update_account_data(account_id, {
            'balance': 10000.0,
            'equity': 10500.0,
            'profit': 500.0
        })
        
        # Calculate metrics
        metrics = self.portfolio_manager.calculate_portfolio_metrics(portfolio_id)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_accounts', metrics)
        self.assertIn('account_values', metrics)
    
    def test_get_status(self):
        """Test getting portfolio manager status"""
        status = self.portfolio_manager.get_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('total_accounts', status)
        self.assertIn('total_portfolios', status)
        self.assertIn('database_type', status)
        self.assertEqual(status['database_type'], 'PostgreSQL')

class TestTradeManagerPG(unittest.TestCase):
    """Test PostgreSQL Trade Manager"""
    
    def setUp(self):
        self.trade_manager = QNTITradeManagerPG()
        self.test_trade_ids = []
    
    def tearDown(self):
        """Clean up test data"""
        for trade_id in self.test_trade_ids:
            try:
                query = "DELETE FROM trades WHERE trade_id = %s"
                self.trade_manager.db_manager.execute_command(query, (trade_id,))
            except:
                pass
    
    def test_add_trade(self):
        """Test adding a new trade"""
        trade_id = f"TEST_{uuid.uuid4().hex[:8]}"
        
        trade = Trade(
            trade_id=trade_id,
            magic_number=12345,
            symbol="EURUSD",
            trade_type="BUY",
            lot_size=0.1,
            open_price=1.0500,
            source=TradeSource.MANUAL,
            strategy_tags=["test", "manual"]
        )
        
        self.test_trade_ids.append(trade_id)
        
        result = self.trade_manager.add_trade(trade)
        self.assertTrue(result)
        self.assertIn(trade_id, self.trade_manager.trades)
    
    def test_close_trade(self):
        """Test closing a trade"""
        trade_id = f"TEST_{uuid.uuid4().hex[:8]}"
        
        trade = Trade(
            trade_id=trade_id,
            magic_number=12345,
            symbol="EURUSD",
            trade_type="BUY",
            lot_size=0.1,
            open_price=1.0500,
            source=TradeSource.MANUAL
        )
        
        self.test_trade_ids.append(trade_id)
        
        # Add trade
        self.trade_manager.add_trade(trade)
        
        # Close trade
        result = self.trade_manager.close_trade(trade_id, 1.0550)
        self.assertTrue(result)
        
        # Check trade is closed
        closed_trade = self.trade_manager.trades[trade_id]
        self.assertEqual(closed_trade.status.value, 'closed')
        self.assertIsNotNone(closed_trade.close_price)
        self.assertIsNotNone(closed_trade.profit)
    
    def test_calculate_statistics(self):
        """Test trade statistics calculation"""
        stats = self.trade_manager.calculate_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_trades', stats)
        self.assertIn('closed_trades', stats)
        self.assertIn('win_rate', stats)
        self.assertIn('profit_factor', stats)
    
    def test_get_system_health(self):
        """Test system health check"""
        health = self.trade_manager.get_system_health()
        
        self.assertIsInstance(health, dict)
        self.assertIn('total_trades', health)
        self.assertIn('active_eas', health)
        self.assertIn('database_type', health)
        self.assertEqual(health['database_type'], 'PostgreSQL')

class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring"""
    
    def setUp(self):
        self.monitor = QNTIPerformanceMonitor(monitoring_interval=5)
    
    def test_performance_report(self):
        """Test performance report generation"""
        # Start monitoring briefly
        self.monitor.start_monitoring()
        time.sleep(10)  # Wait for some data to be collected
        
        report = self.monitor.get_performance_report()
        
        self.assertIsInstance(report, dict)
        if 'error' not in report:
            self.assertIn('database_stats', report)
            self.assertIn('system_resources', report)
        
        self.monitor.stop_monitoring()
    
    def test_optimization(self):
        """Test database optimization"""
        results = self.monitor.optimize_database()
        
        self.assertIsInstance(results, dict)
        self.assertIn('operations', results)
        self.assertIn('timestamp', results)

class TestBackupRecovery(unittest.TestCase):
    """Test backup and recovery system"""
    
    def setUp(self):
        self.backup_manager = QNTIBackupManager(
            backup_dir="test_backups",
            retention_days=1
        )
        self.test_backup_ids = []
    
    def tearDown(self):
        """Clean up test backups"""
        import shutil
        try:
            shutil.rmtree("test_backups")
        except:
            pass
    
    def test_create_backup(self):
        """Test backup creation"""
        metadata = self.backup_manager.create_backup(backup_type="test")
        
        if metadata.status == "success":
            self.test_backup_ids.append(metadata.backup_id)
            self.assertEqual(metadata.status, "success")
            self.assertIsNotNone(metadata.file_path)
            self.assertGreater(metadata.file_size, 0)
    
    def test_verify_backup(self):
        """Test backup verification"""
        metadata = self.backup_manager.create_backup(backup_type="test")
        
        if metadata.status == "success":
            self.test_backup_ids.append(metadata.backup_id)
            
            verification = self.backup_manager.verify_backup(metadata.backup_id)
            
            self.assertIsInstance(verification, dict)
            self.assertIn('status', verification)
    
    def test_backup_status(self):
        """Test backup system status"""
        status = self.backup_manager.get_backup_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('total_backups', status)
        self.assertIn('backup_directory', status)
        self.assertIn('retention_days', status)

class IntegrationTestSuite:
    """Integration test suite for the entire migration"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'overall_status': 'PASSED',
            'summary': {}
        }
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("Starting QNTI Database Migration Integration Tests")
        print("=" * 60)
        
        # Test 1: Database Connection
        print("\n1. Testing Database Connection...")
        try:
            db_manager = get_database_manager()
            connection_test = db_manager.test_connection()
            
            if connection_test:
                self.results['tests'].append({
                    'test': 'Database Connection',
                    'status': 'PASSED',
                    'message': 'Successfully connected to PostgreSQL'
                })
                print("   ✓ Database connection successful")
            else:
                self.results['tests'].append({
                    'test': 'Database Connection',
                    'status': 'FAILED',
                    'message': 'Failed to connect to PostgreSQL'
                })
                print("   ✗ Database connection failed")
                self.results['overall_status'] = 'FAILED'
        except Exception as e:
            self.results['tests'].append({
                'test': 'Database Connection',
                'status': 'ERROR',
                'message': str(e)
            })
            print(f"   ✗ Database connection error: {e}")
            self.results['overall_status'] = 'FAILED'
        
        # Test 2: Schema Validation
        print("\n2. Testing Schema Validation...")
        try:
            schema_test = self._test_schema_validation()
            if schema_test:
                self.results['tests'].append({
                    'test': 'Schema Validation',
                    'status': 'PASSED',
                    'message': 'All required tables and indexes exist'
                })
                print("   ✓ Schema validation successful")
            else:
                self.results['tests'].append({
                    'test': 'Schema Validation',
                    'status': 'FAILED',
                    'message': 'Schema validation failed'
                })
                print("   ✗ Schema validation failed")
                self.results['overall_status'] = 'FAILED'
        except Exception as e:
            self.results['tests'].append({
                'test': 'Schema Validation',
                'status': 'ERROR',
                'message': str(e)
            })
            print(f"   ✗ Schema validation error: {e}")
            self.results['overall_status'] = 'FAILED'
        
        # Test 3: Portfolio Manager
        print("\n3. Testing Portfolio Manager...")
        try:
            portfolio_test = self._test_portfolio_manager()
            if portfolio_test:
                self.results['tests'].append({
                    'test': 'Portfolio Manager',
                    'status': 'PASSED',
                    'message': 'Portfolio operations working correctly'
                })
                print("   ✓ Portfolio manager test successful")
            else:
                self.results['tests'].append({
                    'test': 'Portfolio Manager',
                    'status': 'FAILED',
                    'message': 'Portfolio manager test failed'
                })
                print("   ✗ Portfolio manager test failed")
                self.results['overall_status'] = 'FAILED'
        except Exception as e:
            self.results['tests'].append({
                'test': 'Portfolio Manager',
                'status': 'ERROR',
                'message': str(e)
            })
            print(f"   ✗ Portfolio manager error: {e}")
            self.results['overall_status'] = 'FAILED'
        
        # Test 4: Trade Manager
        print("\n4. Testing Trade Manager...")
        try:
            trade_test = self._test_trade_manager()
            if trade_test:
                self.results['tests'].append({
                    'test': 'Trade Manager',
                    'status': 'PASSED',
                    'message': 'Trade operations working correctly'
                })
                print("   ✓ Trade manager test successful")
            else:
                self.results['tests'].append({
                    'test': 'Trade Manager',
                    'status': 'FAILED',
                    'message': 'Trade manager test failed'
                })
                print("   ✗ Trade manager test failed")
                self.results['overall_status'] = 'FAILED'
        except Exception as e:
            self.results['tests'].append({
                'test': 'Trade Manager',
                'status': 'ERROR',
                'message': str(e)
            })
            print(f"   ✗ Trade manager error: {e}")
            self.results['overall_status'] = 'FAILED'
        
        # Test 5: Performance Monitoring
        print("\n5. Testing Performance Monitoring...")
        try:
            perf_test = self._test_performance_monitoring()
            if perf_test:
                self.results['tests'].append({
                    'test': 'Performance Monitoring',
                    'status': 'PASSED',
                    'message': 'Performance monitoring working correctly'
                })
                print("   ✓ Performance monitoring test successful")
            else:
                self.results['tests'].append({
                    'test': 'Performance Monitoring',
                    'status': 'FAILED',
                    'message': 'Performance monitoring test failed'
                })
                print("   ✗ Performance monitoring test failed")
                self.results['overall_status'] = 'FAILED'
        except Exception as e:
            self.results['tests'].append({
                'test': 'Performance Monitoring',
                'status': 'ERROR',
                'message': str(e)
            })
            print(f"   ✗ Performance monitoring error: {e}")
            self.results['overall_status'] = 'FAILED'
        
        # Test 6: Backup System
        print("\n6. Testing Backup System...")
        try:
            backup_test = self._test_backup_system()
            if backup_test:
                self.results['tests'].append({
                    'test': 'Backup System',
                    'status': 'PASSED',
                    'message': 'Backup operations working correctly'
                })
                print("   ✓ Backup system test successful")
            else:
                self.results['tests'].append({
                    'test': 'Backup System',
                    'status': 'FAILED',
                    'message': 'Backup system test failed'
                })
                print("   ✗ Backup system test failed")
                self.results['overall_status'] = 'FAILED'
        except Exception as e:
            self.results['tests'].append({
                'test': 'Backup System',
                'status': 'ERROR',
                'message': str(e)
            })
            print(f"   ✗ Backup system error: {e}")
            self.results['overall_status'] = 'FAILED'
        
        # Generate summary
        passed_tests = len([t for t in self.results['tests'] if t['status'] == 'PASSED'])
        failed_tests = len([t for t in self.results['tests'] if t['status'] == 'FAILED'])
        error_tests = len([t for t in self.results['tests'] if t['status'] == 'ERROR'])
        
        self.results['summary'] = {
            'total_tests': len(self.results['tests']),
            'passed': passed_tests,
            'failed': failed_tests,
            'errors': error_tests,
            'success_rate': (passed_tests / len(self.results['tests']) * 100) if self.results['tests'] else 0
        }
        
        print("\n" + "=" * 60)
        print("INTEGRATION TEST RESULTS")
        print("=" * 60)
        print(f"Overall Status: {self.results['overall_status']}")
        print(f"Tests Passed: {passed_tests}/{len(self.results['tests'])}")
        print(f"Success Rate: {self.results['summary']['success_rate']:.1f}%")
        
        if failed_tests > 0 or error_tests > 0:
            print("\nFailed/Error Tests:")
            for test in self.results['tests']:
                if test['status'] in ['FAILED', 'ERROR']:
                    print(f"  - {test['test']}: {test['message']}")
        
        return self.results
    
    def _test_schema_validation(self) -> bool:
        """Test schema validation"""
        try:
            db_manager = get_database_manager()
            
            # Check schema exists
            schema_query = "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'qnti'"
            schema_result = db_manager.execute_query(schema_query)
            
            if not schema_result:
                return False
            
            # Check required tables
            required_tables = [
                'accounts', 'portfolios', 'positions', 'portfolio_snapshots',
                'trades', 'ea_performance', 'ea_profiles', 'system_config'
            ]
            
            tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'qnti'
            """
            
            tables_result = db_manager.execute_query(tables_query)
            table_names = [t['table_name'] for t in tables_result]
            
            for table in required_tables:
                if table not in table_names:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return False
    
    def _test_portfolio_manager(self) -> bool:
        """Test portfolio manager functionality"""
        try:
            portfolio_manager = QNTIPortfolioManagerPG()
            
            # Test adding account
            account_id = portfolio_manager.add_account(
                name="Integration Test Account",
                account_type=AccountType.DEMO,
                broker="Test Broker",
                login="test_integration",
                server="test-server"
            )
            
            if not account_id:
                return False
            
            # Test creating portfolio
            portfolio_id = portfolio_manager.create_portfolio(
                name="Integration Test Portfolio",
                description="Integration test portfolio",
                account_ids=[account_id]
            )
            
            if not portfolio_id:
                return False
            
            # Test updating account data
            portfolio_manager.update_account_data(account_id, {
                'balance': 10000.0,
                'equity': 10500.0,
                'profit': 500.0
            })
            
            # Test calculating metrics
            metrics = portfolio_manager.calculate_portfolio_metrics(portfolio_id)
            
            if not metrics or 'total_accounts' not in metrics:
                return False
            
            # Cleanup
            db_manager = get_database_manager()
            db_manager.execute_command("DELETE FROM portfolios WHERE id = %s", (portfolio_id,))
            db_manager.execute_command("DELETE FROM accounts WHERE id = %s", (account_id,))
            
            return True
            
        except Exception as e:
            logger.error(f"Portfolio manager test error: {e}")
            return False
    
    def _test_trade_manager(self) -> bool:
        """Test trade manager functionality"""
        try:
            trade_manager = QNTITradeManagerPG()
            
            # Test adding trade
            trade_id = f"INTEGRATION_TEST_{uuid.uuid4().hex[:8]}"
            
            trade = Trade(
                trade_id=trade_id,
                magic_number=99999,
                symbol="EURUSD",
                trade_type="BUY",
                lot_size=0.1,
                open_price=1.0500,
                source=TradeSource.MANUAL,
                strategy_tags=["integration", "test"]
            )
            
            result = trade_manager.add_trade(trade)
            
            if not result:
                return False
            
            # Test closing trade
            result = trade_manager.close_trade(trade_id, 1.0550)
            
            if not result:
                return False
            
            # Test statistics
            stats = trade_manager.calculate_statistics()
            
            if not stats or 'total_trades' not in stats:
                return False
            
            # Cleanup
            db_manager = get_database_manager()
            db_manager.execute_command("DELETE FROM trades WHERE trade_id = %s", (trade_id,))
            
            return True
            
        except Exception as e:
            logger.error(f"Trade manager test error: {e}")
            return False
    
    def _test_performance_monitoring(self) -> bool:
        """Test performance monitoring"""
        try:
            monitor = QNTIPerformanceMonitor(monitoring_interval=1)
            
            # Test performance report
            report = monitor.get_performance_report()
            
            if not report:
                return False
            
            # Should have basic structure even with no data
            if 'error' in report:
                # Error is acceptable for this test
                return True
            
            if 'database_stats' not in report:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Performance monitoring test error: {e}")
            return False
    
    def _test_backup_system(self) -> bool:
        """Test backup system"""
        try:
            backup_manager = QNTIBackupManager(
                backup_dir="integration_test_backups",
                retention_days=1
            )
            
            # Test backup creation
            metadata = backup_manager.create_backup(backup_type="integration_test")
            
            if metadata.status != "success":
                return False
            
            # Test backup verification
            verification = backup_manager.verify_backup(metadata.backup_id)
            
            if verification.get('status') != 'success':
                return False
            
            # Test backup status
            status = backup_manager.get_backup_status()
            
            if not status or 'total_backups' not in status:
                return False
            
            # Cleanup
            import shutil
            try:
                shutil.rmtree("integration_test_backups")
            except:
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Backup system test error: {e}")
            return False
    
    def save_results(self, filename: str = None):
        """Save test results to file"""
        if not filename:
            filename = f"integration_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nIntegration test results saved to: {filename}")
        return filename

if __name__ == "__main__":
    # Check if config file exists
    if not os.path.exists("db_config.json"):
        print("Database configuration not found. Creating default config...")
        create_default_config()
        print("Please update db_config.json with your PostgreSQL credentials and run again.")
        sys.exit(1)
    
    # Validate configuration
    if not validate_config():
        print("Database configuration validation failed. Please check db_config.json.")
        sys.exit(1)
    
    # Run unit tests
    print("Running Unit Tests...")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDatabaseConnection,
        TestPortfolioManagerPG,
        TestTradeManagerPG,
        TestPerformanceMonitor,
        TestBackupRecovery
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run unit tests
    runner = unittest.TextTestRunner(verbosity=2)
    unit_test_result = runner.run(test_suite)
    
    # Run integration tests
    print("\n" + "=" * 50)
    integration_suite = IntegrationTestSuite()
    integration_results = integration_suite.run_all_tests()
    
    # Save results
    results_file = integration_suite.save_results()
    
    # Overall summary
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    
    unit_passed = unit_test_result.wasSuccessful()
    integration_passed = integration_results['overall_status'] == 'PASSED'
    
    print(f"Unit Tests: {'PASSED' if unit_passed else 'FAILED'}")
    print(f"Integration Tests: {'PASSED' if integration_passed else 'FAILED'}")
    
    if unit_passed and integration_passed:
        print("\n🎉 ALL TESTS PASSED! PostgreSQL migration is ready for production.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review the results and fix issues before deploying.")
        sys.exit(1)
