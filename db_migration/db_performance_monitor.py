#!/usr/bin/env python3
"""
QNTI Database Performance Monitor
Monitors PostgreSQL database performance and provides optimization recommendations
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import threading
import psutil
import pandas as pd

from database_config import get_database_manager, execute_query

logger = logging.getLogger(__name__)

@dataclass
class QueryPerformanceMetric:
    """Query performance metric"""
    query: str
    execution_time: float
    rows_returned: int
    timestamp: datetime
    database_name: str
    
@dataclass
class DatabaseStats:
    """Database statistics"""
    timestamp: datetime
    total_connections: int
    active_connections: int
    idle_connections: int
    database_size: float  # MB
    cache_hit_ratio: float
    transactions_per_second: float
    average_query_time: float
    slow_queries_count: int
    deadlocks_count: int
    
class QNTIPerformanceMonitor:
    """PostgreSQL performance monitoring system"""
    
    def __init__(self, monitoring_interval: int = 60):
        self.monitoring_interval = monitoring_interval
        self.db_manager = get_database_manager()
        self.running = False
        self.monitor_thread = None
        
        # Performance metrics storage
        self.query_metrics: List[QueryPerformanceMetric] = []
        self.database_stats: List[DatabaseStats] = []
        
        # Thresholds for alerts
        self.slow_query_threshold = 1.0  # seconds
        self.high_connection_threshold = 80  # percent of max connections
        self.low_cache_hit_threshold = 0.90  # 90%
        
        logger.info("QNTI Performance Monitor initialized")
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect database statistics
                stats = self._collect_database_stats()
                if stats:
                    self.database_stats.append(stats)
                    
                    # Keep only last 24 hours of stats
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    self.database_stats = [s for s in self.database_stats if s.timestamp >= cutoff_time]
                    
                    # Check for performance alerts
                    self._check_performance_alerts(stats)
                
                # Collect slow query information
                self._collect_slow_queries()
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_database_stats(self) -> Optional[DatabaseStats]:
        """Collect database statistics"""
        try:
            # Get connection statistics
            connections_query = """
            SELECT 
                count(*) as total_connections,
                count(*) FILTER (WHERE state = 'active') as active_connections,
                count(*) FILTER (WHERE state = 'idle') as idle_connections
            FROM pg_stat_activity
            WHERE datname = current_database()
            """
            
            conn_result = execute_query(connections_query, fetch_all=False)
            
            # Get database size
            size_query = """
            SELECT pg_size_pretty(pg_database_size(current_database())) as size_pretty,
                   pg_database_size(current_database()) as size_bytes
            """
            
            size_result = execute_query(size_query, fetch_all=False)
            
            # Get cache hit ratio
            cache_query = """
            SELECT 
                sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as cache_hit_ratio
            FROM pg_statio_user_tables
            WHERE heap_blks_hit + heap_blks_read > 0
            """
            
            cache_result = execute_query(cache_query, fetch_all=False)
            
            # Get transaction statistics
            txn_query = """
            SELECT 
                xact_commit + xact_rollback as total_transactions,
                xact_commit,
                xact_rollback
            FROM pg_stat_database
            WHERE datname = current_database()
            """
            
            txn_result = execute_query(txn_query, fetch_all=False)
            
            # Get slow queries count
            slow_query_query = """
            SELECT count(*) as slow_queries
            FROM pg_stat_statements
            WHERE mean_exec_time > %s
            """
            
            try:
                slow_query_result = execute_query(slow_query_query, (self.slow_query_threshold * 1000,), fetch_all=False)
                slow_queries_count = int(slow_query_result['slow_queries']) if slow_query_result else 0
            except:
                # pg_stat_statements might not be enabled
                slow_queries_count = 0
            
            # Calculate TPS (approximate)
            tps = 0.0
            if len(self.database_stats) > 0:
                prev_stats = self.database_stats[-1]
                if txn_result and prev_stats:
                    time_diff = (datetime.now() - prev_stats.timestamp).total_seconds()
                    if time_diff > 0:
                        # This is a simplified TPS calculation
                        tps = float(txn_result['total_transactions']) / time_diff
            
            return DatabaseStats(
                timestamp=datetime.now(),
                total_connections=int(conn_result['total_connections']) if conn_result else 0,
                active_connections=int(conn_result['active_connections']) if conn_result else 0,
                idle_connections=int(conn_result['idle_connections']) if conn_result else 0,
                database_size=float(size_result['size_bytes']) / (1024 * 1024) if size_result else 0.0,  # MB
                cache_hit_ratio=float(cache_result['cache_hit_ratio']) if cache_result and cache_result['cache_hit_ratio'] else 0.0,
                transactions_per_second=tps,
                average_query_time=0.0,  # Will be calculated separately
                slow_queries_count=slow_queries_count,
                deadlocks_count=0  # Will be implemented if needed
            )
            
        except Exception as e:
            logger.error(f"Error collecting database stats: {e}")
            return None
    
    def _collect_slow_queries(self):
        """Collect slow query information"""
        try:
            # Get slow queries from pg_stat_statements if available
            slow_queries_query = """
            SELECT 
                query,
                calls,
                mean_exec_time,
                total_exec_time,
                rows,
                100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
            FROM pg_stat_statements
            WHERE mean_exec_time > %s
            ORDER BY mean_exec_time DESC
            LIMIT 10
            """
            
            slow_queries = execute_query(slow_queries_query, (self.slow_query_threshold * 1000,))
            
            for query_info in slow_queries:
                metric = QueryPerformanceMetric(
                    query=query_info['query'][:200],  # Truncate long queries
                    execution_time=float(query_info['mean_exec_time']) / 1000.0,  # Convert to seconds
                    rows_returned=int(query_info['rows']),
                    timestamp=datetime.now(),
                    database_name=self.db_manager.config.database
                )
                
                self.query_metrics.append(metric)
                
                # Log slow query warning
                logger.warning(f"Slow query detected: {metric.execution_time:.2f}s - {metric.query}")
            
            # Keep only last 100 query metrics
            if len(self.query_metrics) > 100:
                self.query_metrics = self.query_metrics[-100:]
                
        except Exception as e:
            logger.debug(f"Could not collect slow queries (pg_stat_statements may not be enabled): {e}")
    
    def _check_performance_alerts(self, stats: DatabaseStats):
        """Check for performance alerts"""
        try:
            # Check connection usage
            max_connections = self._get_max_connections()
            if max_connections and stats.total_connections > (max_connections * self.high_connection_threshold / 100):
                logger.warning(f"High connection usage: {stats.total_connections}/{max_connections} ({stats.total_connections/max_connections*100:.1f}%)")
            
            # Check cache hit ratio
            if stats.cache_hit_ratio < self.low_cache_hit_threshold:
                logger.warning(f"Low cache hit ratio: {stats.cache_hit_ratio:.2f} (threshold: {self.low_cache_hit_threshold:.2f})")
            
            # Check for too many slow queries
            if stats.slow_queries_count > 10:
                logger.warning(f"High number of slow queries: {stats.slow_queries_count}")
                
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    def _get_max_connections(self) -> Optional[int]:
        """Get maximum connections setting"""
        try:
            max_conn_query = "SHOW max_connections"
            result = execute_query(max_conn_query, fetch_all=False)
            return int(result['max_connections']) if result else None
        except Exception as e:
            logger.error(f"Error getting max connections: {e}")
            return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        try:
            if not self.database_stats:
                return {"error": "No performance data available"}
            
            # Get latest stats
            latest_stats = self.database_stats[-1]
            
            # Calculate averages over last hour
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_stats = [s for s in self.database_stats if s.timestamp >= one_hour_ago]
            
            if recent_stats:
                avg_connections = sum(s.total_connections for s in recent_stats) / len(recent_stats)
                avg_cache_hit = sum(s.cache_hit_ratio for s in recent_stats) / len(recent_stats)
                avg_tps = sum(s.transactions_per_second for s in recent_stats) / len(recent_stats)
            else:
                avg_connections = latest_stats.total_connections
                avg_cache_hit = latest_stats.cache_hit_ratio
                avg_tps = latest_stats.transactions_per_second
            
            # Get system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get table statistics
            table_stats = self._get_table_statistics()
            
            # Get index usage
            index_stats = self._get_index_statistics()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "database_stats": {
                    "current_connections": latest_stats.total_connections,
                    "active_connections": latest_stats.active_connections,
                    "idle_connections": latest_stats.idle_connections,
                    "database_size_mb": round(latest_stats.database_size, 2),
                    "cache_hit_ratio": round(latest_stats.cache_hit_ratio, 4),
                    "transactions_per_second": round(latest_stats.transactions_per_second, 2),
                    "slow_queries_count": latest_stats.slow_queries_count
                },
                "averages_last_hour": {
                    "avg_connections": round(avg_connections, 1),
                    "avg_cache_hit_ratio": round(avg_cache_hit, 4),
                    "avg_tps": round(avg_tps, 2)
                },
                "system_resources": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "disk_usage_percent": disk.percent,
                    "disk_free_gb": round(disk.free / (1024**3), 2)
                },
                "table_statistics": table_stats,
                "index_statistics": index_stats,
                "recent_slow_queries": [
                    {
                        "query": metric.query,
                        "execution_time": metric.execution_time,
                        "rows_returned": metric.rows_returned,
                        "timestamp": metric.timestamp.isoformat()
                    }
                    for metric in self.query_metrics[-10:]  # Last 10 slow queries
                ],
                "recommendations": self._get_performance_recommendations()
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}
    
    def _get_table_statistics(self) -> List[Dict[str, Any]]:
        """Get table statistics"""
        try:
            table_stats_query = """
            SELECT 
                schemaname,
                tablename,
                n_tup_ins as inserts,
                n_tup_upd as updates,
                n_tup_del as deletes,
                n_live_tup as live_tuples,
                n_dead_tup as dead_tuples,
                last_vacuum,
                last_autovacuum,
                last_analyze,
                last_autoanalyze
            FROM pg_stat_user_tables
            WHERE schemaname = 'qnti'
            ORDER BY n_live_tup DESC
            LIMIT 10
            """
            
            table_stats = execute_query(table_stats_query)
            
            return [
                {
                    "schema": row['schemaname'],
                    "table": row['tablename'],
                    "inserts": int(row['inserts']),
                    "updates": int(row['updates']),
                    "deletes": int(row['deletes']),
                    "live_tuples": int(row['live_tuples']),
                    "dead_tuples": int(row['dead_tuples']),
                    "last_vacuum": row['last_vacuum'].isoformat() if row['last_vacuum'] else None,
                    "last_analyze": row['last_analyze'].isoformat() if row['last_analyze'] else None
                }
                for row in table_stats
            ]
            
        except Exception as e:
            logger.error(f"Error getting table statistics: {e}")
            return []
    
    def _get_index_statistics(self) -> List[Dict[str, Any]]:
        """Get index usage statistics"""
        try:
            index_stats_query = """
            SELECT 
                schemaname,
                tablename,
                indexname,
                idx_tup_read,
                idx_tup_fetch,
                idx_scan
            FROM pg_stat_user_indexes
            WHERE schemaname = 'qnti' AND idx_scan > 0
            ORDER BY idx_scan DESC
            LIMIT 10
            """
            
            index_stats = execute_query(index_stats_query)
            
            return [
                {
                    "schema": row['schemaname'],
                    "table": row['tablename'],
                    "index": row['indexname'],
                    "tuples_read": int(row['idx_tup_read']),
                    "tuples_fetched": int(row['idx_tup_fetch']),
                    "scans": int(row['idx_scan'])
                }
                for row in index_stats
            ]
            
        except Exception as e:
            logger.error(f"Error getting index statistics: {e}")
            return []
    
    def _get_performance_recommendations(self) -> List[str]:
        """Get performance recommendations"""
        recommendations = []
        
        try:
            if not self.database_stats:
                return recommendations
            
            latest_stats = self.database_stats[-1]
            
            # Connection recommendations
            max_connections = self._get_max_connections()
            if max_connections and latest_stats.total_connections > (max_connections * 0.8):
                recommendations.append("Consider increasing max_connections or implementing connection pooling")
            
            # Cache recommendations
            if latest_stats.cache_hit_ratio < 0.90:
                recommendations.append("Consider increasing shared_buffers to improve cache hit ratio")
            
            # Slow query recommendations
            if latest_stats.slow_queries_count > 5:
                recommendations.append("Review and optimize slow queries, consider adding indexes")
            
            # Database size recommendations
            if latest_stats.database_size > 1000:  # 1GB
                recommendations.append("Consider implementing data archiving or partitioning for large tables")
            
            # Check for unused indexes
            unused_indexes = self._get_unused_indexes()
            if unused_indexes:
                recommendations.append(f"Consider dropping {len(unused_indexes)} unused indexes to reduce storage and improve write performance")
            
            # Check for missing indexes
            missing_indexes = self._suggest_missing_indexes()
            if missing_indexes:
                recommendations.append(f"Consider adding {len(missing_indexes)} suggested indexes for better query performance")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _get_unused_indexes(self) -> List[Dict[str, Any]]:
        """Get unused indexes"""
        try:
            unused_indexes_query = """
            SELECT 
                schemaname,
                tablename,
                indexname,
                idx_scan
            FROM pg_stat_user_indexes
            WHERE schemaname = 'qnti' AND idx_scan = 0
            ORDER BY tablename, indexname
            """
            
            return execute_query(unused_indexes_query)
            
        except Exception as e:
            logger.error(f"Error getting unused indexes: {e}")
            return []
    
    def _suggest_missing_indexes(self) -> List[Dict[str, Any]]:
        """Suggest missing indexes based on query patterns"""
        try:
            # This is a simplified approach - in production, you'd want more sophisticated analysis
            missing_indexes_query = """
            SELECT 
                schemaname,
                tablename,
                seq_scan,
                seq_tup_read,
                idx_scan,
                idx_tup_fetch
            FROM pg_stat_user_tables
            WHERE schemaname = 'qnti' 
              AND seq_scan > 100  -- Tables with many sequential scans
              AND seq_tup_read > 10000  -- That read many tuples
            ORDER BY seq_scan DESC
            """
            
            return execute_query(missing_indexes_query)
            
        except Exception as e:
            logger.error(f"Error suggesting missing indexes: {e}")
            return []
    
    def optimize_database(self) -> Dict[str, Any]:
        """Perform database optimization tasks"""
        optimization_results = {
            "timestamp": datetime.now().isoformat(),
            "operations": []
        }
        
        try:
            # Vacuum and analyze tables
            vacuum_results = self._vacuum_analyze_tables()
            optimization_results["operations"].append({
                "operation": "vacuum_analyze",
                "status": "completed",
                "results": vacuum_results
            })
            
            # Update table statistics
            stats_results = self._update_table_statistics()
            optimization_results["operations"].append({
                "operation": "update_statistics",
                "status": "completed",
                "results": stats_results
            })
            
            # Reindex if needed
            reindex_results = self._reindex_tables()
            optimization_results["operations"].append({
                "operation": "reindex",
                "status": "completed",
                "results": reindex_results
            })
            
            logger.info("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Error during database optimization: {e}")
            optimization_results["error"] = str(e)
        
        return optimization_results
    
    def _vacuum_analyze_tables(self) -> List[str]:
        """Vacuum and analyze tables"""
        results = []
        
        try:
            # Get tables that need vacuuming
            tables_query = """
            SELECT tablename
            FROM pg_stat_user_tables
            WHERE schemaname = 'qnti'
              AND (last_vacuum IS NULL OR last_vacuum < NOW() - INTERVAL '1 day')
            """
            
            tables = execute_query(tables_query)
            
            for table in tables:
                table_name = table['tablename']
                try:
                    # Use ANALYZE instead of VACUUM for safety in production
                    analyze_query = f"ANALYZE qnti.{table_name}"
                    execute_query(analyze_query)
                    results.append(f"Analyzed table: {table_name}")
                    logger.info(f"Analyzed table: {table_name}")
                except Exception as e:
                    results.append(f"Failed to analyze table {table_name}: {str(e)}")
                    logger.error(f"Failed to analyze table {table_name}: {e}")
            
        except Exception as e:
            logger.error(f"Error vacuuming/analyzing tables: {e}")
        
        return results
    
    def _update_table_statistics(self) -> List[str]:
        """Update table statistics"""
        results = []
        
        try:
            # Update statistics for all tables
            update_stats_query = """
            SELECT 'ANALYZE ' || schemaname || '.' || tablename as analyze_cmd
            FROM pg_stat_user_tables
            WHERE schemaname = 'qnti'
            """
            
            commands = execute_query(update_stats_query)
            
            for cmd in commands:
                try:
                    execute_query(cmd['analyze_cmd'])
                    results.append(f"Updated statistics: {cmd['analyze_cmd']}")
                except Exception as e:
                    results.append(f"Failed: {cmd['analyze_cmd']} - {str(e)}")
            
        except Exception as e:
            logger.error(f"Error updating table statistics: {e}")
        
        return results
    
    def _reindex_tables(self) -> List[str]:
        """Reindex tables if needed"""
        results = []
        
        try:
            # Check for indexes that might benefit from reindexing
            # This is a conservative approach - only reindex if there's clear benefit
            reindex_query = """
            SELECT 
                schemaname,
                tablename,
                indexname
            FROM pg_stat_user_indexes
            WHERE schemaname = 'qnti'
              AND idx_scan > 1000  -- Only reindex frequently used indexes
            ORDER BY idx_scan DESC
            LIMIT 5  -- Limit to avoid long-running operations
            """
            
            indexes = execute_query(reindex_query)
            
            for idx in indexes:
                index_name = idx['indexname']
                try:
                    # REINDEX can be expensive, so we'll just log the recommendation
                    results.append(f"Recommended for reindexing: {index_name}")
                    logger.info(f"Index {index_name} could benefit from reindexing")
                except Exception as e:
                    results.append(f"Failed to check index {index_name}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error checking indexes for reindexing: {e}")
        
        return results
    
    def export_performance_data(self, filename: str = None) -> str:
        """Export performance data to CSV"""
        try:
            if not filename:
                filename = f"qnti_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Convert database stats to DataFrame
            stats_data = []
            for stat in self.database_stats:
                stats_data.append({
                    'timestamp': stat.timestamp.isoformat(),
                    'total_connections': stat.total_connections,
                    'active_connections': stat.active_connections,
                    'idle_connections': stat.idle_connections,
                    'database_size_mb': stat.database_size,
                    'cache_hit_ratio': stat.cache_hit_ratio,
                    'transactions_per_second': stat.transactions_per_second,
                    'slow_queries_count': stat.slow_queries_count
                })
            
            df = pd.DataFrame(stats_data)
            df.to_csv(filename, index=False)
            
            logger.info(f"Performance data exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting performance data: {e}")
            raise

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create performance monitor
    monitor = QNTIPerformanceMonitor(monitoring_interval=30)  # 30 second intervals for testing
    
    try:
        print("Starting QNTI Performance Monitor...")
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Wait a bit for some data to be collected
        time.sleep(60)
        
        # Generate performance report
        print("\nGenerating performance report...")
        report = monitor.get_performance_report()
        print(json.dumps(report, indent=2, default=str))
        
        # Run optimization
        print("\nRunning database optimization...")
        optimization_results = monitor.optimize_database()
        print(json.dumps(optimization_results, indent=2, default=str))
        
        # Export performance data
        print("\nExporting performance data...")
        filename = monitor.export_performance_data()
        print(f"Performance data exported to: {filename}")
        
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        monitor.stop_monitoring()
    
    print("Performance monitoring completed.")
