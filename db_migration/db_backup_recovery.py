#!/usr/bin/env python3
"""
QNTI Database Backup and Recovery System
Comprehensive backup and recovery solution for PostgreSQL
"""

import os
import subprocess
import logging
import json
import gzip
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import schedule
import boto3
from botocore.exceptions import ClientError

from database_config import get_database_manager, DatabaseConfig

logger = logging.getLogger(__name__)

@dataclass
class BackupMetadata:
    """Backup metadata"""
    backup_id: str
    timestamp: datetime
    backup_type: str  # full, incremental, differential
    database_name: str
    file_path: str
    file_size: int
    compression: str
    status: str  # success, failed, in_progress
    duration: float
    error_message: Optional[str] = None
    
class QNTIBackupManager:
    """PostgreSQL backup and recovery manager"""
    
    def __init__(self, backup_dir: str = "backups", 
                 retention_days: int = 30,
                 s3_bucket: str = None,
                 s3_region: str = "us-east-1"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.retention_days = retention_days
        self.s3_bucket = s3_bucket
        self.s3_region = s3_region
        
        # Database configuration
        self.db_manager = get_database_manager()
        self.db_config = self.db_manager.config
        
        # Backup metadata
        self.backup_metadata: List[BackupMetadata] = []
        self.metadata_file = self.backup_dir / "backup_metadata.json"
        
        # AWS S3 client (if configured)
        self.s3_client = None
        if self.s3_bucket:
            try:
                self.s3_client = boto3.client('s3', region_name=self.s3_region)
                logger.info(f"S3 backup configured for bucket: {self.s3_bucket}")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
        
        # Load existing metadata
        self._load_metadata()
        
        # Scheduler for automated backups
        self.scheduler_thread = None
        self.scheduler_running = False
        
        logger.info(f"QNTI Backup Manager initialized - Backup dir: {self.backup_dir}")
    
    def _load_metadata(self):
        """Load backup metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata_list = json.load(f)
                    
                for item in metadata_list:
                    metadata = BackupMetadata(
                        backup_id=item['backup_id'],
                        timestamp=datetime.fromisoformat(item['timestamp']),
                        backup_type=item['backup_type'],
                        database_name=item['database_name'],
                        file_path=item['file_path'],
                        file_size=item['file_size'],
                        compression=item['compression'],
                        status=item['status'],
                        duration=item['duration'],
                        error_message=item.get('error_message')
                    )
                    self.backup_metadata.append(metadata)
                    
                logger.info(f"Loaded {len(self.backup_metadata)} backup metadata records")
                
        except Exception as e:
            logger.error(f"Error loading backup metadata: {e}")
    
    def _save_metadata(self):
        """Save backup metadata to file"""
        try:
            metadata_list = []
            for metadata in self.backup_metadata:
                metadata_list.append({
                    'backup_id': metadata.backup_id,
                    'timestamp': metadata.timestamp.isoformat(),
                    'backup_type': metadata.backup_type,
                    'database_name': metadata.database_name,
                    'file_path': metadata.file_path,
                    'file_size': metadata.file_size,
                    'compression': metadata.compression,
                    'status': metadata.status,
                    'duration': metadata.duration,
                    'error_message': metadata.error_message
                })
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_list, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving backup metadata: {e}")
    
    def create_backup(self, backup_type: str = "full", compress: bool = True) -> BackupMetadata:
        """Create a database backup"""
        backup_id = f"qnti_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = time.time()
        
        # Create backup metadata
        metadata = BackupMetadata(
            backup_id=backup_id,
            timestamp=datetime.now(),
            backup_type=backup_type,
            database_name=self.db_config.database,
            file_path="",
            file_size=0,
            compression="gzip" if compress else "none",
            status="in_progress",
            duration=0.0
        )
        
        try:
            logger.info(f"Starting {backup_type} backup: {backup_id}")
            
            # Create backup file path
            backup_filename = f"{backup_id}.sql"
            if compress:
                backup_filename += ".gz"
            
            backup_file = self.backup_dir / backup_filename
            metadata.file_path = str(backup_file)
            
            # Create pg_dump command
            pg_dump_cmd = [
                "pg_dump",
                "-h", self.db_config.host,
                "-p", str(self.db_config.port),
                "-U", self.db_config.username,
                "-d", self.db_config.database,
                "--verbose",
                "--no-password",
                "--format=custom" if not compress else "--format=plain",
                "--compress=9" if not compress else ""
            ]
            
            # Remove empty arguments
            pg_dump_cmd = [arg for arg in pg_dump_cmd if arg]
            
            # Set environment variables
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_config.password
            
            # Execute backup
            if compress:
                # Use gzip compression
                with gzip.open(backup_file, 'wt') as gz_file:
                    process = subprocess.Popen(
                        pg_dump_cmd,
                        stdout=gz_file,
                        stderr=subprocess.PIPE,
                        env=env,
                        text=True
                    )
                    
                    _, stderr = process.communicate()
                    
                    if process.returncode != 0:
                        raise Exception(f"pg_dump failed: {stderr}")
            else:
                # No compression
                with open(backup_file, 'w') as f:
                    process = subprocess.Popen(
                        pg_dump_cmd,
                        stdout=f,
                        stderr=subprocess.PIPE,
                        env=env,
                        text=True
                    )
                    
                    _, stderr = process.communicate()
                    
                    if process.returncode != 0:
                        raise Exception(f"pg_dump failed: {stderr}")
            
            # Get file size
            metadata.file_size = backup_file.stat().st_size
            metadata.duration = time.time() - start_time
            metadata.status = "success"
            
            logger.info(f"Backup completed successfully: {backup_id} ({metadata.file_size} bytes, {metadata.duration:.2f}s)")
            
            # Upload to S3 if configured
            if self.s3_client:
                self._upload_to_s3(backup_file, backup_id)
            
        except Exception as e:
            metadata.status = "failed"
            metadata.error_message = str(e)
            metadata.duration = time.time() - start_time
            logger.error(f"Backup failed: {backup_id} - {e}")
        
        # Save metadata
        self.backup_metadata.append(metadata)
        self._save_metadata()
        
        return metadata
    
    def _upload_to_s3(self, local_file: Path, backup_id: str):
        """Upload backup to S3"""
        try:
            s3_key = f"qnti-backups/{backup_id}/{local_file.name}"
            
            logger.info(f"Uploading backup to S3: {s3_key}")
            
            self.s3_client.upload_file(
                str(local_file),
                self.s3_bucket,
                s3_key,
                ExtraArgs={
                    'StorageClass': 'STANDARD_IA',  # Infrequent Access storage class
                    'ServerSideEncryption': 'AES256'
                }
            )
            
            logger.info(f"Backup uploaded to S3 successfully: {s3_key}")
            
        except ClientError as e:
            logger.error(f"Failed to upload backup to S3: {e}")
            raise
    
    def restore_backup(self, backup_id: str, target_database: str = None) -> bool:
        """Restore a database backup"""
        try:
            # Find backup metadata
            backup_metadata = None
            for metadata in self.backup_metadata:
                if metadata.backup_id == backup_id:
                    backup_metadata = metadata
                    break
            
            if not backup_metadata:
                raise Exception(f"Backup not found: {backup_id}")
            
            if backup_metadata.status != "success":
                raise Exception(f"Cannot restore failed backup: {backup_id}")
            
            backup_file = Path(backup_metadata.file_path)
            
            # Download from S3 if file doesn't exist locally
            if not backup_file.exists() and self.s3_client:
                self._download_from_s3(backup_file, backup_id)
            
            if not backup_file.exists():
                raise Exception(f"Backup file not found: {backup_file}")
            
            target_db = target_database or self.db_config.database
            
            logger.info(f"Starting restore of backup {backup_id} to database {target_db}")
            
            # Create restore command
            if backup_metadata.compression == "gzip":
                # Compressed backup
                restore_cmd = [
                    "psql",
                    "-h", self.db_config.host,
                    "-p", str(self.db_config.port),
                    "-U", self.db_config.username,
                    "-d", target_db,
                    "--no-password"
                ]
                
                env = os.environ.copy()
                env['PGPASSWORD'] = self.db_config.password
                
                # Use gunzip to decompress and pipe to psql
                with gzip.open(backup_file, 'rt') as gz_file:
                    process = subprocess.Popen(
                        restore_cmd,
                        stdin=gz_file,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=env,
                        text=True
                    )
                    
                    stdout, stderr = process.communicate()
                    
                    if process.returncode != 0:
                        raise Exception(f"Restore failed: {stderr}")
            else:
                # Uncompressed backup
                restore_cmd = [
                    "pg_restore",
                    "-h", self.db_config.host,
                    "-p", str(self.db_config.port),
                    "-U", self.db_config.username,
                    "-d", target_db,
                    "--verbose",
                    "--no-password",
                    "--clean",
                    "--if-exists",
                    str(backup_file)
                ]
                
                env = os.environ.copy()
                env['PGPASSWORD'] = self.db_config.password
                
                process = subprocess.Popen(
                    restore_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    text=True
                )
                
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    logger.warning(f"Restore completed with warnings: {stderr}")
                else:
                    logger.info(f"Restore completed successfully: {backup_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def _download_from_s3(self, local_file: Path, backup_id: str):
        """Download backup from S3"""
        try:
            s3_key = f"qnti-backups/{backup_id}/{local_file.name}"
            
            logger.info(f"Downloading backup from S3: {s3_key}")
            
            self.s3_client.download_file(
                self.s3_bucket,
                s3_key,
                str(local_file)
            )
            
            logger.info(f"Backup downloaded from S3 successfully: {local_file}")
            
        except ClientError as e:
            logger.error(f"Failed to download backup from S3: {e}")
            raise
    
    def cleanup_old_backups(self):
        """Remove old backups based on retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            backups_to_remove = []
            for metadata in self.backup_metadata:
                if metadata.timestamp < cutoff_date:
                    backups_to_remove.append(metadata)
            
            for metadata in backups_to_remove:
                try:
                    # Remove local file
                    backup_file = Path(metadata.file_path)
                    if backup_file.exists():
                        backup_file.unlink()
                        logger.info(f"Removed old backup file: {backup_file}")
                    
                    # Remove from S3 if configured
                    if self.s3_client:
                        s3_key = f"qnti-backups/{metadata.backup_id}/{backup_file.name}"
                        try:
                            self.s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
                            logger.info(f"Removed old backup from S3: {s3_key}")
                        except ClientError as e:
                            logger.warning(f"Failed to remove S3 backup: {e}")
                    
                    # Remove from metadata
                    self.backup_metadata.remove(metadata)
                    
                except Exception as e:
                    logger.error(f"Error removing old backup {metadata.backup_id}: {e}")
            
            if backups_to_remove:
                self._save_metadata()
                logger.info(f"Cleaned up {len(backups_to_remove)} old backups")
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def get_backup_list(self) -> List[Dict[str, Any]]:
        """Get list of all backups"""
        return [
            {
                'backup_id': metadata.backup_id,
                'timestamp': metadata.timestamp.isoformat(),
                'backup_type': metadata.backup_type,
                'database_name': metadata.database_name,
                'file_path': metadata.file_path,
                'file_size': metadata.file_size,
                'file_size_mb': round(metadata.file_size / (1024 * 1024), 2),
                'compression': metadata.compression,
                'status': metadata.status,
                'duration': metadata.duration,
                'error_message': metadata.error_message
            }
            for metadata in sorted(self.backup_metadata, key=lambda x: x.timestamp, reverse=True)
        ]
    
    def verify_backup(self, backup_id: str) -> Dict[str, Any]:
        """Verify backup integrity"""
        try:
            # Find backup metadata
            backup_metadata = None
            for metadata in self.backup_metadata:
                if metadata.backup_id == backup_id:
                    backup_metadata = metadata
                    break
            
            if not backup_metadata:
                return {"status": "error", "message": f"Backup not found: {backup_id}"}
            
            backup_file = Path(backup_metadata.file_path)
            
            # Check if file exists
            if not backup_file.exists():
                return {"status": "error", "message": f"Backup file not found: {backup_file}"}
            
            # Check file size
            actual_size = backup_file.stat().st_size
            if actual_size != backup_metadata.file_size:
                return {
                    "status": "error",
                    "message": f"File size mismatch: expected {backup_metadata.file_size}, actual {actual_size}"
                }
            
            # Test decompression if compressed
            if backup_metadata.compression == "gzip":
                try:
                    with gzip.open(backup_file, 'rt') as gz_file:
                        # Read first few lines to test decompression
                        for _ in range(10):
                            line = gz_file.readline()
                            if not line:
                                break
                except Exception as e:
                    return {"status": "error", "message": f"Decompression test failed: {e}"}
            
            # Test SQL syntax (basic check)
            try:
                if backup_metadata.compression == "gzip":
                    with gzip.open(backup_file, 'rt') as f:
                        content = f.read(1000)  # Read first 1000 characters
                else:
                    with open(backup_file, 'r') as f:
                        content = f.read(1000)
                
                # Check for SQL dump header
                if "PostgreSQL database dump" not in content and "pg_dump" not in content:
                    return {"status": "warning", "message": "File doesn't appear to be a PostgreSQL dump"}
                
            except Exception as e:
                return {"status": "error", "message": f"Content verification failed: {e}"}
            
            return {
                "status": "success",
                "message": "Backup verification passed",
                "file_size": actual_size,
                "compression": backup_metadata.compression,
                "created": backup_metadata.timestamp.isoformat()
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Verification failed: {e}"}
    
    def setup_automated_backups(self, schedule_config: Dict[str, Any]):
        """Setup automated backup scheduling"""
        try:
            # Clear existing schedules
            schedule.clear()
            
            # Daily backup
            if schedule_config.get('daily_backup', {}).get('enabled', False):
                daily_time = schedule_config['daily_backup'].get('time', '02:00')
                schedule.every().day.at(daily_time).do(self._automated_backup, 'daily')
                logger.info(f"Daily backup scheduled at {daily_time}")
            
            # Weekly backup
            if schedule_config.get('weekly_backup', {}).get('enabled', False):
                weekly_day = schedule_config['weekly_backup'].get('day', 'sunday')
                weekly_time = schedule_config['weekly_backup'].get('time', '01:00')
                getattr(schedule.every(), weekly_day).at(weekly_time).do(self._automated_backup, 'weekly')
                logger.info(f"Weekly backup scheduled on {weekly_day} at {weekly_time}")
            
            # Monthly backup
            if schedule_config.get('monthly_backup', {}).get('enabled', False):
                monthly_day = schedule_config['monthly_backup'].get('day', 1)
                monthly_time = schedule_config['monthly_backup'].get('time', '00:00')
                # Note: schedule library doesn't support monthly directly, this is a simplification
                schedule.every().day.at(monthly_time).do(self._monthly_backup_check, monthly_day)
                logger.info(f"Monthly backup scheduled on day {monthly_day} at {monthly_time}")
            
            # Start scheduler thread
            if not self.scheduler_running:
                self.scheduler_running = True
                self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
                self.scheduler_thread.start()
                logger.info("Automated backup scheduler started")
            
        except Exception as e:
            logger.error(f"Error setting up automated backups: {e}")
    
    def _automated_backup(self, backup_type: str):
        """Perform automated backup"""
        try:
            logger.info(f"Starting automated {backup_type} backup")
            metadata = self.create_backup(backup_type=backup_type)
            
            if metadata.status == "success":
                logger.info(f"Automated {backup_type} backup completed successfully")
                
                # Cleanup old backups
                self.cleanup_old_backups()
            else:
                logger.error(f"Automated {backup_type} backup failed: {metadata.error_message}")
                
        except Exception as e:
            logger.error(f"Error in automated backup: {e}")
    
    def _monthly_backup_check(self, target_day: int):
        """Check if monthly backup should run"""
        if datetime.now().day == target_day:
            self._automated_backup('monthly')
    
    def _scheduler_loop(self):
        """Scheduler loop"""
        while self.scheduler_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)
    
    def stop_automated_backups(self):
        """Stop automated backups"""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        schedule.clear()
        logger.info("Automated backup scheduler stopped")
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get backup system status"""
        try:
            total_backups = len(self.backup_metadata)
            successful_backups = len([m for m in self.backup_metadata if m.status == "success"])
            failed_backups = len([m for m in self.backup_metadata if m.status == "failed"])
            
            # Get latest backup
            latest_backup = None
            if self.backup_metadata:
                latest_backup = max(self.backup_metadata, key=lambda x: x.timestamp)
            
            # Calculate total backup size
            total_size = sum(m.file_size for m in self.backup_metadata if m.status == "success")
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_backups": total_backups,
                "successful_backups": successful_backups,
                "failed_backups": failed_backups,
                "success_rate": (successful_backups / total_backups * 100) if total_backups > 0 else 0,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "backup_directory": str(self.backup_dir),
                "retention_days": self.retention_days,
                "s3_backup_enabled": self.s3_client is not None,
                "s3_bucket": self.s3_bucket,
                "automated_backups_running": self.scheduler_running,
                "latest_backup": {
                    "backup_id": latest_backup.backup_id,
                    "timestamp": latest_backup.timestamp.isoformat(),
                    "status": latest_backup.status,
                    "file_size_mb": round(latest_backup.file_size / (1024 * 1024), 2)
                } if latest_backup else None
            }
            
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create backup manager
    backup_manager = QNTIBackupManager(
        backup_dir="test_backups",
        retention_days=7,
        s3_bucket=None  # Set to your S3 bucket name if using S3
    )
    
    try:
        print("QNTI Backup Manager Test")
        print("="*50)
        
        # Create a backup
        print("\nCreating backup...")
        backup_metadata = backup_manager.create_backup(backup_type="test", compress=True)
        print(f"Backup created: {backup_metadata.backup_id} - Status: {backup_metadata.status}")
        
        if backup_metadata.status == "success":
            # Verify backup
            print("\nVerifying backup...")
            verification = backup_manager.verify_backup(backup_metadata.backup_id)
            print(f"Verification: {verification}")
            
            # List backups
            print("\nBackup list:")
            backups = backup_manager.get_backup_list()
            for backup in backups:
                print(f"  {backup['backup_id']} - {backup['timestamp']} - {backup['status']} - {backup['file_size_mb']}MB")
        
        # Get backup status
        print("\nBackup system status:")
        status = backup_manager.get_backup_status()
        print(json.dumps(status, indent=2, default=str))
        
        # Setup automated backups (example)
        print("\nSetting up automated backups...")
        schedule_config = {
            'daily_backup': {
                'enabled': True,
                'time': '02:00'
            },
            'weekly_backup': {
                'enabled': True,
                'day': 'sunday',
                'time': '01:00'
            }
        }
        
        backup_manager.setup_automated_backups(schedule_config)
        print("Automated backups configured")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        backup_manager.stop_automated_backups()
    
    print("\nBackup manager test completed.")
