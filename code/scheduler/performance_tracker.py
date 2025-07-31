"""
Performance Tracker for LLM Evaluation
Module for collecting execution data and providing statistics-based predictions
"""
import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, asdict
import logging
import psutil
import GPUtil

# Logging setup
logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    """Execution record data class"""
    # Basic information
    run_id: str
    timestamp: str
    mode: str  # baseline or optimized
    
    # Model information
    model_id: str
    model_name: str
    model_size: str  # 0.5B, 2B, 8B etc.
    
    # Task information
    task_name: str
    task_type: str  # harness or custom
    num_fewshot: int
    batch_size: int
    sample_limit: Optional[int]  # None for full run
    
    # Execution information
    gpu_id: str
    device: str
    start_time: float
    end_time: Optional[float] = None
    
    # Result information
    status: str = "running"  # running, completed, failed, oom
    execution_time: Optional[float] = None
    
    # Resource usage
    gpu_memory_start: Optional[float] = None
    gpu_memory_peak: Optional[float] = None
    gpu_memory_end: Optional[float] = None
    gpu_utilization_avg: Optional[float] = None
    cpu_percent_avg: Optional[float] = None
    ram_usage_peak: Optional[float] = None
    
    # Result metrics
    result_metrics: Optional[Dict] = None
    error_message: Optional[str] = None
    
    # Additional metadata
    metadata: Optional[Dict] = None


class PerformanceTracker:
    """LLM evaluation performance tracking and prediction class"""
    
    def __init__(self, mode: str = "baseline", db_path: Optional[Path] = None):
        """
        Args:
            mode: Execution mode (baseline or optimized)
            db_path: SQLite DB path
        """
        self.mode = mode
        self.run_id = f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # DB path setup - FIXED PATH
        if db_path is None:
            # Use project root path (3 levels up from code/scheduler/)
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "data" / "performanceDB" / "performance_history.db"
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # DB connection and initialization
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        
        # Track currently running records
        self.active_records: Dict[str, ExecutionRecord] = {}
        
        logger.info(f"PerformanceTracker initialized: mode={mode}, run_id={self.run_id}")
    
    def _create_tables(self):
        """Create DB tables"""
        cursor = self.conn.cursor()
        
        # Execution records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                mode TEXT NOT NULL,
                
                -- Model information
                model_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_size TEXT,
                
                -- Task information
                task_name TEXT NOT NULL,
                task_type TEXT,
                num_fewshot INTEGER,
                batch_size INTEGER,
                sample_limit INTEGER,
                
                -- Execution information
                gpu_id TEXT,
                device TEXT,
                start_time REAL NOT NULL,
                end_time REAL,
                
                -- Result information
                status TEXT NOT NULL,
                execution_time REAL,
                
                -- Resource usage
                gpu_memory_start REAL,
                gpu_memory_peak REAL,
                gpu_memory_end REAL,
                gpu_utilization_avg REAL,
                cpu_percent_avg REAL,
                ram_usage_peak REAL,
                
                -- Results and metadata
                result_metrics TEXT,  -- JSON
                error_message TEXT,
                metadata TEXT,  -- JSON
                
                -- For indexing
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Statistics summary table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                model_size TEXT,
                task_name TEXT NOT NULL,
                
                -- Statistical information
                num_executions INTEGER DEFAULT 0,
                num_successes INTEGER DEFAULT 0,
                num_failures INTEGER DEFAULT 0,
                num_ooms INTEGER DEFAULT 0,
                
                -- Time statistics
                avg_execution_time REAL,
                std_execution_time REAL,
                min_execution_time REAL,
                max_execution_time REAL,
                
                -- Memory statistics
                avg_memory_peak REAL,
                std_memory_peak REAL,
                min_memory_peak REAL,
                max_memory_peak REAL,
                
                -- GPU utilization statistics
                avg_gpu_utilization REAL,
                
                -- Update time
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(model_id, task_name)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_task ON execution_records(model_id, task_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON execution_records(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON execution_records(status)")
        
        self.conn.commit()
    
    def _extract_model_size(self, model_id: str) -> str:
        """Extract model size from model ID"""
        model_id_lower = model_id.lower()
        
        # Pattern matching for model size extraction
        size_patterns = [
            "0.5b", "0-5b", "500m",
            "1.5b", "1-5b", 
            "2.1b", "2-1b", "2.4b", "2-4b",
            "3b", "4b", "7b", "7.8b", "7-8b", "8b",
            "12b", "13b", "21.4b", "21-4b", "30b", "32b", "70b"
        ]
        
        for pattern in size_patterns:
            if pattern in model_id_lower:
                return pattern.upper().replace("-", ".")
        
        # Return unknown if size cannot be found
        return "unknown"
    
    def _get_gpu_memory(self) -> Tuple[float, float]:
        """Get current GPU memory usage and utilization (GB, %)"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # First GPU
                memory_used = gpu.memoryUsed / 1024  # MB to GB
                utilization = gpu.load * 100  # percentage
                return memory_used, utilization
        except:
            pass
        return 0.0, 0.0
    
    def _get_cpu_ram_usage(self) -> Tuple[float, float]:
        """Get CPU and RAM usage (%, GB)"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory()
            ram_used_gb = ram.used / (1024**3)  # bytes to GB
            return cpu_percent, ram_used_gb
        except:
            return 0.0, 0.0
    
    def record_start(self, model_id: str, model_name: str, task_name: str, config: Dict[str, Any]) -> str:
        """Record execution start"""
        record_key = f"{model_id}_{task_name}_{time.time()}"
        
        # GPU memory information
        gpu_memory, gpu_util = self._get_gpu_memory()
        cpu_percent, ram_usage = self._get_cpu_ram_usage()
        
        # Create execution record
        record = ExecutionRecord(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            mode=self.mode,
            model_id=model_id,
            model_name=model_name,
            model_size=self._extract_model_size(model_id),
            task_name=task_name,
            task_type="harness",  # TODO: Add custom task distinction
            num_fewshot=config.get("num_fewshot", 0),
            batch_size=config.get("batch_size", 1),
            sample_limit=config.get("limit"),
            gpu_id=config.get("gpu_id", "0"),
            device=config.get("device", "cuda"),
            start_time=time.time(),
            gpu_memory_start=gpu_memory,
            cpu_percent_avg=cpu_percent,
            ram_usage_peak=ram_usage,
            metadata=config.get("metadata", {})
        )
        
        self.active_records[record_key] = record
        logger.info(f"Started tracking: {model_name} on {task_name}")
        
        return record_key
    
    def record_end(self, record_key: str, status: str = "completed", 
                   results: Optional[Dict] = None, error_message: Optional[str] = None):
        """Record execution end"""
        if record_key not in self.active_records:
            logger.warning(f"No active record found for key: {record_key}")
            return
        
        record = self.active_records[record_key]
        record.end_time = time.time()
        record.execution_time = record.end_time - record.start_time
        record.status = status
        
        # GPU memory information
        gpu_memory, gpu_util = self._get_gpu_memory()
        record.gpu_memory_end = gpu_memory
        record.gpu_utilization_avg = gpu_util
        
        # Update CPU/RAM information
        cpu_percent, ram_usage = self._get_cpu_ram_usage()
        record.cpu_percent_avg = (record.cpu_percent_avg + cpu_percent) / 2
        record.ram_usage_peak = max(record.ram_usage_peak, ram_usage)
        
        # Save results
        if results:
            record.result_metrics = results
        if error_message:
            record.error_message = error_message
        
        # Estimate peak memory (actual monitoring thread would be needed)
        record.gpu_memory_peak = max(record.gpu_memory_start or 0, record.gpu_memory_end or 0)
        
        # Save to DB
        self._save_record(record)
        
        # Update statistics
        self._update_statistics(record)
        
        # Remove from active records
        del self.active_records[record_key]
        
        logger.info(f"Completed tracking: {record.model_name} on {record.task_name} "
                   f"- Status: {status}, Time: {record.execution_time:.2f}s")
    
    def _save_record(self, record: ExecutionRecord):
        """Save record to DB"""
        cursor = self.conn.cursor()
        
        # Convert dataclass to dict
        data = asdict(record)
        
        # Serialize JSON fields
        data['result_metrics'] = json.dumps(data['result_metrics']) if data['result_metrics'] else None
        data['metadata'] = json.dumps(data['metadata']) if data['metadata'] else None
        
        # SQL insertion
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO execution_records ({columns}) VALUES ({placeholders})"
        
        cursor.execute(query, list(data.values()))
        self.conn.commit()
    
    def _update_statistics(self, record: ExecutionRecord):
        """Update statistics table"""
        if record.status not in ["completed", "failed", "oom"]:
            return
        
        cursor = self.conn.cursor()
        
        # Query existing statistics
        cursor.execute("""
            SELECT * FROM execution_stats 
            WHERE model_id = ? AND task_name = ?
        """, (record.model_id, record.task_name))
        
        stats = cursor.fetchone()
        
        if stats:
            # Update existing statistics
            self._update_existing_stats(cursor, record)
        else:
            # Create new statistics
            self._create_new_stats(cursor, record)
        
        self.conn.commit()
    
    def _update_existing_stats(self, cursor, record: ExecutionRecord):
        """Update existing statistics"""
        # Query all successful records for this model-task
        cursor.execute("""
            SELECT execution_time, gpu_memory_peak, gpu_utilization_avg
            FROM execution_records
            WHERE model_id = ? AND task_name = ? AND status = 'completed'
        """, (record.model_id, record.task_name))
        
        records = cursor.fetchall()
        
        if records:
            exec_times = [r['execution_time'] for r in records if r['execution_time']]
            memory_peaks = [r['gpu_memory_peak'] for r in records if r['gpu_memory_peak']]
            gpu_utils = [r['gpu_utilization_avg'] for r in records if r['gpu_utilization_avg']]
            
            # Calculate statistics
            stats_update = {
                'num_executions': len(records),
                'num_successes': len([r for r in records]),
                'avg_execution_time': np.mean(exec_times) if exec_times else None,
                'std_execution_time': np.std(exec_times) if exec_times else None,
                'min_execution_time': np.min(exec_times) if exec_times else None,
                'max_execution_time': np.max(exec_times) if exec_times else None,
                'avg_memory_peak': np.mean(memory_peaks) if memory_peaks else None,
                'std_memory_peak': np.std(memory_peaks) if memory_peaks else None,
                'min_memory_peak': np.min(memory_peaks) if memory_peaks else None,
                'max_memory_peak': np.max(memory_peaks) if memory_peaks else None,
                'avg_gpu_utilization': np.mean(gpu_utils) if gpu_utils else None,
            }
            
            # OOM and failure counts
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN status = 'oom' THEN 1 ELSE 0 END) as num_ooms,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as num_failures
                FROM execution_records
                WHERE model_id = ? AND task_name = ?
            """, (record.model_id, record.task_name))
            
            counts = cursor.fetchone()
            stats_update['num_ooms'] = counts['num_ooms'] or 0
            stats_update['num_failures'] = counts['num_failures'] or 0
            
            # Update query
            set_clause = ', '.join([f"{k} = ?" for k in stats_update.keys()])
            values = list(stats_update.values()) + [record.model_id, record.task_name]
            
            cursor.execute(f"""
                UPDATE execution_stats 
                SET {set_clause}, last_updated = CURRENT_TIMESTAMP
                WHERE model_id = ? AND task_name = ?
            """, values)
    
    def _create_new_stats(self, cursor, record: ExecutionRecord):
        """Create new statistics record"""
        if record.status == 'completed':
            cursor.execute("""
                INSERT INTO execution_stats (
                    model_id, model_size, task_name,
                    num_executions, num_successes, num_failures, num_ooms,
                    avg_execution_time, min_execution_time, max_execution_time,
                    avg_memory_peak, min_memory_peak, max_memory_peak,
                    avg_gpu_utilization
                ) VALUES (?, ?, ?, 1, 1, 0, 0, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.model_id, record.model_size, record.task_name,
                record.execution_time, record.execution_time, record.execution_time,
                record.gpu_memory_peak, record.gpu_memory_peak, record.gpu_memory_peak,
                record.gpu_utilization_avg
            ))
        else:
            # Failed/OOM cases
            num_ooms = 1 if record.status == 'oom' else 0
            num_failures = 1 if record.status == 'failed' else 0
            
            cursor.execute("""
                INSERT INTO execution_stats (
                    model_id, model_size, task_name,
                    num_executions, num_successes, num_failures, num_ooms
                ) VALUES (?, ?, ?, 1, 0, ?, ?)
            """, (record.model_id, record.model_size, record.task_name, num_failures, num_ooms))
    
    def predict_execution(self, model_id: str, task_name: str) -> Dict[str, Any]:
        """Predict execution time and memory usage"""
        cursor = self.conn.cursor()
        
        # Query statistics
        cursor.execute("""
            SELECT * FROM execution_stats 
            WHERE model_id = ? AND task_name = ?
        """, (model_id, task_name))
        
        stats = cursor.fetchone()
        
        if not stats:
            # Estimate with similar model size
            model_size = self._extract_model_size(model_id)
            cursor.execute("""
                SELECT * FROM execution_stats 
                WHERE model_size = ? AND task_name = ?
                LIMIT 1
            """, (model_size, task_name))
            stats = cursor.fetchone()
        
        if stats:
            return {
                'predicted_time': stats['avg_execution_time'],
                'time_std': stats['std_execution_time'],
                'predicted_memory': stats['avg_memory_peak'],
                'memory_std': stats['std_memory_peak'],
                'success_rate': stats['num_successes'] / stats['num_executions'] if stats['num_executions'] > 0 else 0,
                'oom_rate': stats['num_ooms'] / stats['num_executions'] if stats['num_executions'] > 0 else 0,
                'num_samples': stats['num_executions']
            }
        
        # Unpredictable case
        return {
            'predicted_time': None,
            'predicted_memory': None,
            'success_rate': None,
            'oom_rate': None,
            'num_samples': 0
        }
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Overall statistics summary - FIXED NULL HANDLING"""
        cursor = self.conn.cursor()
        
        try:
            # Overall execution statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_runs,
                    SUM(CASE WHEN status = 'oom' THEN 1 ELSE 0 END) as oom_runs,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
                    AVG(execution_time) as avg_execution_time,
                    SUM(execution_time) as total_execution_time
                FROM execution_records
                WHERE mode = ?
            """, (self.mode,))
            
            overall_result = cursor.fetchone()
            
            # Handle case where no records exist
            if not overall_result or overall_result['total_runs'] == 0:
                overall = {
                    'total_runs': 0,
                    'successful_runs': 0,
                    'oom_runs': 0,
                    'failed_runs': 0,
                    'avg_execution_time': 0.0,
                    'total_execution_time': 0.0
                }
            else:
                overall = {
                    'total_runs': overall_result['total_runs'] or 0,
                    'successful_runs': overall_result['successful_runs'] or 0,
                    'oom_runs': overall_result['oom_runs'] or 0,
                    'failed_runs': overall_result['failed_runs'] or 0,
                    'avg_execution_time': overall_result['avg_execution_time'] or 0.0,
                    'total_execution_time': overall_result['total_execution_time'] or 0.0
                }
            
            # Model-wise statistics
            cursor.execute("""
                SELECT 
                    model_name, model_size,
                    COUNT(*) as runs,
                    AVG(execution_time) as avg_time,
                    AVG(gpu_memory_peak) as avg_memory
                FROM execution_records
                WHERE mode = ? AND status = 'completed'
                GROUP BY model_id
            """, (self.mode,))
            
            model_results = cursor.fetchall()
            
            # Convert to list of dicts with null handling
            model_stats = []
            for m in model_results:
                model_stats.append({
                    'model_name': m['model_name'] or 'unknown',
                    'model_size': m['model_size'] or 'unknown',
                    'runs': m['runs'] or 0,
                    'avg_time': m['avg_time'] or 0.0,
                    'avg_memory': m['avg_memory'] or 0.0
                })
            
            return {
                'mode': self.mode,
                'overall': overall,
                'by_model': model_stats
            }
            
        except Exception as e:
            logger.error(f"Error in get_statistics_summary: {e}")
            # Return safe default values
            return {
                'mode': self.mode,
                'overall': {
                    'total_runs': 0,
                    'successful_runs': 0,
                    'oom_runs': 0,
                    'failed_runs': 0,
                    'avg_execution_time': 0.0,
                    'total_execution_time': 0.0
                },
                'by_model': []
            }
    
    def close(self):
        """Close DB connection"""
        self.conn.close()