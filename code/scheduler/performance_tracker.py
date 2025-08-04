"""
Fixed Performance Tracker for LLM Evaluation
Enhanced mode isolation, error handling, and stability improvements
WITH GPU THERMAL TRACKING
"""
import sqlite3
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, asdict
import logging
import psutil
import GPUtil
import os
import uuid
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    run_id: str
    timestamp: str
    mode: str
    
    model_id: str
    model_name: str
    model_size: str
    
    task_name: str
    task_type: str
    num_fewshot: int
    batch_size: int
    sample_limit: Optional[int]
    
    gpu_id: str
    device: str
    start_time: float
    end_time: Optional[float] = None
    
    status: str = "running"
    execution_time: Optional[float] = None
    
    gpu_memory_start: Optional[float] = None
    gpu_memory_peak: Optional[float] = None
    gpu_memory_end: Optional[float] = None
    gpu_utilization_avg: Optional[float] = None
    cpu_percent_avg: Optional[float] = None
    ram_usage_peak: Optional[float] = None
    
    # NEW: GPU Thermal Fields
    gpu_temperature_start: Optional[float] = None    # °C at task start
    gpu_temperature_peak: Optional[float] = None     # °C peak during execution
    gpu_temperature_avg: Optional[float] = None      # °C average during execution
    gpu_temperature_end: Optional[float] = None      # °C at task end
    
    result_metrics: Optional[Dict] = None
    error_message: Optional[str] = None
    
    metadata: Optional[Dict] = None


class PerformanceTracker:
    def __init__(self, mode: str = "baseline", db_path: Optional[Path] = None):
        if mode not in ["baseline", "optimized"]:
            raise ValueError(f"Mode must be 'baseline' or 'optimized', got: {mode}")
        
        self.mode = mode
        
        self.run_id = self._generate_unique_run_id()
        
        self._db_lock = threading.Lock()
        
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "data" / "performanceDB" / "performance_history.db"
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.conn = sqlite3.connect(
                str(self.db_path), 
                check_same_thread=False,
                timeout=30.0,
                isolation_level='DEFERRED'
            )
            self.conn.row_factory = sqlite3.Row
            
            essential_pragmas = {
                "busy_timeout": "30000",
                "journal_mode": "WAL",
                "synchronous": "NORMAL",
                "foreign_keys": "ON"
            }
            
            for pragma, value in essential_pragmas.items():
                self.conn.execute(f"PRAGMA {pragma} = {value}")
                
            logger.info(f"Database configured with essential settings")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
        
        self._create_tables()
        
        self.active_records: Dict[str, ExecutionRecord] = {}
        
        logger.info(f"PerformanceTracker initialized: mode={mode}, run_id={self.run_id}")
    
    def _generate_unique_run_id(self) -> str:
        entropy_sources = [
            str(uuid.uuid4()),
            str(time.time_ns()),
            str(os.getpid()),
            str(threading.current_thread().ident),
            self.mode,
            str(hash(frozenset(os.environ.items())))
        ]
        
        combined = "_".join(entropy_sources)
        hash_digest = hashlib.sha256(combined.encode()).hexdigest()[:16]
        
        return f"{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash_digest}"
    
    def _create_tables(self):
        with self._db_lock:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    
                    model_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_size TEXT,
                    
                    task_name TEXT NOT NULL,
                    task_type TEXT,
                    num_fewshot INTEGER,
                    batch_size INTEGER,
                    sample_limit INTEGER,
                    
                    gpu_id TEXT,
                    device TEXT,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    
                    status TEXT NOT NULL,
                    execution_time REAL,
                    
                    gpu_memory_start REAL,
                    gpu_memory_peak REAL,
                    gpu_memory_end REAL,
                    gpu_utilization_avg REAL,
                    cpu_percent_avg REAL,
                    ram_usage_peak REAL,
                    
                    gpu_temperature_start REAL,
                    gpu_temperature_peak REAL,
                    gpu_temperature_avg REAL,
                    gpu_temperature_end REAL,
                    
                    result_metrics TEXT,
                    error_message TEXT,
                    metadata TEXT,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    model_size TEXT,
                    task_name TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    run_id_pattern TEXT,
                    
                    num_executions INTEGER DEFAULT 0,
                    num_successes INTEGER DEFAULT 0,
                    num_failures INTEGER DEFAULT 0,
                    num_ooms INTEGER DEFAULT 0,
                    
                    avg_execution_time REAL,
                    std_execution_time REAL,
                    min_execution_time REAL,
                    max_execution_time REAL,
                    
                    avg_memory_peak REAL,
                    std_memory_peak REAL,
                    min_memory_peak REAL,
                    max_memory_peak REAL,
                    
                    avg_gpu_utilization REAL,
                    
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(model_id, task_name, mode, run_id_pattern)
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_task_mode_run ON execution_records(model_id, task_name, mode, run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp_mode ON execution_records(timestamp, mode)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status_mode_run ON execution_records(status, mode, run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_run_id ON execution_records(run_id)")
            
            self.conn.commit()
            logger.debug("Database tables created/verified successfully")
    
    def _extract_model_size(self, model_id: str) -> str:
        model_id_lower = model_id.lower()
        
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
        
        return "unknown"
    
    def _get_gpu_memory_info(self):
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'used': gpu.memoryUsed,
                    'free': gpu.memoryFree,
                    'total': gpu.memoryTotal,
                    'utilization': gpu.load * 100
                }
        except Exception as e:
            logger.debug(f"Error getting GPU info: {e}")
        return {'used': 0, 'free': 0, 'total': 0, 'utilization': 0}
    
    def _get_gpu_memory(self) -> Tuple[float, float]:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                memory_used = gpu.memoryUsed / 1024
                utilization = gpu.load * 100
                return memory_used, utilization
        except Exception as e:
            logger.debug(f"Error getting GPU memory: {e}")
        return 0.0, 0.0
    
    def _get_gpu_temperature(self) -> float:
        """Get current GPU temperature"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return gpu.temperature
        except Exception as e:
            logger.debug(f"Error getting GPU temperature: {e}")
        return 0.0
    
    def _get_cpu_ram_usage(self) -> Tuple[float, float]:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory()
            ram_used_gb = ram.used / (1024**3)
            return cpu_percent, ram_used_gb
        except Exception as e:
            logger.debug(f"Error getting CPU/RAM usage: {e}")
        return 0.0, 0.0
    
    def record_start(self, model_id: str, model_name: str, task_name: str, config: Dict[str, Any]) -> str:
        record_key = f"{model_id}_{task_name}_{time.time_ns()}_{os.getpid()}"
        
        gpu_memory, gpu_util = self._get_gpu_memory()
        cpu_percent, ram_usage = self._get_cpu_ram_usage()
        gpu_temperature = self._get_gpu_temperature()  # NEW: Get thermal data
        
        record = ExecutionRecord(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            mode=self.mode,
            model_id=model_id,
            model_name=model_name,
            model_size=self._extract_model_size(model_id),
            task_name=task_name,
            task_type="harness",
            num_fewshot=config.get("num_fewshot", 0),
            batch_size=config.get("batch_size", 1),
            sample_limit=config.get("limit"),
            gpu_id=config.get("gpu_id", "0"),
            device=config.get("device", "cuda"),
            start_time=time.time(),
            gpu_memory_start=gpu_memory,
            gpu_temperature_start=gpu_temperature,  # NEW
            cpu_percent_avg=cpu_percent,
            ram_usage_peak=ram_usage,
            metadata=config.get("metadata", {})
        )
        
        self.active_records[record_key] = record
        logger.info(f"Started tracking: {model_name} on {task_name} (run_id: {self.run_id}, temp: {gpu_temperature:.1f}°C)")
        
        return record_key
    
    def record_end(self, record_key: str, status: str = "completed", 
                   results: Optional[Dict] = None, error_message: Optional[str] = None):
        if record_key not in self.active_records:
            logger.warning(f"No active record found for key: {record_key}")
            return
        
        record = self.active_records[record_key]
        record.end_time = time.time()
        record.execution_time = record.end_time - record.start_time
        record.status = status
        
        gpu_memory, gpu_util = self._get_gpu_memory()
        gpu_temperature_end = self._get_gpu_temperature()  # NEW: Get end temperature
        
        record.gpu_memory_end = gpu_memory
        record.gpu_utilization_avg = gpu_util
        
        # NEW: Update thermal data
        record.gpu_temperature_end = gpu_temperature_end
        record.gpu_temperature_peak = max(record.gpu_temperature_start or 0, gpu_temperature_end)
        record.gpu_temperature_avg = (record.gpu_temperature_start + gpu_temperature_end) / 2 if record.gpu_temperature_start else gpu_temperature_end
        
        cpu_percent, ram_usage = self._get_cpu_ram_usage()
        record.cpu_percent_avg = (record.cpu_percent_avg + cpu_percent) / 2
        record.ram_usage_peak = max(record.ram_usage_peak, ram_usage)
        
        if results:
            record.result_metrics = results
        if error_message:
            record.error_message = error_message
        
        record.gpu_memory_peak = max(record.gpu_memory_start or 0, record.gpu_memory_end or 0)
        
        self._save_record(record)
        self._update_statistics_simplified(record)
        
        del self.active_records[record_key]
        
        logger.info(f"Completed tracking: {record.model_name} on {record.task_name} "
                   f"- Status: {status}, Time: {record.execution_time:.2f}s, Temp: {gpu_temperature_end:.1f}°C")
    
    def _save_record(self, record: ExecutionRecord):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._db_lock:
                    cursor = self.conn.cursor()
                    
                    data = asdict(record)
                    
                    data['result_metrics'] = json.dumps(data['result_metrics']) if data['result_metrics'] else None
                    data['metadata'] = json.dumps(data['metadata']) if data['metadata'] else None
                    
                    columns = ', '.join(data.keys())
                    placeholders = ', '.join(['?' for _ in data])
                    query = f"INSERT INTO execution_records ({columns}) VALUES ({placeholders})"
                    
                    cursor.execute(query, list(data.values()))
                    self.conn.commit()
                    break
                    
            except sqlite3.OperationalError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Database locked, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(0.5 * (attempt + 1))
                else:
                    logger.error(f"Failed to save record after {max_retries} attempts: {e}")
                    raise
    
    def _update_statistics_simplified(self, record: ExecutionRecord):
        if record.status not in ["completed", "failed", "oom"]:
            return
        
        run_id_pattern = f"{record.mode}_%"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._db_lock:
                    cursor = self.conn.cursor()
                    cursor.execute("PRAGMA busy_timeout = 5000")
                    
                    cursor.execute("""
                        SELECT * FROM execution_stats 
                        WHERE model_id = ? AND task_name = ? AND mode = ? AND run_id_pattern = ?
                    """, (record.model_id, record.task_name, record.mode, run_id_pattern))
                    
                    stats = cursor.fetchone()
                    
                    if stats:
                        self._update_existing_stats(cursor, record, run_id_pattern)
                    else:
                        self._create_new_stats(cursor, record, run_id_pattern)
                    
                    self.conn.commit()
                    break
                    
            except sqlite3.OperationalError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Database locked during stats update, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(0.5 * (attempt + 1))
                else:
                    logger.error(f"Failed to update statistics after {max_retries} attempts: {e}")
                    raise
    
    def _update_existing_stats(self, cursor, record: ExecutionRecord, run_id_pattern: str):
        cursor.execute("""
            SELECT execution_time, gpu_memory_peak, gpu_utilization_avg
            FROM execution_records
            WHERE model_id = ? AND task_name = ? AND mode = ? AND run_id LIKE ? AND status = 'completed'
        """, (record.model_id, record.task_name, record.mode, run_id_pattern))
        
        records = cursor.fetchall()
        
        if records:
            exec_times = [r['execution_time'] for r in records if r['execution_time']]
            memory_peaks = [r['gpu_memory_peak'] for r in records if r['gpu_memory_peak']]
            gpu_utils = [r['gpu_utilization_avg'] for r in records if r['gpu_utilization_avg']]
            
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
            
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN status = 'oom' THEN 1 ELSE 0 END) as num_ooms,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as num_failures
                FROM execution_records
                WHERE model_id = ? AND task_name = ? AND mode = ? AND run_id LIKE ?
            """, (record.model_id, record.task_name, record.mode, run_id_pattern))
            
            counts = cursor.fetchone()
            stats_update['num_ooms'] = counts['num_ooms'] or 0
            stats_update['num_failures'] = counts['num_failures'] or 0
            
            set_clause = ', '.join([f"{k} = ?" for k in stats_update.keys()])
            values = list(stats_update.values()) + [record.model_id, record.task_name, record.mode, run_id_pattern]
            
            cursor.execute(f"""
                UPDATE execution_stats 
                SET {set_clause}, last_updated = CURRENT_TIMESTAMP
                WHERE model_id = ? AND task_name = ? AND mode = ? AND run_id_pattern = ?
            """, values)
    
    def _create_new_stats(self, cursor, record: ExecutionRecord, run_id_pattern: str):
        if record.status == 'completed':
            cursor.execute("""
                INSERT INTO execution_stats (
                    model_id, model_size, task_name, mode, run_id_pattern,
                    num_executions, num_successes, num_failures, num_ooms,
                    avg_execution_time, min_execution_time, max_execution_time,
                    avg_memory_peak, min_memory_peak, max_memory_peak,
                    avg_gpu_utilization
                ) VALUES (?, ?, ?, ?, ?, 1, 1, 0, 0, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.model_id, record.model_size, record.task_name, record.mode, run_id_pattern,
                record.execution_time, record.execution_time, record.execution_time,
                record.gpu_memory_peak, record.gpu_memory_peak, record.gpu_memory_peak,
                record.gpu_utilization_avg
            ))
        else:
            num_ooms = 1 if record.status == 'oom' else 0
            num_failures = 1 if record.status == 'failed' else 0
            
            cursor.execute("""
                INSERT INTO execution_stats (
                    model_id, model_size, task_name, mode, run_id_pattern,
                    num_executions, num_successes, num_failures, num_ooms
                ) VALUES (?, ?, ?, ?, ?, 1, 0, ?, ?)
            """, (record.model_id, record.model_size, record.task_name, record.mode, run_id_pattern, num_failures, num_ooms))
    
    def predict_execution_safe(self, model_id: str, task_name: str) -> Dict[str, Any]:
        default_result = {
            'predicted_time': None,
            'predicted_memory': None,
            'success_rate': None,
            'oom_rate': None,
            'num_samples': 0,
            'error': None
        }
        
        try:
            return self._predict_execution_internal(model_id, task_name)
            
        except sqlite3.OperationalError as e:
            default_result['error'] = f"Database error: {str(e)[:100]}"
            logger.warning(f"Database error in prediction: {e}")
            
        except Exception as e:
            default_result['error'] = f"Prediction error: {str(e)[:100]}"
            logger.error(f"Unexpected error in prediction: {e}")
            
        return default_result
    
    def _predict_execution_internal(self, model_id: str, task_name: str) -> Dict[str, Any]:
        run_id_pattern = f"{self.mode}_%"
        
        with self._db_lock:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT * FROM execution_stats 
                WHERE model_id = ? AND task_name = ? AND mode = ? AND run_id_pattern = ?
            """, (model_id, task_name, self.mode, run_id_pattern))
            
            stats = cursor.fetchone()
            
            if not stats:
                model_size = self._extract_model_size(model_id)
                cursor.execute("""
                    SELECT * FROM execution_stats 
                    WHERE model_size = ? AND task_name = ? AND mode = ? AND run_id_pattern = ?
                    LIMIT 1
                """, (model_size, task_name, self.mode, run_id_pattern))
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
            
            return {
                'predicted_time': None,
                'predicted_memory': None,
                'success_rate': None,
                'oom_rate': None,
                'num_samples': 0
            }
    
    def predict_execution(self, model_id: str, task_name: str) -> Dict[str, Any]:
        return self.predict_execution_safe(model_id, task_name)
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        run_id_pattern = f"{self.mode}_%"
        
        try:
            with self._db_lock:
                cursor = self.conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_runs,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_runs,
                        SUM(CASE WHEN status = 'oom' THEN 1 ELSE 0 END) as oom_runs,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
                        AVG(execution_time) as avg_execution_time,
                        SUM(execution_time) as total_execution_time
                    FROM execution_records
                    WHERE mode = ? AND run_id LIKE ?
                """, (self.mode, run_id_pattern))
                
                overall_result = cursor.fetchone()
                
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
                
                cursor.execute("""
                    SELECT 
                        model_name, model_size,
                        COUNT(*) as runs,
                        AVG(execution_time) as avg_time,
                        AVG(gpu_memory_peak) as avg_memory
                    FROM execution_records
                    WHERE mode = ? AND run_id LIKE ? AND status = 'completed'
                    GROUP BY model_id
                """, (self.mode, run_id_pattern))
                
                model_results = cursor.fetchall()
                
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
                    'run_id': self.run_id,
                    'overall': overall,
                    'by_model': model_stats
                }
                
        except Exception as e:
            logger.error(f"Error in get_statistics_summary: {e}")
            return {
                'mode': self.mode,
                'run_id': self.run_id,
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
    
    def get_recent_executions(self, limit: int = 50) -> List[Dict]:
        try:
            with self._db_lock:
                cursor = self.conn.cursor()
                cursor.execute("PRAGMA busy_timeout = 3000")
                
                cursor.execute("""
                    SELECT model_name, task_name, status, execution_time, 
                           gpu_memory_peak, gpu_temperature_peak, timestamp
                    FROM execution_records
                    WHERE mode = ? AND run_id LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (self.mode, f"{self.mode}_%", limit))
                
                records = cursor.fetchall()
                
                return [
                    {
                        'model_name': r['model_name'],
                        'task_name': r['task_name'],
                        'status': r['status'],
                        'execution_time': r['execution_time'],
                        'gpu_memory_peak': r['gpu_memory_peak'],
                        'gpu_temperature_peak': r['gpu_temperature_peak'],  # NEW
                        'timestamp': r['timestamp']
                    }
                    for r in records
                ]
                
        except Exception as e:
            logger.error(f"Error getting recent executions: {e}")
            return []
    
    def get_model_performance_summary(self, model_id: str) -> Dict[str, Any]:
        try:
            with self._db_lock:
                cursor = self.conn.cursor()
                cursor.execute("PRAGMA busy_timeout = 3000")
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_runs,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                        SUM(CASE WHEN status = 'oom' THEN 1 ELSE 0 END) as oom_failures,
                        AVG(CASE WHEN status = 'completed' THEN execution_time END) as avg_time,
                        AVG(CASE WHEN status = 'completed' THEN gpu_memory_peak END) as avg_memory,
                        AVG(CASE WHEN status = 'completed' THEN gpu_temperature_avg END) as avg_temperature
                    FROM execution_records
                    WHERE model_id = ? AND mode = ? AND run_id LIKE ?
                """, (model_id, self.mode, f"{self.mode}_%"))
                
                result = cursor.fetchone()
                
                if result and result['total_runs'] > 0:
                    return {
                        'model_id': model_id,
                        'total_runs': result['total_runs'],
                        'success_rate': result['completed'] / result['total_runs'],
                        'oom_rate': result['oom_failures'] / result['total_runs'],
                        'avg_execution_time': result['avg_time'],
                        'avg_memory_usage': result['avg_memory'],
                        'avg_temperature': result['avg_temperature']  # NEW
                    }
                else:
                    return {
                        'model_id': model_id,
                        'total_runs': 0,
                        'success_rate': 0.0,
                        'oom_rate': 0.0,
                        'avg_execution_time': None,
                        'avg_memory_usage': None,
                        'avg_temperature': None  # NEW
                    }
                    
        except Exception as e:
            logger.error(f"Error getting model performance summary: {e}")
            return {
                'model_id': model_id,
                'error': str(e),
                'total_runs': 0,
                'success_rate': 0.0,
                'oom_rate': 0.0,
                'avg_execution_time': None,
                'avg_memory_usage': None,
                'avg_temperature': None
            }
    
    def cleanup_old_records(self, retention_days: int = 90):
        try:
            with self._db_lock:
                cursor = self.conn.cursor()
                cursor.execute("PRAGMA busy_timeout = 10000")
                
                cursor.execute("""
                    DELETE FROM execution_records
                    WHERE created_at < datetime('now', '-{} days')
                    AND mode = ?
                """.format(retention_days), (self.mode,))
                
                deleted_count = cursor.rowcount
                
                cursor.execute("""
                    DELETE FROM execution_stats
                    WHERE last_updated < datetime('now', '-{} days')
                    AND mode = ?
                """.format(retention_days), (self.mode,))
                
                stats_deleted = cursor.rowcount
                
                self.conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old execution records and {stats_deleted} old stats for mode {self.mode}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old records: {e}")
    
    def export_data(self, output_path: Path, include_raw_records: bool = True):
        try:
            data = {
                'metadata': {
                    'mode': self.mode,
                    'run_id': self.run_id,
                    'export_timestamp': datetime.now().isoformat(),
                },
                'summary': self.get_statistics_summary()
            }
            
            if include_raw_records:
                with self._db_lock:
                    cursor = self.conn.cursor()
                    cursor.execute("""
                        SELECT * FROM execution_records
                        WHERE mode = ? AND run_id LIKE ?
                        ORDER BY timestamp DESC
                    """, (self.mode, f"{self.mode}_%"))
                    
                    records = cursor.fetchall()
                    data['raw_records'] = [dict(record) for record in records]
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Performance data exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
    
    def get_failure_analysis(self) -> Dict[str, Any]:
        try:
            with self._db_lock:
                cursor = self.conn.cursor()
                cursor.execute("PRAGMA busy_timeout = 3000")
                
                cursor.execute("""
                    SELECT 
                        model_name,
                        task_name,
                        status,
                        error_message,
                        COUNT(*) as failure_count
                    FROM execution_records
                    WHERE mode = ? AND run_id LIKE ? AND status IN ('failed', 'oom')
                    GROUP BY model_name, task_name, status, error_message
                    ORDER BY failure_count DESC
                """, (self.mode, f"{self.mode}_%"))
                
                failures = cursor.fetchall()
                
                analysis = {
                    'total_failures': sum(f['failure_count'] for f in failures),
                    'failure_patterns': [],
                    'oom_models': [],
                    'frequent_errors': {}
                }
                
                for failure in failures:
                    pattern = {
                        'model': failure['model_name'],
                        'task': failure['task_name'],
                        'status': failure['status'],
                        'count': failure['failure_count'],
                        'error': failure['error_message']
                    }
                    analysis['failure_patterns'].append(pattern)
                    
                    if failure['status'] == 'oom':
                        analysis['oom_models'].append(failure['model_name'])
                    
                    if failure['error_message']:
                        error_key = failure['error_message'][:100]
                        analysis['frequent_errors'][error_key] = analysis['frequent_errors'].get(error_key, 0) + failure['failure_count']
                
                return analysis
                
        except Exception as e:
            logger.error(f"Error in failure analysis: {e}")
            return {
                'total_failures': 0,
                'failure_patterns': [],
                'oom_models': [],
                'frequent_errors': {},
                'error': str(e)
            }
    
    def close(self):
        try:
            if hasattr(self, 'conn') and self.conn:
                with self._db_lock:
                    self.conn.close()
                logger.info(f"Database connection closed for mode: {self.mode}")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
    
    def __del__(self):
        try:
            self.close()
        except:
            pass


class RecordBuffer:
    def __init__(self, performance_tracker: PerformanceTracker, max_size=1000, flush_interval=60):
        self.performance_tracker = performance_tracker
        self.buffer = []
        self.max_size = max_size
        self.flush_interval = flush_interval
        self.last_flush = time.time()
        self._buffer_lock = threading.Lock()
        
    def add_record(self, record: ExecutionRecord):
        with self._buffer_lock:
            self.buffer.append(record)
            
            if (len(self.buffer) >= self.max_size or 
                time.time() - self.last_flush > self.flush_interval):
                self.flush()
    
    def flush(self):
        if not self.buffer:
            return
            
        try:
            with self._buffer_lock:
                cursor = self.performance_tracker.conn.cursor()
                
                for record in self.buffer:
                    self.performance_tracker._save_record(record)
                
                logger.debug(f"Flushed {len(self.buffer)} records to database")
                self.buffer.clear()
                self.last_flush = time.time()
                
        except Exception as e:
            logger.error(f"Batch flush failed: {e}")
            self._fallback_individual_insert()
    
    def _fallback_individual_insert(self):
        failed_records = []
        for record in self.buffer:
            try:
                self.performance_tracker._save_record(record)
            except Exception as e:
                logger.error(f"Failed to save individual record: {e}")
                failed_records.append(record)
        
        self.buffer = failed_records
        if failed_records:
            logger.warning(f"{len(failed_records)} records could not be saved")


def create_performance_tracker(mode: str, db_path: Optional[Path] = None) -> PerformanceTracker:
    if mode not in ["baseline", "optimized"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'baseline' or 'optimized'")
    
    return PerformanceTracker(mode=mode, db_path=db_path)


def get_performance_summary(mode: str, db_path: Optional[Path] = None) -> Dict[str, Any]:
    tracker = create_performance_tracker(mode, db_path)
    try:
        return tracker.get_statistics_summary()
    finally:
        tracker.close()


def compare_mode_performance(baseline_db: Optional[Path] = None, 
                           optimized_db: Optional[Path] = None) -> Dict[str, Any]:
    baseline_tracker = create_performance_tracker("baseline", baseline_db)
    optimized_tracker = create_performance_tracker("optimized", optimized_db)
    
    try:
        baseline_stats = baseline_tracker.get_statistics_summary()
        optimized_stats = optimized_tracker.get_statistics_summary()
        
        comparison = {
            'baseline': baseline_stats,
            'optimized': optimized_stats,
            'improvements': {}
        }
        
        if (baseline_stats['overall']['total_execution_time'] > 0 and 
            optimized_stats['overall']['total_execution_time'] > 0):
            
            time_improvement = (
                baseline_stats['overall']['total_execution_time'] - 
                optimized_stats['overall']['total_execution_time']
            ) / baseline_stats['overall']['total_execution_time'] * 100
            
            comparison['improvements']['time_saved_percent'] = time_improvement
        
        if (baseline_stats['overall']['oom_runs'] >= 0 and 
            optimized_stats['overall']['oom_runs'] >= 0):
            
            oom_reduction = baseline_stats['overall']['oom_runs'] - optimized_stats['overall']['oom_runs']
            comparison['improvements']['oom_reduction'] = oom_reduction
        
        return comparison
        
    finally:
        baseline_tracker.close()
        optimized_tracker.close()