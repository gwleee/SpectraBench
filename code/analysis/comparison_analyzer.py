"""
Comparison Analyzer for LLM Benchmark
Baseline vs Optimized mode comparison analysis
WITH GPU THERMAL ANALYSIS
"""
import sqlite3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import defaultdict

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Comparison analysis result"""
    # Overall summary
    total_time_baseline: float
    total_time_optimized: float
    time_improvement_percent: float
    
    # Success rates
    success_rate_baseline: float
    success_rate_optimized: float
    
    # OOM rates
    oom_rate_baseline: float
    oom_rate_optimized: float
    oom_reduction_percent: float
    
    # GPU utilization
    avg_gpu_util_baseline: float
    avg_gpu_util_optimized: float
    
    # Memory usage
    avg_memory_baseline: float
    avg_memory_optimized: float
    memory_efficiency_gain: float
    
    # NEW: Thermal data
    avg_temp_baseline: float
    avg_temp_optimized: float
    peak_temp_baseline: float
    peak_temp_optimized: float
    thermal_improvement_percent: float
    high_temp_events_baseline: int
    high_temp_events_optimized: int
    thermal_events_reduction_percent: float
    
    # Task-wise improvements
    task_improvements: Dict[str, float]
    model_improvements: Dict[str, float]
    
    # Detailed data
    baseline_records: List[Dict]
    optimized_records: List[Dict]


class ComparisonAnalyzer:
    """Baseline vs Optimized comparison analysis class"""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Args:
            db_path: Performance Tracker DB path
        """
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "data" / "performanceDB" / "performance_history.db"
        
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        
        logger.info(f"ComparisonAnalyzer initialized with DB: {db_path}")
    
    def load_baseline_data(self, run_id_pattern: Optional[str] = None) -> pd.DataFrame:
        """Load Baseline mode data"""
        query = """
            SELECT * FROM execution_records 
            WHERE mode = 'baseline'
        """
        if run_id_pattern:
            query += f" AND run_id LIKE '%{run_id_pattern}%'"
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, self.conn)
        logger.info(f"Loaded {len(df)} baseline records")
        return df
    
    def load_optimized_data(self, run_id_pattern: Optional[str] = None) -> pd.DataFrame:
        """Load Optimized mode data"""
        query = """
            SELECT * FROM execution_records 
            WHERE mode = 'optimized'
        """
        if run_id_pattern:
            query += f" AND run_id LIKE '%{run_id_pattern}%'"
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, self.conn)
        logger.info(f"Loaded {len(df)} optimized records")
        return df
    
    def get_thermal_analysis(self) -> Dict[str, Any]:
        """GPU thermal management analysis"""
        try:
            query = """
                SELECT 
                    mode,
                    model_name,
                    model_size,
                    AVG(gpu_temperature_avg) as avg_temp,
                    MAX(gpu_temperature_peak) as peak_temp,
                    MIN(gpu_temperature_start) as min_temp,
                    COUNT(CASE WHEN gpu_temperature_peak > 85 THEN 1 END) as high_temp_count,
                    COUNT(CASE WHEN gpu_temperature_peak > 90 THEN 1 END) as critical_temp_count,
                    COUNT(*) as total_tasks
                FROM execution_records
                WHERE gpu_temperature_avg IS NOT NULL
                GROUP BY mode, model_name, model_size
                ORDER BY mode, model_size DESC
            """
            
            df = pd.read_sql_query(query, self.conn)
            
            # Separate baseline and optimized data
            baseline_thermal = df[df['mode'] == 'baseline']
            optimized_thermal = df[df['mode'] == 'optimized']
            
            analysis = {
                'baseline_thermal': baseline_thermal.to_dict('records'),
                'optimized_thermal': optimized_thermal.to_dict('records'),
                'thermal_comparison': {},
                'cooling_effectiveness': {}
            }
            
            # Calculate thermal improvements
            if len(baseline_thermal) > 0 and len(optimized_thermal) > 0:
                baseline_avg_temp = baseline_thermal['avg_temp'].mean()
                optimized_avg_temp = optimized_thermal['avg_temp'].mean()
                
                analysis['thermal_comparison'] = {
                    'baseline_avg_temp': baseline_avg_temp,
                    'optimized_avg_temp': optimized_avg_temp,
                    'temperature_reduction': baseline_avg_temp - optimized_avg_temp,
                    'temperature_improvement_percent': ((baseline_avg_temp - optimized_avg_temp) / baseline_avg_temp * 100) if baseline_avg_temp > 0 else 0
                }
                
                # High temperature event comparison
                baseline_high_temp = baseline_thermal['high_temp_count'].sum()
                optimized_high_temp = optimized_thermal['high_temp_count'].sum()
                
                analysis['cooling_effectiveness'] = {
                    'baseline_high_temp_events': int(baseline_high_temp),
                    'optimized_high_temp_events': int(optimized_high_temp),
                    'high_temp_reduction': int(baseline_high_temp - optimized_high_temp),
                    'high_temp_reduction_percent': ((baseline_high_temp - optimized_high_temp) / baseline_high_temp * 100) if baseline_high_temp > 0 else 0
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in thermal analysis: {e}")
            return {'error': str(e)}
    
    def calculate_improvements(self, 
                             baseline_df: Optional[pd.DataFrame] = None,
                             optimized_df: Optional[pd.DataFrame] = None) -> ComparisonResult:
        """Calculate improvement metrics"""
        # Load data
        if baseline_df is None:
            baseline_df = self.load_baseline_data()
        if optimized_df is None:
            optimized_df = self.load_optimized_data()
        
        # Filter completed tasks only
        baseline_completed = baseline_df[baseline_df['status'] == 'completed']
        optimized_completed = optimized_df[optimized_df['status'] == 'completed']
        
        # Total execution time
        total_time_baseline = baseline_df['execution_time'].sum()
        total_time_optimized = optimized_df['execution_time'].sum()
        time_improvement = ((total_time_baseline - total_time_optimized) / total_time_baseline * 100) if total_time_baseline > 0 else 0
        
        # Success rates
        success_rate_baseline = len(baseline_completed) / len(baseline_df) if len(baseline_df) > 0 else 0
        success_rate_optimized = len(optimized_completed) / len(optimized_df) if len(optimized_df) > 0 else 0
        
        # OOM rates
        baseline_oom = baseline_df[baseline_df['status'] == 'oom']
        optimized_oom = optimized_df[optimized_df['status'] == 'oom']
        oom_rate_baseline = len(baseline_oom) / len(baseline_df) if len(baseline_df) > 0 else 0
        oom_rate_optimized = len(optimized_oom) / len(optimized_df) if len(optimized_df) > 0 else 0
        oom_reduction = ((oom_rate_baseline - oom_rate_optimized) / oom_rate_baseline * 100) if oom_rate_baseline > 0 else 0
        
        # GPU utilization (completed tasks only)
        avg_gpu_util_baseline = baseline_completed['gpu_utilization_avg'].mean() if len(baseline_completed) > 0 else 0
        avg_gpu_util_optimized = optimized_completed['gpu_utilization_avg'].mean() if len(optimized_completed) > 0 else 0
        
        # Memory usage
        avg_memory_baseline = baseline_completed['gpu_memory_peak'].mean() if len(baseline_completed) > 0 else 0
        avg_memory_optimized = optimized_completed['gpu_memory_peak'].mean() if len(optimized_completed) > 0 else 0
        memory_efficiency = ((avg_memory_baseline - avg_memory_optimized) / avg_memory_baseline * 100) if avg_memory_baseline > 0 else 0
        
        # NEW: Thermal calculations
        # Average temperatures (completed tasks only)
        avg_temp_baseline = baseline_completed['gpu_temperature_avg'].mean() if len(baseline_completed) > 0 and 'gpu_temperature_avg' in baseline_completed.columns else 0
        avg_temp_optimized = optimized_completed['gpu_temperature_avg'].mean() if len(optimized_completed) > 0 and 'gpu_temperature_avg' in optimized_completed.columns else 0
        
        # Peak temperatures
        peak_temp_baseline = baseline_df['gpu_temperature_peak'].max() if len(baseline_df) > 0 and 'gpu_temperature_peak' in baseline_df.columns else 0
        peak_temp_optimized = optimized_df['gpu_temperature_peak'].max() if len(optimized_df) > 0 and 'gpu_temperature_peak' in optimized_df.columns else 0
        
        # Thermal improvement
        thermal_improvement = ((avg_temp_baseline - avg_temp_optimized) / avg_temp_baseline * 100) if avg_temp_baseline > 0 else 0
        
        # High temperature events (>85°C)
        high_temp_baseline = len(baseline_df[baseline_df['gpu_temperature_peak'] > 85]) if 'gpu_temperature_peak' in baseline_df.columns else 0
        high_temp_optimized = len(optimized_df[optimized_df['gpu_temperature_peak'] > 85]) if 'gpu_temperature_peak' in optimized_df.columns else 0
        thermal_events_reduction = ((high_temp_baseline - high_temp_optimized) / high_temp_baseline * 100) if high_temp_baseline > 0 else 0
        
        # Task-wise improvements
        task_improvements = self._calculate_task_improvements(baseline_df, optimized_df)
        
        # Model-wise improvements
        model_improvements = self._calculate_model_improvements(baseline_df, optimized_df)
        
        return ComparisonResult(
            total_time_baseline=total_time_baseline,
            total_time_optimized=total_time_optimized,
            time_improvement_percent=time_improvement,
            success_rate_baseline=success_rate_baseline,
            success_rate_optimized=success_rate_optimized,
            oom_rate_baseline=oom_rate_baseline,
            oom_rate_optimized=oom_rate_optimized,
            oom_reduction_percent=oom_reduction,
            avg_gpu_util_baseline=avg_gpu_util_baseline,
            avg_gpu_util_optimized=avg_gpu_util_optimized,
            avg_memory_baseline=avg_memory_baseline,
            avg_memory_optimized=avg_memory_optimized,
            memory_efficiency_gain=memory_efficiency,
            
            # NEW: Thermal fields
            avg_temp_baseline=avg_temp_baseline,
            avg_temp_optimized=avg_temp_optimized,
            peak_temp_baseline=peak_temp_baseline,
            peak_temp_optimized=peak_temp_optimized,
            thermal_improvement_percent=thermal_improvement,
            high_temp_events_baseline=high_temp_baseline,
            high_temp_events_optimized=high_temp_optimized,
            thermal_events_reduction_percent=thermal_events_reduction,
            
            task_improvements=task_improvements,
            model_improvements=model_improvements,
            baseline_records=baseline_df.to_dict('records'),
            optimized_records=optimized_df.to_dict('records')
        )
    
    def _calculate_task_improvements(self, baseline_df: pd.DataFrame, optimized_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate task-wise execution time improvements"""
        improvements = {}
        
        # Group by task
        baseline_by_task = baseline_df.groupby('task_name')['execution_time'].sum()
        optimized_by_task = optimized_df.groupby('task_name')['execution_time'].sum()
        
        # Compare common tasks only
        common_tasks = set(baseline_by_task.index) & set(optimized_by_task.index)
        
        for task in common_tasks:
            baseline_time = baseline_by_task[task]
            optimized_time = optimized_by_task[task]
            improvement = ((baseline_time - optimized_time) / baseline_time * 100) if baseline_time > 0 else 0
            improvements[task] = improvement
        
        return improvements
    
    def _calculate_model_improvements(self, baseline_df: pd.DataFrame, optimized_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate model-wise execution time improvements"""
        improvements = {}
        
        # Group by model
        baseline_by_model = baseline_df.groupby('model_name')['execution_time'].sum()
        optimized_by_model = optimized_df.groupby('model_name')['execution_time'].sum()
        
        # Compare common models only
        common_models = set(baseline_by_model.index) & set(optimized_by_model.index)
        
        for model in common_models:
            baseline_time = baseline_by_model[model]
            optimized_time = optimized_by_model[model]
            improvement = ((baseline_time - optimized_time) / baseline_time * 100) if baseline_time > 0 else 0
            improvements[model] = improvement
        
        return improvements
    
    def get_execution_timeline(self, mode: str) -> List[Dict]:
        """Query execution timeline data"""
        query = """
            SELECT 
                model_name,
                task_name,
                start_time,
                end_time,
                execution_time,
                gpu_memory_peak,
                gpu_temperature_peak,
                status
            FROM execution_records
            WHERE mode = ?
            ORDER BY start_time
        """
        
        cursor = self.conn.cursor()
        cursor.execute(query, (mode,))
        
        timeline = []
        for row in cursor.fetchall():
            timeline.append({
                'model': row['model_name'],
                'task': row['task_name'],
                'start': row['start_time'],
                'end': row['end_time'],
                'duration': row['execution_time'],
                'memory': row['gpu_memory_peak'],
                'temperature': row['gpu_temperature_peak'],  # NEW
                'status': row['status']
            })
        
        return timeline
    
    def get_batch_size_analysis(self) -> Dict[str, Any]:
        """Batch size analysis"""
        query = """
            SELECT 
                mode,
                batch_size,
                COUNT(*) as count,
                AVG(execution_time) as avg_time,
                AVG(gpu_memory_peak) as avg_memory,
                AVG(gpu_temperature_avg) as avg_temperature,
                SUM(CASE WHEN status = 'oom' THEN 1 ELSE 0 END) as oom_count
            FROM execution_records
            WHERE batch_size IS NOT NULL
            GROUP BY mode, batch_size
        """
        
        df = pd.read_sql_query(query, self.conn)
        
        return {
            'baseline': df[df['mode'] == 'baseline'].to_dict('records'),
            'optimized': df[df['mode'] == 'optimized'].to_dict('records')
        }
    
    def get_learning_curve(self) -> Dict[str, List[float]]:
        """Performance improvement trends by execution iteration"""
        # Execution history for each model-task combination
        query = """
            SELECT 
                model_id,
                task_name,
                mode,
                execution_time,
                timestamp
            FROM execution_records
            WHERE status = 'completed'
            ORDER BY model_id, task_name, timestamp
        """
        
        df = pd.read_sql_query(query, self.conn)
        
        # Calculate average time by execution iteration
        learning_curves = {'baseline': [], 'optimized': []}
        
        for mode in ['baseline', 'optimized']:
            mode_df = df[df['mode'] == mode]
            if len(mode_df) == 0:
                continue
            
            # Sort by time and calculate cumulative average
            mode_df = mode_df.sort_values('timestamp')
            cumulative_avg = []
            
            for i in range(1, len(mode_df) + 1):
                avg_time = mode_df.iloc[:i]['execution_time'].mean()
                cumulative_avg.append(avg_time)
            
            learning_curves[mode] = cumulative_avg
        
        return learning_curves
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate comprehensive comparison report"""
        result = self.calculate_improvements()
        
        # Add timestamp to filename only if not already present
        if output_path and not any(char.isdigit() for char in output_path.stem[-8:]):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stem = output_path.stem
            suffix = output_path.suffix
            output_path = output_path.parent / f"{stem}_{timestamp}{suffix}"
        
        report = []
        report.append("=" * 80)
        report.append("LLM Benchmark Performance Comparison Report")
        report.append("Baseline vs Optimized Mode Analysis")
        report.append("=" * 80)
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive summary
        report.append("## EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Execution Time:")
        report.append(f"  - Baseline:  {result.total_time_baseline/3600:.2f} hours")
        report.append(f"  - Optimized: {result.total_time_optimized/3600:.2f} hours")
        report.append(f"  - Improvement: {result.time_improvement_percent:.1f}% ⬆️")
        report.append("")
        
        report.append(f"Success Rate:")
        report.append(f"  - Baseline:  {result.success_rate_baseline*100:.1f}%")
        report.append(f"  - Optimized: {result.success_rate_optimized*100:.1f}%")
        improvement = (result.success_rate_optimized - result.success_rate_baseline) * 100
        if improvement > 0:
            report.append(f"  - Improvement: +{improvement:.1f}% ⬆️")
        report.append("")
        
        report.append(f"OOM Rate:")
        report.append(f"  - Baseline:  {result.oom_rate_baseline*100:.1f}%")
        report.append(f"  - Optimized: {result.oom_rate_optimized*100:.1f}%")
        report.append(f"  - Reduction: {result.oom_reduction_percent:.1f}% ⬇️")
        report.append("")
        
        report.append(f"Resource Utilization:")
        report.append(f"  - GPU Utilization:")
        report.append(f"    - Baseline:  {result.avg_gpu_util_baseline:.1f}%")
        report.append(f"    - Optimized: {result.avg_gpu_util_optimized:.1f}%")
        report.append(f"  - Memory Usage:")
        report.append(f"    - Baseline:  {result.avg_memory_baseline:.1f} GB")
        report.append(f"    - Optimized: {result.avg_memory_optimized:.1f} GB")
        report.append(f"    - Efficiency Gain: {result.memory_efficiency_gain:.1f}% ⬇️")
        report.append("")
        
        # NEW: Thermal section
        report.append(f"Thermal Management:")
        report.append(f"  - Average GPU Temperature:")
        report.append(f"    - Baseline:  {result.avg_temp_baseline:.1f}°C")
        report.append(f"    - Optimized: {result.avg_temp_optimized:.1f}°C")
        if result.thermal_improvement_percent > 0:
            report.append(f"    - Temperature Reduction: {result.thermal_improvement_percent:.1f}% ⬇️")
        report.append(f"  - Peak GPU Temperature:")
        report.append(f"    - Baseline:  {result.peak_temp_baseline:.1f}°C")
        report.append(f"    - Optimized: {result.peak_temp_optimized:.1f}°C")
        report.append(f"  - High Temperature Events (>85°C):")
        report.append(f"    - Baseline:  {result.high_temp_events_baseline}")
        report.append(f"    - Optimized: {result.high_temp_events_optimized}")
        if result.thermal_events_reduction_percent > 0:
            report.append(f"    - Event Reduction: {result.thermal_events_reduction_percent:.1f}% ⬇️")
        report.append("")
        
        # Task-wise improvements
        if result.task_improvements:
            report.append("## TASK-WISE IMPROVEMENTS")
            report.append("-" * 40)
            sorted_tasks = sorted(result.task_improvements.items(), key=lambda x: x[1], reverse=True)
            for task, improvement in sorted_tasks[:10]:  # Top 10
                report.append(f"  {task:<30} {improvement:>6.1f}% ⬆️")
            if len(sorted_tasks) > 10:
                report.append(f"  ... and {len(sorted_tasks) - 10} more tasks")
            report.append("")
        
        # Model-wise improvements
        if result.model_improvements:
            report.append("## MODEL-WISE IMPROVEMENTS")
            report.append("-" * 40)
            sorted_models = sorted(result.model_improvements.items(), key=lambda x: x[1], reverse=True)
            for model, improvement in sorted_models:
                report.append(f"  {model:<30} {improvement:>6.1f}% ⬆️")
            report.append("")
        
        # Key findings
        report.append("## KEY FINDINGS")
        report.append("-" * 40)
        
        if result.time_improvement_percent > 30:
            report.append(f"✓ Significant time reduction of {result.time_improvement_percent:.1f}% achieved")
        
        if result.oom_reduction_percent > 50:
            report.append(f"✓ OOM errors reduced by {result.oom_reduction_percent:.1f}%")
        
        if result.memory_efficiency_gain > 20:
            report.append(f"✓ Memory usage optimized by {result.memory_efficiency_gain:.1f}%")
        
        # NEW: Thermal findings
        if result.thermal_improvement_percent > 5:
            report.append(f"✓ GPU temperature reduced by {result.thermal_improvement_percent:.1f}% on average")
        
        if result.thermal_events_reduction_percent > 50:
            report.append(f"✓ High temperature events reduced by {result.thermal_events_reduction_percent:.1f}%")
        
        # Best improvement task
        if result.task_improvements:
            best_task = max(result.task_improvements.items(), key=lambda x: x[1])
            report.append(f"✓ Best improvement: {best_task[0]} ({best_task[1]:.1f}% faster)")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save to file
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text)
            logger.info(f"Report saved to {output_path}")
        
        return report_text
    
    def export_detailed_data(self, output_dir: Path):
        """Export detailed data"""
        # Add timestamp only if not already present
        if not any(char.isdigit() for char in output_dir.name[-8:]):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = output_dir.parent / f"{output_dir.name}_{timestamp}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Overall comparison result
        result = self.calculate_improvements()
        
        # Save as JSON
        with open(output_dir / "comparison_result.json", 'w') as f:
            json.dump({
                'summary': {
                    'total_time_baseline': result.total_time_baseline,
                    'total_time_optimized': result.total_time_optimized,
                    'time_improvement_percent': result.time_improvement_percent,
                    'success_rate_baseline': result.success_rate_baseline,
                    'success_rate_optimized': result.success_rate_optimized,
                    'oom_rate_baseline': result.oom_rate_baseline,
                    'oom_rate_optimized': result.oom_rate_optimized,
                    'oom_reduction_percent': result.oom_reduction_percent,
                    'avg_gpu_util_baseline': result.avg_gpu_util_baseline,
                    'avg_gpu_util_optimized': result.avg_gpu_util_optimized,
                    'avg_memory_baseline': result.avg_memory_baseline,
                    'avg_memory_optimized': result.avg_memory_optimized,
                    'memory_efficiency_gain': result.memory_efficiency_gain,
                    
                    # NEW: Thermal summary
                    'avg_temp_baseline': result.avg_temp_baseline,
                    'avg_temp_optimized': result.avg_temp_optimized,
                    'peak_temp_baseline': result.peak_temp_baseline,
                    'peak_temp_optimized': result.peak_temp_optimized,
                    'thermal_improvement_percent': result.thermal_improvement_percent,
                    'high_temp_events_baseline': result.high_temp_events_baseline,
                    'high_temp_events_optimized': result.high_temp_events_optimized,
                    'thermal_events_reduction_percent': result.thermal_events_reduction_percent,
                },
                'task_improvements': result.task_improvements,
                'model_improvements': result.model_improvements,
            }, f, indent=2)
        
        # NEW: Save thermal analysis
        thermal_analysis = self.get_thermal_analysis()
        with open(output_dir / "thermal_analysis.json", 'w') as f:
            json.dump(thermal_analysis, f, indent=2)
        
        # Save as CSV
        baseline_df = pd.DataFrame(result.baseline_records)
        optimized_df = pd.DataFrame(result.optimized_records)
        
        baseline_df.to_csv(output_dir / "baseline_records.csv", index=False)
        optimized_df.to_csv(output_dir / "optimized_records.csv", index=False)
        
        logger.info(f"Detailed data exported to {output_dir}")
    
    def close(self):
        """Close DB connection"""
        self.conn.close()


def find_latest_experiment_dir() -> Optional[Path]:
    """Find the latest experiment directory"""
    experiments_root = Path("experiments_results")
    if not experiments_root.exists():
        return None
    
    exp_dirs = list(experiments_root.glob("exp_*"))
    if not exp_dirs:
        return None
    
    # Sort by directory name (which includes timestamp)
    latest_dir = max(exp_dirs, key=lambda x: x.name)
    return latest_dir


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Baseline vs Optimized performance")
    parser.add_argument("--db-path", type=Path, help="Path to performance DB")
    parser.add_argument("--output", type=Path, help="Output report path")
    parser.add_argument("--export-dir", type=Path, help="Directory to export detailed data")
    parser.add_argument("--experiment-dir", type=Path, help="Experiment directory to save results")
    
    args = parser.parse_args()
    
    # Handle experiment directory
    if args.experiment_dir:
        # Check if provided experiment directory exists
        if not args.experiment_dir.exists():
            logger.warning(f"Experiment directory does not exist: {args.experiment_dir}")
            # Try to find latest experiment directory
            latest_dir = find_latest_experiment_dir()
            if latest_dir:
                logger.info(f"Using latest experiment directory: {latest_dir}")
                args.experiment_dir = latest_dir
            else:
                logger.error("No experiment directories found")
                exit(1)
    else:
        # Auto-find latest experiment directory
        latest_dir = find_latest_experiment_dir()
        if latest_dir:
            logger.info(f"Auto-detected experiment directory: {latest_dir}")
            args.experiment_dir = latest_dir
    
    # Determine output paths
    if args.experiment_dir:
        # Use experiment directory structure
        if not args.output:
            args.output = args.experiment_dir / "analysis_reports" / "comparison_report.txt"
        if not args.export_dir:
            args.export_dir = args.experiment_dir / "analysis_reports" / "detailed_data"
    else:
        # Use default paths with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not args.output:
            project_root = Path(__file__).parent.parent.parent
            args.output = project_root / "experiments_results" / f"comparison_report_{timestamp}.txt"
        if not args.export_dir:
            project_root = Path(__file__).parent.parent.parent
            args.export_dir = project_root / "experiments_results" / f"detailed_data_{timestamp}"
    
    # Run analysis
    analyzer = ComparisonAnalyzer(db_path=args.db_path)
    
    # Generate report
    report = analyzer.generate_report(output_path=args.output)
    print(report)
    
    # Export detailed data
    if args.export_dir:
        analyzer.export_detailed_data(args.export_dir)
    
    analyzer.close()