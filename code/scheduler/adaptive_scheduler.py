"""
Adaptive Scheduler for LLM Evaluation 
Machine Learning-based scheduler using Random Forest for optimal resource allocation
ENHANCED: Advanced confidence calculation, uncertainty quantification, and improved features
FIXED: DB connection compatibility with new PerformanceTracker
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import joblib

# Import existing components
try:
    from .performance_tracker import PerformanceTracker
    from .resource_monitor import ResourceMonitor
    from .intelligent_scheduler import TaskPriority
except ImportError:
    from performance_tracker import PerformanceTracker
    from resource_monitor import ResourceMonitor
    from intelligent_scheduler import TaskPriority

# Logging setup
logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    """Machine Learning prediction result"""
    execution_time: float
    memory_usage: float
    oom_probability: float
    success_probability: float
    suggested_batch_size: int
    suggested_num_fewshot: int
    confidence_score: float
    feature_importance: Dict[str, float]


class AdaptiveScheduler:
    """ML-based adaptive scheduler using Random Forest with improved features"""
    
    def __init__(self, 
                 performance_tracker: PerformanceTracker,
                 resource_monitor: ResourceMonitor,
                 num_gpus: int = 1,
                 model_save_path: Optional[Path] = None):
        """
        Args:
            performance_tracker: Performance tracker instance
            resource_monitor: Resource monitor instance
            num_gpus: Number of available GPUs
            model_save_path: Path to save/load trained models
        """
        self.performance_tracker = performance_tracker
        self.resource_monitor = resource_monitor
        self.num_gpus = num_gpus
        
        # Model save path
        if model_save_path is None:
            project_root = Path(__file__).parent.parent.parent
            model_save_path = project_root / "data" / "ml_models"
        self.model_save_path = model_save_path
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # ML Models
        self.time_predictor = None
        self.memory_predictor = None
        self.oom_classifier = None
        self.success_classifier = None
        self.batch_size_predictor = None
        self.fewshot_predictor = None
        
        # Feature encoding
        self.label_encoders = {}
        self.feature_names = []
        
        # Model metadata
        self.last_training_time = None
        self.training_data_size = 0
        self.model_performance = {}
        
        # Thresholds
        self.min_training_samples = 50
        self.retrain_threshold = 100  # Retrain every 100 new samples
        self.confidence_threshold = 0.7
        
        logger.info(f"AdaptiveScheduler initialized with {num_gpus} GPUs")
        
        # Try to load existing models
        self._load_models()
    
    def _extract_features(self, model_config: Dict, task_name: str, 
                         current_resources: Optional[Any] = None) -> Dict[str, Any]:
        """Extract features for ML prediction - IMPROVED VERSION"""
        features = {}
        
        # Model features
        model_id = model_config.get("id", "")
        model_name = model_config.get("name", model_id.split("/")[-1])
        
        # Extract model features
        features['model_size_numeric'] = self._extract_model_size_numeric(model_id)
        features['model_size_category'] = self._extract_model_size_category(model_id)
        features['model_type'] = self._extract_model_type(model_id)
        
        # NEW: Advanced model features
        features['quantization_type'] = self._extract_quantization_type(model_config)
        features['model_architecture'] = self._extract_model_architecture(model_id)
        
        # Task features
        features['task_name'] = task_name
        features['task_type'] = self._extract_task_type(task_name)
        features['task_complexity'] = self._extract_task_complexity(task_name)
        
        # NEW: Sequence length estimation (critical for memory usage)
        features['estimated_sequence_length'] = self._estimate_sequence_length(task_name)
        
        # Resource features (FILTERED - removed less relevant ones)
        if current_resources:
            features['gpu_memory_available'] = current_resources.gpu_memory_total - current_resources.gpu_memory_used
            features['gpu_utilization'] = current_resources.gpu_utilization
            features['gpu_memory_utilization'] = current_resources.gpu_memory_percent
        else:
            # Default values if no resource info
            features['gpu_memory_available'] = 40.0
            features['gpu_utilization'] = 0.0
            features['gpu_memory_utilization'] = 0.0
        
        # NEW: Concurrent execution features
        features['concurrent_tasks'] = self._get_concurrent_task_count()
        features['system_load_factor'] = self._calculate_system_load_factor()
        
        # Historical features (enhanced)
        historical = self._get_historical_features(model_id, task_name)
        features.update(historical)
        
        # NEW: Context-aware features
        features['memory_pressure'] = self._calculate_memory_pressure(current_resources)
        features['batch_size_hint'] = self._get_optimal_batch_size_hint(model_id, task_name)
        
        # System features
        features['num_gpus'] = self.num_gpus
        
        return features
    
    def _extract_quantization_type(self, model_config: Dict) -> str:
        """Extract quantization type from model configuration"""
        model_id = model_config.get("id", "").lower()
        model_name = model_config.get("name", "").lower()
        
        # Check for quantization indicators in model ID/name
        if any(q in model_id for q in ['4bit', '4-bit', 'bnb-4bit']):
            return '4bit'
        elif any(q in model_id for q in ['8bit', '8-bit', 'int8']):
            return '8bit'
        elif any(q in model_id for q in ['fp16', 'half']):
            return 'fp16'
        elif any(q in model_id for q in ['fp32', 'float32']):
            return 'fp32'
        else:
            # Default assumption based on model size
            model_size = self._extract_model_size_numeric(model_id)
            if model_size >= 30:  # Large models likely quantized
                return '4bit'
            elif model_size >= 10:
                return 'fp16'
            else:
                return 'fp16'  # Default
    
    def _extract_model_architecture(self, model_id: str) -> str:
        """Extract model architecture type"""
        model_id_lower = model_id.lower()
        
        # Mixture of Experts models
        if any(moe in model_id_lower for moe in ['moe', 'mixtral']):
            return 'moe'
        
        # Transformer variants
        if 'mamba' in model_id_lower:
            return 'mamba'
        elif 'rwkv' in model_id_lower:
            return 'rwkv'
        else:
            return 'transformer'  # Default
    
    def _estimate_sequence_length(self, task_name: str) -> int:
        """Estimate sequence length based on task characteristics"""
        task_name_lower = task_name.lower()
        
        # Task-specific sequence length patterns
        length_map = {
            # Long context tasks
            'mmlu_pro': 2048,
            'bbh': 1536,
            'agieval': 1536,
            
            # Medium context tasks
            'mmlu': 1024,
            'gsm8k': 1024,
            'arc_challenge': 1024,
            'kmmlu': 1024,
            
            # Short context tasks
            'hellaswag': 512,
            'winogrande': 512,
            'piqa': 512,
            'arc_easy': 512,
            
            # Code tasks (variable length)
            'humaneval': 1024,
            'mbpp': 1024,
            
            # Korean tasks
            'haerae': 1024,
            'kobest': 1024,
        }
        
        # Find matching task
        for task_pattern, length in length_map.items():
            if task_pattern in task_name_lower:
                return length
        
        # Default based on task complexity
        complexity = self._extract_task_complexity(task_name)
        if complexity >= 0.8:
            return 2048
        elif complexity >= 0.6:
            return 1024
        else:
            return 512
    
    def _get_concurrent_task_count(self) -> int:
        """Get number of currently running tasks - FIXED DB ACCESS"""
        try:
            with self.performance_tracker._db_lock:
                conn = self.performance_tracker._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT COUNT(*) as running_tasks
                    FROM execution_records 
                    WHERE status = 'running'
                """)
                
                result = cursor.fetchone()
                return result['running_tasks'] if result else 0
                
        except Exception as e:
            logger.debug(f"Error getting concurrent task count: {e}")
            return 0
    
    def _calculate_system_load_factor(self) -> float:
        """Calculate overall system load factor (0.0 to 1.0)"""
        try:
            current_resources = self.resource_monitor.get_current_snapshot()
            if not current_resources:
                return 0.0
            
            # Weighted combination of resource utilizations
            gpu_load = current_resources.gpu_utilization / 100.0
            memory_load = current_resources.gpu_memory_percent / 100.0
            
            # GPU utilization has higher weight than memory for load calculation
            system_load = (0.7 * gpu_load) + (0.3 * memory_load)
            
            return min(system_load, 1.0)
            
        except Exception as e:
            logger.debug(f"Error calculating system load factor: {e}")
            return 0.0
    
    def _calculate_memory_pressure(self, current_resources: Optional[Any]) -> float:
        """Calculate memory pressure indicator (0.0 to 1.0)"""
        if not current_resources:
            return 0.0
        
        try:
            memory_usage_percent = current_resources.gpu_memory_percent
            
            # Non-linear mapping: pressure increases rapidly above 70%
            if memory_usage_percent >= 90:
                pressure = 1.0
            elif memory_usage_percent >= 80:
                pressure = 0.7 + (memory_usage_percent - 80) * 0.03  # 0.7 to 1.0
            elif memory_usage_percent >= 70:
                pressure = 0.4 + (memory_usage_percent - 70) * 0.03  # 0.4 to 0.7
            else:
                pressure = memory_usage_percent / 175.0  # 0.0 to 0.4
            
            return min(pressure, 1.0)
            
        except Exception as e:
            logger.debug(f"Error calculating memory pressure: {e}")
            return 0.0
    
    def _get_optimal_batch_size_hint(self, model_id: str, task_name: str) -> int:
        """Get optimal batch size hint based on model and task characteristics"""
        try:
            # Get model size
            model_size = self._extract_model_size_numeric(model_id)
            
            # Get estimated sequence length
            seq_length = self._estimate_sequence_length(task_name)
            
            # Calculate hint based on memory requirements
            # Rough estimation: memory ∝ model_size * seq_length * batch_size
            
            if model_size >= 30:  # Large models
                if seq_length >= 2048:
                    return 1
                elif seq_length >= 1024:
                    return 2
                else:
                    return 4
            elif model_size >= 10:  # Medium models
                if seq_length >= 2048:
                    return 2
                elif seq_length >= 1024:
                    return 4
                else:
                    return 8
            else:  # Small models
                if seq_length >= 2048:
                    return 4
                elif seq_length >= 1024:
                    return 8
                else:
                    return 16
                    
        except Exception as e:
            logger.debug(f"Error calculating batch size hint: {e}")
            return 1
    
    def _extract_model_size_numeric(self, model_id: str) -> float:
        """Extract numeric model size in billions"""
        model_id_lower = model_id.lower()
        
        size_patterns = {
            '0.5b': 0.5, '0-5b': 0.5, '500m': 0.5,
            '1.5b': 1.5, '1-5b': 1.5,
            '2.1b': 2.1, '2-1b': 2.1, '2.4b': 2.4, '2-4b': 2.4,
            '3b': 3.0, '4b': 4.0, '7b': 7.0, '7.8b': 7.8, '7-8b': 7.8,
            '8b': 8.0, '12b': 12.0, '13b': 13.0, '21.4b': 21.4, '21-4b': 21.4,
            '30b': 30.0, '32b': 32.0, '70b': 70.0
        }
        
        for pattern, size in size_patterns.items():
            if pattern in model_id_lower:
                return size
        
        return 7.0  # Default assumption
    
    def _extract_model_size_category(self, model_id: str) -> str:
        """Extract model size category"""
        size = self._extract_model_size_numeric(model_id)
        
        if size <= 3:
            return 'small'
        elif size <= 8:
            return 'medium'
        elif size <= 15:
            return 'large'
        else:
            return 'xlarge'
    
    def _extract_model_type(self, model_id: str) -> str:
        """Extract model type/family"""
        model_id_lower = model_id.lower()
        
        if 'gemma' in model_id_lower:
            return 'gemma'
        elif 'llama' in model_id_lower:
            return 'llama'
        elif 'exaone' in model_id_lower:
            return 'exaone'
        elif 'mistral' in model_id_lower or 'mixtral' in model_id_lower:
            return 'mistral'
        elif 'phi' in model_id_lower:
            return 'phi'
        elif 'qwen' in model_id_lower:
            return 'qwen'
        else:
            return 'other'
    
    def _extract_task_type(self, task_name: str) -> str:
        """Extract task type category"""
        task_name_lower = task_name.lower()
        
        knowledge_tasks = ['mmlu', 'mmlu_pro', 'gpqa']
        reasoning_tasks = ['bbh', 'gsm8k', 'agieval', 'arc']
        coding_tasks = ['humaneval', 'mbpp', 'apps']
        language_tasks = ['hellaswag', 'winogrande', 'piqa']
        korean_tasks = ['haerae', 'kobest', 'klue', 'kmmlu']
        
        if any(t in task_name_lower for t in knowledge_tasks):
            return 'knowledge'
        elif any(t in task_name_lower for t in reasoning_tasks):
            return 'reasoning'
        elif any(t in task_name_lower for t in coding_tasks):
            return 'coding'
        elif any(t in task_name_lower for t in language_tasks):
            return 'language'
        elif any(t in task_name_lower for t in korean_tasks):
            return 'korean'
        else:
            return 'other'
    
    def _extract_task_complexity(self, task_name: str) -> float:
        """Extract task complexity score (0-1)"""
        task_name_lower = task_name.lower()
        
        complexity_map = {
            # High complexity
            'mmlu_pro': 0.9, 'bbh': 0.9, 'gpqa': 0.9, 'humaneval': 0.9,
            # Medium-high complexity
            'mmlu': 0.7, 'gsm8k': 0.7, 'agieval': 0.7, 'mbpp': 0.7,
            # Medium complexity
            'arc_challenge': 0.6, 'kmmlu': 0.6, 'haerae': 0.6,
            # Lower complexity
            'arc_easy': 0.4, 'hellaswag': 0.4, 'winogrande': 0.4, 'piqa': 0.4,
            # Simple tasks
            'openbookqa': 0.3, 'copa': 0.3
        }
        
        for task, complexity in complexity_map.items():
            if task in task_name_lower:
                return complexity
        
        return 0.5  # Default medium complexity
    
    def _get_historical_features(self, model_id: str, task_name: str) -> Dict[str, float]:
        """Get historical performance features"""
        features = {}
        
        # Get past performance for this model-task combination
        prediction = self.performance_tracker.predict_execution(model_id, task_name)
        
        features['historical_avg_time'] = prediction.get('predicted_time', 3600.0)
        features['historical_avg_memory'] = prediction.get('predicted_memory', 20.0)
        features['historical_success_rate'] = prediction.get('success_rate', 0.8)
        features['historical_oom_rate'] = prediction.get('oom_rate', 0.1)
        features['historical_sample_count'] = prediction.get('num_samples', 0)
        
        # Get model-level statistics
        try:
            with self.performance_tracker._db_lock:
                conn = self.performance_tracker._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        AVG(execution_time) as avg_time,
                        AVG(gpu_memory_peak) as avg_memory,
                        COUNT(*) as total_runs
                    FROM execution_records 
                    WHERE model_id = ? AND status = 'completed'
                """, (model_id,))
                
                result = cursor.fetchone()
                if result and result['total_runs'] > 0:
                    features['model_avg_time'] = result['avg_time'] or 3600.0
                    features['model_avg_memory'] = result['avg_memory'] or 20.0
                    features['model_run_count'] = result['total_runs']
                else:
                    features['model_avg_time'] = 3600.0
                    features['model_avg_memory'] = 20.0
                    features['model_run_count'] = 0
                    
        except Exception as e:
            logger.warning(f"Error getting historical features: {e}")
            features['model_avg_time'] = 3600.0
            features['model_avg_memory'] = 20.0
            features['model_run_count'] = 0
        
        return features
    
    def _prepare_training_data(self) -> Optional[pd.DataFrame]:
        """Prepare training data from performance tracker - FIXED DB ACCESS"""
        try:
            with self.performance_tracker._db_lock:
                conn = self.performance_tracker._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM execution_records 
                    WHERE status IN ('completed', 'failed', 'oom')
                    ORDER BY timestamp DESC
                """)
                
                records = cursor.fetchall()
                
                if len(records) < self.min_training_samples:
                    logger.info(f"Insufficient training data: {len(records)} < {self.min_training_samples}")
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame([dict(record) for record in records])
                
                # Add features for each record
                feature_data = []
                for _, row in df.iterrows():
                    # Create mock model config
                    model_config = {
                        "id": row['model_id'],
                        "name": row['model_name']
                    }
                    
                    features = self._extract_features(model_config, row['task_name'])
                    
                    # Add target variables
                    features['target_execution_time'] = row['execution_time'] if row['execution_time'] else 3600.0
                    features['target_memory_usage'] = row['gpu_memory_peak'] if row['gpu_memory_peak'] else 20.0
                    features['target_oom'] = 1 if row['status'] == 'oom' else 0
                    features['target_success'] = 1 if row['status'] == 'completed' else 0
                    features['target_batch_size'] = row['batch_size'] if row['batch_size'] else 1
                    features['target_num_fewshot'] = row['num_fewshot'] if row['num_fewshot'] else 0
                    
                    feature_data.append(features)
                
                training_df = pd.DataFrame(feature_data)
                logger.info(f"Prepared training data with {len(training_df)} samples")
                
                return training_df
                
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None
    
    def _encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features with backward compatibility"""
        categorical_features = [
            'model_size_category', 'model_type', 'task_name', 'task_type',
            'quantization_type', 'model_architecture'  # NEW features
        ]
        
        df_encoded = df.copy()
        
        for feature in categorical_features:
            if feature in df_encoded.columns:
                if fit:
                    if feature not in self.label_encoders:
                        self.label_encoders[feature] = LabelEncoder()
                    
                    # Handle unseen categories
                    df_encoded[feature] = df_encoded[feature].astype(str)
                    self.label_encoders[feature].fit(df_encoded[feature])
                    df_encoded[feature] = self.label_encoders[feature].transform(df_encoded[feature])
                else:
                    # Transform with existing encoder
                    df_encoded[feature] = df_encoded[feature].astype(str)
                    # Handle unseen categories
                    seen_categories = set(self.label_encoders[feature].classes_)
                    df_encoded[feature] = df_encoded[feature].apply(
                        lambda x: x if x in seen_categories else 'unknown'
                    )
                    
                    # Add 'unknown' class if not exists
                    if 'unknown' not in seen_categories:
                        new_classes = np.append(self.label_encoders[feature].classes_, 'unknown')
                        self.label_encoders[feature].classes_ = new_classes
                    
                    df_encoded[feature] = self.label_encoders[feature].transform(df_encoded[feature])
            else:
                # Handle missing features for backward compatibility
                if not fit and feature not in df_encoded.columns:
                    logger.debug(f"Missing feature {feature}, using default value")
                    df_encoded[feature] = 0  # Default encoded value
        
        return df_encoded
    
    def train_models(self) -> bool:
        """Train all ML models"""
        logger.info("Starting ML model training with improved features...")
        
        # Prepare training data
        training_df = self._prepare_training_data()
        if training_df is None:
            return False
        
        # Encode categorical features
        training_df = self._encode_categorical_features(training_df, fit=True)
        
        # Define feature columns (exclude target columns)
        target_columns = ['target_execution_time', 'target_memory_usage', 'target_oom', 
                         'target_success', 'target_batch_size', 'target_num_fewshot']
        feature_columns = [col for col in training_df.columns if col not in target_columns]
        self.feature_names = feature_columns
        
        X = training_df[feature_columns]
        
        # Train individual models
        success_count = 0
        
        # 1. Execution time predictor
        if self._train_time_predictor(X, training_df['target_execution_time']):
            success_count += 1
        
        # 2. Memory usage predictor
        if self._train_memory_predictor(X, training_df['target_memory_usage']):
            success_count += 1
        
        # 3. OOM classifier
        if self._train_oom_classifier(X, training_df['target_oom']):
            success_count += 1
        
        # 4. Success classifier
        if self._train_success_classifier(X, training_df['target_success']):
            success_count += 1
        
        # 5. Batch size predictor
        if self._train_batch_size_predictor(X, training_df['target_batch_size']):
            success_count += 1
        
        # 6. Few-shot predictor
        if self._train_fewshot_predictor(X, training_df['target_num_fewshot']):
            success_count += 1
        
        # Update metadata
        self.last_training_time = datetime.now()
        self.training_data_size = len(training_df)
        
        # Save models
        self._save_models()
        
        logger.info(f"Model training completed with improved features: {success_count}/6 models trained successfully")
        return success_count >= 4  # At least 4 models should be trained
    
    def _train_time_predictor(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Train execution time predictor"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.time_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                oob_score=True  # Enable OOB scoring for confidence calculation
            )
            
            self.time_predictor.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.time_predictor.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            self.model_performance['time_predictor'] = {
                'mae': mae,
                'relative_error': mae / y_test.mean(),
                'oob_score': getattr(self.time_predictor, 'oob_score_', None)
            }
            
            logger.info(f"Time predictor trained - MAE: {mae:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error training time predictor: {e}")
            return False
    
    def _train_memory_predictor(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Train memory usage predictor"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.memory_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                oob_score=True
            )
            
            self.memory_predictor.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.memory_predictor.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            self.model_performance['memory_predictor'] = {
                'mae': mae,
                'relative_error': mae / y_test.mean(),
                'oob_score': getattr(self.memory_predictor, 'oob_score_', None)
            }
            
            logger.info(f"Memory predictor trained - MAE: {mae:.2f}GB")
            return True
            
        except Exception as e:
            logger.error(f"Error training memory predictor: {e}")
            return False
    
    def _train_oom_classifier(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Train OOM classifier"""
        try:
            if y.sum() < 5:  # Not enough positive samples
                logger.warning("Insufficient OOM samples for training classifier")
                return False
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.oom_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                oob_score=True
            )
            
            self.oom_classifier.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.oom_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.model_performance['oom_classifier'] = {
                'accuracy': accuracy,
                'oob_score': getattr(self.oom_classifier, 'oob_score_', None)
            }
            
            logger.info(f"OOM classifier trained - Accuracy: {accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training OOM classifier: {e}")
            return False
    
    def _train_success_classifier(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Train success classifier"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.success_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                oob_score=True
            )
            
            self.success_classifier.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.success_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.model_performance['success_classifier'] = {
                'accuracy': accuracy,
                'oob_score': getattr(self.success_classifier, 'oob_score_', None)
            }
            
            logger.info(f"Success classifier trained - Accuracy: {accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training success classifier: {e}")
            return False
    
    def _train_batch_size_predictor(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Train batch size predictor"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.batch_size_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
                oob_score=True
            )
            
            self.batch_size_predictor.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.batch_size_predictor.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            self.model_performance['batch_size_predictor'] = {
                'mae': mae,
                'oob_score': getattr(self.batch_size_predictor, 'oob_score_', None)
            }
            
            logger.info(f"Batch size predictor trained - MAE: {mae:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training batch size predictor: {e}")
            return False
    
    def _train_fewshot_predictor(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Train few-shot predictor"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.fewshot_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
                oob_score=True
            )
            
            self.fewshot_predictor.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.fewshot_predictor.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            self.model_performance['fewshot_predictor'] = {
                'mae': mae,
                'oob_score': getattr(self.fewshot_predictor, 'oob_score_', None)
            }
            
            logger.info(f"Few-shot predictor trained - MAE: {mae:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training few-shot predictor: {e}")
            return False
    
    def predict(self, model_config: Dict, task_name: str) -> MLPrediction:
        """Make ML prediction for model-task combination"""
        try:
            # Extract features
            current_resources = self.resource_monitor.get_current_snapshot()
            features = self._extract_features(model_config, task_name, current_resources)
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Encode categorical features
            feature_df = self._encode_categorical_features(feature_df, fit=False)
            
            # Select feature columns
            X = feature_df[self.feature_names]
            
            # Make predictions
            predictions = {}
            confidence_scores = {}
            
            # Execution time
            if self.time_predictor:
                pred_time = self.time_predictor.predict(X)[0]
                predictions['execution_time'] = max(pred_time, 60.0)  # Minimum 1 minute
                
                # Calculate confidence based on feature importance
                feature_importance = dict(zip(self.feature_names, self.time_predictor.feature_importances_))
                confidence_scores['time'] = self._calculate_prediction_confidence(X, feature_importance)
            else:
                predictions['execution_time'] = 3600.0  # Default 1 hour
                confidence_scores['time'] = 0.0
            
            # Memory usage
            if self.memory_predictor:
                pred_memory = self.memory_predictor.predict(X)[0]
                predictions['memory_usage'] = max(pred_memory, 1.0)  # Minimum 1GB
                
                feature_importance = dict(zip(self.feature_names, self.memory_predictor.feature_importances_))
                confidence_scores['memory'] = self._calculate_prediction_confidence(X, feature_importance)
            else:
                predictions['memory_usage'] = 20.0  # Default 20GB
                confidence_scores['memory'] = 0.0
            
            # OOM probability
            if self.oom_classifier:
                oom_proba = self.oom_classifier.predict_proba(X)[0]
                predictions['oom_probability'] = oom_proba[1] if len(oom_proba) > 1 else 0.1
                confidence_scores['oom'] = max(oom_proba)
            else:
                predictions['oom_probability'] = 0.1  # Default 10%
                confidence_scores['oom'] = 0.0
            
            # Success probability
            if self.success_classifier:
                success_proba = self.success_classifier.predict_proba(X)[0]
                predictions['success_probability'] = success_proba[1] if len(success_proba) > 1 else 0.8
                confidence_scores['success'] = max(success_proba)
            else:
                predictions['success_probability'] = 0.8  # Default 80%
                confidence_scores['success'] = 0.0
            
            # Batch size
            if self.batch_size_predictor:
                pred_batch = self.batch_size_predictor.predict(X)[0]
                predictions['suggested_batch_size'] = max(1, min(int(round(pred_batch)), 16))
            else:
                predictions['suggested_batch_size'] = 1
            
            # Few-shot
            if self.fewshot_predictor:
                pred_fewshot = self.fewshot_predictor.predict(X)[0]
                predictions['suggested_num_fewshot'] = max(0, min(int(round(pred_fewshot)), 10))
            else:
                predictions['suggested_num_fewshot'] = 5
            
            # Overall confidence using enhanced calculation
            overall_confidence = self._calculate_prediction_confidence(X, 
                dict(zip(self.feature_names, self.time_predictor.feature_importances_)) if self.time_predictor else {})
            
            # Feature importance (from time predictor as main model)
            if self.time_predictor:
                feature_importance = dict(zip(self.feature_names, self.time_predictor.feature_importances_))
            else:
                feature_importance = {}
            
            return MLPrediction(
                execution_time=predictions['execution_time'],
                memory_usage=predictions['memory_usage'],
                oom_probability=predictions['oom_probability'],
                success_probability=predictions['success_probability'],
                suggested_batch_size=predictions['suggested_batch_size'],
                suggested_num_fewshot=predictions['suggested_num_fewshot'],
                confidence_score=overall_confidence,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            logger.error(f"Error making ML prediction: {e}")
            # Return fallback prediction
            return MLPrediction(
                execution_time=3600.0,
                memory_usage=20.0,
                oom_probability=0.1,
                success_probability=0.8,
                suggested_batch_size=1,
                suggested_num_fewshot=5,
                confidence_score=0.0,
                feature_importance={}
            )
    
    def _calculate_prediction_confidence(self, X: pd.DataFrame, feature_importance: Dict[str, float]) -> float:
        """Enhanced prediction confidence calculation using multiple uncertainty metrics"""
        try:
            confidence_scores = []
            
            # 1. Feature importance entropy
            if feature_importance:
                importance_values = np.array(list(feature_importance.values()))
                importance_values = importance_values / importance_values.sum()
                
                entropy = -np.sum(importance_values * np.log(importance_values + 1e-10))
                max_entropy = np.log(len(importance_values))
                entropy_confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
                confidence_scores.append(entropy_confidence)
            
            # 2. Ensemble agreement confidence
            if hasattr(self, 'time_predictor') and self.time_predictor:
                ensemble_confidence = self._calculate_ensemble_confidence(X)
                if ensemble_confidence is not None:
                    confidence_scores.append(ensemble_confidence)
            
            # 3. Prediction stability confidence
            stability_confidence = self._calculate_prediction_stability(X)
            if stability_confidence is not None:
                confidence_scores.append(stability_confidence)
            
            # 4. Historical accuracy confidence
            historical_confidence = self._get_historical_accuracy_confidence()
            if historical_confidence is not None:
                confidence_scores.append(historical_confidence)
            
            # 5. Out-of-bag confidence
            oob_confidence = self._calculate_oob_confidence()
            if oob_confidence is not None:
                confidence_scores.append(oob_confidence)
            
            # Combine all confidence scores with weights
            if confidence_scores:
                weights = [0.25, 0.25, 0.2, 0.15, 0.15][:len(confidence_scores)]
                weights = np.array(weights) / sum(weights)
                
                final_confidence = np.average(confidence_scores, weights=weights)
                return min(max(final_confidence, 0.0), 1.0)
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"Error calculating prediction confidence: {e}")
            return 0.5

    def _calculate_ensemble_confidence(self, X: pd.DataFrame) -> Optional[float]:
        """Calculate confidence based on Random Forest ensemble agreement"""
        try:
            if not hasattr(self.time_predictor, 'estimators_'):
                return None
            
            tree_predictions = []
            for estimator in self.time_predictor.estimators_:
                pred = estimator.predict(X)
                tree_predictions.append(pred[0] if len(pred) > 0 else 0)
            
            if len(tree_predictions) < 2:
                return None
            
            pred_mean = np.mean(tree_predictions)
            pred_std = np.std(tree_predictions)
            
            if pred_mean > 0:
                coefficient_of_variation = pred_std / pred_mean
                confidence = 1 / (1 + np.exp(coefficient_of_variation - 0.5))
            else:
                confidence = 0.5
            
            return confidence
            
        except Exception as e:
            logger.debug(f"Error calculating ensemble confidence: {e}")
            return None

    def _calculate_prediction_stability(self, X: pd.DataFrame) -> Optional[float]:
        """Calculate confidence based on prediction stability with small input perturbations"""
        try:
            if not self.time_predictor:
                return None
            
            original_pred = self.time_predictor.predict(X)[0]
            
            perturbed_predictions = []
            num_perturbations = 5
            
            for _ in range(num_perturbations):
                X_perturbed = X.copy()
                
                numerical_cols = X.select_dtypes(include=[np.number]).columns
                
                for col in numerical_cols:
                    if X[col].std() > 0:
                        noise_scale = X[col].std() * 0.01
                        noise = np.random.normal(0, noise_scale, size=len(X))
                        X_perturbed[col] = X[col] + noise
                
                perturbed_pred = self.time_predictor.predict(X_perturbed)[0]
                perturbed_predictions.append(perturbed_pred)
            
            if len(perturbed_predictions) > 1:
                pred_std = np.std(perturbed_predictions)
                if original_pred > 0:
                    stability = 1 / (1 + (pred_std / original_pred))
                else:
                    stability = 0.5
            else:
                stability = 0.5
            
            return stability
            
        except Exception as e:
            logger.debug(f"Error calculating prediction stability: {e}")
            return None

    def _get_historical_accuracy_confidence(self) -> Optional[float]:
        """Calculate confidence based on historical prediction accuracy"""
        try:
            if not hasattr(self, 'model_performance') or not self.model_performance:
                return None
            
            accuracy_scores = []
            
            for model_name, performance in self.model_performance.items():
                if 'accuracy' in performance:
                    accuracy_scores.append(performance['accuracy'])
                elif 'relative_error' in performance:
                    accuracy = max(0, 1 - performance['relative_error'])
                    accuracy_scores.append(accuracy)
            
            if accuracy_scores:
                mean_accuracy = np.mean(accuracy_scores)
                confidence = 1 / (1 + np.exp(-(mean_accuracy - 0.7) * 10))
                return confidence
            
            return None
            
        except Exception as e:
            logger.debug(f"Error calculating historical accuracy confidence: {e}")
            return None

    def _calculate_oob_confidence(self) -> Optional[float]:
        """Calculate confidence based on Out-of-Bag error"""
        try:
            if not hasattr(self.time_predictor, 'oob_score_'):
                return None
            
            oob_score = self.time_predictor.oob_score_
            
            if oob_score is not None:
                confidence = 1 / (1 + np.exp(-(oob_score - 0.5) * 5))
                return confidence
            
            return None
            
        except Exception as e:
            logger.debug(f"Error calculating OOB confidence: {e}")
            return None

    def get_enhanced_model_confidence(self) -> Dict[str, float]:
        """Enhanced version of get_model_confidence with multiple confidence metrics"""
        if not self.is_trained():
            return {
                'overall_confidence': 0.0,
                'training_data_confidence': 0.0,
                'model_performance_confidence': 0.0,
                'prediction_stability_confidence': 0.0,
                'ensemble_agreement_confidence': 0.0,
                'temporal_stability_confidence': 0.0
            }
        
        confidence_metrics = {}
        
        # 1. Training data size confidence
        data_confidence = min(self.training_data_size / 200.0, 1.0)
        confidence_metrics['training_data_confidence'] = data_confidence
        
        # 2. Model performance confidence
        if self.model_performance:
            performance_scores = []
            for model_name, perf in self.model_performance.items():
                if 'accuracy' in perf:
                    performance_scores.append(perf['accuracy'])
                elif 'relative_error' in perf:
                    accuracy = max(0, 1 - perf['relative_error'])
                    performance_scores.append(accuracy)
            
            if performance_scores:
                avg_performance = np.mean(performance_scores)
                perf_confidence = 1 / (1 + np.exp(-(avg_performance - 0.7) * 8))
            else:
                perf_confidence = 0.5
        else:
            perf_confidence = 0.0
        
        confidence_metrics['model_performance_confidence'] = perf_confidence
        
        # 3. Prediction stability confidence
        try:
            sample_features = pd.DataFrame({
                'model_size_numeric': [7.0],
                'model_size_category': [1],
                'model_type': [1],
                'task_name': [1],
                'task_type': [1],
                'task_complexity': [0.5],
                'quantization_type': [1],
                'model_architecture': [1],
                'estimated_sequence_length': [1024],
                'gpu_memory_available': [30.0],
                'gpu_utilization': [20.0],
                'gpu_memory_utilization': [50.0],
                'concurrent_tasks': [0],
                'system_load_factor': [0.3],
                'historical_avg_time': [3600.0],
                'historical_avg_memory': [15.0],
                'historical_success_rate': [0.8],
                'historical_oom_rate': [0.1],
                'historical_sample_count': [10],
                'model_avg_time': [3600.0],
                'model_avg_memory': [15.0],
                'model_run_count': [5],
                'memory_pressure': [0.3],
                'batch_size_hint': [4],
                'num_gpus': [1]
            })
            
            if hasattr(self, 'feature_names') and self.feature_names:
                for feature in self.feature_names:
                    if feature not in sample_features.columns:
                        sample_features[feature] = 0
                
                sample_features = sample_features[self.feature_names]
            
            stability_confidence = self._calculate_prediction_stability(sample_features)
            if stability_confidence is None:
                stability_confidence = 0.5
                
        except Exception as e:
            logger.debug(f"Error calculating stability confidence: {e}")
            stability_confidence = 0.5
        
        confidence_metrics['prediction_stability_confidence'] = stability_confidence
        
        # 4. Ensemble agreement confidence
        try:
            ensemble_confidence = self._calculate_ensemble_confidence(sample_features)
            if ensemble_confidence is None:
                ensemble_confidence = 0.5
        except Exception as e:
            logger.debug(f"Error calculating ensemble confidence: {e}")
            ensemble_confidence = 0.5
        
        confidence_metrics['ensemble_agreement_confidence'] = ensemble_confidence
        
        # 5. Temporal stability confidence
        if self.last_training_time:
            time_since_training = datetime.now() - self.last_training_time
            hours_since_training = time_since_training.total_seconds() / 3600
            
            if hours_since_training <= 24:
                temporal_confidence = 1.0
            else:
                temporal_confidence = np.exp(-0.693 * (hours_since_training - 24) / 72)
        else:
            temporal_confidence = 0.0
        
        confidence_metrics['temporal_stability_confidence'] = temporal_confidence
        
        # 6. Overall confidence
        weights = {
            'training_data_confidence': 0.3,
            'model_performance_confidence': 0.25,
            'prediction_stability_confidence': 0.2,
            'ensemble_agreement_confidence': 0.15,
            'temporal_stability_confidence': 0.1
        }
        
        overall_confidence = sum(
            confidence_metrics[metric] * weight 
            for metric, weight in weights.items()
        )
        
        confidence_metrics['overall_confidence'] = overall_confidence
        
        return confidence_metrics

    def get_model_confidence(self) -> float:
        """Get overall model confidence"""
        enhanced_confidence = self.get_enhanced_model_confidence()
        return enhanced_confidence.get('overall_confidence', 0.0)
    
    def create_optimal_schedule(self, models: List[Dict], tasks: List[str]) -> List[TaskPriority]:
        """Create optimal schedule using ML predictions"""
        schedule = []
        
        # Generate ML predictions for all model-task combinations
        for model in models:
            for task in tasks:
                ml_pred = self.predict(model, task)
                
                # Create TaskPriority with ML predictions
                priority = self._create_task_priority_from_ml(model, task, ml_pred)
                schedule.append(priority)
        
        # Sort by priority score
        schedule.sort(key=lambda x: x.priority_score, reverse=True)
        
        # Optimize GPU assignment
        self._optimize_gpu_assignment(schedule)
        
        logger.info(f"Created ML-based optimal schedule with {len(schedule)} tasks")
        return schedule
    
    def _create_task_priority_from_ml(self, model: Dict, task: str, ml_pred: MLPrediction) -> TaskPriority:
        """Create TaskPriority object from ML prediction"""
        model_id = model.get("id", "")
        model_name = model.get("name", model_id.split("/")[-1])
        
        # Calculate priority score using ML predictions
        priority_score = self._compute_ml_priority_score(ml_pred)
        
        # Generate rationale
        rationale = self._generate_ml_rationale(ml_pred, model_name, task)
        
        return TaskPriority(
            model_id=model_id,
            task_name=task,
            priority_score=priority_score,
            estimated_time=ml_pred.execution_time,
            estimated_memory=ml_pred.memory_usage,
            success_probability=ml_pred.success_probability,
            suggested_gpu=0,  # Will be optimized later
            suggested_batch_size=ml_pred.suggested_batch_size,
            suggested_num_fewshot=ml_pred.suggested_num_fewshot,
            rationale=rationale
        )
    
    def _compute_ml_priority_score(self, ml_pred: MLPrediction) -> float:
        """Compute priority score from ML prediction"""
        # Base score from success probability
        score = ml_pred.success_probability * 100
        
        # Time efficiency
        time_efficiency = 1 / (1 + np.log1p(ml_pred.execution_time / 3600))
        score *= time_efficiency
        
        # Memory efficiency
        memory_efficiency = 1 / (1 + np.log1p(ml_pred.memory_usage / 40))
        score *= memory_efficiency
        
        # OOM risk penalty
        score *= (1 - ml_pred.oom_probability)
        
        # Confidence bonus
        confidence_bonus = 1 + (ml_pred.confidence_score * 0.2)
        score *= confidence_bonus
        
        return score
    
    def _generate_ml_rationale(self, ml_pred: MLPrediction, model_name: str, task: str) -> str:
        """Generate rationale for ML-based scheduling decision"""
        rationale_parts = []
        
        hours = ml_pred.execution_time / 3600
        rationale_parts.append(f"ML Est. time: {hours:.1f}h")
        rationale_parts.append(f"ML Est. memory: {ml_pred.memory_usage:.1f}GB")
        
        rationale_parts.append(f"Success: {ml_pred.success_probability*100:.1f}%")
        
        if ml_pred.oom_probability > 0.5:
            rationale_parts.append(f"High OOM risk ({ml_pred.oom_probability*100:.0f}%)")
        elif ml_pred.oom_probability > 0.2:
            rationale_parts.append(f"Medium OOM risk ({ml_pred.oom_probability*100:.0f}%)")
        
        rationale_parts.append(f"Confidence: {ml_pred.confidence_score*100:.0f}%")
        
        return " | ".join(rationale_parts)
    
    def _optimize_gpu_assignment(self, schedule: List[TaskPriority]):
        """Optimize GPU assignment"""
        if self.num_gpus == 1:
            for task in schedule:
                task.suggested_gpu = 0
            return
        
        gpu_loads = [0.0] * self.num_gpus
        
        for task in schedule:
            min_load_gpu = np.argmin(gpu_loads)
            task.suggested_gpu = min_load_gpu
            gpu_loads[min_load_gpu] += task.estimated_time
        
        logger.info(f"ML-based GPU load distribution: {gpu_loads}")
    
    def should_retrain(self) -> bool:
        """Check if models should be retrained - FIXED: DB 연결 오류 수정"""
        try:
            cursor = self.performance_tracker.conn.cursor()
            cursor.execute("PRAGMA busy_timeout = 5000")  # 5초 타임아웃
            cursor.execute("""
                SELECT COUNT(*) as total_records
                FROM execution_records 
                WHERE status IN ('completed', 'failed', 'oom')
                AND mode = ?
            """, (self.performance_tracker.mode,))
            
            result = cursor.fetchone()
            current_data_size = result['total_records'] if result else 0
            
            # Check if we have enough new data
            new_data = current_data_size - self.training_data_size
            
            return (new_data >= self.retrain_threshold or 
                    current_data_size >= self.min_training_samples and self.last_training_time is None)
                    
        except Exception as e:
            logger.error(f"Error checking retrain status: {e}")
            return False
    
    def is_trained(self) -> bool:
        """Check if models are trained and ready"""
        required_models = [self.time_predictor, self.memory_predictor, 
                          self.success_classifier]
        return all(model is not None for model in required_models)
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            models_to_save = {
                'time_predictor': self.time_predictor,
                'memory_predictor': self.memory_predictor,
                'oom_classifier': self.oom_classifier,
                'success_classifier': self.success_classifier,
                'batch_size_predictor': self.batch_size_predictor,
                'fewshot_predictor': self.fewshot_predictor
            }
            
            for name, model in models_to_save.items():
                if model is not None:
                    model_path = self.model_save_path / f"{name}_{timestamp}.pkl"
                    joblib.dump(model, model_path)
            
            # Save metadata
            metadata = {
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'training_data_size': self.training_data_size,
                'model_performance': self.model_performance,
                'feature_names': self.feature_names,
                'label_encoders': {k: {'classes_': v.classes_.tolist()} for k, v in self.label_encoders.items()}
            }
            
            metadata_path = self.model_save_path / f"metadata_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Models saved to {self.model_save_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self):
        """Load most recent trained models from disk"""
        try:
            model_files = list(self.model_save_path.glob("time_predictor_*.pkl"))
            if not model_files:
                logger.info("No pre-trained models found")
                return
            
            latest_timestamp = max(f.stem.split('_')[-1] for f in model_files)
            
            model_names = ['time_predictor', 'memory_predictor', 'oom_classifier', 
                          'success_classifier', 'batch_size_predictor', 'fewshot_predictor']
            
            for name in model_names:
                model_path = self.model_save_path / f"{name}_{latest_timestamp}.pkl"
                if model_path.exists():
                    setattr(self, name, joblib.load(model_path))
                    logger.debug(f"Loaded {name}")
            
            metadata_path = self.model_save_path / f"metadata_{latest_timestamp}.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.last_training_time = datetime.fromisoformat(metadata['last_training_time']) if metadata['last_training_time'] else None
                self.training_data_size = metadata.get('training_data_size', 0)
                self.model_performance = metadata.get('model_performance', {})
                self.feature_names = metadata.get('feature_names', [])
                
                # Restore label encoders
                for name, encoder_data in metadata.get('label_encoders', {}).items():
                    encoder = LabelEncoder()
                    encoder.classes_ = np.array(encoder_data['classes_'])
                    self.label_encoders[name] = encoder
            
            logger.info(f"Pre-trained models loaded from {latest_timestamp}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def get_feature_importance_analysis(self) -> Dict[str, Any]:
        """Get detailed feature importance analysis"""
        if not self.is_trained():
            return {}
        
        analysis = {}
        
        if self.time_predictor:
            time_importance = dict(zip(self.feature_names, self.time_predictor.feature_importances_))
            time_importance_sorted = sorted(time_importance.items(), key=lambda x: x[1], reverse=True)
            analysis['time_prediction'] = {
                'top_features': time_importance_sorted[:10],
                'feature_importance': time_importance
            }
        
        if self.memory_predictor:
            memory_importance = dict(zip(self.feature_names, self.memory_predictor.feature_importances_))
            memory_importance_sorted = sorted(memory_importance.items(), key=lambda x: x[1], reverse=True)
            analysis['memory_prediction'] = {
                'top_features': memory_importance_sorted[:10],
                'feature_importance': memory_importance
            }
        
        if self.time_predictor and self.memory_predictor:
            overall_importance = {}
            for feature in self.feature_names:
                time_imp = time_importance.get(feature, 0)
                memory_imp = memory_importance.get(feature, 0)
                overall_importance[feature] = (time_imp + memory_imp) / 2
            
            overall_sorted = sorted(overall_importance.items(), key=lambda x: x[1], reverse=True)
            analysis['overall'] = {
                'top_features': overall_sorted[:15],
                'feature_importance': overall_importance
            }
        
        return analysis
    
    def export_model_analysis(self, filepath: Path):
        """Export comprehensive model analysis"""
        analysis = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'is_trained': self.is_trained(),
                'training_data_size': self.training_data_size,
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'num_features': len(self.feature_names),
                'feature_improvements': {
                    'removed_features': ['gpu_temperature', 'hour', 'day_of_week', 'cpu_percent', 'ram_available'],
                    'added_features': ['quantization_type', 'model_architecture', 'estimated_sequence_length', 
                                     'concurrent_tasks', 'system_load_factor', 'memory_pressure', 'batch_size_hint'],
                    'enhancement_focus': 'Memory usage prediction and context awareness'
                }
            },
            'model_performance': self.model_performance,
            'confidence_metrics': self.get_enhanced_model_confidence(),
            'feature_analysis': self.get_feature_importance_analysis(),
            'feature_names': self.feature_names
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Model analysis exported to {filepath}")
    
    def get_prediction_summary(self, models: List[Dict], tasks: List[str]) -> Dict[str, Any]:
        """Get prediction summary for all model-task combinations"""
        if not self.is_trained():
            return {'error': 'Models not trained'}
        
        predictions = []
        
        for model in models:
            for task in tasks:
                try:
                    ml_pred = self.predict(model, task)
                    
                    predictions.append({
                        'model_id': model.get('id', ''),
                        'model_name': model.get('name', ''),
                        'task_name': task,
                        'execution_time_hours': ml_pred.execution_time / 3600,
                        'memory_usage_gb': ml_pred.memory_usage,
                        'success_probability': ml_pred.success_probability,
                        'oom_probability': ml_pred.oom_probability,
                        'confidence_score': ml_pred.confidence_score,
                        'suggested_batch_size': ml_pred.suggested_batch_size,
                        'suggested_num_fewshot': ml_pred.suggested_num_fewshot
                    })
                    
                except Exception as e:
                    logger.error(f"Error predicting for {model.get('id', '')} on {task}: {e}")
        
        # Calculate summary statistics
        if predictions:
            summary = {
                'total_predictions': len(predictions),
                'avg_execution_time_hours': np.mean([p['execution_time_hours'] for p in predictions]),
                'avg_memory_usage_gb': np.mean([p['memory_usage_gb'] for p in predictions]),
                'avg_success_probability': np.mean([p['success_probability'] for p in predictions]),
                'avg_confidence_score': np.mean([p['confidence_score'] for p in predictions]),
                'high_confidence_predictions': len([p for p in predictions if p['confidence_score'] > 0.7]),
                'high_risk_predictions': len([p for p in predictions if p['oom_probability'] > 0.3]),
                'predictions': predictions
            }
        else:
            summary = {'total_predictions': 0, 'predictions': []}
        
        return summary