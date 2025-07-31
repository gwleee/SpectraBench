"""
Adaptive Scheduler for LLM Evaluation
Machine Learning-based scheduler using Random Forest for optimal resource allocation
ENHANCED: Advanced confidence calculation and uncertainty quantification
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
    """ML-based adaptive scheduler using Random Forest"""
    
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
        """Extract features for ML prediction"""
        features = {}
        
        # Model features
        model_id = model_config.get("id", "")
        model_name = model_config.get("name", model_id.split("/")[-1])
        
        # Extract model size
        features['model_size_numeric'] = self._extract_model_size_numeric(model_id)
        features['model_size_category'] = self._extract_model_size_category(model_id)
        features['model_type'] = self._extract_model_type(model_id)
        
        # Task features
        features['task_name'] = task_name
        features['task_type'] = self._extract_task_type(task_name)
        features['task_complexity'] = self._extract_task_complexity(task_name)
        
        # Resource features
        if current_resources:
            features['gpu_memory_available'] = current_resources.gpu_memory_total - current_resources.gpu_memory_used
            features['gpu_utilization'] = current_resources.gpu_utilization
            features['gpu_temperature'] = current_resources.gpu_temperature
            features['cpu_percent'] = current_resources.cpu_percent
            features['ram_available'] = current_resources.ram_available
        else:
            # Default values if no resource info
            features['gpu_memory_available'] = 40.0  # Assume 40GB available
            features['gpu_utilization'] = 0.0
            features['gpu_temperature'] = 25.0
            features['cpu_percent'] = 10.0
            features['ram_available'] = 100.0
        
        # Historical features
        historical = self._get_historical_features(model_id, task_name)
        features.update(historical)
        
        # Time features
        now = datetime.now()
        features['hour'] = now.hour
        features['day_of_week'] = now.weekday()
        
        # System features
        features['num_gpus'] = self.num_gpus
        
        return features
    
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
        elif 'mistral' in model_id_lower:
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
            cursor = self.performance_tracker.conn.cursor()
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
        """Prepare training data from performance tracker"""
        try:
            cursor = self.performance_tracker.conn.cursor()
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
        """Encode categorical features"""
        categorical_features = ['model_size_category', 'model_type', 'task_name', 'task_type']
        
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
        
        return df_encoded
    
    def train_models(self) -> bool:
        """Train all ML models"""
        logger.info("Starting ML model training...")
        
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
        
        logger.info(f"Model training completed: {success_count}/6 models trained successfully")
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
        """
        Enhanced prediction confidence calculation using multiple uncertainty metrics
        
        Args:
            X: Input features
            feature_importance: Feature importance dictionary
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            confidence_scores = []
            
            # 1. Feature importance entropy (existing method, improved)
            if feature_importance:
                importance_values = np.array(list(feature_importance.values()))
                importance_values = importance_values / importance_values.sum()  # Normalize
                
                # Calculate entropy
                entropy = -np.sum(importance_values * np.log(importance_values + 1e-10))
                max_entropy = np.log(len(importance_values))
                entropy_confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
                confidence_scores.append(entropy_confidence)
            
            # 2. Ensemble agreement confidence (Random Forest specific)
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
            
            # 5. Out-of-bag confidence (if available)
            oob_confidence = self._calculate_oob_confidence()
            if oob_confidence is not None:
                confidence_scores.append(oob_confidence)
            
            # Combine all confidence scores with weights
            if confidence_scores:
                weights = [0.25, 0.25, 0.2, 0.15, 0.15][:len(confidence_scores)]
                weights = np.array(weights) / sum(weights)  # Normalize weights
                
                final_confidence = np.average(confidence_scores, weights=weights)
                return min(max(final_confidence, 0.0), 1.0)
            else:
                return 0.5  # Default if no confidence metrics available
                
        except Exception as e:
            logger.warning(f"Error calculating prediction confidence: {e}")
            return 0.5

    def _calculate_ensemble_confidence(self, X: pd.DataFrame) -> Optional[float]:
        """
        Calculate confidence based on Random Forest ensemble agreement
        
        Args:
            X: Input features
            
        Returns:
            Ensemble confidence score or None if not available
        """
        try:
            if not hasattr(self.time_predictor, 'estimators_'):
                return None
            
            # Get predictions from all trees
            tree_predictions = []
            for estimator in self.time_predictor.estimators_:
                pred = estimator.predict(X)
                tree_predictions.append(pred[0] if len(pred) > 0 else 0)
            
            if len(tree_predictions) < 2:
                return None
            
            # Calculate variance among tree predictions
            pred_mean = np.mean(tree_predictions)
            pred_std = np.std(tree_predictions)
            
            # Convert variance to confidence (lower variance = higher confidence)
            if pred_mean > 0:
                coefficient_of_variation = pred_std / pred_mean
                # Use sigmoid function to map CV to confidence
                confidence = 1 / (1 + np.exp(coefficient_of_variation - 0.5))
            else:
                confidence = 0.5
            
            return confidence
            
        except Exception as e:
            logger.debug(f"Error calculating ensemble confidence: {e}")
            return None

    def _calculate_prediction_stability(self, X: pd.DataFrame) -> Optional[float]:
        """
        Calculate confidence based on prediction stability with small input perturbations
        
        Args:
            X: Input features
            
        Returns:
            Stability confidence score or None if not available
        """
        try:
            if not self.time_predictor:
                return None
            
            original_pred = self.time_predictor.predict(X)[0]
            
            # Add small random perturbations to numerical features
            perturbed_predictions = []
            num_perturbations = 5
            
            for _ in range(num_perturbations):
                X_perturbed = X.copy()
                
                # Identify numerical columns
                numerical_cols = X.select_dtypes(include=[np.number]).columns
                
                for col in numerical_cols:
                    if X[col].std() > 0:
                        noise_scale = X[col].std() * 0.01  # 1% of standard deviation
                        noise = np.random.normal(0, noise_scale, size=len(X))
                        X_perturbed[col] = X[col] + noise
                
                perturbed_pred = self.time_predictor.predict(X_perturbed)[0]
                perturbed_predictions.append(perturbed_pred)
            
            # Calculate stability as inverse of prediction variance
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
        """
        Calculate confidence based on historical prediction accuracy
        
        Returns:
            Historical accuracy confidence or None if not available
        """
        try:
            if not hasattr(self, 'model_performance') or not self.model_performance:
                return None
            
            accuracy_scores = []
            
            # Extract accuracy metrics from model performance
            for model_name, performance in self.model_performance.items():
                if 'accuracy' in performance:
                    accuracy_scores.append(performance['accuracy'])
                elif 'relative_error' in performance:
                    # Convert relative error to accuracy-like score
                    accuracy = max(0, 1 - performance['relative_error'])
                    accuracy_scores.append(accuracy)
            
            if accuracy_scores:
                mean_accuracy = np.mean(accuracy_scores)
                # Apply sigmoid transformation to emphasize high accuracy
                confidence = 1 / (1 + np.exp(-(mean_accuracy - 0.7) * 10))
                return confidence
            
            return None
            
        except Exception as e:
            logger.debug(f"Error calculating historical accuracy confidence: {e}")
            return None

    def _calculate_oob_confidence(self) -> Optional[float]:
        """
        Calculate confidence based on Out-of-Bag (OOB) error
        
        Returns:
            OOB confidence score or None if not available
        """
        try:
            if not hasattr(self.time_predictor, 'oob_score_'):
                return None
            
            oob_score = self.time_predictor.oob_score_
            
            # OOB score is R² for regression, convert to confidence
            if oob_score is not None:
                # Apply sigmoid transformation
                confidence = 1 / (1 + np.exp(-(oob_score - 0.5) * 5))
                return confidence
            
            return None
            
        except Exception as e:
            logger.debug(f"Error calculating OOB confidence: {e}")
            return None

    def get_enhanced_model_confidence(self) -> Dict[str, float]:
        """
        Enhanced version of get_model_confidence with multiple confidence metrics
        
        Returns:
            Dictionary containing various confidence metrics
        """
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
                    # Convert error to accuracy-like score
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
        
        # 3. Prediction stability confidence (test with sample data)
        try:
            # Create sample feature vector for testing
            sample_features = pd.DataFrame({
                'model_size_numeric': [7.0],
                'model_size_category': [1],  # Encoded value
                'model_type': [1],           # Encoded value
                'task_name': [1],            # Encoded value
                'task_type': [1],            # Encoded value
                'task_complexity': [0.5],
                'gpu_memory_available': [30.0],
                'gpu_utilization': [20.0],
                'gpu_temperature': [35.0],
                'cpu_percent': [15.0],
                'ram_available': [80.0],
                'historical_avg_time': [3600.0],
                'historical_avg_memory': [15.0],
                'historical_success_rate': [0.8],
                'historical_oom_rate': [0.1],
                'historical_sample_count': [10],
                'model_avg_time': [3600.0],
                'model_avg_memory': [15.0],
                'model_run_count': [5],
                'hour': [14],
                'day_of_week': [2],
                'num_gpus': [1]
            })
            
            # Ensure all expected feature columns are present
            if hasattr(self, 'feature_names') and self.feature_names:
                for feature in self.feature_names:
                    if feature not in sample_features.columns:
                        sample_features[feature] = 0  # Default value
                
                # Reorder columns to match training
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
        
        # 5. Temporal stability confidence (based on retraining frequency)
        if self.last_training_time:
            time_since_training = datetime.now() - self.last_training_time
            hours_since_training = time_since_training.total_seconds() / 3600
            
            # Confidence decreases over time (models become stale)
            # Full confidence for first 24 hours, then decay
            if hours_since_training <= 24:
                temporal_confidence = 1.0
            else:
                # Exponential decay with half-life of 72 hours
                temporal_confidence = np.exp(-0.693 * (hours_since_training - 24) / 72)
        else:
            temporal_confidence = 0.0
        
        confidence_metrics['temporal_stability_confidence'] = temporal_confidence
        
        # 6. Overall confidence (weighted combination)
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
        """
        Get overall model confidence (enhanced version)
        
        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        enhanced_confidence = self.get_enhanced_model_confidence()
        return enhanced_confidence.get('overall_confidence', 0.0)

    def predict_with_uncertainty(self, model_config: Dict, task_name: str) -> Dict[str, Any]:
        """
        Make prediction with uncertainty quantification
        
        Args:
            model_config: Model configuration
            task_name: Task name
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        try:
            # Get standard prediction
            prediction = self.predict(model_config, task_name)
            
            # Extract features for uncertainty calculation
            current_resources = self.resource_monitor.get_current_snapshot()
            features = self._extract_features(model_config, task_name, current_resources)
            feature_df = pd.DataFrame([features])
            feature_df = self._encode_categorical_features(feature_df, fit=False)
            
            if hasattr(self, 'feature_names') and self.feature_names:
                X = feature_df[self.feature_names]
            else:
                X = feature_df
            
            # Calculate prediction intervals using quantile forests (approximation)
            uncertainty_metrics = {}
            
            # Time prediction uncertainty
            if self.time_predictor and hasattr(self.time_predictor, 'estimators_'):
                time_predictions = []
                for estimator in self.time_predictor.estimators_:
                    pred = estimator.predict(X)[0]
                    time_predictions.append(pred)
                
                time_predictions = np.array(time_predictions)
                uncertainty_metrics['time_prediction_std'] = np.std(time_predictions)
                uncertainty_metrics['time_prediction_ci_lower'] = np.percentile(time_predictions, 5)
                uncertainty_metrics['time_prediction_ci_upper'] = np.percentile(time_predictions, 95)
            
            # Memory prediction uncertainty
            if self.memory_predictor and hasattr(self.memory_predictor, 'estimators_'):
                memory_predictions = []
                for estimator in self.memory_predictor.estimators_:
                    pred = estimator.predict(X)[0]
                    memory_predictions.append(pred)
                
                memory_predictions = np.array(memory_predictions)
                uncertainty_metrics['memory_prediction_std'] = np.std(memory_predictions)
                uncertainty_metrics['memory_prediction_ci_lower'] = np.percentile(memory_predictions, 5)
                uncertainty_metrics['memory_prediction_ci_upper'] = np.percentile(memory_predictions, 95)
            
            # Success probability uncertainty
            if self.success_classifier and hasattr(self.success_classifier, 'estimators_'):
                success_predictions = []
                for estimator in self.success_classifier.estimators_:
                    pred_proba = estimator.predict_proba(X)[0]
                    prob = pred_proba[1] if len(pred_proba) > 1 else 0.5
                    success_predictions.append(prob)
                
                success_predictions = np.array(success_predictions)
                uncertainty_metrics['success_prediction_std'] = np.std(success_predictions)
                uncertainty_metrics['success_prediction_ci_lower'] = np.percentile(success_predictions, 5)
                uncertainty_metrics['success_prediction_ci_upper'] = np.percentile(success_predictions, 95)
            
            # Enhanced confidence calculation
            enhanced_confidence = self._calculate_prediction_confidence(X, prediction.feature_importance)
            
            return {
                'prediction': prediction,
                'uncertainty_metrics': uncertainty_metrics,
                'enhanced_confidence': enhanced_confidence,
                'confidence_breakdown': self.get_enhanced_model_confidence()
            }
            
        except Exception as e:
            logger.error(f"Error in uncertainty prediction: {e}")
            return {
                'prediction': self.predict(model_config, task_name),
                'uncertainty_metrics': {},
                'enhanced_confidence': 0.5,
                'confidence_breakdown': {}
            }

    def get_confidence_adjusted_thresholds(self, base_min_threshold: int = 50, 
                                         base_stable_threshold: int = 200) -> Dict[str, Any]:
        """
        Get confidence-adjusted thresholds for stage transitions
        
        Args:
            base_min_threshold: Base minimum learning data threshold
            base_stable_threshold: Base stable learning data threshold
            
        Returns:
            Dictionary with adjusted thresholds
        """
        confidence_metrics = self.get_enhanced_model_confidence()
        overall_confidence = confidence_metrics.get('overall_confidence', 0.0)
        
        # Adjust thresholds based on confidence
        # High confidence -> lower thresholds (earlier transition)
        # Low confidence -> higher thresholds (more conservative)
        
        confidence_factor = 1.0 - (overall_confidence - 0.5)  # Range: 0.5 to 1.5
        
        adjusted_min = int(base_min_threshold * confidence_factor)
        adjusted_stable = int(base_stable_threshold * confidence_factor)
        
        # Ensure reasonable bounds
        adjusted_min = max(20, min(adjusted_min, 150))
        adjusted_stable = max(100, min(adjusted_stable, 400))
        
        # Ensure min < stable
        if adjusted_min >= adjusted_stable:
            adjusted_stable = adjusted_min + 50
        
        return {
            'min_learning_data': adjusted_min,
            'stable_learning_data': adjusted_stable,
            'confidence_factor': confidence_factor,
            'overall_confidence': overall_confidence
        }
    
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
        
        # Time efficiency (prefer shorter tasks when uncertainty is high)
        time_efficiency = 1 / (1 + np.log1p(ml_pred.execution_time / 3600))
        score *= time_efficiency
        
        # Memory efficiency
        memory_efficiency = 1 / (1 + np.log1p(ml_pred.memory_usage / 40))
        score *= memory_efficiency
        
        # OOM risk penalty
        score *= (1 - ml_pred.oom_probability)
        
        # Confidence bonus (prioritize high-confidence predictions)
        confidence_bonus = 1 + (ml_pred.confidence_score * 0.2)
        score *= confidence_bonus
        
        return score
    
    def _generate_ml_rationale(self, ml_pred: MLPrediction, model_name: str, task: str) -> str:
        """Generate rationale for ML-based scheduling decision"""
        rationale_parts = []
        
        # Time and memory
        hours = ml_pred.execution_time / 3600
        rationale_parts.append(f"ML Est. time: {hours:.1f}h")
        rationale_parts.append(f"ML Est. memory: {ml_pred.memory_usage:.1f}GB")
        
        # Success and OOM rates
        rationale_parts.append(f"Success: {ml_pred.success_probability*100:.1f}%")
        
        if ml_pred.oom_probability > 0.5:
            rationale_parts.append(f"High OOM risk ({ml_pred.oom_probability*100:.0f}%)")
        elif ml_pred.oom_probability > 0.2:
            rationale_parts.append(f"Medium OOM risk ({ml_pred.oom_probability*100:.0f}%)")
        
        # Confidence
        rationale_parts.append(f"Confidence: {ml_pred.confidence_score*100:.0f}%")
        
        return " | ".join(rationale_parts)
    
    def _optimize_gpu_assignment(self, schedule: List[TaskPriority]):
        """Optimize GPU assignment (reuse from IntelligentScheduler)"""
        if self.num_gpus == 1:
            for task in schedule:
                task.suggested_gpu = 0
            return
        
        # Multi-GPU load balancing
        gpu_loads = [0.0] * self.num_gpus
        
        for task in schedule:
            min_load_gpu = np.argmin(gpu_loads)
            task.suggested_gpu = min_load_gpu
            gpu_loads[min_load_gpu] += task.estimated_time
        
        logger.info(f"ML-based GPU load distribution: {gpu_loads}")
    
    def should_retrain(self) -> bool:
        """Check if models should be retrained"""
        try:
            cursor = self.performance_tracker.conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as total_records
                FROM execution_records 
                WHERE status IN ('completed', 'failed', 'oom')
            """)
            
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
            # Find most recent model files
            model_files = list(self.model_save_path.glob("time_predictor_*.pkl"))
            if not model_files:
                logger.info("No pre-trained models found")
                return
            
            # Get most recent timestamp
            latest_timestamp = max(f.stem.split('_')[-1] for f in model_files)
            
            # Load models
            model_names = ['time_predictor', 'memory_predictor', 'oom_classifier', 
                          'success_classifier', 'batch_size_predictor', 'fewshot_predictor']
            
            for name in model_names:
                model_path = self.model_save_path / f"{name}_{latest_timestamp}.pkl"
                if model_path.exists():
                    setattr(self, name, joblib.load(model_path))
                    logger.debug(f"Loaded {name}")
            
            # Load metadata
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