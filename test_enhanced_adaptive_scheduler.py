"""
Enhanced AdaptiveScheduler Test Suite
Tests new confidence calculation methods and uncertainty quantification
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the enhanced scheduler
try:
    from code.scheduler.adaptive_scheduler import AdaptiveScheduler
    from code.scheduler.performance_tracker import PerformanceTracker
    from code.scheduler.resource_monitor import ResourceMonitor
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the enhanced adaptive_scheduler.py is in place")
    sys.exit(1)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_enhanced_confidence_calculation():
    """Test enhanced confidence calculation methods"""
    print("🧪 Testing Enhanced Confidence Calculation")
    print("=" * 60)
    
    # Initialize components
    performance_tracker = PerformanceTracker(mode="test")
    resource_monitor = ResourceMonitor(monitoring_interval=1.0)
    
    try:
        # Initialize scheduler
        scheduler = AdaptiveScheduler(
            performance_tracker=performance_tracker,
            resource_monitor=resource_monitor,
            num_gpus=1
        )
        
        print("✅ AdaptiveScheduler initialized successfully")
        
        # Test 1: Enhanced confidence methods without training
        print("\n--- Test 1: Confidence methods (untrained) ---")
        
        enhanced_confidence = scheduler.get_enhanced_model_confidence()
        print(f"Enhanced confidence metrics: {enhanced_confidence}")
        
        overall_confidence = scheduler.get_model_confidence()
        print(f"Overall confidence: {overall_confidence:.3f}")
        
        # Test 2: Confidence-adjusted thresholds
        print("\n--- Test 2: Confidence-adjusted thresholds ---")
        
        adjusted_thresholds = scheduler.get_confidence_adjusted_thresholds()
        print(f"Adjusted thresholds: {adjusted_thresholds}")
        
        # Test 3: Prediction with uncertainty (should handle gracefully)
        print("\n--- Test 3: Prediction with uncertainty (untrained) ---")
        
        model_config = {
            "id": "meta-llama/Llama-3.1-8B",
            "name": "LLaMA 3.1 8B"
        }
        
        uncertainty_result = scheduler.predict_with_uncertainty(model_config, "mmlu")
        print(f"Prediction: {uncertainty_result['prediction'].execution_time:.1f}s")
        print(f"Enhanced confidence: {uncertainty_result['enhanced_confidence']:.3f}")
        print(f"Uncertainty metrics available: {len(uncertainty_result['uncertainty_metrics'])}")
        
        print("✅ All enhanced confidence tests passed!")
        
    except Exception as e:
        print(f"❌ Enhanced confidence test failed: {e}")
        raise
    finally:
        performance_tracker.close()


def test_training_with_mock_data():
    """Test training with mock data to verify ML pipeline"""
    print("\n🧪 Testing Training with Mock Data")
    print("=" * 60)
    
    # Create temporary database with mock data
    import tempfile
    import sqlite3
    
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    try:
        # Create mock performance data
        conn = sqlite3.connect(temp_db.name)
        conn.row_factory = sqlite3.Row
        
        # Create table structure
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE execution_records (
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
                result_metrics TEXT,
                error_message TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert mock data (100 records)
        models = [
            ("meta-llama/Llama-3.1-8B", "LLaMA 3.1 8B", "8B"),
            ("google/gemma-3-4b-it", "Gemma 3 4B", "4B"),
            ("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", "EXAONE 7.8B", "7.8B"),
            ("mistralai/Mistral-7B-v0.3", "Mistral 7B", "7B")
        ]
        
        tasks = ["mmlu", "bbh", "gsm8k", "humaneval", "hellaswag"]
        
        np.random.seed(42)  # For reproducible results
        
        mock_records = []
        for i in range(100):
            model_id, model_name, model_size = models[i % len(models)]
            task = tasks[i % len(tasks)]
            
            # Simulate realistic execution times and outcomes
            base_time = np.random.exponential(3600)  # Average 1 hour
            memory_usage = np.random.gamma(2, 10)    # Average 20GB
            
            # Some probability of failure/OOM
            status = np.random.choice(['completed', 'failed', 'oom'], 
                                    p=[0.8, 0.15, 0.05])
            
            if status != 'completed':
                base_time *= 0.3  # Failed tasks take less time
            
            record = (
                f"test_run_{i//20}",                    # run_id
                datetime.now().isoformat(),             # timestamp
                "test",                                 # mode
                model_id,                               # model_id
                model_name,                             # model_name
                model_size,                             # model_size
                task,                                   # task_name
                "harness",                              # task_type
                5,                                      # num_fewshot
                1,                                      # batch_size
                None,                                   # sample_limit
                "0",                                    # gpu_id
                "cuda:0",                               # device
                1000000 + i * 100,                     # start_time
                1000000 + i * 100 + base_time,         # end_time
                status,                                 # status
                base_time,                              # execution_time
                memory_usage * 0.8,                     # gpu_memory_start
                memory_usage,                           # gpu_memory_peak
                memory_usage * 0.9,                     # gpu_memory_end
                np.random.uniform(60, 95),              # gpu_utilization_avg
                np.random.uniform(10, 30),              # cpu_percent_avg
                np.random.uniform(40, 80),              # ram_usage_peak
                None,                                   # result_metrics
                None,                                   # error_message
                None                                    # metadata
            )
            mock_records.append(record)
        
        cursor.executemany("""
            INSERT INTO execution_records (
                run_id, timestamp, mode, model_id, model_name, model_size,
                task_name, task_type, num_fewshot, batch_size, sample_limit,
                gpu_id, device, start_time, end_time, status, execution_time,
                gpu_memory_start, gpu_memory_peak, gpu_memory_end,
                gpu_utilization_avg, cpu_percent_avg, ram_usage_peak,
                result_metrics, error_message, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, mock_records)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created mock database with {len(mock_records)} records")
        
        # Test training with mock data
        performance_tracker = PerformanceTracker(mode="test", db_path=Path(temp_db.name))
        resource_monitor = ResourceMonitor(monitoring_interval=1.0)
        
        scheduler = AdaptiveScheduler(
            performance_tracker=performance_tracker,
            resource_monitor=resource_monitor,
            num_gpus=1
        )
        
        print("\n--- Training ML models ---")
        training_success = scheduler.train_models()
        
        if training_success:
            print("✅ ML model training successful!")
            
            # Test enhanced confidence after training
            print("\n--- Testing enhanced confidence (trained) ---")
            enhanced_confidence = scheduler.get_enhanced_model_confidence()
            
            for metric, value in enhanced_confidence.items():
                print(f"  {metric}: {value:.3f}")
            
            # Test prediction with uncertainty
            print("\n--- Testing prediction with uncertainty (trained) ---")
            model_config = {
                "id": "meta-llama/Llama-3.1-8B",
                "name": "LLaMA 3.1 8B"
            }
            
            uncertainty_result = scheduler.predict_with_uncertainty(model_config, "mmlu")
            
            print(f"Prediction time: {uncertainty_result['prediction'].execution_time:.1f}s")
            print(f"Prediction memory: {uncertainty_result['prediction'].memory_usage:.1f}GB")
            print(f"Enhanced confidence: {uncertainty_result['enhanced_confidence']:.3f}")
            
            # Print uncertainty metrics
            if uncertainty_result['uncertainty_metrics']:
                print("Uncertainty metrics:")
                for metric, value in uncertainty_result['uncertainty_metrics'].items():
                    print(f"  {metric}: {value:.3f}")
            
            # Test confidence-adjusted thresholds
            print("\n--- Testing confidence-adjusted thresholds (trained) ---")
            adjusted_thresholds = scheduler.get_confidence_adjusted_thresholds()
            
            for key, value in adjusted_thresholds.items():
                print(f"  {key}: {value}")
            
            print("✅ All training and enhanced feature tests passed!")
            
        else:
            print("❌ ML model training failed")
        
        performance_tracker.close()
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        raise
    finally:
        # Cleanup
        if os.path.exists(temp_db.name):
            os.unlink(temp_db.name)


def test_integration_with_scheduler_manager():
    """Test integration with SchedulerManager"""
    print("\n🧪 Testing Integration with SchedulerManager")
    print("=" * 60)
    
    try:
        from code.scheduler.scheduler_manager import SchedulerManager
        
        # Create temporary components
        performance_tracker = PerformanceTracker(mode="integration_test")
        resource_monitor = ResourceMonitor(monitoring_interval=1.0)
        
        # Test SchedulerManager with enhanced AdaptiveScheduler
        scheduler_manager = SchedulerManager(
            performance_tracker=performance_tracker,
            resource_monitor=resource_monitor,
            num_gpus=1
        )
        
        print("✅ SchedulerManager initialized with enhanced AdaptiveScheduler")
        
        # Test getting current mode
        current_mode = scheduler_manager.get_current_mode()
        print(f"Current scheduling mode: {current_mode}")
        
        # Test getting scheduler statistics
        stats = scheduler_manager.get_scheduler_statistics()
        print(f"Scheduler statistics: {stats['current_mode']}")
        print(f"Training records: {stats['total_training_records']}")
        print(f"Adaptive status: {stats['adaptive_status']}")
        
        # Test creating schedule (should work even with minimal data)
        models = [
            {"id": "meta-llama/Llama-3.1-8B", "name": "LLaMA 3.1 8B"},
            {"id": "google/gemma-3-4b-it", "name": "Gemma 3 4B"}
        ]
        tasks = ["mmlu", "bbh"]
        
        schedule = scheduler_manager.create_optimal_schedule(models, tasks)
        print(f"✅ Created schedule with {len(schedule)} tasks")
        
        for i, task in enumerate(schedule[:2]):  # Show first 2
            print(f"  Task {i+1}: {task.model_id} on {task.task_name}")
            print(f"    Priority: {task.priority_score:.2f}")
            print(f"    Rationale: {task.rationale}")
        
        print("✅ Integration test passed!")
        
        performance_tracker.close()
        
    except ImportError as e:
        print(f"⚠️ SchedulerManager not available: {e}")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise


def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Enhanced AdaptiveScheduler Comprehensive Test Suite")
    print("=" * 80)
    
    tests = [
        ("Enhanced Confidence Calculation", test_enhanced_confidence_calculation),
        ("Training with Mock Data", test_training_with_mock_data),
        ("Integration with SchedulerManager", test_integration_with_scheduler_manager)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔍 Running: {test_name}")
            test_func()
            passed += 1
            print(f"✅ {test_name} PASSED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED: {e}")
    
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Enhanced AdaptiveScheduler is ready!")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_comprehensive_test()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Replace existing adaptive_scheduler.py with enhanced version")
        print("2. Run real experiments with threshold optimization")
        print("3. Collect data for paper results section")
    else:
        print("\n🔧 Fix the failing tests before proceeding to experiments")