"""
Test script for Threshold Optimization Experiment Framework
Tests the automated experiment system for finding optimal thresholds
"""
import sys
import os
import tempfile
import shutil
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_experiment_config():
    """Test experiment configuration creation"""
    print("=" * 60)
    print("TEST 1: Experiment Configuration")
    print("=" * 60)
    
    try:
        from code.experiments.threshold_optimization import ExperimentConfig, create_default_experiment_config
        
        # Test default config creation
        config = create_default_experiment_config()
        
        print(f"✅ Stage 1 thresholds: {config.stage1_to_stage2_thresholds}")
        print(f"✅ Stage 2 thresholds: {config.stage2_to_stage3_thresholds}")
        print(f"✅ Number of models: {len(config.models)}")
        print(f"✅ Number of tasks: {len(config.tasks)}")
        print(f"✅ Runs per threshold: {config.num_runs_per_threshold}")
        print(f"✅ Output directory: {config.output_dir}")
        
        # Verify configuration makes sense
        assert len(config.stage1_to_stage2_thresholds) > 0, "No stage 1 thresholds"
        assert len(config.stage2_to_stage3_thresholds) > 0, "No stage 2 thresholds"
        # Verify that most stage1 thresholds are less than most stage2 thresholds (allowing some overlap)
        min_stage2 = min(config.stage2_to_stage3_thresholds)
        max_stage1 = max(config.stage1_to_stage2_thresholds)
        assert max_stage1 <= min_stage2, f"Invalid threshold ranges: max stage1 ({max_stage1}) should be <= min stage2 ({min_stage2})"
        assert len(config.models) > 0, "No models configured"
        assert len(config.tasks) > 0, "No tasks configured"
        
        print("✅ Experiment configuration test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Experiment configuration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_threshold_optimizer_init():
    """Test ThresholdOptimizer initialization"""
    print("\n" + "=" * 60)
    print("TEST 2: ThresholdOptimizer Initialization")
    print("=" * 60)
    
    try:
        from code.experiments.threshold_optimization import ThresholdOptimizer, create_default_experiment_config
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config with temporary output directory
            config = create_default_experiment_config()
            config.output_dir = Path(temp_dir) / "test_experiments"
            config.num_runs_per_threshold = 1  # Faster for testing
            
            # Initialize optimizer
            optimizer = ThresholdOptimizer(config)
            
            print(f"✅ Optimizer created with experiment ID: {optimizer.experiment_id}")
            print(f"✅ Output directory: {optimizer.config.output_dir}")
            print(f"✅ Output directory exists: {optimizer.config.output_dir.exists()}")
            
            # Check that output directory was created
            assert optimizer.config.output_dir.exists(), "Output directory not created"
            
        print("✅ ThresholdOptimizer initialization test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ ThresholdOptimizer initialization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_task_simulation():
    """Test task execution simulation"""
    print("\n" + "=" * 60)
    print("TEST 3: Task Execution Simulation")
    print("=" * 60)
    
    try:
        from code.experiments.threshold_optimization import ThresholdOptimizer, create_default_experiment_config
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_default_experiment_config()
            config.output_dir = Path(temp_dir) / "test_experiments"
            
            optimizer = ThresholdOptimizer(config)
            
            # Test simulation with different model sizes
            test_models = [
                {"id": "test-model-0.5b", "name": "Test Small Model"},
                {"id": "test-model-7b", "name": "Test Medium Model"},
                {"id": "test-model-32b", "name": "Test Large Model"}
            ]
            
            test_tasks = ["mmlu", "bbh", "humaneval"]
            
            for model in test_models:
                for task in test_tasks:
                    execution_time, status = optimizer._simulate_task_execution(model, task)
                    
                    print(f"✅ {model['name']:20} | {task:10} | {execution_time:8.1f}s | {status}")
                    
                    # Verify simulation results
                    assert execution_time > 0, f"Invalid execution time: {execution_time}"
                    assert status in ["completed", "oom", "failed"], f"Invalid status: {status}"
            
        print("✅ Task execution simulation test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Task execution simulation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_experiment():
    """Test running a single experiment"""
    print("\n" + "=" * 60)
    print("TEST 4: Single Experiment Execution")
    print("=" * 60)
    
    try:
        from code.experiments.threshold_optimization import ThresholdOptimizer, create_default_experiment_config
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal config for faster testing
            config = create_default_experiment_config()
            config.output_dir = Path(temp_dir) / "test_experiments"
            config.models = config.models[:2]  # Use only 2 models
            config.tasks = config.tasks[:2]    # Use only 2 tasks
            config.num_runs_per_threshold = 1
            
            optimizer = ThresholdOptimizer(config)
            
            # Test single experiment
            stage1_thresh = 50
            stage2_thresh = 200
            run_id = 0
            
            print(f"Running experiment: stage1={stage1_thresh}, stage2={stage2_thresh}")
            
            result = optimizer._run_single_experiment(stage1_thresh, stage2_thresh, run_id)
            
            print(f"✅ Experiment completed successfully")
            print(f"✅ Stage 1 threshold: {result.stage1_threshold}")
            print(f"✅ Stage 2 threshold: {result.stage2_threshold}")
            print(f"✅ Total execution time: {result.total_execution_time:.1f}s")
            print(f"✅ Success rate: {result.total_success_rate:.2f}")
            print(f"✅ OOM rate: {result.total_oom_rate:.2f}")
            print(f"✅ Completed tasks: {result.completed_tasks}")
            print(f"✅ Failed tasks: {result.failed_tasks}")
            print(f"✅ OOM tasks: {result.oom_tasks}")
            
            # Verify result
            total_tasks = len(config.models) * len(config.tasks)
            assert result.completed_tasks + result.failed_tasks + result.oom_tasks == total_tasks
            assert 0 <= result.total_success_rate <= 1
            assert 0 <= result.total_oom_rate <= 1
            assert result.total_execution_time > 0
            
        print("✅ Single experiment execution test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Single experiment execution test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_results_processing():
    """Test results processing and analysis"""
    print("\n" + "=" * 60)
    print("TEST 5: Results Processing")
    print("=" * 60)
    
    try:
        from code.experiments.threshold_optimization import (
            ThresholdOptimizer, ExperimentResult, create_default_experiment_config
        )
        from datetime import datetime
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = create_default_experiment_config()
            config.output_dir = Path(temp_dir) / "test_experiments"
            
            optimizer = ThresholdOptimizer(config)
            
            # Create mock results
            mock_results = [
                ExperimentResult(
                    stage1_threshold=25, stage2_threshold=150, run_id=0,
                    total_execution_time=1000, total_success_rate=0.9, total_oom_rate=0.05,
                    memory_efficiency=0.8, stage1_tasks=5, stage2_tasks=10, stage3_tasks=5,
                    completed_tasks=18, failed_tasks=1, oom_tasks=1,
                    experiment_start=datetime.now().isoformat(),
                    experiment_end=datetime.now().isoformat(),
                    experiment_duration=60.0
                ),
                ExperimentResult(
                    stage1_threshold=50, stage2_threshold=200, run_id=0,
                    total_execution_time=1200, total_success_rate=0.85, total_oom_rate=0.1,
                    memory_efficiency=0.75, stage1_tasks=8, stage2_tasks=7, stage3_tasks=5,
                    completed_tasks=17, failed_tasks=2, oom_tasks=1,
                    experiment_start=datetime.now().isoformat(),
                    experiment_end=datetime.now().isoformat(),
                    experiment_duration=65.0
                )
            ]
            
            optimizer.results = mock_results
            
            # Test DataFrame conversion
            results_df = optimizer._convert_results_to_dataframe()
            print(f"✅ Results DataFrame shape: {results_df.shape}")
            print(f"✅ DataFrame columns: {list(results_df.columns)}")
            
            # Test aggregation
            aggregated = optimizer._aggregate_results(results_df)
            print(f"✅ Aggregated results shape: {aggregated.shape}")
            
            # Test best configurations
            best_configs = optimizer._find_best_configurations(aggregated)
            print(f"✅ Best configurations found: {list(best_configs.keys())}")
            
            for config_type, config_data in best_configs.items():
                print(f"   {config_type}: stage1={config_data['stage1_threshold']}, "
                      f"stage2={config_data['stage2_threshold']}")
            
            # Verify results
            assert len(results_df) == len(mock_results), "DataFrame conversion failed"
            assert 'stage1_threshold' in results_df.columns, "Missing required column"
            assert 'total_execution_time' in results_df.columns, "Missing required column"
            
        print("✅ Results processing test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Results processing test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mini_experiment():
    """Test running a very small complete experiment"""
    print("\n" + "=" * 60)
    print("TEST 6: Mini Complete Experiment")
    print("=" * 60)
    
    try:
        from code.experiments.threshold_optimization import ThresholdOptimizer, ExperimentConfig
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal experiment config
            config = ExperimentConfig(
                stage1_to_stage2_thresholds=[25, 50],  # Only 2 values
                stage2_to_stage3_thresholds=[100, 150], # Only 2 values
                models=[
                    {"id": "test-model-7b", "name": "Test Model 7B"},
                    {"id": "test-model-0.5b", "name": "Test Model 0.5B"}
                ],
                tasks=["mmlu", "bbh"],  # Only 2 tasks
                num_runs_per_threshold=1,  # Only 1 run per combination
                full_run=False,
                output_dir=Path(temp_dir) / "mini_experiment"
            )
            
            print(f"Running mini experiment with:")
            print(f"  - Stage 1 thresholds: {config.stage1_to_stage2_thresholds}")
            print(f"  - Stage 2 thresholds: {config.stage2_to_stage3_thresholds}")
            print(f"  - Models: {len(config.models)}")
            print(f"  - Tasks: {len(config.tasks)}")
            print(f"  - Expected total experiments: {len(config.stage1_to_stage2_thresholds) * len(config.stage2_to_stage3_thresholds)}")
            
            optimizer = ThresholdOptimizer(config)
            
            # Run complete experiment
            results_df = optimizer.run_full_experiment()
            
            print(f"✅ Mini experiment completed!")
            print(f"✅ Results shape: {results_df.shape}")
            print(f"✅ Unique threshold combinations: {len(results_df.groupby(['stage1_threshold', 'stage2_threshold']))}")
            
            # Check output files
            output_files = list(config.output_dir.glob("*"))
            print(f"✅ Output files created: {len(output_files)}")
            for file in output_files:
                print(f"   - {file.name}")
            
            # Verify results
            expected_experiments = len(config.stage1_to_stage2_thresholds) * len(config.stage2_to_stage3_thresholds)
            assert len(results_df) == expected_experiments, f"Expected {expected_experiments} results, got {len(results_df)}"
            
        print("✅ Mini complete experiment test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Mini complete experiment test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_experiment_tests():
    """Run all threshold optimization tests"""
    print("🚀 Starting Threshold Optimization Tests")
    print("=" * 80)
    
    # Check dependencies first
    try:
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"❌ Required dependency missing: {e}")
        return False
    
    tests = [
        test_experiment_config,
        test_threshold_optimizer_init,
        test_task_simulation,
        test_single_experiment,
        test_results_processing,
        test_mini_experiment
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 EXPERIMENT TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL EXPERIMENT TESTS PASSED!")
    else:
        print(f"❌ {total - passed} tests failed")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_experiment_tests()
    sys.exit(0 if success else 1)