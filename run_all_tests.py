"""
Comprehensive test runner for SpectraBench
Runs all tests and provides detailed reporting
"""
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_prerequisites():
    """Check if all required files and dependencies exist"""
    print("🔍 Checking Prerequisites")
    print("=" * 60)
    
    issues = []
    
    # Check required files
    required_files = [
        "code/config/scheduler_config.yaml",
        "code/config/config_manager.py", 
        "code/experiments/threshold_optimization.py",
    ]
    
    for file_path in required_files:
        # Try multiple possible locations
        possible_paths = [
            Path(file_path),  # From current directory
            project_root / file_path,  # From project root
            Path.cwd() / file_path  # From working directory
        ]
        
        file_found = False
        for full_path in possible_paths:
            if full_path.exists():
                print(f"✅ Found: {file_path} (at {full_path})")
                file_found = True
                break
        
        if not file_found:
            issues.append(f"Missing file: {file_path}")
    
    # Check required directories  
    required_dirs = [
        "code/config",
        "code/experiments",
        "code/scheduler",
    ]
    
    for dir_path in required_dirs:
        # Try multiple possible locations
        possible_paths = [
            Path(dir_path),  # From current directory
            project_root / dir_path,  # From project root
            Path.cwd() / dir_path  # From working directory
        ]
        
        dir_found = False
        for full_path in possible_paths:
            if full_path.exists():
                print(f"✅ Found: {dir_path} (at {full_path})")
                dir_found = True
                break
        
        if not dir_found:
            issues.append(f"Missing directory: {dir_path}")
    
    # Check Python dependencies
    required_packages = [
        "yaml", "pandas", "numpy", "pathlib", "sqlite3", 
        "dataclasses", "logging", "datetime"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ Package: {package}")
        except ImportError:
            issues.append(f"Missing Python package: {package}")
    
    if issues:
        print("\n❌ Prerequisites check FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nNOTE: If files exist but are not found, this might be a path issue.")
        print("Please run this script from the project root directory.")
        return False
    else:
        print("\n✅ All prerequisites satisfied!")
        return True

def run_config_tests():
    """Run configuration management tests"""
    print("\n" + "🧪 Running Configuration Tests")
    print("=" * 60)
    
    try:
        # Find the correct test script path
        possible_paths = [
            Path("test_config_manager.py"),
            project_root / "test_config_manager.py",
            Path.cwd() / "test_config_manager.py"
        ]
        
        test_script = None
        for path in possible_paths:
            if path.exists():
                test_script = path
                break
        
        if not test_script:
            print("❌ test_config_manager.py not found")
            return False
        
        # Execute as subprocess
        import subprocess
        result = subprocess.run([
            sys.executable, str(test_script)
        ], capture_output=True, text=True, cwd=str(Path.cwd()))
        
        if result.returncode == 0:
            print("✅ Configuration tests PASSED")
            print("Output preview:")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print("❌ Configuration tests FAILED")
            print("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running configuration tests: {e}")
        return False

def run_experiment_tests():
    """Run experiment framework tests"""
    print("\n" + "🧪 Running Experiment Tests")
    print("=" * 60)
    
    try:
        # Find the correct test script path
        possible_paths = [
            Path("test_threshold_optimization.py"),
            project_root / "test_threshold_optimization.py", 
            Path.cwd() / "test_threshold_optimization.py"
        ]
        
        test_script = None
        for path in possible_paths:
            if path.exists():
                test_script = path
                break
        
        if not test_script:
            print("❌ test_threshold_optimization.py not found")
            return False
        
        import subprocess
        result = subprocess.run([
            sys.executable, str(test_script)
        ], capture_output=True, text=True, cwd=str(Path.cwd()))
        
        if result.returncode == 0:
            print("✅ Experiment tests PASSED")
            print("Output preview:")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print("❌ Experiment tests FAILED")
            print("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running experiment tests: {e}")
        return False

def run_integration_tests():
    """Run integration tests between components"""
    print("\n" + "🧪 Running Integration Tests")
    print("=" * 60)
    
    try:
        # Test 1: Config + Experiment integration
        print("Test 1: Configuration and Experiment Integration")
        
        from code.config.config_manager import ConfigManager
        from code.experiments.threshold_optimization import create_default_experiment_config
        
        # Load config
        config_manager = ConfigManager(environment='development')
        experiment_config = create_default_experiment_config()
        
        # Test that experiment can use config values
        stage_thresholds = config_manager.get_domain_thresholds('medium_models')
        
        print(f"✅ Config stage thresholds: {stage_thresholds}")
        print(f"✅ Experiment stage1 options: {experiment_config.stage1_to_stage2_thresholds}")
        print(f"✅ Experiment stage2 options: {experiment_config.stage2_to_stage3_thresholds}")
        
        # Verify compatibility (allow some overlap)
        min_stage2 = min(experiment_config.stage2_to_stage3_thresholds)
        max_stage1 = max(experiment_config.stage1_to_stage2_thresholds)
        assert max_stage1 <= min_stage2, f"Invalid threshold ranges: max stage1 ({max_stage1}) should be <= min stage2 ({min_stage2})"
        
        print("✅ Integration Test 1 PASSED")
        
        # Test 2: Config updates and persistence
        print("\nTest 2: Configuration Updates")
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_config_path = f.name
        
        try:
            # Save and reload config
            config_manager.save_config(temp_config_path)
            new_config = ConfigManager(config_path=temp_config_path, environment='development')
            
            # Test update
            original_min = new_config.stage_transitions.min_learning_data
            new_config.update_thresholds(75, 250)
            
            assert new_config.stage_transitions.min_learning_data == 75
            assert new_config.stage_transitions.stable_learning_data == 250
            
            print("✅ Integration Test 2 PASSED")
            
        finally:
            os.unlink(temp_config_path)
        
        print("✅ All integration tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Integration tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_performance_tests():
    """Run basic performance tests"""
    print("\n" + "🧪 Running Performance Tests")
    print("=" * 60)
    
    try:
        # Test config loading performance
        print("Test 1: Configuration Loading Performance")
        
        start_time = time.time()
        for i in range(10):
            from code.config.config_manager import ConfigManager
            config = ConfigManager(environment='development')
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"✅ Average config loading time: {avg_time:.3f}s")
        
        if avg_time > 1.0:
            print("⚠️  Config loading is slow (>1s)")
        
        # Test experiment simulation performance
        print("\nTest 2: Experiment Simulation Performance")
        
        from code.experiments.threshold_optimization import ThresholdOptimizer, ExperimentConfig
        
        # Small experiment for performance testing
        config = ExperimentConfig(
            stage1_to_stage2_thresholds=[50],
            stage2_to_stage3_thresholds=[200],
            models=[{"id": "test-7b", "name": "Test Model"}],
            tasks=["mmlu"],
            num_runs_per_threshold=1,
            output_dir=Path("temp_performance_test")
        )
        
        optimizer = ThresholdOptimizer(config)
        
        start_time = time.time()
        result = optimizer._run_single_experiment(50, 200, 0)
        end_time = time.time()
        
        experiment_time = end_time - start_time
        print(f"✅ Single experiment time: {experiment_time:.3f}s")
        
        if experiment_time > 5.0:
            print("⚠️  Experiment simulation is slow (>5s)")
        
        # Cleanup
        if config.output_dir.exists():
            import shutil
            shutil.rmtree(config.output_dir)
        
        print("✅ Performance tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Performance tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_test_report(results):
    """Generate comprehensive test report"""
    print("\n" + "📊 COMPREHENSIVE TEST REPORT")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Test run completed at: {timestamp}")
    print()
    
    # Test results summary
    test_categories = [
        ("Prerequisites", results.get('prerequisites', False)),
        ("Configuration", results.get('config', False)),
        ("Experiments", results.get('experiments', False)),
        ("Integration", results.get('integration', False)),
        ("Performance", results.get('performance', False)),
    ]
    
    passed_count = 0
    total_count = len(test_categories)
    
    for category, passed in test_categories:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{category:15} {status}")
        if passed:
            passed_count += 1
    
    print(f"\nOverall Results: {passed_count}/{total_count} categories passed")
    print(f"Success Rate: {passed_count/total_count*100:.1f}%")
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS")
    print("-" * 40)
    
    if results.get('prerequisites', False):
        print("✅ Prerequisites are satisfied")
    else:
        print("❌ Fix missing prerequisites before proceeding")
    
    if results.get('config', False):
        print("✅ Configuration system is working")
    else:
        print("❌ Fix configuration issues before running experiments")
    
    if results.get('experiments', False):
        print("✅ Experiment framework is ready")
        if results.get('config', False):
            print("✅ You can proceed with threshold optimization experiments")
    else:
        print("❌ Fix experiment framework issues")
    
    if results.get('integration', False):
        print("✅ Component integration is working")
    else:
        print("❌ Fix integration issues for reliable operation")
    
    if not results.get('performance', False):
        print("⚠️  Performance issues detected - may affect large experiments")
    
    # Next steps
    print("\n🚀 NEXT STEPS")
    print("-" * 40)
    
    if passed_count >= 4:  # At least 4/5 categories passed
        print("1. ✅ Start running threshold optimization experiments")
        print("2. ✅ Modify existing scheduler code to use new config system")
        print("3. ✅ Begin paper experiments for threshold validation")
    elif passed_count >= 2:
        print("1. 🔧 Fix remaining test failures")
        print("2. 🔧 Re-run tests to verify fixes")
        print("3. ⏳ Then proceed with experiments")
    else:
        print("1. 🚨 Address critical test failures")
        print("2. 🚨 Review code and configuration")
        print("3. 🚨 Get help if needed")
    
    return passed_count >= 4

def main():
    """Main test runner"""
    print("🚀 SpectraBench Comprehensive Test Suite")
    print("=" * 80)
    print("This will test all new components and their integration")
    print()
    
    start_time = time.time()
    
    # Run all test categories
    results = {}
    
    # Prerequisites
    results['prerequisites'] = check_prerequisites()
    if not results['prerequisites']:
        print("\n❌ Cannot proceed without prerequisites")
        return False
    
    # Configuration tests
    results['config'] = run_config_tests()
    
    # Experiment tests
    results['experiments'] = run_experiment_tests()
    
    # Integration tests
    results['integration'] = run_integration_tests()
    
    # Performance tests
    results['performance'] = run_performance_tests()
    
    # Generate final report
    end_time = time.time()
    total_time = end_time - start_time
    
    success = generate_test_report(results)
    
    print(f"\nTotal test time: {total_time:.1f} seconds")
    
    if success:
        print("\n🎉 TESTS COMPLETED SUCCESSFULLY!")
        print("You can now proceed with the next development phase.")
    else:
        print("\n❌ TESTS FAILED!")
        print("Please address the issues before proceeding.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)