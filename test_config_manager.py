"""
Test script for Configuration Manager
Tests loading, validation, and usage of configuration files
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_config_loading():
    """Test basic configuration loading"""
    print("=" * 60)
    print("TEST 1: Configuration Loading")
    print("=" * 60)
    
    try:
        from code.config.config_manager import ConfigManager
        
        # Test with different environments
        environments = ['development', 'testing', 'production']
        
        for env in environments:
            print(f"\n--- Testing {env} environment ---")
            config = ConfigManager(environment=env)
            
            print(f"✅ Environment: {config.environment}")
            print(f"✅ Min learning data: {config.stage_transitions.min_learning_data}")
            print(f"✅ Stable learning data: {config.stage_transitions.stable_learning_data}")
            print(f"✅ Random Forest estimators: {config.ml_models.random_forest['n_estimators']}")
            
            # Test validation
            warnings = config.validate_config()
            if warnings:
                print(f"⚠️  Warnings: {len(warnings)}")
                for warning in warnings[:3]:  # Show first 3 warnings
                    print(f"    - {warning}")
            else:
                print("✅ Configuration validation passed")
        
        print("\n✅ Configuration loading test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_domain_specific_thresholds():
    """Test domain-specific threshold functionality"""
    print("\n" + "=" * 60)
    print("TEST 2: Domain-Specific Thresholds")
    print("=" * 60)
    
    try:
        from code.config.config_manager import ConfigManager
        
        config = ConfigManager(environment='development')
        
        # Test different model sizes
        model_sizes = ['small_models', 'medium_models', 'large_models', '7b', '32b']
        
        for size in model_sizes:
            thresholds = config.get_domain_thresholds(size)
            print(f"✅ {size:15} -> min: {thresholds['min_learning_data']:3d}, stable: {thresholds['stable_learning_data']:3d}")
        
        print("\n✅ Domain-specific thresholds test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Domain-specific thresholds test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_updates():
    """Test configuration updates"""
    print("\n" + "=" * 60)
    print("TEST 3: Configuration Updates")
    print("=" * 60)
    
    try:
        from code.config.config_manager import ConfigManager
        
        config = ConfigManager(environment='development')
        
        # Original values
        original_min = config.stage_transitions.min_learning_data
        original_stable = config.stage_transitions.stable_learning_data
        print(f"✅ Original values - min: {original_min}, stable: {original_stable}")
        
        # Update values
        new_min = 75
        new_stable = 250
        config.update_thresholds(new_min, new_stable)
        
        # Check updated values
        updated_min = config.stage_transitions.min_learning_data
        updated_stable = config.stage_transitions.stable_learning_data
        print(f"✅ Updated values - min: {updated_min}, stable: {updated_stable}")
        
        # Verify
        assert updated_min == new_min, f"Min threshold not updated correctly: {updated_min} != {new_min}"
        assert updated_stable == new_stable, f"Stable threshold not updated correctly: {updated_stable} != {new_stable}"
        
        print("✅ Configuration updates test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Configuration updates test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convenience_functions():
    """Test convenience functions"""
    print("\n" + "=" * 60)
    print("TEST 4: Convenience Functions")
    print("=" * 60)
    
    try:
        from code.config.config_manager import get_config, get_stage_thresholds, get_ml_config
        
        # Test global config instance
        config1 = get_config('development')
        config2 = get_config('development')
        assert config1 is config2, "Global config instance not working"
        print("✅ Global config instance working")
        
        # Test convenience functions
        thresholds = get_stage_thresholds()
        print(f"✅ Default thresholds: {thresholds}")
        
        small_thresholds = get_stage_thresholds('small_models')
        print(f"✅ Small model thresholds: {small_thresholds}")
        
        ml_config = get_ml_config()
        print(f"✅ ML config keys: {list(ml_config.keys())}")
        
        print("✅ Convenience functions test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Convenience functions test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_operations():
    """Test file save/load operations"""
    print("\n" + "=" * 60)
    print("TEST 5: File Operations")
    print("=" * 60)
    
    try:
        from code.config.config_manager import ConfigManager
        import tempfile
        import os
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config = ConfigManager(environment='development')
            
            # Save configuration
            config.save_config(temp_path)
            print(f"✅ Configuration saved to: {temp_path}")
            
            # Load configuration from saved file
            new_config = ConfigManager(config_path=temp_path, environment='development')
            print("✅ Configuration loaded from saved file")
            
            # Compare key values
            assert config.stage_transitions.min_learning_data == new_config.stage_transitions.min_learning_data
            assert config.ml_models.random_forest['n_estimators'] == new_config.ml_models.random_forest['n_estimators']
            print("✅ Saved and loaded configurations match")
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        print("✅ File operations test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ File operations test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_config_tests():
    """Run all configuration tests"""
    print("🚀 Starting Configuration Manager Tests")
    print("=" * 80)
    
    tests = [
        test_config_loading,
        test_domain_specific_thresholds,
        test_config_updates,
        test_convenience_functions,
        test_file_operations
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"❌ {total - passed} tests failed")
    
    return passed == total

if __name__ == "__main__":
    # Make sure the config file exists
    config_file = Path(__file__).parent / "code" / "config" / "scheduler_config.yaml"
    if not config_file.exists():
        print(f"❌ Configuration file not found: {config_file}")
        print("Please create the scheduler_config.yaml file first!")
        print("Expected location: code/config/scheduler_config.yaml")
        sys.exit(1)
    
    success = run_all_config_tests()
    sys.exit(0 if success else 1)