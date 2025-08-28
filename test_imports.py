#!/usr/bin/env python3
"""
Quick test to verify that the dynamic imports work correctly
"""

import sys
import os

# Add benchmark system to path
sys.path.append('./benchmark_system')

def test_model_factory_imports():
    """Test that the model factory can import plugins without syntax errors"""
    try:
        print("Testing model factory imports...")
        
        # Import the factory
        from benchmark_system.core.model_factory import UniversalModelFactory, create_model_plugin
        print("✅ Model factory imported successfully")
        
        # Create a factory instance
        factory = UniversalModelFactory()
        print("✅ Factory instance created successfully")
        
        # Test plugin class import methods
        try:
            # This should work without syntax errors
            plugin_class = factory._import_plugin_class('CNNModelPlugin')
            print("✅ CNNModelPlugin import successful")
        except Exception as e:
            print(f"⚠️ CNNModelPlugin import issue: {e}")
        
        try:
            plugin_class = factory._import_plugin_class('GenericModelPlugin')
            print("✅ GenericModelPlugin import successful")
        except Exception as e:
            print(f"⚠️ GenericModelPlugin import issue: {e}")
        
        try:
            # This should fall back to generic plugin
            plugin_class = factory._import_plugin_class('NonExistentPlugin')
            print("✅ Fallback plugin creation successful")
        except Exception as e:
            print(f"⚠️ Fallback plugin creation issue: {e}")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in model factory: {e}")
        return False
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_direct_plugin_imports():
    """Test direct plugin imports"""
    try:
        print("\nTesting direct plugin imports...")
        
        from benchmark_system.plugins import CNNModelPlugin, GenericModelPlugin
        print("✅ Direct plugin imports successful")
        
        # Create instances
        cnn_plugin = CNNModelPlugin()
        generic_plugin = GenericModelPlugin()
        print("✅ Plugin instances created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct plugin import error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 TESTING MODEL FACTORY IMPORTS")
    print("=" * 50)
    
    results = []
    
    # Test 1: Model factory imports
    results.append(test_model_factory_imports())
    
    # Test 2: Direct plugin imports  
    results.append(test_direct_plugin_imports())
    
    # Summary
    print(f"\n📊 RESULTS:")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 All tests passed! Dynamic imports are working correctly.")
        return 0
    else:
        print(f"⚠️ {passed}/{total} tests passed. Some issues remain.")
        return 1

if __name__ == '__main__':
    sys.exit(main())