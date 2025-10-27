"""
Test script to verify the comparison framework works correctly
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.simple_stego import SimpleSTEGOModel
from utils.constants import Constants
from utils.root_config import get_root_config


def test_stego_model():
    """Test STEGO model creation and basic functionality"""
    print("Testing STEGO model...")
    
    try:
        # Create model
        model = SimpleSTEGOModel(num_classes=4, device="cpu")
        print("✓ STEGO model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        expected_shape = (1, 4, 8, 8)  # (batch, classes, height, width)
        if output.shape == expected_shape:
            print("✓ STEGO model forward pass successful")
        else:
            print(f"✗ STEGO model output shape mismatch: {output.shape} vs {expected_shape}")
            return False
        
        # Test prediction
        pred = model.predict(dummy_input)
        if pred.shape == expected_shape:
            print("✓ STEGO model prediction successful")
        else:
            print(f"✗ STEGO model prediction shape mismatch: {pred.shape} vs {expected_shape}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ STEGO model test failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        ROOT, config = get_root_config(__file__, Constants)
        print("✓ Configuration loaded successfully")
        
        # Check required keys
        required_keys = ['vars', 'dirs']
        for key in required_keys:
            if key in config:
                print(f"✓ {key} section found in config")
            else:
                print(f"✗ {key} section missing from config")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False


def test_data_structure():
    """Test data directory structure"""
    print("\nTesting data structure...")
    
    try:
        ROOT, config = get_root_config(__file__, Constants)
        
        # Check test data directories
        test_img_dir = ROOT / config['dirs']['data_dir'] / config['dirs']['test_dir'] / config['dirs']['image_dir']
        test_mask_dir = ROOT / config['dirs']['data_dir'] / config['dirs']['test_dir'] / config['dirs']['mask_dir']
        
        if test_img_dir.exists():
            print("✓ Test images directory exists")
            img_files = list(test_img_dir.glob("*.tif"))
            print(f"  Found {len(img_files)} test images")
        else:
            print("✗ Test images directory not found")
            return False
            
        if test_mask_dir.exists():
            print("✓ Test masks directory exists")
            mask_files = list(test_mask_dir.glob("*.tif"))
            print(f"  Found {len(mask_files)} test masks")
        else:
            print("✗ Test masks directory not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Data structure test failed: {e}")
        return False


def test_model_loading():
    """Test U-Net model loading"""
    print("\nTesting U-Net model loading...")
    
    try:
        ROOT, config = get_root_config(__file__, Constants)
        model_dir = ROOT / config['dirs']['model_dir']
        model_name = config['vars']['model_name']
        model_path = model_dir / model_name
        
        if model_path.exists():
            print(f"✓ U-Net model file found: {model_path}")
            
            # Try to load model
            device = config['vars']['device']
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                print("  CUDA not available, using CPU")
            
            model = torch.load(model_path, map_location=device, weights_only=False)
            print("✓ U-Net model loaded successfully")
            return True
        else:
            print(f"✗ U-Net model file not found: {model_path}")
            return False
            
    except Exception as e:
        print(f"✗ U-Net model loading failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*50)
    print("TESTING COMPARISON FRAMEWORK")
    print("="*50)
    
    tests = [
        test_config_loading,
        test_data_structure,
        test_stego_model,
        test_model_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*50)
    
    if passed == total:
        print("✓ All tests passed! The comparison framework is ready to use.")
        print("\nNext steps:")
        print("1. Run: python train_simple_stego.py")
        print("2. Run: python compare_models.py")
        print("3. Or run: python run_comparison.py")
    else:
        print("✗ Some tests failed. Please check the issues above.")
    
    return passed == total


if __name__ == "__main__":
    main()
